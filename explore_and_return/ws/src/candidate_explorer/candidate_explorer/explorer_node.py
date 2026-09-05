"""Return-Aware Utility Frontier Exploration (RA-UFE).

    Map -> Frontiers -> Score frontiers -> Navigate -> Re-evaluate

The strategy lives in ra_ufe.py (pure numpy, no rclpy). This file is the ROS
glue: it owns the /map subscription, the /navigate_to_pose action client, the
TF lookups, the state machine, and the time budget that guarantees the return
leg.

External contract, unchanged from the stub:
  - subscribe /map (nav_msgs/OccupancyGrid) -- published live by slam_toolbox
  - send goals via the /navigate_to_pose action (nav2_msgs/action/NavigateToPose)
  - call the /finish_exploration service (std_srvs/srv/Trigger) when done

"home" is the odom frame's origin, re-derived in the map frame every time it
is needed -- slam_toolbox corrects map->odom as it refines the pose graph, so
a value cached at t=0 goes stale even though the robot never moved in odom.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .ra_ufe import Planner, PlannerConfig


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def quaternion_to_yaw(q: Quaternion) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ExplorerNode(Node):
    def __init__(self) -> None:
        # The stack runs entirely on sim time. eval_runner.sh passes
        # use_sim_time:=true but the README's manual two-terminal recipe does
        # not, and a wall-clock stamp on a goal makes every TF lookup fail
        # with an extrapolation error. Force it so both paths behave.
        super().__init__(
            "candidate_explorer",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        self.declare_parameter("time_limit_s", 5400.0)
        self.declare_parameter("goal_timeout_scale", 3.0)
        self.declare_parameter("goal_timeout_floor_s", 40.0)
        self.declare_parameter("replan_period_s", 4.0)
        self.declare_parameter("switch_margin", 0.25)
        self.declare_parameter("stuck_window_s", 25.0)
        self.declare_parameter("stuck_distance_m", 0.15)
        self.declare_parameter("home_tolerance_m", 0.15)
        self.declare_parameter("max_home_attempts", 10)
        self.declare_parameter("publish_markers", True)

        self.time_limit_s = float(self.get_parameter("time_limit_s").value)
        self.replan_period_s = float(self.get_parameter("replan_period_s").value)
        self.switch_margin = float(self.get_parameter("switch_margin").value)
        self.home_tolerance_m = float(self.get_parameter("home_tolerance_m").value)
        self.max_home_attempts = int(self.get_parameter("max_home_attempts").value)

        self.planner = Planner(PlannerConfig())

        # /map is latched by slam_toolbox (RELIABLE + TRANSIENT_LOCAL, depth
        # 1). Match it, or a late subscriber waits for the next update.
        self.latest_map: OccupancyGrid | None = None
        self.map_count = 0
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self._on_map, map_qos)

        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.finish_client = self.create_client(Trigger, "/finish_exploration")
        self.marker_pub = (
            self.create_publisher(MarkerArray, "/ra_ufe/candidates", 1)
            if self.get_parameter("publish_markers").value
            else None
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.state = "WAITING_FOR_MAP"
        self._goal_handle = None
        self._goal_in_progress = False
        self._goal_seq = 0
        self._goal_target: tuple[float, float] | None = None
        self._goal_utility = -math.inf
        self._goal_started_s = 0.0
        self._goal_deadline_s = math.inf
        self._progress_ref: tuple[float, float] | None = None
        self._progress_ref_s = 0.0
        self._last_plan_s = -math.inf
        self._home_attempts = 0
        self._return_rank = 0
        self._finish_sent = False
        self._no_candidate_cycles = 0
        self._time_pressure = 0.0
        self._stuck_events = 0
        self._escaping = False

        self._fetch_sim_time_limit()
        self.timer = self.create_timer(1.0, self._tick)
        self.get_logger().info("RA-UFE explorer up; waiting for /map.")

    # ------------------------------------------------------------- plumbing

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg
        self.map_count += 1
        if self.map_count == 1:
            info = msg.info
            self.get_logger().info(
                f"/map: {info.width}x{info.height} @ {info.resolution:.3f} m/cell, "
                f"origin ({info.origin.position.x:.2f}, {info.origin.position.y:.2f})"
            )

    def _fetch_sim_time_limit(self) -> None:
        """Ask the simulator for its own time_limit_s so the return reserve
        is right for short debug runs too, not just the 5400 s default."""
        client = self.create_client(GetParameters, "/challenge_sim_node/get_parameters")
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                f"sim node params unavailable; assuming time_limit_s={self.time_limit_s:.0f}"
            )
            return
        req = GetParameters.Request(names=["time_limit_s"])
        future = client.call_async(req)

        def _on_response(fut) -> None:
            try:
                values = fut.result().values
            except Exception as exc:  # noqa: BLE001 - fall back to the default
                self.get_logger().warn(f"could not read sim time_limit_s: {exc}")
                return
            if not values:
                return
            v = values[0]
            limit = float(v.double_value) if v.type == 3 else float(v.integer_value)
            if limit > 0:
                self.time_limit_s = limit
                self.get_logger().info(f"time budget from sim: {limit:.0f} s")

        future.add_done_callback(_on_response)

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def elapsed_s(self) -> float:
        """Sim seconds consumed by the session. challenge_sim is the sole
        /clock authority and starts it at zero, so sim-time "now" already is
        the session's elapsed time -- including the few seconds of Nav2
        bringup that pass before this node exists."""
        return self.now_s()

    def remaining_s(self) -> float:
        return self.time_limit_s - self.elapsed_s()

    # ------------------------------------------------------------------ TF

    def robot_pose_in_map(self) -> tuple[float, float, float] | None:
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        return (
            t.transform.translation.x,
            t.transform.translation.y,
            quaternion_to_yaw(t.transform.rotation),
        )

    def get_home_pose_in_map_frame(self) -> PoseStamped | None:
        try:
            t = self.tf_buffer.lookup_transform("map", "odom", rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"no map->odom transform yet: {ex}")
            return None
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = t.transform.translation.x
        pose.pose.position.y = t.transform.translation.y
        pose.pose.orientation = t.transform.rotation
        return pose

    def distance_to_home_m(self) -> float | None:
        """Our own best estimate of the graded distance_to_home_m: the robot's
        position in the odom frame, whose origin *is* home by definition."""
        try:
            t = self.tf_buffer.lookup_transform("odom", "base_link", rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        return math.hypot(t.transform.translation.x, t.transform.translation.y)

    # -------------------------------------------------------------- actions

    def send_nav_goal(self, pose: PoseStamped, on_done) -> None:
        """Dispatch a goal, tagged with a sequence number.

        Preempting a goal makes Nav2 deliver its CANCELED result *after* the
        replacement has already been sent. Without the tag that late result
        looks like the new goal failing, and the node blacklists the very
        viewpoint it is currently driving to -- which cascades into
        blacklisting everything within seconds.
        """
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("/navigate_to_pose action server not available")
            on_done(False)
            return
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._goal_seq += 1
        seq = self._goal_seq
        self._goal_in_progress = True
        send_future = self.nav_client.send_goal_async(goal)

        def _on_goal_response(fut):
            handle = fut.result()
            if seq != self._goal_seq:
                return  # superseded before it was even accepted
            if not handle.accepted:
                self.get_logger().warn("goal rejected")
                self._goal_in_progress = False
                self._goal_handle = None
                on_done(False)
                return
            self._goal_handle = handle
            result_future = handle.get_result_async()

            def _on_result(fut2):
                if seq != self._goal_seq:
                    return  # stale result from a goal we already preempted
                status = fut2.result().status
                self._goal_in_progress = False
                self._goal_handle = None
                on_done(status == GoalStatus.STATUS_SUCCEEDED)

            result_future.add_done_callback(_on_result)

        send_future.add_done_callback(_on_goal_response)

    def cancel_goal(self) -> None:
        """Abandon the running goal. Bumping the sequence number first means
        its eventual CANCELED result is ignored rather than mistaken for a
        failure of whatever we send next."""
        self._goal_seq += 1
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
        self._goal_in_progress = False
        self._goal_handle = None
        self._goal_target = None
        self._goal_utility = -math.inf
        self._goal_deadline_s = math.inf
        self._escaping = False

    def call_finish_exploration(self) -> None:
        if not self.finish_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/finish_exploration service not available")
            return
        future = self.finish_client.call_async(Trigger.Request())

        def _on_response(fut):
            res = fut.result()
            self.get_logger().info(f"Session result: {res.message}")

        future.add_done_callback(_on_response)

    # ------------------------------------------------------------ machinery

    def _current_view(self):
        msg = self.latest_map
        if msg is None:
            return None
        data = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        return self.planner.build_view(
            data,
            msg.info.resolution,
            msg.info.origin.position.x,
            msg.info.origin.position.y,
        )

    def _goal_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation = yaw_to_quaternion(yaw)
        return pose

    def _dispatch(self, cand, robot) -> None:
        """Send a viewpoint goal, arm its watchdogs, and remember its utility
        so a later re-evaluation can decide whether switching is worth it."""
        yaw = math.atan2(cand.y - robot[1], cand.x - robot[0])
        self._goal_target = (cand.x, cand.y)
        self._goal_utility = cand.utility
        self._goal_started_s = self.elapsed_s()
        budget = self.planner.travel_time(cand.path_cost)
        self._goal_deadline_s = self._goal_started_s + max(
            float(self.get_parameter("goal_timeout_floor_s").value),
            budget * float(self.get_parameter("goal_timeout_scale").value),
        )
        self._progress_ref = (robot[0], robot[1])
        self._progress_ref_s = self._goal_started_s

        self.get_logger().info(
            f"-> viewpoint ({cand.x:.2f}, {cand.y:.2f}) U={cand.utility:+.2f} "
            f"gain={cand.gain} path={cand.path_cost:.2f}m home={cand.home_cost:.2f}m "
            f"revisit={cand.revisit:.0f} src={cand.source} left={self.remaining_s():.0f}s"
        )

        target = (cand.x, cand.y)

        def _on_done(success: bool) -> None:
            if not success:
                self.planner.note_failure(*target)
                self.get_logger().warn(f"viewpoint failed, blacklisting ({target[0]:.2f}, {target[1]:.2f})")
            self._goal_target = None
            self._goal_utility = -math.inf
            self._goal_deadline_s = math.inf

        self.send_nav_goal(self._goal_pose(cand.x, cand.y, yaw), _on_done)

    def _watchdogs(self, robot) -> bool:
        """Abort a goal that has overrun its time budget or stopped making
        progress. Returns True if the goal was cancelled."""
        now = self.elapsed_s()
        if now > self._goal_deadline_s:
            self.get_logger().warn("goal overran its time budget; cancelling")
            if self._goal_target and self.state == "EXPLORING":
                self.planner.note_failure(*self._goal_target)
            if self.state == "RETURNING":
                self._return_rank += 1
            self.cancel_goal()
            return True

        window = float(self.get_parameter("stuck_window_s").value)
        if self._progress_ref is not None and now - self._progress_ref_s > window:
            moved = math.hypot(robot[0] - self._progress_ref[0], robot[1] - self._progress_ref[1])
            if moved < float(self.get_parameter("stuck_distance_m").value):
                self._stuck_events += 1
                self.get_logger().warn(
                    f"no progress ({moved:.2f} m in {window:.0f} s); cancelling "
                    f"[stuck #{self._stuck_events}]"
                )
                if self._goal_target and self.state == "EXPLORING":
                    self.planner.note_failure(*self._goal_target)
                if self.state == "RETURNING":
                    self._return_rank += 1
                self.cancel_goal()
                return True
            self._progress_ref = (robot[0], robot[1])
            self._progress_ref_s = now
        return False

    def _publish_markers(self, cands) -> None:
        if self.marker_pub is None:
            return
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "candidates"
        m.id = 0
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.18
        m.color.a = 0.9
        best = cands[0].utility if cands else 1.0
        worst = cands[-1].utility if cands else 0.0
        span = max(1e-6, best - worst)
        for c in cands:
            t = (c.utility - worst) / span
            m.points.append(Point(x=c.x, y=c.y, z=0.1))
            m.colors.append(ColorRGBA(r=float(1.0 - t), g=float(t), b=0.1, a=0.9))
        arr.markers.append(m)
        self.marker_pub.publish(arr)

    # ------------------------------------------------------------ the ticks

    def _tick(self) -> None:
        if self.state == "WAITING_FOR_MAP":
            if self.latest_map is not None and self.robot_pose_in_map() is not None:
                self.get_logger().info("Got first /map and TF; exploring.")
                self.state = "EXPLORING"
            return

        if self.state == "EXPLORING":
            self._explore_tick()
            return

        if self.state == "RETURNING":
            self._return_tick()
            return

        if self.state == "FINISHING":
            if not self._finish_sent:
                self._finish_sent = True
                d = self.distance_to_home_m()
                self.get_logger().info(
                    f"finishing at {d:.2f} m from home after {self.elapsed_s():.0f} s"
                    if d is not None
                    else "finishing"
                )
                self.call_finish_exploration()
            self.state = "DONE"
            return

    def _explore_tick(self) -> None:
        robot = self.robot_pose_in_map()
        if robot is None:
            return
        self.planner.note_visit(robot[0], robot[1])

        if self._goal_in_progress and self._watchdogs(robot):
            return

        now = self.elapsed_s()
        if self._goal_in_progress and now - self._last_plan_s < self.replan_period_s:
            return
        self._last_plan_s = now

        view = self._current_view()
        if view is None:
            return

        # Two failed goals in a row usually means the footprint is wedged
        # against something. Nav2's spin/back-up recoveries bail out with
        # "Collision Ahead" in that state, so drive to open space explicitly.
        if self._stuck_events >= 2 and not self._escaping:
            target = self.planner.escape_target(view, (robot[0], robot[1]))
            if target is not None:
                self._stuck_events = 0
                self._escaping = True
                self.get_logger().warn(f"wedged; escaping to open space ({target[0]:.2f}, {target[1]:.2f})")

                def _on_escape(success: bool) -> None:
                    self._escaping = False
                    self.get_logger().info(f"escape finished, success={success}")

                self._goal_deadline_s = now + 60.0
                self._progress_ref = (robot[0], robot[1])
                self._progress_ref_s = now
                yaw = math.atan2(target[1] - robot[1], target[0] - robot[0])
                self.send_nav_goal(self._goal_pose(target[0], target[1], yaw), _on_escape)
                return

        home = self.get_home_pose_in_map_frame()
        if home is None:
            return
        home_xy = (home.pose.position.x, home.pose.position.y)

        # Pressure carried from the previous cycle seeds the scoring; it is
        # refreshed below once this cycle's true return cost is known.
        result = self.planner.plan(
            view, (robot[0], robot[1]), robot[2], home_xy, self._time_pressure
        )

        # Hard return gate: if getting home is about to stop being affordable,
        # stop exploring now regardless of how good the frontiers look.
        reserve = self.planner.reserve_for_return(result.robot_home_cost)
        self._time_pressure = reserve / max(1.0, self.remaining_s())
        if reserve >= self.remaining_s():
            self.get_logger().warn(
                f"time reserve reached ({self.remaining_s():.0f} s left, "
                f"need ~{reserve:.0f} s for {result.robot_home_cost:.1f} m home); heading home"
            )
            self.cancel_goal()
            self.state = "RETURNING"
            return

        cands = [c for c in result.candidates if self.planner.affordable(c, self.remaining_s())]
        self._publish_markers(cands)

        if not cands:
            self._no_candidate_cycles += 1
            # Two consecutive empty cycles: either the frontier set is
            # genuinely exhausted or everything left is unreachable or too
            # expensive. Either way this is "good enough", not a failure.
            if self._no_candidate_cycles >= 2 and not self._goal_in_progress:
                self.get_logger().info("no affordable frontiers left; exploration complete")
                self.state = "RETURNING"
            return
        self._no_candidate_cycles = 0

        best = cands[0]
        if not self._goal_in_progress:
            n_probe = sum(1 for c in cands if c.source == "probe")
            self.get_logger().info(
                f"{len(cands)} candidates ({n_probe} probe), pressure={self._time_pressure:.2f}"
            )
            self._dispatch(best, robot)
            return

        # Re-evaluate: only preempt a running goal when the new pick is
        # clearly better, otherwise we thrash and never arrive anywhere.
        if best.utility > self._goal_utility + self.switch_margin:
            if self._goal_target is None or math.hypot(
                best.x - self._goal_target[0], best.y - self._goal_target[1]
            ) > 0.5:
                self.get_logger().info(
                    f"switching goals: U {self._goal_utility:+.2f} -> {best.utility:+.2f}"
                )
                self.cancel_goal()
                self._dispatch(best, robot)

    def _return_tick(self) -> None:
        if self._goal_in_progress:
            robot = self.robot_pose_in_map()
            if robot is not None:
                self._watchdogs(robot)
            return

        d = self.distance_to_home_m()
        if d is not None and d <= self.home_tolerance_m:
            self.get_logger().info(f"home ({d:.2f} m); finishing")
            self.state = "FINISHING"
            return

        if self._home_attempts >= self.max_home_attempts:
            self.get_logger().warn(
                f"giving up on refining the home approach at {d:.2f} m"
                if d is not None
                else "giving up on refining the home approach"
            )
            self.state = "FINISHING"
            return

        # Out of time entirely: stop retrying and report where we are.
        if self.remaining_s() <= 5.0:
            self.state = "FINISHING"
            return

        home = self.get_home_pose_in_map_frame()
        if home is None:
            return
        home_xy = (home.pose.position.x, home.pose.position.y)

        # Re-derived every attempt: map->odom keeps being corrected and the
        # map itself keeps growing, so both home and what is reachable move.
        view = self._current_view()
        robot = self.robot_pose_in_map()
        if view is not None and robot is not None:
            ranked = self.planner.return_targets(view, home_xy, (robot[0], robot[1]))
            if ranked:
                idx = min(self._return_rank, len(ranked) - 1)
                (tx, ty), offset = ranked[idx]
                if offset > 0.02:
                    self.get_logger().info(
                        f"home ({home_xy[0]:.2f}, {home_xy[1]:.2f}) not occupiable; "
                        f"rank {idx} target ({tx:.2f}, {ty:.2f}), {offset:.2f} m off"
                    )
                home.pose.position.x, home.pose.position.y = tx, ty
                home_xy = (tx, ty)

        self._home_attempts += 1
        self.get_logger().info(
            f"return-home attempt {self._home_attempts}"
            + (f" (currently {d:.2f} m away)" if d is not None else "")
        )
        self._goal_target = home_xy
        self._goal_started_s = self.elapsed_s()
        # Short deadline: a refused target must be detected fast so the next
        # ranked candidate gets a turn, rather than one bad cell eating the
        # whole return budget.
        self._goal_deadline_s = self._goal_started_s + max(
            35.0, self.planner.travel_time(d if d is not None else 5.0) * 2.5
        )
        self._progress_ref = None

        def _on_done(success: bool) -> None:
            self._goal_target = None
            self._goal_deadline_s = math.inf
            if not success:
                # Nav2 refused or failed this one; fall through to the next
                # ranked candidate rather than retrying the same cell.
                self._return_rank += 1
            self.get_logger().info(f"return-home goal finished, success={success}")

        self.send_nav_goal(home, _on_done)


def main() -> None:
    rclpy.init()
    node = ExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
