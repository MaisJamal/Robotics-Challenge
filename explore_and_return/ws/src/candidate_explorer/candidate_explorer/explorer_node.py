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
from collections import deque

import numpy as np
import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped, Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Empty, Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .ra_ufe import FREE, Planner, PlannerConfig


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def quaternion_to_yaw(q: Quaternion) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ExplorerNode(Node):
    # How far the active goal may sit from a candidate and still be
    # recognised as the same intent. Just above viewpoint_min_sep_m (0.9 m),
    # the spacing the planner enforces between proposals, so the match picks
    # up the same frontier re-proposed a cell or two over without ever
    # reaching a genuinely different viewpoint.
    INCUMBENT_MATCH_M = 1.0

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
        self.declare_parameter("time_pressure_gain", 1.5)
        self.declare_parameter("goal_timeout_scale", 3.0)
        self.declare_parameter("goal_timeout_floor_s", 40.0)
        self.declare_parameter("replan_period_s", 4.0)
        # Now that both sides of the comparison come from the same cycle
        # (see _incumbent), this is real hysteresis rather than noise
        # tolerance, so it can afford to be stricter about preempting.
        self.declare_parameter("switch_margin", 0.4)
        self.declare_parameter("stuck_window_s", 25.0)
        self.declare_parameter("stuck_distance_m", 0.15)
        self.declare_parameter("home_tolerance_m", 0.15)
        self.declare_parameter("max_home_attempts", 40)
        self.declare_parameter("finish_reserve_s", 60.0)
        self.declare_parameter("home_attempt_gap_s", 15.0)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("unstick_duration_s", 5.0)

        self.time_limit_s = float(self.get_parameter("time_limit_s").value)
        self.replan_period_s = float(self.get_parameter("replan_period_s").value)
        self.switch_margin = float(self.get_parameter("switch_margin").value)
        self.home_tolerance_m = float(self.get_parameter("home_tolerance_m").value)
        self.max_home_attempts = int(self.get_parameter("max_home_attempts").value)
        self.finish_reserve_s = float(self.get_parameter("finish_reserve_s").value)
        self.home_attempt_gap_s = float(self.get_parameter("home_attempt_gap_s").value)

        self.planner = Planner(PlannerConfig(
            time_pressure_gain=float(self.get_parameter("time_pressure_gain").value),
        ))

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

        # Direct velocity control, used only to break a wedge. When the
        # footprint ends up inside a costmap obstacle, NavFn has no legal
        # start cell, so *every* goal fails to plan -- including the escape
        # goal -- and Nav2's own recoveries abort with "Collision Ahead".
        # Nothing routed through the action interface can help there.
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.clear_local = self.create_client(Empty, "/local_costmap/clear_entirely_local_costmap")
        self.clear_global = self.create_client(Empty, "/global_costmap/clear_entirely_global_costmap")
        self.unstick_duration_s = float(self.get_parameter("unstick_duration_s").value)
        self._unstick_until_s = -math.inf
        self._unstick_active = False
        self._unstick_driving = False
        self._unstick_wait_until_s = -math.inf
        self._unstick_dir = 1.0
        self._unstick_count = 0
        self.unstick_timer = self.create_timer(0.1, self._unstick_tick)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.state = "WAITING_FOR_MAP"
        self._goal_handle = None
        self._goal_in_progress = False
        self._goal_source = None
        self._goal_seq = 0
        self._nav_outstanding = set()
        self._return_unverified_since = None
        self._goal_target: tuple[float, float] | None = None
        self._goal_started_s = 0.0
        self._goal_deadline_s = math.inf
        self._progress_ref: tuple[float, float] | None = None
        self._progress_ref_s = 0.0
        self._last_plan_s = -math.inf
        self._home_attempts = 0
        self._return_rank = 0
        self._last_return_success = False
        self._best_home_d = math.inf
        self._last_home_attempt_s = -math.inf
        self._wait_ticks = 0
        self._travelled_m = 0.0
        self._free_history: deque = deque()
        self._last_odom_xy: tuple[float, float] | None = None
        self._finish_sent = False
        self._no_candidate_cycles = 0
        self._time_pressure = 0.0
        self._stuck_events = 0

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

    # ------------------------------------------------------------- unstick

    def unsticking(self) -> bool:
        return self._unstick_active

    def stop_unstick(self) -> None:
        if self._unstick_driving:
            self.cmd_pub.publish(Twist())
        self._unstick_active = False
        self._unstick_driving = False
        self._unstick_until_s = -math.inf

    def begin_unstick(self, reason: str) -> None:
        """Wait for navigation to stop before a bounded reverse-and-turn."""
        self.cancel_goal()
        self._unstick_count += 1
        self._unstick_dir = -self._unstick_dir
        self._unstick_active = True
        self._unstick_driving = False
        self._unstick_wait_until_s = self.elapsed_s() + 5.0
        self._stuck_events = 0
        self.get_logger().warn(f"unstick #{self._unstick_count} ({reason})")

    def _unstick_tick(self) -> None:
        if not self.unsticking():
            return
        now = self.elapsed_s()
        if self.remaining_s() <= self.finish_reserve_s or self.state not in ("EXPLORING", "RETURNING"):
            self.stop_unstick()
            return
        if self._nav_outstanding:
            if now >= self._unstick_wait_until_s:
                self.get_logger().warn("recovery skipped: navigation has not stopped")
                self.stop_unstick()
            return
        if not self._unstick_driving:
            self._unstick_driving = True
            self._unstick_until_s = now + self.unstick_duration_s
            for client in (self.clear_local, self.clear_global):
                if client.service_is_ready():
                    client.call_async(Empty.Request())
        if now >= self._unstick_until_s:
            self.stop_unstick()
            return
        cmd = Twist()
        cmd.linear.x = -0.12
        cmd.angular.z = 0.9 * self._unstick_dir
        self.cmd_pub.publish(cmd)

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

    def _odom_xy(self) -> tuple[float, float] | None:
        try:
            t = self.tf_buffer.lookup_transform("odom", "base_link", rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        return t.transform.translation.x, t.transform.translation.y

    def _update_measured_speed(self) -> None:
        """Accumulate path length in the odom frame and feed the planner the
        robot's real average speed.

        Odom rather than map: slam_toolbox's pose-graph corrections make the
        map-frame pose jump, and those jumps are not travel.
        """
        xy = self._odom_xy()
        if xy is None:
            return
        if self._last_odom_xy is not None:
            step = math.hypot(xy[0] - self._last_odom_xy[0], xy[1] - self._last_odom_xy[1])
            if step < 1.0:  # ignore any implausible jump
                self._travelled_m += step
        self._last_odom_xy = xy
        elapsed = self.elapsed_s()
        # Wait for enough motion that the average is not dominated by bringup.
        if self._travelled_m > 2.0 and elapsed > 60.0:
            self.planner.measured_speed = self._travelled_m / elapsed

    def distance_to_home_m(self) -> float | None:
        """Our own best estimate of the graded distance_to_home_m: the robot's
        position in the odom frame, whose origin *is* home by definition."""
        try:
            t = self.tf_buffer.lookup_transform("odom", "base_link", rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        return math.hypot(t.transform.translation.x, t.transform.translation.y)

    # -------------------------------------------------------------- actions

    def send_nav_goal(self, pose: PoseStamped, on_done) -> bool:
        """Dispatch a goal, tagged with a sequence number. Returns whether the
        goal was actually handed to Nav2.

        Preempting a goal makes Nav2 deliver its CANCELED result *after* the
        replacement has already been sent. Without the tag that late result
        looks like the new goal failing, and the node blacklists the very
        viewpoint it is currently driving to -- which cascades into
        blacklisting everything within seconds.
        """
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            # Infrastructure, not a bad viewpoint. Reporting this through
            # on_done would blacklist a perfectly good frontier -- and during
            # Nav2 bringup it would burn through several before the server
            # even appears. Leave the target untouched and retry next tick.
            self.get_logger().error("/navigate_to_pose action server not available; will retry")
            self._goal_in_progress = False
            self._goal_target = None
            self._goal_deadline_s = math.inf
            return False
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._goal_seq += 1
        seq = self._goal_seq
        self._goal_in_progress = True
        self._nav_outstanding.add(seq)
        send_future = self.nav_client.send_goal_async(goal)

        def _on_goal_response(fut):
            try:
                handle = fut.result()
            except Exception as exc:
                # The server may have accepted the request; do not grant
                # direct velocity ownership when its state is unknown.
                self.get_logger().error(f"goal acceptance unavailable: {exc}")
                return
            if not handle.accepted:
                self._nav_outstanding.discard(seq)
                if seq == self._goal_seq:
                    self._goal_in_progress = False
                    self._goal_handle = None
                    on_done(False)
                return
            result_future = handle.get_result_async()

            def _on_result(fut2):
                try:
                    status = fut2.result().status
                except Exception as exc:
                    self.get_logger().error(f"goal result unavailable: {exc}")
                    return
                self._nav_outstanding.discard(seq)
                if seq != self._goal_seq:
                    return
                self._goal_in_progress = False
                self._goal_handle = None
                on_done(status == GoalStatus.STATUS_SUCCEEDED)

            result_future.add_done_callback(_on_result)
            if seq != self._goal_seq:
                # Cancellation can happen before the acceptance arrives.
                handle.cancel_goal_async()
            else:
                self._goal_handle = handle

        send_future.add_done_callback(_on_goal_response)
        return True

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
        self._goal_deadline_s = math.inf

    def call_finish_exploration(self) -> None:
        if not self.finish_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/finish_exploration service not available")
            self._finish_sent = False
            return
        future = self.finish_client.call_async(Trigger.Request())

        def _on_response(fut):
            try:
                res = fut.result()
                self.get_logger().info(f"Session result: {res.message}")
                if res.success:
                    self.state = "DONE"
                else:
                    self._finish_sent = False
            except Exception as exc:
                self._finish_sent = False
                self.get_logger().error(f"finish request failed: {exc}")

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

    def _coverage_plateaued(self, view) -> bool:
        """Has exploring stopped paying for itself?

        Uses known-free cell count as the proxy for coverage, since the real
        coverage_fraction is ground truth we never see. Without this the
        robot keeps chasing 8-cell frontiers long after the map is
        effectively complete, and ends the session far from home -- which
        costs far more than the coverage it is still chasing.
        """
        cfg = self.planner.cfg
        now = self.elapsed_s()
        free = int((view.codes == FREE).sum())
        self._free_history.append((now, free))
        while self._free_history and now - self._free_history[0][0] > cfg.plateau_window_s:
            self._free_history.popleft()

        if now < cfg.plateau_min_elapsed_s or len(self._free_history) < 5:
            return False
        oldest_t, oldest_free = self._free_history[0]
        if now - oldest_t < cfg.plateau_window_s * 0.8 or oldest_free <= 0:
            return False
        return (free - oldest_free) / float(oldest_free) < cfg.plateau_min_growth

    def _goal_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation = yaw_to_quaternion(yaw)
        return pose

    def _dispatch(self, cand, robot) -> None:
        """Send a viewpoint goal and arm its watchdogs.

        The utility is deliberately not remembered: a later re-evaluation
        re-scores this goal against its own cycle's candidates instead (see
        _incumbent), because utilities are not comparable across cycles."""
        yaw = math.atan2(cand.y - robot[1], cand.x - robot[0])
        self._goal_source = cand.source
        self._goal_target = (cand.x, cand.y)
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
        # Session deadlines outrank both navigation and direct recovery.
        if self.state not in ("DONE", "FINISHING") and self.remaining_s() <= self.finish_reserve_s:
            self.stop_unstick()
            self.cancel_goal()
            self.state = "FINISHING"
        if self.state == "EXPLORING" and self.elapsed_s() >= self.time_limit_s * self.planner.cfg.explore_budget_fraction:
            self.stop_unstick()
            self.cancel_goal()
            self.state = "RETURNING"
        # While the unstick manoeuvre owns /cmd_vel, stay out of its way:
        # dispatching a goal here would have Nav2 and the manoeuvre fighting
        # over the same topic.
        if self.unsticking():
            return

        if self.state == "WAITING_FOR_MAP":
            # Nav2 takes ~40 s to bring bt_navigator through its lifecycle to
            # active. The README says to start this node "after a few seconds",
            # which is not enough, so wait for the action server rather than
            # relying on how long the operator happened to pause. Without this
            # the first goals are dispatched into a void.
            nav_ready = self.nav_client.server_is_ready()
            have_map = self.latest_map is not None
            have_tf = self.robot_pose_in_map() is not None
            if nav_ready and have_map and have_tf:
                self.get_logger().info("Got /map, TF and /navigate_to_pose; exploring.")
                self.state = "EXPLORING"
                return
            self._wait_ticks += 1
            if self._wait_ticks % 5 == 0:
                missing = [
                    name
                    for name, ok in (
                        ("/map", have_map),
                        ("TF map->base_link", have_tf),
                        ("/navigate_to_pose", nav_ready),
                    )
                    if not ok
                ]
                self.get_logger().info(f"waiting for: {', '.join(missing)}")
            return

        if self.state == "EXPLORING":
            self._explore_tick()
            return

        if self.state == "RETURNING":
            self._return_tick()
            return

        if self.state == "FINISHING":
            if self._nav_outstanding:
                return
            if not self._finish_sent:
                self._finish_sent = True
                d = self.distance_to_home_m()
                self.get_logger().info(
                    f"finishing at {d:.2f} m from home (best {self._best_home_d:.2f} m, "
                    f"{self._home_attempts} attempts) after {self.elapsed_s():.0f} s"
                    if d is not None
                    else "finishing"
                )
                self.call_finish_exploration()
            return

    def _explore_tick(self) -> None:
        robot = self.robot_pose_in_map()
        if robot is None:
            return
        # The planner ages its failure memory against sim time, so it has to
        # be told what time it is before anything reads that memory.
        self.planner.now_s = self.elapsed_s()
        self._update_measured_speed()
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

        # Two failed goals in a row usually means the footprint is wedged.
        # Break it with direct velocity control first; a navigation goal
        # cannot help while the start pose itself is unplannable.
        if self._stuck_events >= 2:
            self.begin_unstick("exploration stalled")
            return
        home = self.get_home_pose_in_map_frame()
        if home is None:
            return
        home_xy = (home.pose.position.x, home.pose.position.y)

        # The planner computes pressure using the current return field and
        # elapsed budget before ranking this cycle's candidates.
        home_distance = self.distance_to_home_m()
        result = self.planner.plan(
            view, (robot[0], robot[1]), robot[2], home_xy, self._time_pressure,
            at_home=(home_distance is not None and home_distance <= self.home_tolerance_m),
            elapsed_s=now, time_limit_s=self.time_limit_s,
        )
        #######
        if result.est_coverage >= self.planner.cfg.coverage_target:
            self.cancel_goal()
            self.state = "RETURNING"
            return
        #######
        # Hard return gate: if getting home is about to stop being affordable,
        # stop exploring now regardless of how good the frontiers look.
        if not result.home_reachable:
            # A bootstrap probe was budgeted as an out-and-back observation
            # maneuver. Let it gather evidence until arrival or its watchdog,
            # rather than canceling it because the startup map is still sparse.
            if self._goal_in_progress and self._goal_source == "probe":
                self._return_unverified_since = None
                return
            if self._return_unverified_since is None:
                self._return_unverified_since = now
                self.get_logger().warn("return route unverified; pausing new exploration goals")
            # Allow map updates and an already-running bootstrap probe to
            # establish free space before committing to return recovery.
            if now - self._return_unverified_since >= 10.0:
                self.get_logger().warn(
                    f"returning: route absent at coarse and full resolution; "
                    f"robot=({robot[0]:.2f}, {robot[1]:.2f}), "
                    f"home=({home_xy[0]:.2f}, {home_xy[1]:.2f}), "
                    f"map_update={self.map_count}, left={self.remaining_s():.0f}s"
                )
                self.cancel_goal()
                self.state = "RETURNING"
            return
        self._return_unverified_since = None
        reserve = self.planner.reserve_for_return(result.robot_home_cost)
        self._time_pressure = self.planner.time_pressure(
            result.robot_home_cost, now, self.time_limit_s
        )
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
            self.get_logger().warn(
                f"no candidates (cycle {self._no_candidate_cycles}): "
                f"{result.n_clusters} frontier clusters, {result.n_proposals} proposals, "
                f"dropped={result.drops or '{}'}, "
                f"{len(result.candidates)} scored before affordability, "
                f"{self.remaining_s():.0f}s left"
            )
            if self._no_candidate_cycles >= 2 and not self._goal_in_progress:
                self.get_logger().info("no affordable frontiers left; exploration complete")
                self.state = "RETURNING"
            return
        self._no_candidate_cycles = 0

        best = cands[0]
        if not self._goal_in_progress:
            n_probe = sum(1 for c in cands if c.source == "probe")
            self.get_logger().info(
                f"{len(cands)} candidates ({n_probe} probe), est_cov="
                f"{result.est_coverage:.1%}, pressure={self._time_pressure:.2f}"
            )
            self._dispatch(best, robot)
            return

        # Re-evaluate: only preempt a running goal when the new pick is
        # clearly better, otherwise we thrash and never arrive anywhere.
        #
        # The incumbent is re-scored against *this* cycle's candidate set
        # rather than compared to the utility stored at dispatch. Utilities
        # are normalised per cycle, so a stored one was computed on a
        # different basis; a candidate merely entering or leaving the set
        # could shift the numbers past switch_margin with nothing real having
        # changed, and every replan_period_s was another chance to flip. That
        # is the ping-pong, not a genuine disagreement about where to go.
        incumbent = self._incumbent(cands)
        if incumbent is None:
            # The active goal is not in this cycle's set -- typically because
            # we are now inside min_goal_distance_m of it, which is exactly
            # when abandoning it would be worst. Nothing here can compare
            # fairly, so hold; the deadline and stuck watchdogs still apply.
            return

        if best.utility > incumbent.utility + self.switch_margin:
            if self._goal_target is None or math.hypot(
                best.x - self._goal_target[0], best.y - self._goal_target[1]
            ) > 0.5:
                self.get_logger().info(
                    f"switching goals: U {incumbent.utility:+.2f} -> {best.utility:+.2f} "
                    f"(gain {incumbent.gain}->{best.gain}, "
                    f"path {incumbent.path_cost:.1f}->{best.path_cost:.1f}m)"
                )
                self.cancel_goal()
                self._dispatch(best, robot)

    def _incumbent(self, cands):
        """This cycle's candidate for the goal already being pursued.

        Matched by position within INCUMBENT_MATCH_M rather than by identity:
        candidates are regenerated from scratch every cycle, so the same
        frontier reappears as a different cell as the map fills in. Returns
        None when the active goal has no counterpart this cycle.
        """
        if self._goal_target is None:
            return None
        gx, gy = self._goal_target
        best = None
        best_d = self.INCUMBENT_MATCH_M
        for c in cands:
            d = math.hypot(c.x - gx, c.y - gy)
            if d < best_d:
                best, best_d = c, d
        return best

    def _return_tick(self) -> None:
        if self._goal_in_progress:
            robot = self.robot_pose_in_map()
            if robot is not None:
                self._watchdogs(robot)
            return

        self.planner.now_s = self.elapsed_s()
        # Budget first: an escape loop must never be able to run out the
        # clock. One run ping-ponged on the same unreachable escape target
        # until the session died, because this check sat below it.
        if self.remaining_s() <= self.finish_reserve_s:
            d_now = self.distance_to_home_m()
            self.get_logger().warn(
                f"return budget exhausted at {d_now:.2f} m from home"
                if d_now is not None
                else "return budget exhausted"
            )
            self.state = "FINISHING"
            return

        d = self.distance_to_home_m()
        if d is not None and d <= self.home_tolerance_m:
            self.get_logger().info(f"home ({d:.2f} m); finishing")
            self.state = "FINISHING"
            return

        # Wedged on the way home: drive to open space before trying again.
        # Without this the return just re-issues goals a stuck robot cannot
        # act on, and Nav2's own recoveries have already given up by then.
        if self._stuck_events >= 2:
            self.begin_unstick(f"return stalled at {d:.2f} m" if d is not None else "return stalled")
            return
        if d is not None:
            self._best_home_d = min(self._best_home_d, d)

        # Hard cap purely as a loop guard, not as a policy. The real stop
        # condition is the budget check at the top of this function: a
        # previous run ended 0.29915 m from home -- inside the 0.3 m tolerance
        # by less than one map cell -- because it hit a 10-attempt cap with
        # 2896 s of budget still unspent. Time is the constraint, not tries.
        if self._home_attempts >= self.max_home_attempts:
            self.get_logger().warn(
                f"giving up on refining the home approach at {d:.2f} m after "
                f"{self._home_attempts} attempts"
                if d is not None
                else "giving up on refining the home approach"
            )
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
        aim_xy = home_xy
        if self._last_return_success and d is not None and d > self.home_tolerance_m and robot is not None:
            # Nav2 reports success anywhere inside xy_goal_tolerance (0.25 m),
            # so aiming *at* home reliably stops short of it -- and re-sending
            # the same goal just succeeds instantly without moving. Aim past
            # home along the approach direction so that stopping tolerance
            # brackets home instead of ending in front of it.
            vx, vy = home_xy[0] - robot[0], home_xy[1] - robot[1]
            norm = math.hypot(vx, vy)
            if norm > 1e-3:
                push = min(0.35, max(0.15, d))
                aim_xy = (home_xy[0] + vx / norm * push, home_xy[1] + vy / norm * push)
                self.get_logger().info(
                    f"stopped {d:.2f} m short; aiming {push:.2f} m past home at "
                    f"({aim_xy[0]:.2f}, {aim_xy[1]:.2f})"
                )
        if view is not None and robot is not None:
            ranked = self.planner.return_targets(view, aim_xy, (robot[0], robot[1]))
            if ranked:
                # Cycle rather than clamp: with up to 40 attempts, clamping
                # would pin every later try on the worst candidate.
                idx = self._return_rank % len(ranked)
                (tx, ty), offset = ranked[idx]
                if offset > 0.02:
                    self.get_logger().info(
                        f"aim ({aim_xy[0]:.2f}, {aim_xy[1]:.2f}) not occupiable; "
                        f"rank {idx} target ({tx:.2f}, {ty:.2f}), {offset:.2f} m off"
                    )
                home.pose.position.x, home.pose.position.y = tx, ty
                home_xy = (tx, ty)

        # Space attempts out in sim time. When the target is already inside
        # Nav2's goal tolerance it returns success instantly, and without this
        # the whole attempt budget evaporates in under a second with the robot
        # motionless -- 40 attempts in 0.6 s in one run.
        if self.elapsed_s() - self._last_home_attempt_s < self.home_attempt_gap_s:
            return
        self._last_home_attempt_s = self.elapsed_s()

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
        # Arm the no-progress watchdog for the return too. It used to be
        # disabled here (_progress_ref = None), so a robot wedged on the way
        # home was never detected: one run burned 40 identical attempts and
        # 3200 s of budget frozen at 4.14 m, grinding up 90k collisions.
        if robot is not None:
            self._progress_ref = (robot[0], robot[1])
            self._progress_ref_s = self._goal_started_s
        else:
            self._progress_ref = None

        def _on_done(success: bool) -> None:
            self._goal_target = None
            self._goal_deadline_s = math.inf
            self._last_return_success = success
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
        node.stop_unstick()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
