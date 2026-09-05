"""Your task: replace the EXPLORING state's body (marked below) with real
exploration + coverage logic. Everything else in this file is infrastructure
you're free to use as-is, restructure, or throw away entirely — the only
external contract that matters is:

  - subscribe /map (nav_msgs/OccupancyGrid) — published live by slam_toolbox
  - send goals via the /navigate_to_pose action (nav2_msgs/action/NavigateToPose)
  - call the /finish_exploration service (std_srvs/srv/Trigger) when done

See README.md for the full brief and scoring rubric.

Note: "home" is the odom frame's origin (see get_home_pose_in_map_frame
below) — re-derive its pose in the map frame every time you need it, don't
cache it from t=0. slam_toolbox corrects the map->odom transform as it
refines the pose graph, so where home sits in the map frame can drift even
though the robot itself never moves in the odom frame.
"""
from __future__ import annotations

import math

import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class ExplorerNode(Node):
    def __init__(self) -> None:
        super().__init__("candidate_explorer")

        # /map is published by slam_toolbox with a latched QoS
        # (RELIABLE + TRANSIENT_LOCAL, depth 1). Match it so we (a) are
        # compatible with the publisher at all and (b) get the last map
        # immediately on subscribe instead of waiting for the next update.
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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.state = "WAITING_FOR_MAP"
        self._goal_in_progress = False

        self.timer = self.create_timer(1.0, self._tick)

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg
        self.map_count += 1
        if self.map_count == 1:
            info = msg.info
            self.get_logger().info(
                f"/map: {info.width}x{info.height} @ {info.resolution:.3f} m/cell, "
                f"origin ({info.origin.position.x:.2f}, {info.origin.position.y:.2f})"
            )

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        """Map-frame metres -> (col, row) index into latest_map.data, or None
        if the point falls outside the current grid."""
        if self.latest_map is None:
            return None
        info = self.latest_map.info
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)
        if 0 <= col < info.width and 0 <= row < info.height:
            return col, row
        return None

    def cell_to_world(self, col: int, row: int) -> tuple[float, float]:
        """(col, row) -> map-frame metres at the centre of that cell."""
        info = self.latest_map.info
        return (
            info.origin.position.x + (col + 0.5) * info.resolution,
            info.origin.position.y + (row + 0.5) * info.resolution,
        )

    def cell_value(self, col: int, row: int) -> int:
        """Occupancy at (col, row): -1 unknown, 0 free, 100 occupied."""
        return self.latest_map.data[row * self.latest_map.info.width + col]

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

    def send_nav_goal(self, pose: PoseStamped, on_done) -> None:
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("/navigate_to_pose action server not available")
            return
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._goal_in_progress = True
        send_future = self.nav_client.send_goal_async(goal)

        def _on_goal_response(fut):
            handle = fut.result()
            if not handle.accepted:
                self.get_logger().warn("goal rejected")
                self._goal_in_progress = False
                on_done(False)
                return
            result_future = handle.get_result_async()

            def _on_result(fut2):
                status = fut2.result().status
                self._goal_in_progress = False
                on_done(status == GoalStatus.STATUS_SUCCEEDED)

            result_future.add_done_callback(_on_result)

        send_future.add_done_callback(_on_goal_response)

    def call_finish_exploration(self) -> None:
        if not self.finish_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/finish_exploration service not available")
            return
        future = self.finish_client.call_async(Trigger.Request())

        def _on_response(fut):
            res = fut.result()
            self.get_logger().info(f"Session result: {res.message}")

        future.add_done_callback(_on_response)

    def _tick(self) -> None:
        if self.state == "WAITING_FOR_MAP":
            if self.latest_map is not None:
                self.get_logger().info("Got first /map.")
                self.state = "EXPLORING"
            return

        if self.state == "EXPLORING":
            if self._goal_in_progress:
                return

            def GoToPose(x,y,yaw,then):
                pose = PoseStamped()
                pose.header.frame_id = "odom"
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.orientation = yaw_to_quaternion(yaw)
                self.send_nav_goal(pose, then)

            def AfterFirst(success: bool) -> None:
                self.get_logger().info(f"called after reaching first pose. Success:{success}")
                GoToPose(2.5, 1.0, 0.8, AfterSecond)    # inside the room

            def AfterSecond(success: bool) -> None:
                self.get_logger().info(f"called after reaching second pose. Success:{success}")
                self.state = "RETURNING"
            # ============================================================
            # TODO: replace this. Placeholder baseline: send exactly one
            # goal, a fixed distance straight ahead in the odom frame, then
            # declare exploration "done" regardless of actual coverage. This
            # proves map subscription + action client + TF + finish service
            # all work end-to-end. It does not meaningfully explore anything.
            # ============================================================
            
            GoToPose(1.0, 0.0, 0.0, AfterFirst)

            def _on_done(success: bool) -> None:
                self.get_logger().info(f"placeholder exploration goal finished, success={success}")
                self.state = "RETURNING"

            # self.send_nav_goal(pose, _on_done)
            
            self.get_logger().info(f"testing ... ")


            return

        if self.state == "RETURNING":
            if self._goal_in_progress:
                return
            home = self.get_home_pose_in_map_frame()
            if home is None:
                return  # try again next tick

            def _on_done(success: bool) -> None:
                self.get_logger().info(f"return-home goal finished, success={success}")
                self.state = "FINISHING"

            self.send_nav_goal(home, _on_done)
            return

        if self.state == "FINISHING":
            self.state = "DONE"
            self.call_finish_exploration()
            return

        # DONE: nothing left to do.


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
