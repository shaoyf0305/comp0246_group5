import rclpy
from rclpy.node import Node
import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray
import os

from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.duration import Duration
import matplotlib.pyplot as plt

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import PyKDL
from youbot_kinematics.urdf import treeFromUrdfModel
from youbot_kinematics.urdf_parser import URDF

from ament_index_python.packages import get_package_share_directory

import yaml


class TrajDemo(Node):
    def __init__(self):
        super().__init__("traj_demo")

        self.start_time = None

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 5
        )
        self.traj_publisher = self.create_publisher(
            JointTrajectory, "/franka_arm_controller/joint_trajectory", 5
        )

        # --- Load robot model from URDF ---
        self.declare_parameter("urdf_package", "franka_description")
        self.urdf_package = (
            self.get_parameter("urdf_package").get_parameter_value().string_value
        )
        self.urdf_package_path = get_package_share_directory(self.urdf_package)
        self.declare_parameter("urdf_path_in_package", "urdfs/fr3.urdf")
        self.urdf_path_in_package = (
            self.get_parameter("urdf_path_in_package").get_parameter_value().string_value
        )
        self.urdf_name_path = os.path.join(
            self.urdf_package_path, self.urdf_path_in_package
        )

        self.get_logger().info(f"Loading robot from URDF: {self.urdf_name_path}")
        robot = URDF.from_xml_file(self.urdf_name_path)
        (ok, self.kine_tree) = treeFromUrdfModel(robot)
        if not ok:
            raise RuntimeError("Couldn't load URDF into KDL tree successfully")

        default_base = robot.get_root() if hasattr(robot, "get_root") else "base"
        self.declare_parameter("base_link", default_base)
        self.declare_parameter("ee_link", "fr3_link8")
        self.base_link = self.get_parameter("base_link").get_parameter_value().string_value
        self.ee_link = self.get_parameter("ee_link").get_parameter_value().string_value

        self.kine_chain = self.kine_tree.getChain(self.base_link, self.ee_link)
        self.NJoints = self.kine_chain.getNrOfJoints()
        self.current_joint_position = PyKDL.JntArray(self.NJoints)
        self.current_joint_velocity = PyKDL.JntArray(self.NJoints)

        self.ik_solver = PyKDL.ChainIkSolverPos_LMA(self.kine_chain)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Load trajectory config ---
        self.declare_parameter("traj_cfg_pkg", "trajectory_demo")
        self.traj_cfg_pkg = (
            self.get_parameter("traj_cfg_pkg").get_parameter_value().string_value
        )
        self.traj_cfg_pkg_path = get_package_share_directory(self.traj_cfg_pkg)
        self.declare_parameter("traj_cfg_path_within_pkg", "cfg/traj_waypoints.yaml")
        self.traj_cfg_path_within_pkg = (
            self.get_parameter("traj_cfg_path_within_pkg")
            .get_parameter_value()
            .string_value
        )
        self.traj_cfg_path = os.path.join(
            self.traj_cfg_pkg_path, self.traj_cfg_path_within_pkg
        )

        with open(self.traj_cfg_path) as stream:
            self.traj_cfg = yaml.safe_load(stream)

        assert type(self.traj_cfg["joint_names"]) == list
        assert len(self.traj_cfg["joint_names"]) == self.NJoints

        self.created_traj = False

        # --- Data buffers ---
        self.joint_history = []
        self.joint_vel_history = []
        self.ee_history = []  # stores [x, y, z, roll, pitch, yaw]

    # =====================================================
    # CALLBACKS & HELPERS
    # =====================================================

    def joint_state_callback(self, msg: JointState):
        """Callback for /joint_states."""
        for i in range(len(msg.name)):
            self.current_joint_position[i] = msg.position[i]
            self.current_joint_velocity[i] = msg.velocity[i]

        joint_pos = self.kdl_to_np(self.current_joint_position)
        joint_vel = self.kdl_to_np(self.current_joint_velocity)
        self.joint_history.append(np.array(joint_pos, copy=True))
        self.joint_vel_history.append(np.array(joint_vel, copy=True))

        # --- NEW: record EE position + orientation (RPY) ---
        ee_pos, ee_rpy = self.get_ee_pose_ros()
        self.ee_history.append(np.concatenate((ee_pos, ee_rpy)))

        if not self.created_traj:
            joint_traj = self.create_traj()
            self.traj_publisher.publish(joint_traj)
            self.created_traj = True
            self.get_logger().info("Published joint trajectory.")

    def kdl_to_np(self, data: PyKDL.JntArray) -> NDArray:
        is_1d = data.columns() == 1
        np_shape = (data.rows(), data.columns()) if not is_1d else (data.rows(),)
        mat = np.zeros(np_shape)
        for i in range(data.rows()):
            if not is_1d:
                for j in range(data.columns()):
                    mat[i, j] = data[i, j]
            else:
                mat[i] = data[i]
        return mat

    # --- UPDATED: Get both position and orientation ---
    def get_ee_pose_ros(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get EE position + orientation (RPY) using TF2."""
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.base_link, self.ee_link, now, timeout=Duration(seconds=1.0)
            )

            pos = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ])

            quat = trans.transform.rotation
            q = [quat.x, quat.y, quat.z, quat.w]
            rot = PyKDL.Rotation.Quaternion(*q)
            rpy = np.array(rot.GetRPY())  # radians

            return pos, rpy
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return np.full(3, np.nan), np.full(3, np.nan)

    def get_joint_pos(
        self,
        current_angles: Tuple[float],
        target_position: Tuple[float],
        target_orientation: Optional[Tuple[float]] = None,
    ) -> Tuple[float]:
        assert len(target_position) == 3
        assert target_orientation is None or len(target_orientation) == 3

        pos = PyKDL.Vector(*target_position)
        if target_orientation is not None:
            rot = PyKDL.Rotation.RPY(*target_orientation)
            goal_pose = PyKDL.Frame(rot, pos)
        else:
            goal_pose = PyKDL.Frame(pos)

        seed_array = PyKDL.JntArray(self.NJoints)
        for i in range(self.NJoints):
            seed_array[i] = current_angles[i]

        result_angles = PyKDL.JntArray(self.NJoints)
        if self.ik_solver.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            return list(result_angles)
        else:
            raise RuntimeError("IK solver failed.")

    # =====================================================
    # TRAJECTORY CREATION
    # =====================================================

    def create_traj(self) -> JointTrajectory:
        self.start_time = self.get_clock().now()
        cartesian_waypoints = self.traj_cfg["waypoints"]["cartesion"]
        cur_joint_pos = self.kdl_to_np(self.current_joint_position)

        # --- Add current EE pose as first waypoint ---
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.base_link, self.ee_link, now, timeout=Duration(seconds=1.0)
            )
            current_pos = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ]
            cartesian_waypoints.insert(0, {"position": current_pos, "time": 0.0})
        except Exception as e:
            self.get_logger().warn(f"Could not get current EE pose: {e}")

        joint_traj = JointTrajectory()
        joint_traj.header.frame_id = self.base_link
        joint_traj.header.stamp = self.get_clock().now().to_msg()
        joint_traj.joint_names = self.traj_cfg["joint_names"]

        time_since_start = 0.0
        for i, waypoint in enumerate(cartesian_waypoints):
            pos = waypoint["position"]
            time_sec = float(waypoint["time"])
            target_orientation = waypoint.get("orientation", None)
            time_since_start += time_sec

            cur_joint_pos = self.get_joint_pos(cur_joint_pos, pos, target_orientation)

            pt = JointTrajectoryPoint()
            pt.positions = cur_joint_pos
            pt.velocities = [0.0] * self.NJoints
            pt.accelerations = [0.0] * self.NJoints
            pt.time_from_start = Duration(seconds=time_since_start).to_msg()

            joint_traj.points.append(pt)

        self.get_logger().info(f"Planned trajectory duration: {time_since_start:.2f}s")
        return joint_traj

    # =====================================================
    # PLOTTING
    # =====================================================

    def plot_joint_traj(self):
        """Plot EE position (3D + time series), orientation, and joint trajectories."""
        if len(self.ee_history) == 0:
            self.get_logger().warn("No EE data recorded.")
            return

        ee_array = np.array(self.ee_history)
        joint_array = np.array(self.joint_history)
        valid_idx = ~np.isnan(ee_array).any(axis=1)
        ee_valid = ee_array[valid_idx]

        if ee_valid.shape[0] == 0:
            self.get_logger().warn("No valid EE samples.")
            return

        ee_pos = ee_valid[:, :3]
        ee_rpy = ee_valid[:, 3:]

        # --- Plot 3D EE Position ---
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(121, projection="3d")
        ax.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], "-o", markersize=3)
        ax.set_title("End Effector 3D Position Trajectory")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.grid(True)

        # --- Plot Orientation (RPY) ---
        ax2 = fig.add_subplot(122)
        ax2.plot(ee_rpy[:, 0], label="Roll")
        ax2.plot(ee_rpy[:, 1], label="Pitch")
        ax2.plot(ee_rpy[:, 2], label="Yaw")
        ax2.set_title("End Effector Orientation (RPY)")
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Angle [rad]")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # --- NEW: Plot X, Y, Z vs Sample Index ---
        fig3 = plt.figure(figsize=(8, 4))
        ax3 = fig3.add_subplot(111)
        ax3.plot(ee_pos[:, 0], label="X [m]", color="r")
        ax3.plot(ee_pos[:, 1], label="Y [m]", color="g")
        ax3.plot(ee_pos[:, 2], label="Z [m]", color="b")
        ax3.set_title("End Effector Position Components Over Time")
        ax3.set_xlabel("Sample Index")
        ax3.set_ylabel("Position [m]")
        ax3.legend()
        ax3.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Joint position evolution ---
        fig4 = plt.figure(figsize=(8, 4))
        ax4 = fig4.add_subplot(111)
        for j in range(min(joint_array.shape[1], len(self.traj_cfg["joint_names"]))):
            ax4.plot(joint_array[:, j], label=self.traj_cfg["joint_names"][j])
        ax4.set_title("Joint Positions Over Time")
        ax4.set_xlabel("Sample Index")
        ax4.set_ylabel("Joint Angle [rad]")
        ax4.legend()
        ax4.grid(True)
        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    traj_demo = TrajDemo()

    try:
        rclpy.spin(traj_demo)
    except KeyboardInterrupt:
        pass

    traj_demo.plot_joint_traj()
    traj_demo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
