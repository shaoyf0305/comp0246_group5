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
        self.joint_state_sub  # prevent unused variable warning
        self.traj_publisher = self.create_publisher(
            JointTrajectory, "/franka_arm_controller/joint_trajectory", 5
        )
        self.traj_publisher

        # load from urdf file
        self.declare_parameter("urdf_package", "franka_description")
        self.urdf_package = (
            self.get_parameter("urdf_package").get_parameter_value().string_value
        )
        self.urdf_package_path = get_package_share_directory(self.urdf_package)
        self.declare_parameter("urdf_path_in_package", "urdfs/fr3.urdf")
        self.urdf_path_in_package = (
            self.get_parameter("urdf_path_in_package")
            .get_parameter_value()
            .string_value
        )
        self.urdf_name_path = os.path.join(
            self.urdf_package_path, self.urdf_path_in_package
        )

        self.get_logger().info(
            f"loading robot into KDL from urdf: {self.urdf_name_path}"
        )

        robot = URDF.from_xml_file(self.urdf_name_path)

        (ok, self.kine_tree) = treeFromUrdfModel(robot)

        if not ok:
            raise RuntimeError("couldn't load URDF into KDL tree succesfully")

        # Use the URDF root as the default base_link so we don't rely on "base"
        # this part is changed!!
        default_base = robot.get_root() if hasattr(robot, "get_root") else "base"
        self.declare_parameter("base_link", default_base)
        self.declare_parameter("ee_link", "fr3_link8")
        self.base_link = (
            self.get_parameter("base_link").get_parameter_value().string_value
        )
        self.ee_link = self.get_parameter("ee_link").get_parameter_value().string_value

        # Build KDL chain. If the requested base_link isn't present this will raise later.
        self.kine_chain = self.kine_tree.getChain(self.base_link, self.ee_link)
        self.NJoints = self.kine_chain.getNrOfJoints()
        self.current_joint_position = PyKDL.JntArray(self.NJoints)
        self.current_joint_velocity = PyKDL.JntArray(self.NJoints)
        # KDL solvers
        self.ik_solver = PyKDL.ChainIkSolverPos_LMA(self.kine_chain)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # loading trajectory structure
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

        assert type(self.traj_cfg["joint_names"]) == list, type(
            self.traj_cfg["joint_names"]
        )
        assert len(self.traj_cfg["joint_names"]) == self.NJoints

        self.created_traj = False

        self.get_logger().info(f"got traj cfg:\n{self.traj_cfg}")

        # Data structures to record state and EE pose history
        self.joint_history = []      # list of np arrays for joint positions
        self.joint_vel_history = []  # list of np arrays for joint velocities
        self.ee_history = []         # list of 3D positions of the end effector


    def joint_state_callback(self, msg: JointState):
        """Callback for the joint states of the robot arm. It will get joint positions and velocities, and eventually the cartesian position of the end-effector and save this. It allows initializes the trajectory once the first joint state message comes through.

        Args:
            msg (JointState): ROS Joint State Message.
        """
        for i in range(len(msg.name)):
            n = msg.name[i]
            pos = msg.position[i]
            vel = msg.velocity[i]

            self.current_joint_position[i] = pos
            self.current_joint_velocity[i] = vel

        joint_pos, joint_vel = self.kdl_to_np(
            self.current_joint_position
        ), self.kdl_to_np(self.current_joint_velocity)

        # record copies so further modification doesn't change stored history
        self.joint_history.append(np.array(joint_pos, copy=True))
        self.joint_vel_history.append(np.array(joint_vel, copy=True))

        # get and save EE position (may be NaNs if transform not yet available)
        ee_pos = self.get_ee_pos_ros()
        self.ee_history.append(np.array(ee_pos, copy=True))

        # we do this after we have our first callback to have current joint positions
        if not self.created_traj:
            joint_traj = self.create_traj()
            self.traj_publisher.publish(joint_traj)
            self.created_traj = True
            self.get_logger().info("Published joint trajectory.")

        # elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        # self.get_logger().info(f"Trajectory execution time ~{elapsed:.2f} seconds.")
                # self.created_traj = False  # prevent re-reporting


    def kdl_to_np(self, data: PyKDL.JntArray) -> NDArray:
        """Helper Function to go from KDL arrays to numpy arrays"""
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

    def get_ee_pos_ros(self) -> Tuple[float, float, float]:
        """Get current EE position using TF2. Returns NaNs if unavailable."""
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.base_link,
                self.ee_link,
                now,
                timeout=Duration(seconds=1.0),
            )
            pos = (
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            )
            return pos
        except TransformException as ex:
            # don't crash; return NaNs and keep running
            self.get_logger().warn(
                f"Transform lookup failed for {self.base_link} -> {self.ee_link}: {ex}"
            )
            return (np.nan, np.nan, np.nan)

    def get_joint_pos(
        self,
        current_angles: Tuple[float],
        target_position: Tuple[float],
        target_orientation: Optional[Tuple[float]] = None,
    ) -> Tuple[float]:
        """Solve inverse kinematics using KDL."""
        assert len(target_position) == 3
        assert target_orientation is None or len(target_orientation) == 3
        assert len(current_angles) == self.NJoints

        pos = PyKDL.Vector(target_position[0], target_position[1], target_position[2])
        if target_orientation is not None:
            rot = PyKDL.Rotation.RPY(
                target_orientation[0], target_orientation[1], target_orientation[2]
            )

        seed_array = PyKDL.JntArray(self.NJoints)
        for i in range(self.NJoints):
            seed_array[i] = current_angles[i]

        if target_orientation is not None:
            goal_pose = PyKDL.Frame(rot, pos)
        else:
            goal_pose = PyKDL.Frame(pos)
        result_angles = PyKDL.JntArray(self.NJoints)

        if self.ik_solver.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            result = list(result_angles)
            return result
        else:
            raise RuntimeError(
                f"Did not solve for goal_pose: {goal_pose} with initial seed {seed_array}"
            )


    def create_traj(self):
        # Load YAML waypoints
        with open(self.traj_file_path, 'r') as f:
            self.traj_cfg = yaml.safe_load(f)

        waypoints = self.traj_cfg["waypoints"]["cartesion"]

        # --- Get current EE pose ---
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                "base", "fr3_link8", now, timeout=rclpy.duration.Duration(seconds=1.0)
            )
            current_pos = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ]
            # Insert as first waypoint at t=0
            waypoints.insert(0, {"position": current_pos, "time": 0.0})
            self.get_logger().info(f"Added current EE pose {current_pos} as first waypoint.")
        except Exception as e:
            self.get_logger().warn(f"Could not get current EE pose, starting directly: {e}")

        # Continue generating trajectory using updated waypoints
        ...


    def create_traj(self) -> JointTrajectory:
        """Generate a smooth joint trajectory from Cartesian waypoints.
        Automatically adds the current EE pose as the starting waypoint.
        Returns:
            JointTrajectory: ROS JointTrajectory to publish.
        """
        # Record start time (for overall timing)
        self.start_time = self.get_clock().now()

        cartesian_waypoints = self.traj_cfg["waypoints"]["cartesion"]
        cur_joint_pos = self.kdl_to_np(self.current_joint_position)

        # --- Get current EE position and prepend as first waypoint ---
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.base_link, "fr3_link8", now, timeout=rclpy.duration.Duration(seconds=1.0)
            )
            current_pos = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ]
            self.get_logger().info(f"Adding current EE pose {np.round(current_pos, 3)} as first waypoint.")
            cartesian_waypoints.insert(0, {"position": current_pos, "time": 0.0})
        except Exception as e:
            self.get_logger().warn(f"Could not get current EE pose, starting directly: {e}")

        # --- Initialize data containers ---
        goal_positions = []
        goal_times = []
        time_since_start = 0.0

        # --- Initialize trajectory message ---
        joint_traj = JointTrajectory()
        joint_traj.header.frame_id = self.base_link
        joint_traj.header.stamp = self.get_clock().now().to_msg()
        joint_traj.joint_names = self.traj_cfg["joint_names"]

        # --- Process waypoints ---
        for i, waypoint in enumerate(cartesian_waypoints):
            pos = waypoint["position"]
            time_sec = float(waypoint["time"])  # enforce float
            assert len(pos) == 3, f"Invalid position format in waypoint {i}"

            # accumulate absolute timing
            time_since_start += time_sec
            goal_times.append(time_since_start)

            # Solve IK for each Cartesian waypoint
            cur_joint_pos = self.get_joint_pos(cur_joint_pos, pos)
            goal_positions.append(cur_joint_pos)

            pt = JointTrajectoryPoint()
            pt.positions = cur_joint_pos
            pt.velocities = [0.0] * self.NJoints
            pt.accelerations = [0.0] * self.NJoints
            pt.time_from_start = Duration(seconds=time_since_start).to_msg()

            joint_traj.points.append(pt)

        # --- Log planned duration ---
        total_dur = goal_times[-1] if goal_times else 0.0
        self.get_logger().info(f"Planned trajectory total duration: {total_dur:.2f} seconds")

        return joint_traj

    def plot_joint_traj(self):
        """Plot recorded end-effector and joint trajectories after execution."""
        if len(self.ee_history) == 0 or len(self.joint_history) == 0:
            self.get_logger().warn("No recorded data to plot.")
            return

        ee_array = np.array(self.ee_history)
        joint_array = np.array(self.joint_history)

        # Filter out NaN EE positions (in case TF was unavailable at some samples)
        valid_idx = ~np.isnan(ee_array).any(axis=1)
        ee_valid = ee_array[valid_idx]

        if ee_valid.shape[0] == 0:
            self.get_logger().warn("No valid EE pose samples to plot (all NaN).")
            return

        # --- Plot 3D EE trajectory ---
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(ee_valid[:, 0], ee_valid[:, 1], ee_valid[:, 2], "-o", markersize=3)
        ax.set_title("End Effector Trajectory")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot joint positions over time ---
        fig2 = plt.figure(figsize=(8, 4))
        ax2 = fig2.add_subplot(111)
        for j in range(min(joint_array.shape[1], len(self.traj_cfg["joint_names"]))):
            ax2.plot(joint_array[:, j], label=self.traj_cfg["joint_names"][j])
        ax2.set_title("Joint Positions Over Time")
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Joint Angle [rad]")
        ax2.legend(loc="best")
        ax2.grid(True)
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
