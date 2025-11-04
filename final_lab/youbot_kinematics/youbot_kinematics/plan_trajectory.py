import rclpy
from rclpy.node import Node
from scipy.linalg import expm
from scipy.linalg import logm
from itertools import permutations
import time
import threading
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker

import numpy as np
from youbot_kinematics.youbotKineStudent import YoubotKinematicStudent
from youbot_kinematics.target_data import TARGET_JOINT_POSITIONS


class YoubotTrajectoryPlanning(Node):
    def __init__(self):
        # Initialize node
        super().__init__("youbot_trajectory_planner")

        # Save question number for check in main run method
        self.kdl_youbot = YoubotKinematicStudent()

        # Create trajectory publisher and a checkpoint publisher to visualize checkpoints
        self.traj_pub = self.create_publisher(
            JointTrajectory, "/youbot_arm_controller/joint_trajectory", 5
        )
        self.checkpoint_pub = self.create_publisher(Marker, "checkpoint_positions", 100)

    def run(self):
        """This function is the main run function of the class. When called, it runs question 6 by calling the q6()
        function to get the trajectory. Then, the message is filled out and published to the /command topic.
        """
        print("run q6a")
        self.get_logger().info("Waiting 5 seconds for everything to load up.")
        time.sleep(2.0)
        traj = self.q6()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = [
            "arm_joint_1",
            "arm_joint_2",
            "arm_joint_3",
            "arm_joint_4",
            "arm_joint_5",
        ]
        self.traj_pub.publish(traj)

    def q6(self):
        """This is the main q6 function. Here, other methods are called to create the shortest path required for this
        question. Below, a general step-by-step is given as to how to solve the problem.
        Returns:
            traj (JointTrajectory): A list of JointTrajectory points giving the robot joint positions to achieve in a
            given time period.
        """
        # TODO: implement this
        # Steps to solving Q6.
        # 1. Load in targets from the bagfile (checkpoint data and target joint positions).
        # 2. Compute the shortest path achievable visiting each checkpoint Cartesian position.
        # 3. Determine intermediate checkpoints to achieve a linear path between each checkpoint and have a full list of
        #    checkpoints the robot must achieve. You can publish them to see if they look correct. Look at slides 39 in lecture 7
        # 4. Convert all the checkpoints into joint values using an inverse kinematics solver.
        # 5. Create a JointTrajectory message.

        # Your code starts here ------------------------------
        # Load target Cartesian transforms and joint positions
        target_cart_tf, target_joint_positions = self.load_targets()

        # Determine the shortest path ordering among checkpoints (includes starting position at index 0)
        sorted_order, min_dist = self.get_shortest_path(target_cart_tf)

        self.get_logger().info(f"Sorted checkpoint order: {sorted_order.tolist()}, estimated dist: {min_dist:.4f}")

        # Number of intermediate points between checkpoints (tweakable)
        num_intermediate_points = 6

        # Build full set of transforms including intermediate points
        full_checkpoint_tfs = self.intermediate_tfs(sorted_order, target_cart_tf, num_intermediate_points)

        # Publish the transforms for visualization
        try:
            self.publish_traj_tfs(full_checkpoint_tfs)
        except Exception:
            # In case publisher not ready, ignore
            pass

        # Convert full transforms to joint positions using IK
        init_joint_position = target_joint_positions[:, 0].copy()
        q_checkpoints = self.full_checkpoints_to_joints(full_checkpoint_tfs, init_joint_position)

        # Build JointTrajectory message
        traj = JointTrajectory()
        traj.points = []

        # Time allocation: simple uniform time step per waypoint
        time_step = 0.5  # seconds between consecutive points (tunable)
        total_points = q_checkpoints.shape[1]
        for i in range(total_points):
            pt = JointTrajectoryPoint()
            pt.positions = q_checkpoints[:, i].tolist()
            pt.time_from_start.sec = int((i + 1) * time_step)
            pt.time_from_start.nanosec = int(((i + 1) * time_step - int((i + 1) * time_step)) * 1e9)
            traj.points.append(pt)

        # Your code ends here ------------------------------

        assert isinstance(traj, JointTrajectory)
        return traj

    def load_targets(self):
        """This function loads the checkpoint data from the TARGET_JOINT_POSITIONS variable. In this variable you will find each
        row has target joint positions. You need to use forward kinematics to get the goal end-effector position.
        Returns:
            target_cart_tf (4x4x5 np.ndarray): The target 4x4 homogenous transformations of the checkpoints found in the
            bag file. There are a total of 5 transforms (4 checkpoints + 1 initial starting cartesian position).
            target_joint_positions (5x5 np.ndarray): The target joint values for the 4 checkpoints + 1 initial starting
            position.
        """
        num_target_positions = len(TARGET_JOINT_POSITIONS)
        self.get_logger().info(f"{num_target_positions} target positions")
        # Initialize arrays for checkpoint transformations and joint positions
        target_joint_positions = np.zeros((5, num_target_positions + 1))
        # Create a 4x4 transformation matrix, then stack 6 of these matrices together for each checkpoint
        target_cart_tf = np.repeat(
            np.identity(4), num_target_positions + 1, axis=1
        ).reshape((4, 4, num_target_positions + 1))

        # Get the current starting position of the robot
        target_joint_positions[:, 0] = self.kdl_youbot.current_joint_position
        # Initialize the first checkpoint as the current end effector position
        target_cart_tf[:, :, 0] = self.kdl_youbot.forward_kinematics(
            target_joint_positions[:, 0].tolist()
        )

        # TODO: populate the transforms in the target_cart_tf object
        # populate the joint positions in the target_joint_positions object
        # Your code starts here ------------------------------
        # TARGET_JOINT_POSITIONS expected to be an iterable of joint vectors (each length 5)
        for i in range(num_target_positions):
            # store joint positions (column i+1)
            q = np.array(TARGET_JOINT_POSITIONS[i], dtype=float)
            if q.size != 5:
                raise ValueError("target entry must be length 5.")
            target_joint_positions[:, i + 1] = q
            # compute forward kinematics for this target joint configuration
            target_cart_tf[:, :, i + 1] = self.kdl_youbot.forward_kinematics(q.tolist())
        # Your code ends here ------------------------------

        self.get_logger().info(f"{target_cart_tf.shape} target poses")
        assert isinstance(target_cart_tf, np.ndarray)
        assert target_cart_tf.shape == (4, 4, num_target_positions + 1)
        assert isinstance(target_joint_positions, np.ndarray)
        assert target_joint_positions.shape == (5, num_target_positions + 1)

        return target_cart_tf, target_joint_positions

    def get_shortest_path(self, checkpoints_tf):
        """This function takes the checkpoint transformations and computes the order of checkpoints that results
        in the shortest overall path.
        Args:
            checkpoints_tf (np.ndarray): The target checkpoints transformations as a 4x4x5 numpy ndarray.
        Returns:
            sorted_order (np.array): An array of size 5 indicating the order of checkpoint
            min_dist:  (float): The associated distance to the sorted order giving the total estimate for travel
            distance.
        """
        num_checkpoints = checkpoints_tf.shape[2]
        # TODO: implement this method. Make it flexible to accomodate different numbers of targets.
        # Your code starts here ------------------------------
        # assume index 0 is starting pose
        # visit all others in some order minimizing path length
        positions = np.zeros((num_checkpoints, 3))
        for i in range(num_checkpoints):
            positions[i, :] = checkpoints_tf[0:3, 3, i]

        start_idx = 0
        other_indices = list(range(1, num_checkpoints))

        min_dist = float("inf")
        best_order = None

        # brute force permutations of other indices (works for small n)
        for perm in permutations(other_indices):
            order = [start_idx] + list(perm)
            dist = 0.0
            for a, b in zip(order[:-1], order[1:]):
                dist += np.linalg.norm(positions[a] - positions[b])
            if dist < min_dist:
                min_dist = dist
                best_order = order

        sorted_order = np.array(best_order, dtype=int)
        # Your code ends here ------------------------------

        assert isinstance(sorted_order, np.ndarray)
        assert sorted_order.shape == (num_checkpoints,)
        assert isinstance(min_dist, float)

        return sorted_order, min_dist

    def publish_traj_tfs(self, tfs):
        """This function gets a np.ndarray of transforms and publishes them in a color coded fashion to show how the
        Cartesian path of the robot end-effector.
        Args:
            tfs (np.ndarray): A array of 4x4xn homogenous transformations specifying the end-effector trajectory.
        """
        id = 0
        for i in range(0, tfs.shape[2]):
            marker = Marker()
            marker.id = id
            id += 1
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0 + id * 0.05
            marker.color.b = 1.0 - id * 0.05
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = tfs[0, -1, i]
            marker.pose.position.y = tfs[1, -1, i]
            marker.pose.position.z = tfs[2, -1, i]
            self.checkpoint_pub.publish(marker)
            time.sleep(0.05)  # add delay, otherwise trajectory is not complete

    def intermediate_tfs(
        self, sorted_checkpoint_idx, target_checkpoint_tfs, num_points
    ):
        """This function takes the target checkpoint transforms and the desired order based on the shortest path sorting,
        and calls the decoupled_rot_and_trans() function.
        Args:
            sorted_checkpoint_idx (list): List describing order of checkpoints to follow.
            target_checkpoint_tfs (np.ndarray): the state of the robot joints. In a youbot those are revolute
            num_points (int): Number of intermediate points between checkpoints.
        Returns:
            full_checkpoint_tfs: 4x4x(4xnum_points + 5) homogeneous transformations matrices describing the full desired
            poses of the end-effector position.
        """
        # TODO: implement this
        # Your code starts here ------------------------------
        # Build list of transforms: start with the first checkpoint, then for every consecutive pair
        # add the intermediate transforms (num_points) and then the destination checkpoint
        ordered_tfs = []
        # number of checkpoints
        n = target_checkpoint_tfs.shape[2]
        order = list(sorted_checkpoint_idx)
        # Append starting checkpoint
        ordered_tfs.append(target_checkpoint_tfs[:, :, order[0]])

        for idx in range(len(order) - 1):
            a_idx = order[idx]
            b_idx = order[idx + 1]
            a_tf = target_checkpoint_tfs[:, :, a_idx]
            b_tf = target_checkpoint_tfs[:, :, b_idx]
            # get intermediates
            tfs_between = self.decoupled_rot_and_trans(a_tf, b_tf, num_points)
            # append intermediates
            for k in range(tfs_between.shape[2]):
                ordered_tfs.append(tfs_between[:, :, k])
            # append the destination checkpoint
            ordered_tfs.append(b_tf)

        # Convert to numpy array shape 4x4xM
        M = len(ordered_tfs)
        full_checkpoint_tfs = np.zeros((4, 4, M))
        for i in range(M):
            full_checkpoint_tfs[:, :, i] = ordered_tfs[i]
        # Your code ends here ------------------------------

        return full_checkpoint_tfs

    def decoupled_rot_and_trans(self, checkpoint_a_tf, checkpoint_b_tf, num_points):
        """This function takes two checkpoint transforms and computes the intermediate transformations
        that follow a straight line path by decoupling rotation and translation.
        Args:
            checkpoint_a_tf (np.ndarray): 4x4 transformation describing pose of checkpoint a.
            checkpoint_b_tf (np.ndarray): 4x4 transformation describing pose of checkpoint b.
            num_points (int): Number of intermediate points between checkpoint a and checkpoint b.
        Returns:
            tfs: 4x4x(num_points) homogeneous transformations matrices describing the full desired
            poses of the end-effector position from checkpoint a to checkpoint b following a linear path.
        """
        self.get_logger().info("checkpoint a")
        self.get_logger().info(str(checkpoint_a_tf))
        self.get_logger().info("checkpoint b")
        self.get_logger().info(str(checkpoint_b_tf))
        # TODO: implement this
        # Your code starts here ------------------------------
        # Extract rotations and translations
        Ra = checkpoint_a_tf[0:3, 0:3]
        Rb = checkpoint_b_tf[0:3, 0:3]
        ta = checkpoint_a_tf[0:3, 3]
        tb = checkpoint_b_tf[0:3, 3]

        # Compute relative rotation from a to b
        R_rel = Ra.T @ Rb  # For roation matrix T == -1

        # Compute matrix logarithm of R_rel
        try:
            log_R_rel = logm(R_rel)
        except Exception:  # fallback: small-angle approx -> zero
            log_R_rel = np.zeros((3, 3))

        tfs = np.zeros((4, 4, num_points))
        for k in range(num_points):
            s = float(k + 1) / (num_points + 1)  
            t_k = ta + s * (tb - ta) # translation 
            R_k = Ra @ expm(log_R_rel * s) # rotation using expm of scaled log
            tf_k = np.eye(4)
            tf_k[0:3, 0:3] = R_k
            tf_k[0:3, 3] = t_k
            tfs[:, :, k] = tf_k
        # Your code ends here ------------------------------
        return tfs

    def full_checkpoints_to_joints(self, full_checkpoint_tfs, init_joint_position):
        """This function takes the full set of checkpoint transformations, including intermediate checkpoints,
        and computes the associated joint positions by calling the ik_position_only() function.
        Args:
            full_checkpoint_tfs (np.ndarray, 4x4xn): 4x4xn transformations describing all the desired poses of the end-effector
            to follow the desired path.
            init_joint_position (np.ndarray):A 5x1 array for the initial joint position of the robot.
        Returns:
            q_checkpoints (np.ndarray, 5xn): For each pose, the solution of the position IK to get the joint position
            for that pose.
        """
        # TODO: Implement this
        # Your code starts here ------------------------------
        n = full_checkpoint_tfs.shape[2]
        q_checkpoints = np.zeros((5, n))
        # seed with init_joint_position for first IK
        q_seed = init_joint_position.copy()
        for i in range(n):
            tf = full_checkpoint_tfs[:, :, i]
            q_sol, error = self.ik_position_only(tf, q_seed)
            q_checkpoints[:, i] = q_sol
            # next seed is the solution we just found
            q_seed = q_sol.copy()
        # Your code ends here ------------------------------

        return q_checkpoints

    def ik_position_only(self, pose, q0, lam=0.25, num=500):
        """This function implements position only inverse kinematics.
        Args:
            pose (np.ndarray, 4x4): 4x4 transformations describing the pose of the end-effector position.
            q0 (np.ndarray, 5x1):A 5x1 array for the initial starting point of the algorithm.
        Returns:
            q (np.ndarray, 5x1): The IK solution for the given pose.
            error (float): The Cartesian error of the solution.
        """

        # TODO: Implement this
        # Some useful notes:
        # We are only interested in position control - take only the position part of the pose as well as elements of the
        # Jacobian that will affect the position of the error.

        # Your code starts here ------------------------------
        q = q0.astype(float).copy()
        target_pos = pose[0:3, 3]

        max_step = 0.1  # radian limit per iteration (to avoid large jumps)
        for iteration in range(num):
            # compute current end-effector transform for q
            current_tf = self.kdl_youbot.forward_kinematics(q.tolist())
            current_pos = current_tf[0:3, 3]
            error_vec = target_pos - current_pos
            error = np.linalg.norm(error_vec)

            # get analytic Jacobian 
            J6 = self.kdl_youbot.get_jacobian(q.tolist()) 
            J_pos = J6[0:3, :]  # 3x5

            # Damped least squares pseudoinverse: J_pinv = J.T @ inv(J J.T + lam^2 I)
            JJt = J_pos @ J_pos.T
            reg = (lam ** 2) * np.eye(3)
            try:
                inv_term = np.linalg.inv(JJt + reg)
                J_pinv = J_pos.T @ inv_term
            except np.linalg.LinAlgError:
                J_pinv = np.linalg.pinv(J_pos)

            # compute joint update
            dq = J_pinv @ error_vec

            # limit step size
            norm_dq = np.linalg.norm(dq)
            if norm_dq > max_step:
                dq = dq * (max_step / norm_dq)

            # apply update scaled by damping
            q = q + lam * dq

            # enforce joint limits from kinematic base
            try:
                q = np.minimum(q, self.kdl_youbot.joint_limit_max)
                q = np.maximum(q, self.kdl_youbot.joint_limit_min)
            except Exception:
                # if joint limits not present, skip clamping
                pass

        # final error
        final_tf = self.kdl_youbot.forward_kinematics(q.tolist())
        final_pos = final_tf[0:3, 3]
        error_vec = target_pos - final_pos
        error = np.linalg.norm(error_vec)
        # Your code ends here ------------------------------

        return q, error


def main(args=None):
    rclpy.init(args=args)

    youbot_planner = YoubotTrajectoryPlanning()

    youbot_planner.run()

    rclpy.spin(youbot_planner)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    youbot_planner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
