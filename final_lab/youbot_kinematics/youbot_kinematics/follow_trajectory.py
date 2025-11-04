import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np

from youbot_kinematics.youbotKineStudent import YoubotKinematicStudent


class YoubotFKBroadcaster(Node):
    """
    Node that subscribes to the robot joint states, computes forward kinematics,
    and publishes the end-effector transform as a TF frame.
    """

    def __init__(self):
        super().__init__('youbot_fk_broadcaster')

        # Initialize kinematic model (student implementation)
        self.kdl_youbot = YoubotKinematicStudent()

        # TF2 broadcaster to publish transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to joint states from the controller
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',  # or your controller topic, e.g., '/youbot_arm/joint_states'
            self.joint_state_callback,
            10
        )

        self.get_logger().info('Youbot FK Broadcaster initialized and listening to /joint_states')

    def joint_state_callback(self, msg: JointState):
        """
        Callback that runs whenever new joint states are received.
        Computes FK and publishes the end-effector transform.
        """
        if len(msg.position) < 5:
            # Ignore incomplete messages
            return

        # Extract the first 5 joints (youbot arm joints)
        q = np.array(msg.position[:5])

        # Compute forward kinematics using your kinematics class
        T = self.kdl_youbot.forward_kinematics(q.tolist())

        # Extract position and rotation (convert to quaternion)
        position = T[0:3, 3]
        R = T[0:3, 0:3]

        # Convert rotation matrix to quaternion
        quat = self.rotation_matrix_to_quaternion(R)

        # Create and publish the transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'end_effector_link'

        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def rotation_matrix_to_quaternion(R):
        """
        Converts a 3x3 rotation matrix into a quaternion [x, y, z, w].
        """
        q = np.zeros(4)
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[3] = 0.25 / s
            q[0] = (R[2, 1] - R[1, 2]) * s
            q[1] = (R[0, 2] - R[2, 0]) * s
            q[2] = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                q[3] = (R[2, 1] - R[1, 2]) / s
                q[0] = 0.25 * s
                q[1] = (R[0, 1] + R[1, 0]) / s
                q[2] = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                q[3] = (R[0, 2] - R[2, 0]) / s
                q[0] = (R[0, 1] + R[1, 0]) / s
                q[1] = 0.25 * s
                q[2] = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                q[3] = (R[1, 0] - R[0, 1]) / s
                q[0] = (R[0, 2] + R[2, 0]) / s
                q[1] = (R[1, 2] + R[2, 1]) / s
                q[2] = 0.25 * s
        return q


def main(args=None):
    rclpy.init(args=args)
    node = YoubotFKBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
