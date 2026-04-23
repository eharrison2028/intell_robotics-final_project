import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Example internal state
        self.state = 'FOLLOW_LANE'

        # Latest perception values
        self.lane_offset = 0.0
        self.lane_heading_error = 0.0
        self.intersection_detected = False
        self.end_line_detected = False
        self.stop_sign_detected = False
        self.obstacle_ahead = False

        # Main control loop
        self.timer = self.create_timer(0.1, self.control_loop)

    def control_loop(self):
        cmd = Twist()

        if self.state == 'FOLLOW_LANE':
            if self.stop_sign_detected:
                self.state = 'STOP_AT_SIGN'
            elif self.end_line_detected:
                self.state = 'TURN_AROUND'
            elif self.obstacle_ahead:
                self.state = 'AVOID_OBSTACLE'
            elif self.intersection_detected:
                self.state = 'TURN_RIGHT'
            else:
                cmd.linear.x = 0.15
                cmd.angular.z = -1.5 * self.lane_offset - 0.8 * self.lane_heading_error

        elif self.state == 'STOP_AT_SIGN':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            # add timer logic here

        elif self.state == 'TURN_RIGHT':
            cmd.linear.x = 0.05
            cmd.angular.z = -0.8
            # add condition to exit when lane is reacquired

        elif self.state == 'TURN_AROUND':
            cmd.linear.x = 0.0
            cmd.angular.z = 1.0
            # add condition for ~180 turn

        elif self.state == 'AVOID_OBSTACLE':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            # replace with bypass logic if desired

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()