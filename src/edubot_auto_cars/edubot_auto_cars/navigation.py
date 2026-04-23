import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import numpy as np

class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower')
        self.bridge = CvBridge()
        # Parameters
        self.declare_parameter('camera_width', 1800)
        self.camera_width = self.get_parameter('camera_width').get_parameter_value().integer_value
        self.get_logger().info(f"Camera width set to: {self.camera_width}")

        # Topic names as parameters for flexibility
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('lidar_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        self.image_received = False
        self.lidar_received = False

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10)
        self.lidar_sub = self.create_subscription(
            LaserScan,
            lidar_topic,
            self.lidar_callback,
            10)
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.obstacle_detected = False
        self.last_twist = Twist()
        self.get_logger().info(f"Subscribed to image: {image_topic}, lidar: {lidar_topic}, cmd_vel: {cmd_vel_topic}")

    def image_callback(self, msg):
        self.image_received = True
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return
        height, width, _ = cv_image.shape
        roi = cv_image[int(height*0.5):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # White mask (lane lines)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        # Yellow mask (center line)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # Combine masks
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        M = cv2.moments(mask)
        twist = Twist()
        if M['m00'] > 0 and not self.obstacle_detected:
            cx = int(M['m10']/M['m00'])
            err = cx - width // 2
            twist.linear.x = 0.15
            twist.angular.z = -float(err) / 200
            self.get_logger().info(f"Lane detected: cx={cx}, err={err}, linear.x={twist.linear.x:.2f}, angular.z={twist.angular.z:.2f}")
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            if M['m00'] == 0:
                self.get_logger().warn("No lane detected in image.")
            if self.obstacle_detected:
                self.get_logger().warn("Obstacle detected, stopping.")
        self.last_twist = twist
        self.cmd_pub.publish(twist)

    def lidar_callback(self, msg):
        self.lidar_received = True
        # Check for obstacles in front (within 0.5m, +/- 20 degrees)
        ranges = np.array(msg.ranges)
        # Remove inf values
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        front = np.concatenate((ranges[:20], ranges[-20:]))
        if np.any(front < 0.5):
            if not self.obstacle_detected:
                self.get_logger().warn("Obstacle detected in LIDAR, stopping robot.")
            self.obstacle_detected = True
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.cmd_pub.publish(stop)
        else:
            if self.obstacle_detected:
                self.get_logger().info("Obstacle cleared, resuming lane following.")
            self.obstacle_detected = False
            # Republish last twist if lane detected
            self.cmd_pub.publish(self.last_twist)

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    node.get_logger().info("LaneFollower node started. Waiting for image and lidar data...")
    # Timer to check if topics are being received
    def check_topics():
        if not node.image_received:
            node.get_logger().warn("No camera images received yet. Check topic and camera.")
        if not node.lidar_received:
            node.get_logger().warn("No LIDAR scans received yet. Check topic and LIDAR.")
    timer = node.create_timer(2.0, check_topics)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
