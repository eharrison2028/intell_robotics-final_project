import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower')
        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(Image, '/camera_2/image_raw', self.image_callback, 10)

        # --- TUNING PARAMETERS ---
        self.speed = 0.08               # Linear velocity
        self.steering_gain = 1.0        # Increased from 0.2 to fight the "veer"
        self.steering_multiplier = -1.0 # -1.0 assumes +z is Left, and line is on the Right
        
        # Target position of the line (0.0=Left edge, 1.0=Right edge of screen)
        # 0.70 places the line in the right-middle of the view
        self.right_line_target = 0.70 

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        h, w, _ = frame.shape
        
        # 1. TIGHTER ROI: Focus only on the ground immediately in front.
        # This reduces the "angled inward" effect of the lanes.
        roi_top = int(h * 0.80)
        roi_bottom = int(h * 0.95)
        roi = frame[roi_top:roi_bottom, :]
        
        # 2. MASKING: Isolate white/bright pixels
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200]) 
        upper_white = np.array([180, 60, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 3. COORDINATE CALCULATION
        # We find the horizontal average of all white pixels in our narrow slice
        white_pixels = np.where(mask > 0)
        
        cmd = Twist()

        # Check if we have enough pixels to make a decision
        if len(white_pixels[1]) > 100: 
            # Current line position normalized (0.0 to 1.0)
            current_line_x = np.mean(white_pixels[1]) / w
            
            # 4. ERROR & CONTROL
            # Error is how far the line is from our target 0.70
            error = self.right_line_target - current_line_x
            
            cmd.linear.x = self.speed
            # Proportional Control: Steer harder as error increases
            cmd.angular.z = error * self.steering_gain * self.steering_multiplier
            
            self.get_logger().info(f"Line: {current_line_x:.2f} | Error: {error:.2f} | Steer: {cmd.angular.z:.2f}")
        else:
            # SAFETY: If line is lost, stop and spin slowly to find it
            cmd.linear.x = 0.0
            cmd.angular.z = 0.1 
            self.get_logger().warn("LINE LOST - SEARCHING...")

        # 5. PUBLISH
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Standard ROS 2 shutdown pattern
        stop_msg = Twist()
        node.cmd_pub.publish(stop_msg)
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


