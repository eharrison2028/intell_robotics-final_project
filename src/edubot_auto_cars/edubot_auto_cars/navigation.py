import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower_right_blob')
        
        self.subscription = self.create_subscription(Image, '/camera_2/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        # --- LiDAR Subscription ---
        self.obstacle_detected = False
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

        # --- Control Tuning (From Friend's Code) ---
        self.base_speed = 0.07
        self.kp = 0.002
        self.last_steering = 0.0
        self.smoothing_factor = 0.7 
        self.deadzone = 8

        # --- State Machine Variables (From Our Code) ---
        self.state = "FOLLOWING"
        self.state_start_time = 0.0

        # --- E-Stop Variables (From Our Code) ---
        self.e_stop_active = False
        
        # Set up the OpenCV window and mouse callback
        cv2.namedWindow("Right-Most Logic")
        cv2.setMouseCallback("Right-Most Logic", self.mouse_callback)

        # HSV Thresholds
        self.white_low = np.array([0, 10, 220])
        self.white_high = np.array([180, 30, 255])
        self.orange_low = np.array([5, 100, 100])
        self.orange_high = np.array([15, 255, 255])

    def mouse_callback(self, event, x, y, flags, param):
        # E-Stop Trigger
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.e_stop_active:
                self.get_logger().error('E-STOP TRIGGERED BY MOUSE CLICK!')
                self.e_stop_active = True

    def lidar_callback(self, msg: LaserScan):
        # Filter valid ranges
        valid = [r for r in msg.ranges if np.isfinite(r) and r > 0.01]

        if not valid:
            return

        # Minimum distance anywhere around the robot
        min_dist = min(valid)

        # Stop if anything is within ~1 foot (0.30 m)
        self.obstacle_detected = (min_dist < 0.30)

    def get_rightmost_centroid(self, mask):
        """Finds individual contours and picks the one furthest to the right."""
        # Clean up the mask to separate lines that might be touching visually
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cx = None
        max_x = -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:  # Ignore noise
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    # We want the blob whose center is furthest to the right
                    if cx > max_x:
                        max_x = cx
                        best_cx = cx
        
        return best_cx

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            return

        h, w, _ = frame.shape
        
        # Friend's preferred crop (Top 40% removed)
        roi = frame[int(h*0.4):int(h*0.9), :]
        r_h, r_w, _ = roi.shape

        # ==========================================
        # E-STOP OVERRIDE LOGIC (Highest Priority)
        # ==========================================
        if self.e_stop_active:
            stop_msg = Twist()
            self.publisher.publish(stop_msg)

            cv2.rectangle(roi, (0, 0), (r_w, r_h), (0, 0, 255), 10) 
            cv2.putText(roi, "E-STOP ACTIVE", (int(r_w*0.1), int(r_h*0.4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(roi, "Press 'r' to Resume", (int(r_w*0.2), int(r_h*0.6)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Right-Most Logic", roi)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.get_logger().info('E-STOP RESET. Resuming operation.')
                self.e_stop_active = False
                self.state = "FOLLOWING" 
                self.last_steering = 0.0
            return 

        # ==========================================
        # LiDAR OBSTACLE AVOIDANCE OVERRIDE
        # ==========================================
        if self.obstacle_detected:
            stop_msg = Twist()
            self.publisher.publish(stop_msg)

            # Draw yellow warning on the screen
            cv2.rectangle(roi, (0, 0), (r_w, r_h), (0, 255, 255), 10) 
            cv2.putText(roi, "OBSTACLE AHEAD", (int(r_w*0.1), int(r_h*0.4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(roi, "Waiting for path to clear...", (int(r_w*0.15), int(r_h*0.6)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Right-Most Logic", roi)
            cv2.waitKey(1)
            
            # Note: We return here so no driving logic occurs, but as soon as
            # the LiDAR callback sets obstacle_detected to False, this block
            # gets bypassed and normal driving resumes automatically.
            return

        # ==========================================
        # NORMAL NAVIGATION LOGIC
        # ==========================================
        
        # Target: 80% of the image width
        target_x = int(r_w * 0.8)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        twist = Twist()
        now = self.get_clock().now().nanoseconds / 1e9

        if self.state == "FOLLOWING":
            # 1. Check for Orange Dead End First
            orange_mask = cv2.inRange(hsv, self.orange_low, self.orange_high)
            if cv2.countNonZero(orange_mask) > 1500: 
                self.get_logger().info('Orange line detected! Initiating U-Turn.')
                self.state = "TURN_LEFT_1"
                self.state_start_time = now
                return 

            # 2. Friend's Line Following Logic
            white_mask = cv2.inRange(hsv, self.white_low, self.white_high)
            cx = self.get_rightmost_centroid(white_mask)

            if cx is not None:
                # We found a right-most line!
                error = cx - target_x
                if abs(error) < self.deadzone: 
                    error = 0
                
                target_steering = -float(error) * self.kp
                twist.linear.x = self.base_speed
                
                # Visual feedback
                cv2.drawContours(roi, cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 255), 1)
                cv2.circle(roi, (cx, r_h//2), 10, (0, 255, 0), -1)

                # Smoothing
                smooth_steering = (target_steering * (1 - self.smoothing_factor)) + (self.last_steering * self.smoothing_factor)
                twist.angular.z = smooth_steering
                self.last_steering = smooth_steering

            else:
                # SEARCH MODE: Arc Right
                twist.linear.x = 0.04
                target_steering = -0.2
                twist.angular.z = target_steering
                self.last_steering = target_steering # Reset memory so it doesn't snap when found

        # --- U-TURN MANEUVER STATES ---
        elif self.state == "TURN_LEFT_1":
            twist.angular.z = 0.8  
            if now - self.state_start_time > 1.8: 
                self.state = "MOVE_FORWARD"
                self.state_start_time = now
                
        elif self.state == "MOVE_FORWARD":
            twist.linear.x = 0.1 
            if now - self.state_start_time > 4.0: 
                self.state = "TURN_LEFT_2"
                self.state_start_time = now
                
        elif self.state == "TURN_LEFT_2":
            twist.angular.z = 0.8  
            if now - self.state_start_time > 1.8: 
                self.state = "MOVE_FORWARD_2"
                self.state_start_time = now
                
        elif self.state == "MOVE_FORWARD_2":
            twist.linear.x = 0.1
            twist.angular.z = 0.0
            if now - self.state_start_time > 2.0: 
                self.get_logger().info('U-Turn complete. Resuming following.')
                self.state = "FOLLOWING"
                self.last_steering = 0.0 

        # Publish unified movement command
        self.publisher.publish(twist)

        # Draw UI
        cv2.line(roi, (target_x, 0), (target_x, r_h), (255, 0, 0), 2)
        cv2.putText(roi, "Click window or press SPACE to E-STOP", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Right-Most Logic", roi)
        
        # Keyboard Listeners
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q'):
            if not self.e_stop_active:
                self.get_logger().error('E-STOP TRIGGERED BY KEYBOARD!')
                self.e_stop_active = True

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        stop_msg = Twist()
        node.publisher.publish(stop_msg)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#up to date

