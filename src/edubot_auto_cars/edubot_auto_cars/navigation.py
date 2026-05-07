import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower_right_blob')
       
        self.subscription = self.create_subscription(Image, '/camera_2/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        # --- LiDAR Subscription ---
        self.obstacle_detected = False
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

        # --- Control Tuning ---
        self.base_speed = 0.07
        self.kp = 0.0015             # Lowered slightly to reduce snapping
        self.kd = 0.0012             # NEW: Derivative gain to stop oscillation
        self.last_error = 0.0        # NEW: Track error for PD control
        self.last_steering = 0.0
        self.smoothing_factor = 0.7
        self.deadzone = 10           # Increased slightly for stability

        # --- State Machine Variables ---
        self.state = "FOLLOWING"
        self.state_start_time = 0.0

        # --- E-Stop Variables ---
        self.e_stop_active = False
       
        # Set up the OpenCV window and mouse callback
        cv2.namedWindow("Right-Most Logic")
        cv2.setMouseCallback("Right-Most Logic", self.mouse_callback)

        # HSV Thresholds
        self.white_low = np.array([0, 10, 220])
        self.white_high = np.array([180, 30, 255])
        self.orange_low = np.array([5, 100, 100])
        self.orange_high = np.array([15, 255, 255])
       
        # Frame skip variables (as requested to keep)
        self.frame_count = 0
        self.process_every_n_frames = 5 

    def fusion_callback(self, img_msg, scan_msg):
        # 1. Skip frames to save processing power
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        # 2. Run YOLO Detection
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.e_stop_active:
                self.get_logger().error('E-STOP TRIGGERED BY MOUSE CLICK!')
                self.e_stop_active = True

    def lidar_callback(self, msg: LaserScan):
        valid = [r for r in msg.ranges if np.isfinite(r) and r > 0.01]
        if not valid:
            return
        min_dist = min(valid)
        self.obstacle_detected = (min_dist < 0.30)

    def get_rightmost_centroid(self, mask):
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        best_cx = None
        max_x = -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
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
        roi = frame[int(h*0.4):int(h*0.9), :]
        r_h, r_w, _ = roi.shape

        if self.e_stop_active:
            stop_msg = Twist()
            self.publisher.publish(stop_msg)
            cv2.rectangle(roi, (0, 0), (r_w, r_h), (0, 0, 255), 10)
            cv2.putText(roi, "E-STOP ACTIVE", (int(r_w*0.1), int(r_h*0.4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imshow("Right-Most Logic", roi)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.e_stop_active = False
                self.state = "FOLLOWING"
                self.last_steering = 0.0
            return

        if self.obstacle_detected:
            stop_msg = Twist()
            self.publisher.publish(stop_msg)
            cv2.imshow("Right-Most Logic", roi)
            cv2.waitKey(1)
            return

        target_x = int(r_w * 0.8)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        twist = Twist()
        now = self.get_clock().now().nanoseconds / 1e9

        if self.state == "FOLLOWING":
            orange_mask = cv2.inRange(hsv, self.orange_low, self.orange_high)
            orange_cnts, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
           
            trigger_uturn = False
            for c in orange_cnts:
                _, _, cw, _ = cv2.boundingRect(c)
                if cw > (r_w / 2):
                    trigger_uturn = True
                    break

            if trigger_uturn:
                self.state = "TURN_LEFT_1"
                self.state_start_time = now
                return

            white_mask = cv2.inRange(hsv, self.white_low, self.white_high)
            cx = self.get_rightmost_centroid(white_mask)

            if cx is not None:
                error = cx - target_x
                if abs(error) < self.deadzone:
                    error = 0
                
                # --- PD CONTROL LOGIC ---
                d_error = error - self.last_error
                target_steering = -((error * self.kp) + (d_error * self.kd))
                self.last_error = error
                
                twist.linear.x = self.base_speed
                smooth_steering = (target_steering * (1 - self.smoothing_factor)) + (self.last_steering * self.smoothing_factor)
                twist.angular.z = smooth_steering
                self.last_steering = smooth_steering

                cv2.circle(roi, (cx, r_h//2), 10, (0, 255, 0), -1)
            else:
                # --- NEW SEARCH MODE: BARELY CREEP AND TURN RIGHT ---
                twist.linear.x = 0.02
                twist.angular.z = -0.15 
                self.last_steering = -0.15
                self.last_error = 0 

        elif self.state == "TURN_LEFT_1":
            twist.angular.z = 0.8 
            if now - self.state_start_time > 2.2:
                self.state = "MOVE_FORWARD"
                self.state_start_time = now
               
        elif self.state == "MOVE_FORWARD":
            twist.linear.x = 0.1
            if now - self.state_start_time > 6.0:
                self.state = "TURN_LEFT_2"
                self.state_start_time = now
               
        elif self.state == "TURN_LEFT_2":
            twist.angular.z = 0.8 
            if now - self.state_start_time > 2.2:
                self.state = "MOVE_FORWARD_2"
                self.state_start_time = now
               
        elif self.state == "MOVE_FORWARD_2":
            twist.linear.x = 0.1
            twist.angular.z = 0.0
            if now - self.state_start_time > 2.0:
                self.state = "FOLLOWING"
                self.last_steering = 0.0

        self.publisher.publish(twist)
        cv2.line(roi, (target_x, 0), (target_x, r_h), (255, 0, 0), 2)
        cv2.imshow("Right-Most Logic", roi)
       
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q'):
            self.e_stop_active = True

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publisher.publish(Twist())
    finally:
        node.destroy_node()
        cv2.destroyAllWindows() 
        rclpy.shutdown()

if __name__ == '__main__':
    main()


