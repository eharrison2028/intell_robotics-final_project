import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import time


import cv2
import numpy as np


from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


# ── Lane detection defaults ──────────────────────────────────────
WHITE_H_MIN,  WHITE_H_MAX  =   0, 180
WHITE_S_MIN,  WHITE_S_MAX  =  10,  30
WHITE_V_MIN,  WHITE_V_MAX  = 220, 255


YELLOW_H_MIN, YELLOW_H_MAX =  25,  60
YELLOW_S_MIN, YELLOW_S_MAX =  80, 255
YELLOW_V_MIN, YELLOW_V_MAX =  80, 255


ORANGE_H_MIN, ORANGE_H_MAX =   5,  30
ORANGE_S_MIN, ORANGE_S_MAX =  70, 255
ORANGE_V_MIN, ORANGE_V_MAX = 150, 255


MIN_CONTOUR_AREA = 2500


TARGET_WHITE_X_RATIO  = 0.20   # white ~20% from right edge
TARGET_YELLOW_X_RATIO = 0.10   # yellow ~10% from left edge


# ── Navigation defaults ──────────────────────────────────────────
DEFAULT_KP         = 1.2
DEFAULT_KD         = 0.2
DEFAULT_BASE_SPEED = 0.15
DEFAULT_MAX_TURN   = 1.5


RIGHT_TURN_DURATION   = 2.2
FORK_RIGHT_TIMEOUT    = 4.0
RECOVERY_DURATION     = 2.0
RECOVERY_ANGULAR      = 0.4
WHITE_ABSENT_THRESH   = 25


UTURN_SPIN_SPEED    = 0.9
UTURN_SPIN_DURATION = 1.75
UTURN_FWD_DURATION  = 3.5




class NavigationNode(Node):
   def __init__(self):
       super().__init__('navigation_node')


       # ── Lane detection parameters ────────────────────────────
       self.declare_parameter('white_h_min',  WHITE_H_MIN)
       self.declare_parameter('white_h_max',  WHITE_H_MAX)
       self.declare_parameter('white_s_min',  WHITE_S_MIN)
       self.declare_parameter('white_s_max',  WHITE_S_MAX)
       self.declare_parameter('white_v_min',  WHITE_V_MIN)
       self.declare_parameter('white_v_max',  WHITE_V_MAX)


       self.declare_parameter('yellow_h_min', YELLOW_H_MIN)
       self.declare_parameter('yellow_h_max', YELLOW_H_MAX)
       self.declare_parameter('yellow_s_min', YELLOW_S_MIN)
       self.declare_parameter('yellow_s_max', YELLOW_S_MAX)
       self.declare_parameter('yellow_v_min', YELLOW_V_MIN)
       self.declare_parameter('yellow_v_max', YELLOW_V_MAX)


       self.declare_parameter('orange_h_min', ORANGE_H_MIN)
       self.declare_parameter('orange_h_max', ORANGE_H_MAX)
       self.declare_parameter('orange_s_min', ORANGE_S_MIN)
       self.declare_parameter('orange_s_max', ORANGE_S_MAX)
       self.declare_parameter('orange_v_min', ORANGE_V_MIN)
       self.declare_parameter('orange_v_max', ORANGE_V_MAX)


       self.declare_parameter('crop_top_ratio', 0.5)
       self.declare_parameter('crop_bottom_ratio', 0.1)
       self.declare_parameter('crop_left_ratio', 0.1)
       self.declare_parameter('crop_top_orange_ratio', 0.1)
       self.declare_parameter('yellow_target_x_ratio', TARGET_YELLOW_X_RATIO)
       self.declare_parameter('yellow_weight', 0.5)
       self.declare_parameter('right_bias', 0.3)
       self.declare_parameter('yellow_memory_secs', 0.5)
       self.declare_parameter('min_orange_pixels', 4500)
       self.declare_parameter('debug_image', False)


       # ── Navigation parameters ────────────────────────────────
       self.declare_parameter('kp',         DEFAULT_KP)
       self.declare_parameter('kd',         DEFAULT_KD)
       self.declare_parameter('base_speed', DEFAULT_BASE_SPEED)
       self.declare_parameter('max_turn',   DEFAULT_MAX_TURN)


       # ── Lane detection state ────────────────────────────────
       self.bridge = CvBridge()
       self.last_yellow_cx    = None
       self.last_yellow_stamp = 0.0


       self.current_error   = 0.0   # [-1, 1]
       self.end_of_road     = False
       self.white_detected  = True


       # ── Navigation state machine ─────────────────────────────
       self.state             = 'FOLLOWING'
       self.manoeuvre_start   = None
       self.last_error        = 0.0
       self.last_error_time   = time.time()
       self.white_absent_cycles = 0
       self.turn_phase        = 0


       # ── ROS I/O ─────────────────────────────────────────────
       self.create_subscription(Image, '/camera_2/image_raw',
                                self.image_callback, 10)
       self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)


       # Control loop at 20 Hz
       self.create_timer(0.05, self._control_loop)


       self.get_logger().info('Single navigation node started (detection + control).')


   # ─────────────────────────────────────────────────────────────
   # IMAGE CALLBACK: compute error, white_detected, end_of_road
   # ─────────────────────────────────────────────────────────────
   def image_callback(self, msg: Image):
       try:
           frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
       except Exception as e:
           self.get_logger().error(f'cv_bridge error: {e}')
           return


       h, w = frame.shape[:2]


       crop_top    = self.get_parameter('crop_top_ratio').value
       crop_bottom = self.get_parameter('crop_bottom_ratio').value
       crop_left   = self.get_parameter('crop_left_ratio').value


       roi_y     = int(h * crop_top)
       roi_y_bot = int(h * (1.0 - crop_bottom))
       roi_x     = int(w * crop_left)


       roi = frame[roi_y:roi_y_bot, roi_x:w]
       w_roi = roi.shape[1]


       orange_top = int(h * self.get_parameter('crop_top_orange_ratio').value)
       roi_orange = frame[orange_top:roi_y_bot, roi_x:]


       hsv        = cv2.cvtColor(roi,        cv2.COLOR_BGR2HSV)
       hsv_orange = cv2.cvtColor(roi_orange, cv2.COLOR_BGR2HSV)


       white_mask        = self._white_mask(hsv)
       yellow_mask       = self._yellow_mask(hsv)
       orange_mask       = self._orange_mask(hsv_orange)


       # White line centroid
       white_error = 0.0
       white_cx    = None
       white_cnt   = self._largest_contour(white_mask)
       white_available = white_cnt is not None


       if white_cnt is not None:
           M = cv2.moments(white_cnt)
           if M['m00'] > 0:
               white_cx = int(M['m10'] / M['m00'])
               target_white_x = int(w_roi * (1.0 - TARGET_WHITE_X_RATIO))
               white_error = (white_cx - target_white_x) / float(w_roi / 2)


       # Yellow line centroid (with memory)
       yellow_error       = 0.0
       yellow_cx          = None
       yellow_available   = False


       now_sec   = self.get_clock().now().nanoseconds * 1e-9
       yellow_cnt = self._largest_contour(yellow_mask)


       if yellow_cnt is not None:
           M = cv2.moments(yellow_cnt)
           if M['m00'] > 0:
               yellow_cx = int(M['m10'] / M['m00'])
               target_yellow_x = int(w_roi * self.get_parameter('yellow_target_x_ratio').value)
               yellow_error = (yellow_cx - target_yellow_x) / float(w_roi / 2)
               self.last_yellow_cx    = yellow_cx
               self.last_yellow_stamp = now_sec
               yellow_available = True
       else:
           memory_secs = self.get_parameter('yellow_memory_secs').value
           if (self.last_yellow_cx is not None and
                   (now_sec - self.last_yellow_stamp) < memory_secs):
               yellow_cx = self.last_yellow_cx
               target_yellow_x = int(w_roi * self.get_parameter('yellow_target_x_ratio').value)
               yellow_error = (yellow_cx - target_yellow_x) / float(w_roi / 2)
               yellow_available = True


       # Blend errors
       yw = self.get_parameter('yellow_weight').value
       if white_available and yellow_available:
           final_error = (1.0 - yw) * white_error + yw * yellow_error
       elif white_available:
           final_error = white_error
       elif yellow_available:
           bias = self.get_parameter('right_bias').value
           final_error = yellow_error + bias
       else:
           final_error = 0.0


       final_error = max(-1.5, min(1.5, final_error))


       # Orange end-of-road detection
       orange_pixels = int(cv2.countNonZero(orange_mask))
       min_orange    = self.get_parameter('min_orange_pixels').value
       self.end_of_road = orange_pixels >= min_orange


       self.current_error  = float(final_error)
       self.white_detected = white_available


   # ─────────────────────────────────────────────────────────────
   # CONTROL LOOP: PD + state machine
   # ─────────────────────────────────────────────────────────────
   def _control_loop(self):
       cmd = Twist()


       # Handle end-of-road trigger
       if self.end_of_road and self.state in ('FOLLOWING', 'FORK_RIGHT'):
           self.get_logger().info('End of road detected → TURNING_AROUND (phase 0)')
           self.turn_phase = 0
           self._start_manoeuvre('TURNING_AROUND')
           self.end_of_road = False  # consume event


       if self.state == 'STOPPED':
           self.pub_cmd.publish(cmd)
           return


       if self.state == 'TURNING_RIGHT':
           elapsed = time.time() - self.manoeuvre_start
           if elapsed < RIGHT_TURN_DURATION:
               cmd.linear.x  = self.get_parameter('base_speed').value * 0.6
               cmd.angular.z = -self.get_parameter('max_turn').value * 0.8
           else:
               self.get_logger().info('Right turn done → FOLLOWING')
               self.state = 'FOLLOWING'
           self.pub_cmd.publish(cmd)
           return


       if self.state == 'TURNING_AROUND':
           if self.white_detected and self.turn_phase != 0:
               self.get_logger().info('White line found mid-turn → FOLLOWING')
               self.turn_phase = 0
               self.white_absent_cycles = 0
               self.state = 'FOLLOWING'
               return


           elapsed = time.time() - self.manoeuvre_start
           if self.turn_phase == 0:
               if elapsed < UTURN_SPIN_DURATION:
                   cmd.angular.z = UTURN_SPIN_SPEED
               else:
                   self.get_logger().info('U-turn phase 1 → forward')
                   self.turn_phase = 1
                   self.manoeuvre_start = time.time()
           elif self.turn_phase == 1:
               if elapsed < UTURN_FWD_DURATION:
                   cmd.linear.x = self.get_parameter('base_speed').value
               else:
                   self.get_logger().info('U-turn phase 2 → second 90°')
                   self.turn_phase = 2
                   self.manoeuvre_start = time.time()
           elif self.turn_phase == 2:
               if elapsed < UTURN_SPIN_DURATION:
                   cmd.angular.z = UTURN_SPIN_SPEED
               else:
                   self.get_logger().info('U-turn done → LANE_RECOVERY')
                   self.turn_phase = 0
                   self._start_manoeuvre('LANE_RECOVERY')
           self.pub_cmd.publish(cmd)
           return


       if self.state == 'LANE_RECOVERY':
           elapsed = time.time() - self.manoeuvre_start
           if elapsed < RECOVERY_DURATION:
               cmd.linear.x  = self.get_parameter('base_speed').value * 0.5
               cmd.angular.z = RECOVERY_ANGULAR
           else:
               self.get_logger().info('Lane recovery done → FOLLOWING')
               self.state = 'FOLLOWING'
               self.white_absent_cycles = 0
           self.pub_cmd.publish(cmd)
           return


       if self.state == 'FORK_RIGHT':
           elapsed = time.time() - self.manoeuvre_start
           if self.white_detected or elapsed > FORK_RIGHT_TIMEOUT:
               self.get_logger().info('Fork resolved → FOLLOWING')
               self.state = 'FOLLOWING'
               self.white_absent_cycles = 0
               return
           cmd.linear.x  = self.get_parameter('base_speed').value * 0.5
           cmd.angular.z = -self.get_parameter('max_turn').value * 0.6
           self.pub_cmd.publish(cmd)
           return


       # FOLLOWING: PD control on current_error
       if not self.white_detected:
           self.white_absent_cycles += 1
       else:
           self.white_absent_cycles = 0


       if self.white_absent_cycles >= WHITE_ABSENT_THRESH:
           self.get_logger().info('White absent → FORK_RIGHT')
           self.white_absent_cycles = 0
           self._start_manoeuvre('FORK_RIGHT')
           return


       kp = self.get_parameter('kp').value
       kd = self.get_parameter('kd').value
       base_speed = self.get_parameter('base_speed').value
       max_turn   = self.get_parameter('max_turn').value


       now = time.time()
       dt = now - self.last_error_time
       if dt <= 0:
           dt = 0.05
       d_error = (self.current_error - self.last_error) / dt


       raw_turn = -(kp * self.current_error + kd * d_error)
       cmd.angular.z = max(-max_turn, min(max_turn, raw_turn))
       cmd.linear.x  = base_speed * (1.0 - 0.7 * abs(self.current_error))
       cmd.linear.x  = max(0.05, cmd.linear.x)


       self.last_error = self.current_error
       self.last_error_time = now


       self.pub_cmd.publish(cmd)


   # ── Helpers ──────────────────────────────────────────────────
   def _start_manoeuvre(self, state: str):
       self.state = state
       self.manoeuvre_start = time.time()


   def _white_mask(self, hsv):
       lo = np.array([self.get_parameter('white_h_min').value,
                      self.get_parameter('white_s_min').value,
                      self.get_parameter('white_v_min').value])
       hi = np.array([self.get_parameter('white_h_max').value,
                      self.get_parameter('white_s_max').value,
                      self.get_parameter('white_v_max').value])
       mask = cv2.inRange(hsv, lo, hi)
       return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


   def _yellow_mask(self, hsv):
       lo = np.array([self.get_parameter('yellow_h_min').value,
                      self.get_parameter('yellow_s_min').value,
                      self.get_parameter('yellow_v_min').value])
       hi = np.array([self.get_parameter('yellow_h_max').value,
                      self.get_parameter('yellow_s_max').value,
                      self.get_parameter('yellow_v_max').value])
       mask = cv2.inRange(hsv, lo, hi)
       return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


   def _orange_mask(self, hsv):
       lo = np.array([self.get_parameter('orange_h_min').value,
                      self.get_parameter('orange_s_min').value,
                      self.get_parameter('orange_v_min').value])
       hi = np.array([self.get_parameter('orange_h_max').value,
                      self.get_parameter('orange_s_max').value,
                      self.get_parameter('orange_v_max').value])
       mask = cv2.inRange(hsv, lo, hi)
       return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))


   def _largest_contour(self, mask):
       cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
       if not cnts:
           return None
       best = max(cnts, key=cv2.contourArea)
       if cv2.contourArea(best) < MIN_CONTOUR_AREA:
           return None
       return best




def main(args=None):
   rclpy.init(args=args)
   node = NavigationNode()
   try:
       rclpy.spin(node)
   except KeyboardInterrupt:
       pass
   node.destroy_node()
   rclpy.shutdown()




if __name__ == '__main__':
   main()









