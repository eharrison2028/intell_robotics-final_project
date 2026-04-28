import time
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np


class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower')

        self.bridge = CvBridge()

        # ---------------- Topics ----------------
        self.declare_parameter('image_topic', '/camera_2/image_raw')
        self.declare_parameter('lidar_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('debug_image_topic', '/lane/debug_image')

        # ---------------- Conservative driving ----------------
        self.declare_parameter('forward_speed', 0.055)
        self.declare_parameter('turn_gain', 0.12)
        self.declare_parameter('max_angular_speed', 0.16)
        self.declare_parameter('steering_sign', 1.0)

        # ---------------- Right white-line following ----------------
        self.declare_parameter('desired_right_line_offset_ratio', 0.24)
        self.declare_parameter('smoothing_alpha', 0.12)

        # ---------------- Hard white boundary behavior ----------------
        self.declare_parameter('white_boundary_danger_ratio', 0.58)
        self.declare_parameter('white_boundary_warning_ratio', 0.68)
        self.declare_parameter('white_boundary_no_right_turn_ratio', 0.75)
        self.declare_parameter('white_boundary_escape_turn', 0.22)
        self.declare_parameter('white_boundary_warning_turn', 0.16)
        self.declare_parameter('white_boundary_warning_speed', 0.025)

        # ---------------- ROI ----------------
        self.declare_parameter('roi_y_start_ratio', 0.58)
        self.declare_parameter('roi_y_end_ratio', 1.0)

        # ---------------- White tape HSV ----------------
        self.declare_parameter('white_h_min', 0)
        self.declare_parameter('white_h_max', 179)
        self.declare_parameter('white_s_min', 0)
        self.declare_parameter('white_s_max', 90)
        self.declare_parameter('white_v_min', 150)
        self.declare_parameter('white_v_max', 255)

        # ---------------- Yellow tape HSV ----------------
        self.declare_parameter('yellow_h_min', 15)
        self.declare_parameter('yellow_h_max', 40)
        self.declare_parameter('yellow_s_min', 70)
        self.declare_parameter('yellow_s_max', 255)
        self.declare_parameter('yellow_v_min', 90)
        self.declare_parameter('yellow_v_max', 255)

        # ---------------- Orange turnaround line HSV ----------------
        self.declare_parameter('orange_h_min', 5)
        self.declare_parameter('orange_h_max', 18)
        self.declare_parameter('orange_s_min', 90)
        self.declare_parameter('orange_s_max', 255)
        self.declare_parameter('orange_v_min', 100)
        self.declare_parameter('orange_v_max', 255)
        self.declare_parameter('orange_area_ratio_trigger', 0.04)

        # ---------------- Contour filtering ----------------
        self.declare_parameter('min_white_area', 180.0)
        self.declare_parameter('min_yellow_area', 90.0)

        # ---------------- Obstacle behavior ----------------
        self.declare_parameter('use_lidar_obstacle', False)
        self.declare_parameter('obstacle_stop_distance', 0.45)
        self.declare_parameter('front_lidar_degrees', 25.0)

        # ---------------- Turnaround behavior ----------------
        self.declare_parameter('turnaround_angular_speed', 0.45)
        self.declare_parameter('turnaround_duration', 3.2)
        self.declare_parameter('turnaround_cooldown', 4.0)

        # ---------------- Debug ----------------
        self.declare_parameter('show_debug_windows', False)

        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.image_callback,
            10
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            self.get_parameter('lidar_topic').value,
            self.lidar_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            10
        )

        self.debug_pub = self.create_publisher(
            Image,
            self.get_parameter('debug_image_topic').value,
            10
        )

        self.prev_target_x = None
        self.obstacle_detected = False

        self.turning_around = False
        self.turnaround_start_time = 0.0
        self.last_turnaround_time = 0.0

        self.last_warn_times = {}

        self.get_logger().info('Lane follower started with hard white-line boundary behavior.')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        state = self.process_image(frame)
        cmd = self.compute_command(state)

        self.cmd_pub.publish(cmd)
        self.publish_debug(state, cmd)

    def lidar_callback(self, msg):
        if not self.get_parameter('use_lidar_obstacle').value:
            self.obstacle_detected = False
            return

        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)

        front_ranges = []
        front_deg = float(self.get_parameter('front_lidar_degrees').value)

        for i, distance in enumerate(ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if abs(math.degrees(angle)) <= front_deg:
                front_ranges.append(distance)

        if len(front_ranges) == 0:
            self.obstacle_detected = False
            return

        self.obstacle_detected = (
            min(front_ranges) < float(self.get_parameter('obstacle_stop_distance').value)
        )

    def process_image(self, frame):
        h, w, _ = frame.shape

        y1 = int(h * float(self.get_parameter('roi_y_start_ratio').value))
        y2 = int(h * float(self.get_parameter('roi_y_end_ratio').value))

        roi = frame[y1:y2, :].copy()
        roi_h, roi_w, _ = roi.shape
        mid_x = roi_w // 2

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        white_mask = self.make_mask(
            hsv,
            'white_h_min', 'white_s_min', 'white_v_min',
            'white_h_max', 'white_s_max', 'white_v_max'
        )

        yellow_mask = self.make_mask(
            hsv,
            'yellow_h_min', 'yellow_s_min', 'yellow_v_min',
            'yellow_h_max', 'yellow_s_max', 'yellow_v_max'
        )

        orange_mask = self.make_mask(
            hsv,
            'orange_h_min', 'orange_s_min', 'orange_v_min',
            'orange_h_max', 'orange_s_max', 'orange_v_max'
        )

        white_mask = self.clean_mask(white_mask)
        yellow_mask = self.clean_mask(yellow_mask)
        orange_mask = self.clean_mask(orange_mask)

        right_white = self.find_best_right_white(white_mask)
        yellow_left = self.find_best_yellow_left(yellow_mask)

        orange_area_ratio = cv2.countNonZero(orange_mask) / float(roi_w * roi_h)
        orange_detected = orange_area_ratio > float(
            self.get_parameter('orange_area_ratio_trigger').value
        )

        line_found = right_white is not None
        target_x = None
        error = 0.0

        if line_found:
            desired_offset = int(
                roi_w * float(self.get_parameter('desired_right_line_offset_ratio').value)
            )

            raw_target_x = right_white['x'] - desired_offset
            raw_target_x = max(0, min(roi_w - 1, raw_target_x))

            if self.prev_target_x is None:
                target_x = raw_target_x
            else:
                alpha = float(self.get_parameter('smoothing_alpha').value)
                target_x = int((1.0 - alpha) * self.prev_target_x + alpha * raw_target_x)

            self.prev_target_x = target_x
            error = float(target_x - mid_x) / float(mid_x)
        else:
            self.prev_target_x = None

        return {
            'frame': frame,
            'roi': roi,
            'roi_y1': y1,
            'roi_y2': y2,
            'roi_h': roi_h,
            'roi_w': roi_w,
            'mid_x': mid_x,
            'white_mask': white_mask,
            'yellow_mask': yellow_mask,
            'orange_mask': orange_mask,
            'right_white': right_white,
            'yellow_left': yellow_left,
            'target_x': target_x,
            'error': error,
            'line_found': line_found,
            'orange_detected': orange_detected,
            'orange_area_ratio': orange_area_ratio,
        }

    def make_mask(self, hsv, hmin, smin, vmin, hmax, smax, vmax):
        lower = np.array([
            self.get_parameter(hmin).value,
            self.get_parameter(smin).value,
            self.get_parameter(vmin).value
        ], dtype=np.uint8)

        upper = np.array([
            self.get_parameter(hmax).value,
            self.get_parameter(smax).value,
            self.get_parameter(vmax).value
        ], dtype=np.uint8)

        return cv2.inRange(hsv, lower, upper)

    def clean_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask

    def find_best_right_white(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape

        best = None
        best_score = -1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < float(self.get_parameter('min_white_area').value):
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Only accept likely right-side white tape.
            if cx < int(0.40 * w):
                continue

            score = area + 1.0 * cx + 0.4 * cy

            if score > best_score:
                best_score = score
                best = {'x': cx, 'y': cy, 'area': area}

        return best

    def find_best_yellow_left(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape

        best = None
        best_score = -1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < float(self.get_parameter('min_yellow_area').value):
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            score = area + 1.0 * (w - cx) + 0.3 * cy

            if score > best_score:
                best_score = score
                best = {'x': cx, 'y': cy, 'area': area}

        return best

    def compute_command(self, state):
        cmd = Twist()
        now = time.time()

        should_turnaround = state['orange_detected'] or self.obstacle_detected

        if should_turnaround and not self.turning_around:
            if now - self.last_turnaround_time > float(self.get_parameter('turnaround_cooldown').value):
                self.turning_around = True
                self.turnaround_start_time = now
                self.last_turnaround_time = now
                self.get_logger().info('Turnaround triggered.')

        if self.turning_around:
            elapsed = now - self.turnaround_start_time

            if elapsed < float(self.get_parameter('turnaround_duration').value):
                cmd.linear.x = 0.0
                cmd.angular.z = float(self.get_parameter('turnaround_angular_speed').value)
                return cmd

            self.turning_around = False
            self.prev_target_x = None
            self.get_logger().info('Turnaround complete.')

        # If the white boundary is not detected, stop.
        if not state['line_found'] or state['right_white'] is None:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.warn_throttle(
                'line_lost',
                'Right white line lost. Stopping to avoid crossing boundary.',
                1.0
            )
            return cmd

        right_white_x = state['right_white']['x']
        roi_w = state['roi_w']

        danger_x = int(float(self.get_parameter('white_boundary_danger_ratio').value) * roi_w)
        warning_x = int(float(self.get_parameter('white_boundary_warning_ratio').value) * roi_w)
        no_right_turn_x = int(float(self.get_parameter('white_boundary_no_right_turn_ratio').value) * roi_w)

        # HARD BOUNDARY:
        # The right white line must remain far to the right.
        # If it moves toward the center, stop forward motion and steer left.
        if right_white_x < danger_x:
            cmd.linear.x = 0.0
            cmd.angular.z = abs(float(self.get_parameter('white_boundary_escape_turn').value))
            self.warn_throttle(
                'white_boundary_danger',
                'WHITE LINE DANGER: stopping forward motion and steering left.',
                0.5
            )
            return cmd

        if right_white_x < warning_x:
            cmd.linear.x = float(self.get_parameter('white_boundary_warning_speed').value)
            cmd.angular.z = abs(float(self.get_parameter('white_boundary_warning_turn').value))
            self.warn_throttle(
                'white_boundary_warning',
                'White line close. Correcting left.',
                0.8
            )
            return cmd

        # Normal right-white-line following.
        error = state['error']

        if abs(error) < 0.10:
            error = 0.0

        steering_sign = float(self.get_parameter('steering_sign').value)
        turn_gain = float(self.get_parameter('turn_gain').value)

        angular = steering_sign * turn_gain * error

        max_ang = float(self.get_parameter('max_angular_speed').value)
        angular = max(min(angular, max_ang), -max_ang)

        # Extra safety: if close to the white line, do not allow right turns.
        if right_white_x < no_right_turn_x and angular < 0.0:
            angular = 0.0

        cmd.linear.x = float(self.get_parameter('forward_speed').value)
        cmd.angular.z = angular

        return cmd

    def publish_debug(self, state, cmd):
        frame = state['frame'].copy()
        roi = state['roi'].copy()

        roi_h = state['roi_h']
        roi_w = state['roi_w']
        mid_x = state['mid_x']

        danger_x = int(float(self.get_parameter('white_boundary_danger_ratio').value) * roi_w)
        warning_x = int(float(self.get_parameter('white_boundary_warning_ratio').value) * roi_w)

        cv2.line(roi, (mid_x, 0), (mid_x, roi_h), (255, 0, 0), 2)
        cv2.line(roi, (danger_x, 0), (danger_x, roi_h), (0, 0, 255), 2)
        cv2.line(roi, (warning_x, 0), (warning_x, roi_h), (0, 165, 255), 2)

        if state['right_white'] is not None:
            p = state['right_white']
            cv2.circle(roi, (p['x'], p['y']), 8, (255, 255, 255), -1)
            cv2.putText(
                roi,
                'RIGHT WHITE BOUNDARY',
                (max(5, p['x'] - 220), max(25, p['y'])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        if state['yellow_left'] is not None:
            p = state['yellow_left']
            cv2.circle(roi, (p['x'], p['y']), 8, (0, 255, 255), -1)
            cv2.putText(
                roi,
                'YELLOW LEFT',
                (p['x'] + 8, max(25, p['y'])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        if state['target_x'] is not None:
            cv2.line(roi, (state['target_x'], 0), (state['target_x'], roi_h), (0, 255, 0), 2)
            cv2.putText(
                roi,
                'TARGET',
                (state['target_x'] + 8, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        if self.turning_around:
            status = 'TURNAROUND'
        elif state['orange_detected']:
            status = 'ORANGE DETECTED'
        elif not state['line_found']:
            status = 'STOPPED: NO WHITE BOUNDARY'
        elif state['right_white']['x'] < danger_x:
            status = 'WHITE DANGER'
        elif state['right_white']['x'] < warning_x:
            status = 'WHITE WARNING'
        else:
            status = 'FOLLOWING WHITE BOUNDARY'

        cv2.putText(
            roi,
            f"error: {state['error']:+.3f}",
            (20, roi_h - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            roi,
            f"cmd x:{cmd.linear.x:.2f} z:{cmd.angular.z:.2f}",
            (20, roi_h - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            roi,
            f"orange: {state['orange_area_ratio']:.3f}",
            (20, roi_h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            roi,
            status,
            (20, roi_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if status == 'FOLLOWING WHITE BOUNDARY' else (0, 0, 255),
            2
        )

        frame[state['roi_y1']:state['roi_y2'], :] = roi

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().warn(f'Debug image publish failed: {e}')

        if self.get_parameter('show_debug_windows').value:
            cv2.imshow('lane_debug', frame)
            cv2.waitKey(1)

    def warn_throttle(self, key, message, interval):
        now = time.time()
        last = self.last_warn_times.get(key, 0.0)

        if now - last > interval:
            self.get_logger().warn(message)
            self.last_warn_times[key] = now

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        if node.get_parameter('show_debug_windows').value:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()