import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge

import cv2
import numpy as np


class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__('lane_detector_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('debug_image_topic', '/lane/debug_image')
        self.declare_parameter('lane_offset_topic', '/lane/offset')

        # ROI: lower part of image
        self.declare_parameter('roi_y_start_ratio', 0.55)
        self.declare_parameter('roi_y_end_ratio', 1.0)

        # White mask HSV thresholds
        self.declare_parameter('white_h_min', 0)
        self.declare_parameter('white_h_max', 179)
        self.declare_parameter('white_s_min', 0)
        self.declare_parameter('white_s_max', 80)
        self.declare_parameter('white_v_min', 180)
        self.declare_parameter('white_v_max', 255)

        # Yellow mask HSV thresholds
        self.declare_parameter('yellow_h_min', 15)
        self.declare_parameter('yellow_h_max', 40)
        self.declare_parameter('yellow_s_min', 80)
        self.declare_parameter('yellow_s_max', 255)
        self.declare_parameter('yellow_v_min', 100)
        self.declare_parameter('yellow_v_max', 255)

        # Contour filtering
        self.declare_parameter('min_contour_area_white', 250.0)
        self.declare_parameter('min_contour_area_yellow', 120.0)

        # Morphology
        self.declare_parameter('erode_iterations', 1)
        self.declare_parameter('dilate_iterations', 2)

        # Lane geometry
        # Used when only one boundary is visible.
        self.declare_parameter('estimated_lane_width_ratio', 0.35)

        # Smoothing
        self.declare_parameter('smoothing_alpha', 0.3)

        # Visualization
        self.declare_parameter('show_debug_windows', False)

        image_topic = self.get_parameter('image_topic').value
        debug_image_topic = self.get_parameter('debug_image_topic').value
        lane_offset_topic = self.get_parameter('lane_offset_topic').value

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.debug_pub = self.create_publisher(Image, debug_image_topic, 10)
        self.offset_pub = self.create_publisher(Float32, lane_offset_topic, 10)

        # Smoothed state
        self.prev_lane_center_x = None
        self.prev_lane_offset = 0.0

        self.get_logger().info(f'Lane detector listening on {image_topic}')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        debug_frame, lane_offset = self.process_frame(frame)

        offset_msg = Float32()
        offset_msg.data = float(lane_offset)
        self.offset_pub.publish(offset_msg)

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {e}')

        if self.get_parameter('show_debug_windows').value:
            cv2.imshow('Lane Debug', debug_frame)
            cv2.waitKey(1)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        debug_frame = frame.copy()

        # ---------------- ROI ----------------
        y_start_ratio = self.get_parameter('roi_y_start_ratio').value
        y_end_ratio = self.get_parameter('roi_y_end_ratio').value

        y1 = int(h * y_start_ratio)
        y2 = int(h * y_end_ratio)

        roi = frame[y1:y2, :].copy()
        roi_debug = roi.copy()

        roi_h, roi_w, _ = roi.shape
        mid_x = roi_w // 2

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ---------------- White mask ----------------
        white_lower = np.array([
            self.get_parameter('white_h_min').value,
            self.get_parameter('white_s_min').value,
            self.get_parameter('white_v_min').value
        ], dtype=np.uint8)

        white_upper = np.array([
            self.get_parameter('white_h_max').value,
            self.get_parameter('white_s_max').value,
            self.get_parameter('white_v_max').value
        ], dtype=np.uint8)

        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        # ---------------- Yellow mask ----------------
        yellow_lower = np.array([
            self.get_parameter('yellow_h_min').value,
            self.get_parameter('yellow_s_min').value,
            self.get_parameter('yellow_v_min').value
        ], dtype=np.uint8)

        yellow_upper = np.array([
            self.get_parameter('yellow_h_max').value,
            self.get_parameter('yellow_s_max').value,
            self.get_parameter('yellow_v_max').value
        ], dtype=np.uint8)

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        # ---------------- Morphological cleanup ----------------
        erode_iterations = self.get_parameter('erode_iterations').value
        dilate_iterations = self.get_parameter('dilate_iterations').value
        kernel = np.ones((5, 5), np.uint8)

        white_mask = cv2.erode(white_mask, kernel, iterations=erode_iterations)
        white_mask = cv2.dilate(white_mask, kernel, iterations=dilate_iterations)

        yellow_mask = cv2.erode(yellow_mask, kernel, iterations=erode_iterations)
        yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=dilate_iterations)

        # ---------------- Find contours ----------------
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Right boundary: prefer white contour on right side
        right_centroid = self.find_best_contour_centroid(
            contours=white_contours,
            min_area=self.get_parameter('min_contour_area_white').value,
            image_width=roi_w,
            prefer='right'
        )

        # Left divider: prefer yellow contour on left side
        left_centroid = self.find_best_contour_centroid(
            contours=yellow_contours,
            min_area=self.get_parameter('min_contour_area_yellow').value,
            image_width=roi_w,
            prefer='left'
        )

        # Fallback: if no yellow left divider is found, allow white on left
        # for paths where both sides are white.
        left_source = 'yellow'
        if left_centroid is None:
            left_centroid = self.find_best_contour_centroid(
                contours=white_contours,
                min_area=self.get_parameter('min_contour_area_white').value,
                image_width=roi_w,
                prefer='left'
            )
            if left_centroid is not None:
                left_source = 'white'

        # ---------------- Draw guide line ----------------
        cv2.line(roi_debug, (mid_x, 0), (mid_x, roi_h), (255, 0, 0), 2)

        # ---------------- Lane center logic ----------------
        lane_center_x = None
        estimated_lane_width = int(roi_w * self.get_parameter('estimated_lane_width_ratio').value)
        confidence = 0

        if right_centroid is not None:
            confidence += 1
        if left_centroid is not None:
            confidence += 1

        # Best case: both visible
        if right_centroid is not None and left_centroid is not None:
            lane_center_x = int((right_centroid[0] + left_centroid[0]) / 2)

        # Primary mode: right boundary visible
        elif right_centroid is not None:
            lane_center_x = right_centroid[0] - estimated_lane_width

        # Fallback only: left visible
        elif left_centroid is not None:
            lane_center_x = left_centroid[0] + estimated_lane_width

        # Nothing visible: hold previous estimate if available
        else:
            if self.prev_lane_center_x is not None:
                lane_center_x = self.prev_lane_center_x

        # ---------------- Smoothing ----------------
        if lane_center_x is not None:
            if self.prev_lane_center_x is None:
                smoothed_lane_center_x = lane_center_x
            else:
                alpha = self.get_parameter('smoothing_alpha').value
                smoothed_lane_center_x = int(
                    (1.0 - alpha) * self.prev_lane_center_x + alpha * lane_center_x
                )

            self.prev_lane_center_x = smoothed_lane_center_x
        else:
            smoothed_lane_center_x = None

        # ---------------- Compute offset ----------------
        if smoothed_lane_center_x is not None:
            lane_offset_pixels = smoothed_lane_center_x - mid_x
            lane_offset = lane_offset_pixels / float(mid_x)
            self.prev_lane_offset = lane_offset
        else:
            lane_offset = self.prev_lane_offset

        # ---------------- Draw results ----------------
        if left_centroid is not None:
            left_color = (0, 255, 255) if left_source == 'yellow' else (255, 255, 255)
            cv2.circle(roi_debug, left_centroid, 8, left_color, -1)
            cv2.putText(
                roi_debug,
                f'L ({left_source})',
                (left_centroid[0] + 10, max(20, left_centroid[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                left_color,
                2
            )

        if right_centroid is not None:
            cv2.circle(roi_debug, right_centroid, 8, (0, 0, 255), -1)
            cv2.putText(
                roi_debug,
                'R (white)',
                (right_centroid[0] + 10, max(20, right_centroid[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        if smoothed_lane_center_x is not None:
            cv2.circle(roi_debug, (smoothed_lane_center_x, roi_h // 2), 8, (0, 255, 0), -1)
            cv2.line(
                roi_debug,
                (smoothed_lane_center_x, 0),
                (smoothed_lane_center_x, roi_h),
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                roi_debug,
                'NO LANE ESTIMATE',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        # Draw estimated offset text
        cv2.putText(
            roi_debug,
            f'offset: {lane_offset:+.3f}',
            (20, roi_h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            roi_debug,
            f'confidence: {confidence}/2',
            (20, roi_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        # ---------------- Mask previews ----------------
        white_mask_bgr = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        yellow_mask_bgr = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)

        preview_w = w // 4
        preview_h = (y2 - y1) // 4

        white_preview = cv2.resize(white_mask_bgr, (preview_w, preview_h))
        yellow_preview = cv2.resize(yellow_mask_bgr, (preview_w, preview_h))

        cv2.putText(white_preview, 'WHITE', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(yellow_preview, 'YELLOW', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        debug_frame[y1:y2, :] = roi_debug

        # put previews in top-left
        debug_frame[0:preview_h, 0:preview_w] = white_preview
        debug_frame[0:preview_h, preview_w:2 * preview_w] = yellow_preview

        return debug_frame, lane_offset

    def find_best_contour_centroid(self, contours, min_area, image_width, prefer='left'):
        """
        Selects a contour centroid using a simple score:
        - prefers larger contour area
        - prefers the requested side of the image

        prefer = 'left' or 'right'
        """
        best_centroid = None
        best_score = -1e9

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Larger contours are preferred
            score = area

            # Add side preference
            if prefer == 'left':
                # smaller cx is better
                score += (image_width - cx) * 0.5
            elif prefer == 'right':
                # larger cx is better
                score += cx * 0.5

            # Slightly prefer contours lower in image (closer to robot)
            score += cy * 0.2

            if score > best_score:
                best_score = score
                best_centroid = (cx, cy)

        return best_centroid


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if node.get_parameter('show_debug_windows').value:
        cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()