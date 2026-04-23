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

        # -------- Parameters --------
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('debug_image_topic', '/lane/debug_image')
        self.declare_parameter('lane_offset_topic', '/lane/offset')

        # ROI settings: use lower portion of image
        self.declare_parameter('roi_y_start_ratio', 0.55)
        self.declare_parameter('roi_y_end_ratio', 1.0)

        # HSV threshold values
        # These are starter values only. Tune for your tape/floor/lighting.
        self.declare_parameter('h_min', 0)
        self.declare_parameter('h_max', 179)
        self.declare_parameter('s_min', 0)
        self.declare_parameter('s_max', 80)
        self.declare_parameter('v_min', 180)
        self.declare_parameter('v_max', 255)

        # Contour filtering
        self.declare_parameter('min_contour_area', 250.0)

        # Morphology
        self.declare_parameter('erode_iterations', 1)
        self.declare_parameter('dilate_iterations', 2)

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

        self.get_logger().info(f'Lane detector listening on {image_topic}')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        debug_frame, lane_offset = self.process_frame(frame)

        # Publish offset
        offset_msg = Float32()
        offset_msg.data = float(lane_offset)
        self.offset_pub.publish(offset_msg)

        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {e}')

        if self.get_parameter('show_debug_windows').value:
            cv2.imshow('Lane Debug', debug_frame)
            cv2.waitKey(1)

    def process_frame(self, frame):
        """
        Returns:
            debug_frame: annotated BGR image
            lane_offset: normalized offset from image center
                         negative = lane center left of robot center
                         positive = lane center right of robot center
        """
        h, w, _ = frame.shape
        debug_frame = frame.copy()

        # ---------- ROI ----------
        y_start_ratio = self.get_parameter('roi_y_start_ratio').value
        y_end_ratio = self.get_parameter('roi_y_end_ratio').value

        y1 = int(h * y_start_ratio)
        y2 = int(h * y_end_ratio)

        roi = frame[y1:y2, :].copy()
        roi_debug = roi.copy()

        # ---------- HSV threshold ----------
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        h_min = self.get_parameter('h_min').value
        h_max = self.get_parameter('h_max').value
        s_min = self.get_parameter('s_min').value
        s_max = self.get_parameter('s_max').value
        v_min = self.get_parameter('v_min').value
        v_max = self.get_parameter('v_max').value

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # ---------- Morphological cleanup ----------
        erode_iterations = self.get_parameter('erode_iterations').value
        dilate_iterations = self.get_parameter('dilate_iterations').value

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_iterations)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)

        # ---------- Split image into left/right halves ----------
        mid_x = w // 2
        roi_h, roi_w = mask.shape

        left_mask = mask[:, :mid_x]
        right_mask = mask[:, mid_x:]

        left_centroid = self.find_best_contour_centroid(left_mask, x_offset=0)
        right_centroid = self.find_best_contour_centroid(right_mask, x_offset=mid_x)

        # Draw all useful guide lines
        cv2.line(roi_debug, (mid_x, 0), (mid_x, roi_h), (255, 0, 0), 2)

        lane_offset = 0.0
        lane_center_x = None

        # ---------- Draw centroids and compute lane center ----------
        if left_centroid is not None:
            cv2.circle(roi_debug, left_centroid, 8, (0, 255, 0), -1)
            cv2.putText(
                roi_debug,
                'L',
                (left_centroid[0] + 10, left_centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        if right_centroid is not None:
            cv2.circle(roi_debug, right_centroid, 8, (0, 0, 255), -1)
            cv2.putText(
                roi_debug,
                'R',
                (right_centroid[0] + 10, right_centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        if left_centroid is not None and right_centroid is not None:
            lane_center_x = int((left_centroid[0] + right_centroid[0]) / 2)
            cv2.circle(roi_debug, (lane_center_x, roi_h // 2), 8, (0, 255, 255), -1)
            cv2.line(
                roi_debug,
                (lane_center_x, 0),
                (lane_center_x, roi_h),
                (0, 255, 255),
                2
            )

            lane_offset_pixels = lane_center_x - mid_x
            lane_offset = lane_offset_pixels / float(mid_x)

        elif right_centroid is not None:
            # If only right boundary is visible, estimate desired center
            # by assuming robot should stay left of the solid right lane.
            estimated_lane_half_width = int(roi_w * 0.25)
            lane_center_x = right_centroid[0] - estimated_lane_half_width
            lane_center_x = max(0, min(roi_w - 1, lane_center_x))

            cv2.circle(roi_debug, (lane_center_x, roi_h // 2), 8, (0, 255, 255), -1)
            cv2.line(
                roi_debug,
                (lane_center_x, 0),
                (lane_center_x, roi_h),
                (0, 255, 255),
                2
            )

            lane_offset_pixels = lane_center_x - mid_x
            lane_offset = lane_offset_pixels / float(mid_x)

        elif left_centroid is not None:
            # If only left/dividing line is visible, estimate lane center
            estimated_lane_half_width = int(roi_w * 0.25)
            lane_center_x = left_centroid[0] + estimated_lane_half_width
            lane_center_x = max(0, min(roi_w - 1, lane_center_x))

            cv2.circle(roi_debug, (lane_center_x, roi_h // 2), 8, (0, 255, 255), -1)
            cv2.line(
                roi_debug,
                (lane_center_x, 0),
                (lane_center_x, roi_h),
                (0, 255, 255),
                2
            )

            lane_offset_pixels = lane_center_x - mid_x
            lane_offset = lane_offset_pixels / float(mid_x)

        else:
            # No lane found
            lane_offset = 0.0
            cv2.putText(
                roi_debug,
                'NO LANE DETECTED',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        # Draw offset text
        cv2.putText(
            roi_debug,
            f'offset: {lane_offset:+.3f}',
            (20, roi_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Put ROI debug back into full image
        debug_frame[y1:y2, :] = roi_debug

        # Small mask preview in corner
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.resize(mask_bgr, (w // 3, (y2 - y1) // 3))
        mh, mw, _ = mask_bgr.shape
        debug_frame[0:mh, 0:mw] = mask_bgr

        return debug_frame, lane_offset

    def find_best_contour_centroid(self, binary_img, x_offset=0):
        """
        Finds the centroid of the largest contour above minimum area.
        Returns (cx, cy) in ROI coordinates, or None.
        """
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.get_parameter('min_contour_area').value

        best_contour = None
        best_area = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area and area > best_area:
                best_area = area
                best_contour = contour

        if best_contour is None:
            return None

        moments = cv2.moments(best_contour)
        if moments['m00'] == 0:
            return None

        cx = int(moments['m10'] / moments['m00']) + x_offset
        cy = int(moments['m01'] / moments['m00'])

        return (cx, cy)


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