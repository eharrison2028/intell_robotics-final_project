#!/usr/bin/env python3
import math
import os

import cv2
import message_filters
import numpy as np
import rclpy
from ament_index_python import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray


class ConeLidarFusionNode(Node):
    def __init__(self):
        super().__init__('cone_lidar_fusion')

        default_model_path = os.path.join(
            get_package_share_directory('edubot_auto_cars'), 'models', 'cone_best.pt'
        )
        
        test_path = '/home/developer/project_ws/src/edubot_auto_cars/models/cone_best.pt'
        # --- Parameters ---
        self.declare_parameter('model_path', test_path)
        self.declare_parameter('camera_topic', '/camera_1/image_raw')
        self.declare_parameter('lidar_topic', '/scan')
        self.declare_parameter('camera_fov_horizontal', 62.2) # Degrees (Adjust for your camera)
        
        # Load Model
        model_path = self.get_parameter('model_path').value
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.fov_h = math.radians(self.get_parameter('camera_fov_horizontal').value)

        # --- Publishers ---
        self.marker_pub = self.create_publisher(MarkerArray, '/cone_markers', 10)

        # --- Synchronized Subscriptions ---
        # message_filters ensures we get the image and scan that happened at the same time
        self.image_sub = message_filters.Subscriber(self, Image, self.get_parameter('camera_topic').value)
        self.scan_sub = message_filters.Subscriber(self, LaserScan, self.get_parameter('lidar_topic').value)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.scan_sub], 10, 0.1)
        self.ts.registerCallback(self.fusion_callback)

        self.get_logger().info("Cone/LiDAR Fusion Node Started")

    def fusion_callback(self, img_msg, scan_msg):
        # 1. Run YOLO Detection
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        results = self.model.predict(cv_image, conf=0.5, verbose=False)
        
        marker_array = MarkerArray()
        img_w = cv_image.shape[1]

        if results and len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes):
                # Get Bounding Box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2.0
                
                # 2. Calculate Bearing (Angle) from Image
                # center of image is 0 radians. Left is positive, Right is negative.
                bearing = (0.5 - cx / img_w) * self.fov_h

                # 3. Find Range from LiDAR
                # Filter LiDAR points that fall within the angular span of the bounding box
                box_width_px = x2 - x1
                angle_width = (box_width_px / img_w) * self.fov_h
                
                range_to_cone = self.get_lidar_range(scan_msg, bearing, angle_width)

                if range_to_cone is not None:
                    # 4. Calculate X, Y in robot frame (base_link)
                    # Note: LaserScan 0 angle is usually forward (X-axis)
                    x_robot = range_to_cone * math.cos(bearing)
                    y_robot = range_to_cone * math.sin(bearing)

                    # 5. Create Marker
                    marker = self.create_cone_marker(i, x_robot, y_robot, img_msg.header.stamp)
                    marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def get_lidar_range(self, scan, target_angle, angle_width):
        """Finds the closest LiDAR point within a specific angular slice."""
        min_range = float('inf')
        found = False
        
        # Define the search window
        search_min = target_angle - (angle_width / 2.0)
        search_max = target_angle + (angle_width / 2.0)

        for i, r in enumerate(scan.ranges):
            if r < scan.range_min or r > scan.range_max:
                continue
            
            # Current beam angle
            angle = scan.angle_min + (i * scan.angle_increment)
            
            # Normalize angle to -pi to pi
            angle = math.atan2(math.sin(angle), math.cos(angle))

            if search_min <= angle <= search_max:
                if r < min_range:
                    min_range = r
                    found = True
        
        return min_range if found else None

    def create_cone_marker(self, id, x, y, timestamp):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = timestamp
        marker.ns = "cones"
        marker.id = id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1 # Adjust height if on a box
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.3
        marker.color.r = 1.0 # Orange
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
        return marker

def main(args=None):
    rclpy.init(args=args)
    node = ConeLidarFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

