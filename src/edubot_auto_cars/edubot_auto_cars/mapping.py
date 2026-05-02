import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Empty
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import math

class LaneMapper(Node):
    def __init__(self):
        super().__init__('lane_mapper')
        
        # --- ROS Subscriptions & Publishers ---
        self.subscription = self.create_subscription(Image, '/camera_2/image_raw', self.image_callback, 10)
        self.map_publisher = self.create_publisher(OccupancyGrid, '/lane_map', 10)
        self.bridge = CvBridge()

        # --- TF2 Setup ---
        # We need this to get the robot's position in the global SLAM map
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Occupancy Grid Configuration ---
        self.resolution = 0.05  # 5 cm per cell
        self.width = 400        # 20 meters wide
        self.height = 400       # 20 meters tall
        self.origin_x = -10.0   # Center the origin so the robot starts in the middle
        self.origin_y = -10.0
        
        # Initialize the map array with 0 (Transparent space for RViz costmap scheme)
        self.grid_data = np.zeros(self.width * self.height, dtype=np.int8)

        # Create the map reset service
        self.reset_srv = self.create_service(Empty, 'reset_lane_map', self.reset_map_callback)

        # --- Your Custom Calibration Matrix ---
        self.H = np.array([
            [-2.09863060e-04,  7.37222814e-03, -8.35162437e+00],
            [ 1.17726147e-02,  4.69168680e-04, -8.63806519e+00],
            [-1.14045990e-03, -4.10985051e-02,  1.00000000e+00]
        ])

        # HSV Thresholds for lane lines (Updated with V_min = 207)
        self.white_low = np.array([0, 10, 200])
        self.white_high = np.array([180, 30, 255])
        self.yellow_low = np.array([20, 100, 100])
        self.yellow_high = np.array([40, 255, 255])

        self.get_logger().info('Lane Mapper Node Started. Waiting for camera and TF...')

    def get_yaw_from_quaternion(self, q):
        """Helper to convert TF quaternion to Euler Yaw angle."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def reset_map_callback(self, request, response):
        """Wipes the lane map clean without restarting the node."""
        self.grid_data = np.zeros(self.width * self.height, dtype=np.int8)
        self.get_logger().info('Lane map has been wiped clean for a new run!')
        self.publish_map()
        return response

    def image_callback(self, msg):
            try:
                # 1. Get Transform from 'map' to 'base_link'
                t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            except Exception as e:
                return

            # 2. Extract Camera Frame
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            h, w, _ = frame.shape
            roi = frame[int(h*0.4):int(h*0.9), :] # Crop
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

          # 3. Mask the lanes (Pure HSV)
            white_mask = cv2.inRange(hsv, self.white_low, self.white_high)
            yellow_mask = cv2.inRange(hsv, self.yellow_low, self.yellow_high)
            lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

            # --- BULLETPROOF NOISE FILTER (Contour Area) ---
            # 1. Do a very light opening to separate loosely connected noise
            kernel = np.ones((3, 3), np.uint8)
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)

            # 2. Find all isolated white blobs (contours) in the mask
            contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 3. Erase any blob that is too small to be a lane line
            min_blob_area = 750  # Anything smaller than 150 pixels gets deleted
            for cnt in contours:
                if cv2.contourArea(cnt) < min_blob_area:
                    # Paint over the small noise blob with black (0)
                    cv2.drawContours(lane_mask, [cnt], -1, 0, -1)
            # -----------------------------------------------

            # Show the Debug Window
            cv2.imshow("Debug Lane Mask", lane_mask)
            cv2.waitKey(1)






            # --- DEBUG WINDOW ---
            # This pops up a live feed of the mask. 
            # If the window is full of white blobs, your HSV 'V_min' needs to be higher.
            # If it shows clean thin lines, your vision is perfect!
            cv2.imshow("Debug Lane Mask", lane_mask)
            cv2.waitKey(1)
            # --------------------

            # Get coordinates
            y_coords, x_coords = np.nonzero(lane_mask[::2, ::2])
            
            if len(x_coords) == 0:
                self.publish_map()
                return

            x_coords = x_coords * 2
            y_coords = (y_coords * 2) + int(h*0.4)

            # 4. Homography Transformation
            ones = np.ones_like(x_coords)
            pixels = np.vstack((x_coords, y_coords, ones))
            
            physical = self.H @ pixels
            
            x_robot = physical[0, :] / physical[2, :]
            y_robot = physical[1, :] / physical[2, :]

            # --- HORIZON FILTER ---
            # Prevent distant noise from stretching into massive map blobs.
            # Only map pixels that are calculated to be less than 2.5 meters from the robot.
            distances = np.sqrt(x_robot**2 + y_robot**2)
            valid_dist = distances < 2.5  
            
            x_robot = x_robot[valid_dist]
            y_robot = y_robot[valid_dist]

            if len(x_robot) == 0:
                self.publish_map()
                return
            # ----------------------

            # 5. Global Transformation
            robot_x = t.transform.translation.x
            robot_y = t.transform.translation.y
            yaw = self.get_yaw_from_quaternion(t.transform.rotation)

            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            x_map = (x_robot * cos_yaw) - (y_robot * sin_yaw) + robot_x
            y_map = (x_robot * sin_yaw) + (y_robot * cos_yaw) + robot_y

            # 6. Convert to Grid Coordinates
            grid_x = np.floor((x_map - self.origin_x) / self.resolution).astype(int)
            grid_y = np.floor((y_map - self.origin_y) / self.resolution).astype(int)

            valid_idx = (grid_x >= 0) & (grid_x < self.width) & (grid_y >= 0) & (grid_y < self.height)
            grid_x = grid_x[valid_idx]
            grid_y = grid_y[valid_idx]

            indices = grid_x + (grid_y * self.width)
            self.grid_data[indices] = 100

            # 7. Publish
            self.publish_map()

    def publish_map(self):
        """Builds and publishes the OccupancyGrid message."""
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'
        
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        
        # Lift the map 1cm to prevent Z-fighting with the SLAM map
        grid_msg.info.origin.position.z = 0.01 
        grid_msg.info.origin.orientation.w = 1.0
        
        grid_msg.data = self.grid_data.tolist()
        self.map_publisher.publish(grid_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LaneMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


