import rclpy
from rclpy.node import Node


from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from cv_bridge import CvBridge


import cv2
import numpy as np
import math
import time


try:
   from tf2_ros import TransformBroadcaster, Buffer, TransformListener
   HAS_TF2 = True
except ImportError:
   HAS_TF2 = False




# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAP_RESOLUTION    = 0.05   # metres per cell
MAP_WIDTH_M       = 10.0   # initial map width  (metres)
MAP_HEIGHT_M      = 10.0   # initial map height (metres)
OCCUPANCY_FREE    = 0
OCCUPANCY_OCC     = 100
OCCUPANCY_UNKNOWN = -1


# HSV ranges – lanes only (no obstacle detection via camera) ----------------
#  White  – outer lane lines
WHITE_H_MIN, WHITE_H_MAX = 0,   179
WHITE_S_MIN, WHITE_S_MAX = 0,   60
WHITE_V_MIN, WHITE_V_MAX = 180, 255


#  Yellow – dashed centre line
YELLOW_H_MIN, YELLOW_H_MAX = 15,  35
YELLOW_S_MIN, YELLOW_S_MAX = 80,  255
YELLOW_V_MIN, YELLOW_V_MAX = 100, 255


#  Orange – end-of-road marker
ORANGE_H_MIN, ORANGE_H_MAX = 5,  20
ORANGE_S_MIN, ORANGE_S_MAX = 150, 255
ORANGE_V_MIN, ORANGE_V_MAX = 150, 255


WHITE_MIN_AREA  = 300   # pixels²
YELLOW_MIN_AREA = 200
ORANGE_MIN_AREA = 150


# Perspective model for lane pixel → world coordinate ----------------------
CAMERA_HEIGHT_M   = 0.20    # camera above ground (metres)
FOCAL_LENGTH_PX   = 320.0   # approximate focal length (pixels)
ROI_Y_START_RATIO = 0.55    # must match lane_detector settings


# LiDAR obstacle settings ---------------------------------------------------
LIDAR_MAX_RANGE_M      = 4.0    # ignore returns beyond this distance
LIDAR_MIN_RANGE_M      = 0.05   # ignore returns closer than this (self-hits)
LIDAR_OBSTACLE_DEDUP_M = 0.10   # deduplicate obstacle markers within 10 cm
# Beam angles outside this window are ignored (filters out floor/ceiling hits
# on tilted sensors). Set to (-math.pi, math.pi) to accept all bearings.
LIDAR_ANGLE_MIN = -math.pi / 2   # −90°
LIDAR_ANGLE_MAX =  math.pi / 2   #  +90°  (front hemisphere only)




class RoadMapper(Node):
   """
   Builds a 2-D occupancy grid of the road ahead, with separate
   RViz2 marker layers for:
     - white outer lane lines        (HSV camera)
     - yellow centre dashes          (HSV camera)
     - orange end-of-road markers    (HSV camera)
     - obstacles                     (2-D LiDAR /scan)


   The map origin is reset every time the node starts (no persistence).
   Robot pose is tracked via /odom if available; otherwise dead-reckoning
   at a fixed forward speed is used as a fallback.
   """


   def __init__(self):
       super().__init__('road_mapper')


       # ---- Parameters ----
       self.declare_parameter('image_topic',        '/camera/image_raw')
       self.declare_parameter('scan_topic',         '/scan')
       self.declare_parameter('map_topic',          '/road_map')
       self.declare_parameter('marker_topic',       '/road_markers')
       self.declare_parameter('obstacle_marker_topic', '/obstacle_markers')
       self.declare_parameter('map_frame',          'map')
       self.declare_parameter('robot_frame',        'base_link')
       self.declare_parameter('map_resolution',     MAP_RESOLUTION)
       self.declare_parameter('map_width_m',        MAP_WIDTH_M)
       self.declare_parameter('map_height_m',       MAP_HEIGHT_M)
       self.declare_parameter('publish_rate_hz',    2.0)
       self.declare_parameter('show_debug_windows', False)


       # LiDAR parameters
       self.declare_parameter('lidar_max_range_m',      LIDAR_MAX_RANGE_M)
       self.declare_parameter('lidar_min_range_m',      LIDAR_MIN_RANGE_M)
       self.declare_parameter('lidar_angle_min',        LIDAR_ANGLE_MIN)
       self.declare_parameter('lidar_angle_max',        LIDAR_ANGLE_MAX)
       self.declare_parameter('lidar_obstacle_dedup_m', LIDAR_OBSTACLE_DEDUP_M)


       # HSV colour parameters (lane detection only, no obstacle colours)
       for prefix, defaults in [
           ('white',  (WHITE_H_MIN,  WHITE_H_MAX,  WHITE_S_MIN,  WHITE_S_MAX,  WHITE_V_MIN,  WHITE_V_MAX)),
           ('yellow', (YELLOW_H_MIN, YELLOW_H_MAX, YELLOW_S_MIN, YELLOW_S_MAX, YELLOW_V_MIN, YELLOW_V_MAX)),
           ('orange', (ORANGE_H_MIN, ORANGE_H_MAX, ORANGE_S_MIN, ORANGE_S_MAX, ORANGE_V_MIN, ORANGE_V_MAX)),
       ]:
           self.declare_parameter(f'{prefix}_h_min', defaults[0])
           self.declare_parameter(f'{prefix}_h_max', defaults[1])
           self.declare_parameter(f'{prefix}_s_min', defaults[2])
           self.declare_parameter(f'{prefix}_s_max', defaults[3])
           self.declare_parameter(f'{prefix}_v_min', defaults[4])
           self.declare_parameter(f'{prefix}_v_max', defaults[5])


       self.declare_parameter('min_contour_area', 200.0)
       self.declare_parameter('camera_height_m',  CAMERA_HEIGHT_M)
       self.declare_parameter('focal_length_px',  FOCAL_LENGTH_PX)


       # ---- State ----
       self.bridge = CvBridge()


       res  = self.get_parameter('map_resolution').value
       w_m  = self.get_parameter('map_width_m').value
       h_m  = self.get_parameter('map_height_m').value
       self.resolution    = res
       self.map_w_cells   = int(w_m / res)
       self.map_h_cells   = int(h_m / res)


       # Occupancy grid (-1 = unknown, 0 = free, 100 = occupied)
       self.grid = np.full(
           (self.map_h_cells, self.map_w_cells), OCCUPANCY_UNKNOWN, dtype=np.int8
       )


       # Robot starts at map centre
       self.robot_x_m  = w_m / 2.0
       self.robot_y_m  = h_m / 2.0
       self.robot_yaw  = 0.0   # radians; 0 = +x direction (east)


       # Lane marker store: {id: {'wx', 'wy', 'colour'}}
       self._lane_marker_store: dict[int, dict] = {}
       self._next_lane_marker_id = 0


       # Obstacle marker store (LiDAR): {id: {'wx', 'wy'}}
       self._obstacle_marker_store: dict[int, dict] = {}
       self._next_obstacle_marker_id = 0


       # End-of-road flag
       self.end_of_road_detected = False


       # ---- Publishers ----
       self.map_pub      = self.create_publisher(
           OccupancyGrid, self.get_parameter('map_topic').value, 10)
       self.marker_pub   = self.create_publisher(
           MarkerArray, self.get_parameter('marker_topic').value, 10)
       self.obstacle_pub = self.create_publisher(
           MarkerArray, self.get_parameter('obstacle_marker_topic').value, 10)


       # ---- Subscribers ----
       image_topic = self.get_parameter('image_topic').value
       scan_topic  = self.get_parameter('scan_topic').value


       self.image_sub = self.create_subscription(
           Image, image_topic, self.image_callback, 10)
       self.scan_sub  = self.create_subscription(
           LaserScan, scan_topic, self.scan_callback, 10)


       # Optional odometry for robot pose
       try:
           from nav_msgs.msg import Odometry
           self.odom_sub = self.create_subscription(
               Odometry, '/odom', self.odom_callback, 10)
           self._odom_x0   = None
           self._odom_y0   = None
           self._odom_yaw0 = None
           self._has_odom  = True
       except Exception:
           self._has_odom = False


       # TF broadcaster so RViz2 can find map → base_link
       if HAS_TF2:
           self.tf_broadcaster = TransformBroadcaster(self)
       else:
           self.tf_broadcaster = None


       # Periodic publish timer
       rate = self.get_parameter('publish_rate_hz').value
       self.create_timer(1.0 / rate, self.publish_map)


       self._start_time = time.monotonic()


       self.get_logger().info(
           f'RoadMapper started | map {self.map_w_cells}×{self.map_h_cells} cells '
           f'@ {res:.3f} m/cell | camera: {image_topic} | lidar: {scan_topic}'
       )


   # -----------------------------------------------------------------------
   # Odometry callback
   # -----------------------------------------------------------------------
   def odom_callback(self, msg):
       x   = msg.pose.pose.position.x
       y   = msg.pose.pose.position.y
       q   = msg.pose.pose.orientation
       yaw = self._quat_to_yaw(q.x, q.y, q.z, q.w)


       if self._odom_x0 is None:
           self._odom_x0   = x
           self._odom_y0   = y
           self._odom_yaw0 = yaw
           return


       w_m = self.get_parameter('map_width_m').value
       h_m = self.get_parameter('map_height_m').value
       self.robot_x_m = (w_m / 2.0) + (x - self._odom_x0)
       self.robot_y_m = (h_m / 2.0) + (y - self._odom_y0)
       self.robot_yaw = yaw - self._odom_yaw0


   # -----------------------------------------------------------------------
   # LiDAR callback – obstacle detection
   # -----------------------------------------------------------------------
   def scan_callback(self, msg: LaserScan):
       """
       Convert every valid LaserScan beam into a world-frame obstacle point
       and store it in the obstacle marker/grid store.


       The 2-D LiDAR is assumed to be mounted at the robot origin (base_link).
       Beam angles are in the LiDAR frame; we rotate them by robot_yaw to get
       map-frame directions.
       """
       max_r   = self.get_parameter('lidar_max_range_m').value
       min_r   = self.get_parameter('lidar_min_range_m').value
       a_min   = self.get_parameter('lidar_angle_min').value
       a_max   = self.get_parameter('lidar_angle_max').value
       dedup   = self.get_parameter('lidar_obstacle_dedup_m').value


       angle = msg.angle_min
       for r in msg.ranges:
           angle_in_scan = angle
           angle += msg.angle_increment


           # Filter by configured angular window
           if angle_in_scan < a_min or angle_in_scan > a_max:
               continue


           # Filter invalid / out-of-range returns
           if not math.isfinite(r) or r < min_r or r > max_r:
               continue


           # Transform beam endpoint from robot frame to map frame
           beam_angle_world = self.robot_yaw + angle_in_scan
           wx = self.robot_x_m + r * math.cos(beam_angle_world)
           wy = self.robot_y_m + r * math.sin(beam_angle_world)


           self._mark_grid(wx, wy, occupied=True)
           self._add_obstacle_marker(wx, wy, dedup)


   # -----------------------------------------------------------------------
   # Image callback – HSV lane detection
   # -----------------------------------------------------------------------
   def image_callback(self, msg: Image):
       try:
           frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
       except Exception as e:
           self.get_logger().error(f'cv_bridge error: {e}')
           return


       h, w = frame.shape[:2]
       y1  = int(h * ROI_Y_START_RATIO)
       roi = frame[y1:, :]


       hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


       # Lane colours only – obstacles come from LiDAR, not HSV
       masks = {
           'white':  self._hsv_mask(hsv, 'white'),
           'yellow': self._hsv_mask(hsv, 'yellow'),
           'orange': self._hsv_mask(hsv, 'orange'),
       }


       min_area = self.get_parameter('min_contour_area').value


       for colour, mask in masks.items():
           cleaned  = self._morph_clean(mask)
           contours, _ = cv2.findContours(
               cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


           for cnt in contours:
               area = cv2.contourArea(cnt)
               if area < min_area:
                   continue


               M = cv2.moments(cnt)
               if M['m00'] == 0:
                   continue


               cx_px = int(M['m10'] / M['m00'])
               cy_px = int(M['m01'] / M['m00']) + y1   # back to full-frame y


               wx, wy = self._pixel_to_world(cx_px, cy_px, w, h)


               if colour == 'orange':
                   self.end_of_road_detected = True
                   self._add_lane_marker(wx, wy, colour)
                   self._mark_grid(wx, wy, occupied=True)
               elif colour == 'white':
                   self._mark_grid(wx, wy, occupied=True)
                   self._add_lane_marker(wx, wy, colour)
               elif colour == 'yellow':
                   self._add_lane_marker(wx, wy, colour)
                   # Yellow centre dashes: mark as free road surface
                   self._mark_grid(wx, wy, occupied=False)


       if self.get_parameter('show_debug_windows').value:
           debug = frame.copy()
           colour_bgr = {
               'white':  (255, 255, 255),
               'yellow': (0, 215, 255),
               'orange': (0, 165, 255),
           }
           for colour, mask in masks.items():
               cleaned  = self._morph_clean(mask)
               contours, _ = cv2.findContours(
                   cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
               cv2.drawContours(debug[y1:], contours, -1, colour_bgr[colour], 2)
           cv2.imshow('RoadMapper – Lane Debug', debug)
           cv2.waitKey(1)


   # -----------------------------------------------------------------------
   # Map + marker publishing
   # -----------------------------------------------------------------------
   def publish_map(self):
       now = self.get_clock().now().to_msg()
       map_frame = self.get_parameter('map_frame').value


       # ---- TF: map → base_link ----
       if self.tf_broadcaster is not None:
           tf_msg = TransformStamped()
           tf_msg.header.stamp    = now
           tf_msg.header.frame_id = map_frame
           tf_msg.child_frame_id  = self.get_parameter('robot_frame').value
           tf_msg.transform.translation.x = self.robot_x_m
           tf_msg.transform.translation.y = self.robot_y_m
           tf_msg.transform.translation.z = 0.0
           q = self._yaw_to_quat(self.robot_yaw)
           tf_msg.transform.rotation.x = q[0]
           tf_msg.transform.rotation.y = q[1]
           tf_msg.transform.rotation.z = q[2]
           tf_msg.transform.rotation.w = q[3]
           self.tf_broadcaster.sendTransform(tf_msg)


       # ---- OccupancyGrid ----
       occ = OccupancyGrid()
       occ.header.stamp    = now
       occ.header.frame_id = map_frame


       meta = MapMetaData()
       meta.map_load_time        = now
       meta.resolution           = self.resolution
       meta.width                = self.map_w_cells
       meta.height               = self.map_h_cells
       meta.origin.position.x    = 0.0
       meta.origin.position.y    = 0.0
       meta.origin.position.z    = 0.0
       meta.origin.orientation.w = 1.0
       occ.info = meta


       occ.data = self.grid.flatten().tolist()
       self.map_pub.publish(occ)


       # ---- Lane MarkerArray (white / yellow / orange) ----
       lane_array = MarkerArray()


       clear = Marker()
       clear.header.frame_id = map_frame
       clear.header.stamp    = now
       clear.action          = Marker.DELETEALL
       lane_array.markers.append(clear)


       lane_colour_map = {
           'white':  (1.0, 1.0, 1.0, 0.9),
           'yellow': (1.0, 0.85, 0.0, 0.9),
           'orange': (1.0, 0.4,  0.0, 1.0),
       }


       for mid, info in self._lane_marker_store.items():
           m = Marker()
           m.header.frame_id = map_frame
           m.header.stamp    = now
           m.ns              = info['colour']
           m.id              = mid
           m.type            = Marker.CYLINDER
           m.action          = Marker.ADD


           m.pose.position.x    = info['wx']
           m.pose.position.y    = info['wy']
           m.pose.position.z    = 0.05
           m.pose.orientation.w = 1.0


           radius    = 0.04 if info['colour'] in ('white', 'yellow') else 0.10
           m.scale.x = radius
           m.scale.y = radius
           m.scale.z = 0.10


           r, g, b, a = lane_colour_map.get(info['colour'], (0.5, 0.5, 0.5, 1.0))
           m.color.r = r
           m.color.g = g
           m.color.b = b
           m.color.a = a
           m.lifetime.sec = 0


           lane_array.markers.append(m)


       # Robot position marker
       robot_marker                    = Marker()
       robot_marker.header.frame_id    = map_frame
       robot_marker.header.stamp       = now
       robot_marker.ns                 = 'robot'
       robot_marker.id                 = 9999
       robot_marker.type               = Marker.ARROW
       robot_marker.action             = Marker.ADD
       robot_marker.pose.position.x    = self.robot_x_m
       robot_marker.pose.position.y    = self.robot_y_m
       robot_marker.pose.position.z    = 0.1
       q = self._yaw_to_quat(self.robot_yaw)
       robot_marker.pose.orientation.x = q[0]
       robot_marker.pose.orientation.y = q[1]
       robot_marker.pose.orientation.z = q[2]
       robot_marker.pose.orientation.w = q[3]
       robot_marker.scale.x = 0.3
       robot_marker.scale.y = 0.05
       robot_marker.scale.z = 0.05
       robot_marker.color.r = 0.0
       robot_marker.color.g = 1.0
       robot_marker.color.b = 0.4
       robot_marker.color.a = 1.0
       lane_array.markers.append(robot_marker)


       # End-of-road text marker
       if self.end_of_road_detected:
           txt                    = Marker()
           txt.header.frame_id    = map_frame
           txt.header.stamp       = now
           txt.ns                 = 'end_of_road'
           txt.id                 = 10000
           txt.type               = Marker.TEXT_VIEW_FACING
           txt.action             = Marker.ADD
           txt.pose.position.x    = self.robot_x_m
           txt.pose.position.y    = self.robot_y_m
           txt.pose.position.z    = 0.6
           txt.pose.orientation.w = 1.0
           txt.scale.z            = 0.2
           txt.color.r            = 1.0
           txt.color.g            = 0.4
           txt.color.b            = 0.0
           txt.color.a            = 1.0
           txt.text               = 'END OF ROAD'
           lane_array.markers.append(txt)


       self.marker_pub.publish(lane_array)


       # ---- Obstacle MarkerArray (LiDAR) – separate topic ----
       obs_array = MarkerArray()


       obs_clear = Marker()
       obs_clear.header.frame_id = map_frame
       obs_clear.header.stamp    = now
       obs_clear.action          = Marker.DELETEALL
       obs_array.markers.append(obs_clear)


       for oid, info in self._obstacle_marker_store.items():
           m = Marker()
           m.header.frame_id    = map_frame
           m.header.stamp       = now
           m.ns                 = 'obstacle'
           m.id                 = oid
           m.type               = Marker.CUBE
           m.action             = Marker.ADD


           m.pose.position.x    = info['wx']
           m.pose.position.y    = info['wy']
           m.pose.position.z    = 0.1
           m.pose.orientation.w = 1.0


           m.scale.x = 0.12
           m.scale.y = 0.12
           m.scale.z = 0.20


           # Bright red so obstacles stand out from lane markers
           m.color.r = 1.0
           m.color.g = 0.1
           m.color.b = 0.1
           m.color.a = 0.85


           m.lifetime.sec = 0


           obs_array.markers.append(m)


       self.obstacle_pub.publish(obs_array)


   # -----------------------------------------------------------------------
   # Helpers
   # -----------------------------------------------------------------------
   def _hsv_mask(self, hsv_img, colour: str) -> np.ndarray:
       lower = np.array([
           self.get_parameter(f'{colour}_h_min').value,
           self.get_parameter(f'{colour}_s_min').value,
           self.get_parameter(f'{colour}_v_min').value,
       ], dtype=np.uint8)
       upper = np.array([
           self.get_parameter(f'{colour}_h_max').value,
           self.get_parameter(f'{colour}_s_max').value,
           self.get_parameter(f'{colour}_v_max').value,
       ], dtype=np.uint8)
       return cv2.inRange(hsv_img, lower, upper)


   @staticmethod
   def _morph_clean(mask: np.ndarray) -> np.ndarray:
       k    = np.ones((5, 5), np.uint8)
       mask = cv2.erode(mask,  k, iterations=1)
       mask = cv2.dilate(mask, k, iterations=2)
       return mask


   def _pixel_to_world(self, px: int, py: int, img_w: int, img_h: int):
       """
       Rough perspective projection:
         rows near the bottom of the image → closer to the robot
         rows near ROI_Y_START            → farther from the robot
       Returns (world_x, world_y) in the map frame.
       """
       cam_h = self.get_parameter('camera_height_m').value
       focal = self.get_parameter('focal_length_px').value


       row_from_bottom = max(img_h - py, 1)
       dist_ahead      = cam_h * focal / row_from_bottom
       dist_ahead      = min(dist_ahead, 5.0)


       lateral = (px - img_w / 2.0) / focal * dist_ahead


       cos_y = math.cos(self.robot_yaw)
       sin_y = math.sin(self.robot_yaw)


       wx = self.robot_x_m + dist_ahead * cos_y - lateral * sin_y
       wy = self.robot_y_m + dist_ahead * sin_y + lateral * cos_y


       return wx, wy


   def _mark_grid(self, wx: float, wy: float, occupied: bool):
       col = int(wx / self.resolution)
       row = int(wy / self.resolution)
       if 0 <= row < self.map_h_cells and 0 <= col < self.map_w_cells:
           self.grid[row, col] = OCCUPANCY_OCC if occupied else OCCUPANCY_FREE


   def _add_lane_marker(self, wx: float, wy: float, colour: str):
       """Add a lane marker, deduplicating within 5 cm."""
       DEDUP = 0.05
       for info in self._lane_marker_store.values():
           if (info['colour'] == colour
                   and abs(info['wx'] - wx) < DEDUP
                   and abs(info['wy'] - wy) < DEDUP):
               return
       mid = self._next_lane_marker_id
       self._next_lane_marker_id += 1
       self._lane_marker_store[mid] = {'wx': wx, 'wy': wy, 'colour': colour}


   def _add_obstacle_marker(self, wx: float, wy: float, dedup_dist: float):
       """Add a LiDAR obstacle marker, deduplicating within dedup_dist metres."""
       for info in self._obstacle_marker_store.values():
           if (abs(info['wx'] - wx) < dedup_dist
                   and abs(info['wy'] - wy) < dedup_dist):
               return
       oid = self._next_obstacle_marker_id
       self._next_obstacle_marker_id += 1
       self._obstacle_marker_store[oid] = {'wx': wx, 'wy': wy}


   @staticmethod
   def _quat_to_yaw(qx, qy, qz, qw) -> float:
       siny_cosp = 2.0 * (qw * qz + qx * qy)
       cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
       return math.atan2(siny_cosp, cosy_cosp)


   @staticmethod
   def _yaw_to_quat(yaw: float):
       return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))




# ---------------------------------------------------------------------------
def main(args=None):
   rclpy.init(args=args)
   node = RoadMapper()
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



