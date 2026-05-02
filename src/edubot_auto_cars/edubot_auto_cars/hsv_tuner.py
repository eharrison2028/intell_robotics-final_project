import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class HSVTuner(Node):
    def __init__(self):
        super().__init__('hsv_tuner')
        self.subscription = self.create_subscription(Image, '/camera_2/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        
        cv2.namedWindow('HSV Tuner')
        # Create trackbars for White
        cv2.createTrackbar('W_H_Min', 'HSV Tuner', 0, 180, self.nothing)
        cv2.createTrackbar('W_S_Min', 'HSV Tuner', 0, 255, self.nothing)
        cv2.createTrackbar('W_V_Min', 'HSV Tuner', 200, 255, self.nothing) # Start Value high
        cv2.createTrackbar('W_V_Max', 'HSV Tuner', 255, 255, self.nothing)

    def nothing(self, x):
        pass

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w, _ = frame.shape
        roi = frame[int(h*0.4):int(h*0.9), :] # Match your crop
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Get current positions of trackbars
        h_min = cv2.getTrackbarPos('W_H_Min', 'HSV Tuner')
        s_min = cv2.getTrackbarPos('W_S_Min', 'HSV Tuner')
        v_min = cv2.getTrackbarPos('W_V_Min', 'HSV Tuner')
        v_max = cv2.getTrackbarPos('W_V_Max', 'HSV Tuner')

        lower_white = np.array([h_min, s_min, v_min])
        upper_white = np.array([180, 40, v_max]) # Keep Saturation low for white

        mask = cv2.inRange(hsv, lower_white, upper_white)
        result = cv2.bitwise_and(roi, roi, mask=mask)

        cv2.imshow('Original ROI', roi)
        cv2.imshow('Mask', mask)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = HSVTuner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


