import cv2
import numpy as np

# Global list to store the 4 pixel coordinates clicked by the user
image_points = []

def click_event(event, x, y, flags, param):
    """Callback function to record mouse clicks and draw them on the image."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(image_points) < 4:
            image_points.append([x, y])
            # Draw a red dot where clicked
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            # Number the point so you know the order
            cv2.putText(img, str(len(image_points)), (x+10, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Calibration", img)

# --- 1. Load the Image ---
# Replace 'sample.jpg' with the actual filename of your saved camera frame
image_path = 'calibration_img.png' 
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load {image_path}. Check the filename.")
    exit()

# --- 2. Collect Pixel Points via UI ---
cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", click_event)

print("INSTRUCTIONS:")
print("Click the 4 corners of a known physical rectangle on the floor.")
print("ORDER MATTERS! You must click them in this exact sequence:")
print("  1. Bottom-Left  (Closest to robot, left side)")
print("  2. Bottom-Right (Closest to robot, right side)")
print("  3. Top-Right    (Furthest from robot, right side)")
print("  4. Top-Left     (Furthest from robot, left side)")
print("\nAfter clicking 4 points, press any key on your keyboard to continue.")

cv2.imshow("Calibration", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(image_points) != 4:
    print("You didn't click exactly 4 points. Exiting.")
    exit()

src_pixels = np.array(image_points, dtype=np.float32)

# --- 3. Define Real-World Physical Points (IN METERS) ---
# NOTE: You must change these numbers to match your actual physical measurements!
# ROS standard: X is forward, Y is left. The origin (0,0) is your camera lens.
#
# Example scenario: 
# You taped a box on the floor. It is 0.5 meters wide, and 0.6 meters long.
# The closest edge is 0.3 meters directly in front of the camera lens.

x_close = 0.10       # Distance from camera to the closest edge (forward)
x_far = x_close + 0.2159 # Distance from camera to the furthest edge
y_left = 0.1397        # Distance from center to the left edge
y_right = -0.1397      # Distance from center to the right edge (negative in ROS)

dst_meters = np.array([
    [x_close, y_left],  # 1. Bottom-Left
    [x_close, y_right], # 2. Bottom-Right
    [x_far, y_right],   # 3. Top-Right
    [x_far, y_left]     # 4. Top-Left
], dtype=np.float32)

# --- 4. Calculate Homography Matrix ---
# This matrix converts (u, v) pixels directly into (X, Y) physical meters.
H_matrix, _ = cv2.findHomography(src_pixels, dst_meters)

print("\n================ CALIBRATION SUCCESSFUL ================")
print("Copy this array into your ROS 2 Occupancy Grid Node:\n")
print("self.H = np.array(")
print(repr(H_matrix))
print(")")
print("========================================================")

