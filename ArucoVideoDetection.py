"""
ArUco Marker Detection and Tracking System

This script captures video from a camera, applies calibration correction,
and detects ArUco markers in real-time. It displays the video feed with
detected markers highlighted and their IDs labeled.

Requirements:
- OpenCV with ArUco support
- Camera calibration data (MultiMatrix.npz)
- Connected camera device

Author: Toby
"""

# Import required
from cv2 import aruco  # ArUco marker detection functionality
import numpy as np  # Numerical operations and array handling
import cv2  # Computer vision and image processing

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

global_designated_aruco = 6 # Global variable for aruco id selection
global_aruco_detected_flag = 0 # Flag for inter-process communication (IPC)
global_camera_select = 0




# =============================================================================
# ARUCO DETECTION INITIALIZATION
# =============================================================================

# Create ArUco dictionary - defines the specific marker set to detect
# DICT_6X6_250: 6x6 bit markers from a dictionary of 250 unique markers
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Initialize detection parameters with default settings
# These parameters control detection sensitivity, accuracy, and performance
detector_params = aruco.DetectorParameters()

# Create the ArUco detector object that will perform marker detection
detector = aruco.ArucoDetector(marker_dict, detector_params)

# =============================================================================
# CAMERA CALIBRATION DATA
# =============================================================================

# Load pre-computed camera calibration data
data = np.load("./calib_data/MultiMatrix.npz")

# Camera intrinsic matrix - contains focal lengths and optical center
camera_matrix = data["camMatrix"]

# Distortion coefficients - corrects for lens distortion effects
dist_coeffs = data["distCoef"]

# =============================================================================
# VIDEO CAPTURE INITIALIZATION
# =============================================================================

# Initialize video capture from camera (index 0 = default camera)
cap = cv2.VideoCapture(global_camera_select)
if not cap.isOpened():
    print("Error: Could not open camera try a different index")
    exit()  # Exit program if camera cannot be accessed

# User instruction for program termination
print("Press Q to exit")

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

while True:
    # Capture a single frame from the video stream
    # ret: boolean indicating successful frame capture
    # frame: the actual image data as numpy array
    ret, frame = cap.read()

    # Get frame dimensions for camera matrix optimization
    h, w = frame.shape[:2]  # height and width of the frame

    # Calculate optimal camera matrix for the current frame size
    # This improves undistortion quality by optimizing the region of interest
    # Parameters: original matrix, distortion coeffs, image size, alpha, new size
    # alpha=1 retains all pixels, alpha=0 crops to valid pixels only
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # Exit loop if frame capture failed (camera disconnected, end of video, etc.)
    if not ret:
        break

    # =========================================================================
    # CAMERA CALIBRATION APPLICATION
    # =========================================================================

    # Remove lens distortion from the captured frame
    # This corrects for barrel/pincushion distortion and improves detection accuracy
    undistorted_frame = cv2.undistort(
        frame,  # Input distorted image
        camera_matrix,  # Original camera matrix
        dist_coeffs,  # Distortion coefficients
        None,  # Optional rectification transformation (None = identity)
        new_camera_matrix  # Optimized camera matrix
    )

    # =========================================================================
    # ARUCO MARKER DETECTION
    # =========================================================================

    # Convert color image to grayscale for marker detection
    # ArUco detection works on single-channel (grayscale) images
    gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

    # Perform ArUco marker detection on the grayscale image
    # Returns:
    # - marker_corners: list of detected marker corner coordinates
    # - marker_IDs: list of detected marker IDs
    # - reject: list of rejected marker candidates (for debugging)
    marker_corners, marker_IDs, reject = detector.detectMarkers(gray_frame)

    # =========================================================================
    # MARKER VISUALIZATION AND LABELING
    # =========================================================================

    if marker_corners:
        # Loop through each detected marker for custom visualization
        for i, (ids, corners) in enumerate(zip(marker_IDs, marker_corners)):
            # Determine if this is the designated marker
            is_designated = (global_designated_aruco is not None and
                             ids[0] == global_designated_aruco)

            # Set colors based on designation status
            if is_designated:
                marker_color = (0, 255, 0)  # Green for designated marker
                text_color = (0, 255, 0)  # Green text
                thickness = 3  # Thicker border for emphasis
            else:
                marker_color = (0, 0, 255)  # Red for other markers
                text_color = (0, 0, 255)  # Red text
                thickness = 2  # Standard thickness

            # Draw custom colored rectangle around the marker
            # Reshape corner coordinates from (1,4,2) to (4,2) format
            corners_reshaped = corners.reshape(4, 2).astype(int)

            # Draw lines connecting all four corners to form marker outline
            cv2.polylines(undistorted_frame, [corners_reshaped],
                          isClosed=True, color=marker_color, thickness=thickness)

            # Get the top-right corner coordinates for ID label placement
            # corners[0] corresponds to top-right corner in ArUco convention
            top_right = corners_reshaped[0]

            # Add text label showing the marker ID with appropriate color
            cv2.putText(
                undistorted_frame,  # Image to draw on
                f"ID: {ids[0]}",  # Text to display (marker ID)
                tuple(top_right),  # Position (top-right corner)
                cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                0.8,  # Font scale (size)
                text_color,  # Color matches marker outline
                2,  # Thickness
                cv2.LINE_AA,  # Anti-aliasing for smooth text
            )

            # Optional: Add designation status text for the designated marker
            if is_designated:
                # Add "DESIGNATED" label below the ID
                designated_pos = (top_right[0], top_right[1] + 25)
                cv2.putText(
                    undistorted_frame,
                    "DESIGNATED",
                    designated_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),  # Green color
                    2,
                    cv2.LINE_AA,
                )

    # =========================================================================
    # DISPLAY AND USER INTERACTION
    # =========================================================================

    # Display the processed frame with detected markers and labels
    cv2.imshow("Capture", undistorted_frame)

    # Handle user input and window events
    # waitKey(1): wait 1ms for key press, enables real-time processing
    # & 0xFF: ensures compatibility across different platforms
    key = cv2.waitKey(1) & 0xFF

    # Check for exit conditions:
    # 1. 'q' key pressed by user
    # 2. Window closed by clicking X button
    # getWindowProperty checks if window is still visible (returns <1 if closed)
    if key == ord('q') or cv2.getWindowProperty("Capture", cv2.WND_PROP_VISIBLE) < 1:
        break

# =============================================================================
# CLEANUP AND RESOURCE RELEASE
# =============================================================================

# Release the camera resource to make it available for other applications
cap.release()

# Close all OpenCV windows and free associated memory
cv2.destroyAllWindows()