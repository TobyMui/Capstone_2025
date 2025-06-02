"""
ArUco Marker Detection and Tracking System
Detects markers, highlights designated marker in green, others in red.
Shares position data via shared memory for camera control.
"""

from cv2 import aruco
import numpy as np
import cv2
from multiprocessing import shared_memory

# =============================================================================
# CONFIGURATION
# =============================================================================

# Set designated marker ID (change to your target marker)
global_designated_aruco = 24

# IPC flag for other processes
global_aruco_detected_flag = 0

# Camera Select
global_camera_select = 0

# =============================================================================
# SHARED MEMORY FOR POSITION DATA
# =============================================================================

# Create shared memory: [found, center_x, center_y, frame_w, frame_h, error_x, error_y]
position_mem = shared_memory.SharedMemory(create=True, size=28, name="aruco_position")

def update_marker_position(marker_corners, marker_IDs, frame_shape):
    """Update designated marker position in shared memory"""
    h, w = frame_shape[:2]
    frame_center_x, frame_center_y = w // 2, h // 2

    # Default: marker not found
    position_data = np.array([0.0, 0.0, 0.0, float(w), float(h), 0.0, 0.0], dtype=np.float32)

    if marker_corners and global_designated_aruco is not None:
        # Find designated marker
        for i, ids in enumerate(marker_IDs):
            if ids[0] == global_designated_aruco:
                # Calculate marker center
                corners = marker_corners[i].reshape(4, 2)
                marker_center_x = np.mean(corners[:, 0])
                marker_center_y = np.mean(corners[:, 1])

                # Calculate error from frame center
                error_x = marker_center_x - frame_center_x
                error_y = marker_center_y - frame_center_y

                # Update position data
                position_data = np.array([
                    1.0, marker_center_x, marker_center_y,
                    float(w), float(h), error_x, error_y
                ], dtype=np.float32)
                break

    # Write to shared memory
    position_mem.buf[:28] = position_data.tobytes()

# =============================================================================
# ARUCO SETUP
# =============================================================================

# ArUco detection setup
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
detector_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(marker_dict, detector_params)

# Load camera calibration data
data = np.load("./calib_data/MultiMatrix.npz")
camera_matrix = data["camMatrix"]
dist_coeffs = data["distCoef"]

# Initialize camera
cap = cv2.VideoCapture(global_camera_select)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Press Q to exit")

# =============================================================================
# MAIN LOOP
# =============================================================================

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    # Get optimal camera matrix for undistortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # Apply camera calibration
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Convert to grayscale for detection
    gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    marker_corners, marker_IDs, reject = detector.detectMarkers(gray_frame)

    # Update position data for other processes
    update_marker_position(marker_corners, marker_IDs, frame.shape)

    # Visualize markers
    if marker_corners:
        # Draw frame center crosshair
        center_x, center_y = w // 2, h // 2
        cv2.drawMarker(undistorted_frame, (center_x, center_y), (255, 255, 255),
                      cv2.MARKER_CROSS, 20, 2)
        cv2.putText(undistorted_frame, "CENTER", (center_x - 30, center_y - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Process each detected marker
        for i, (ids, corners) in enumerate(zip(marker_IDs, marker_corners)):
            is_designated = (global_designated_aruco is not None and ids[0] == global_designated_aruco)

            # Set colors: green for designated, red for others
            if is_designated:
                marker_color = (0, 255, 0)
                text_color = (0, 255, 0)
                thickness = 3
            else:
                marker_color = (0, 0, 255)
                text_color = (0, 0, 255)
                thickness = 2

            # Draw marker outline
            corners_reshaped = corners.reshape(4, 2).astype(int)
            cv2.polylines(undistorted_frame, [corners_reshaped],
                         isClosed=True, color=marker_color, thickness=thickness)

            # Draw marker center
            marker_center_x = int(np.mean(corners_reshaped[:, 0]))
            marker_center_y = int(np.mean(corners_reshaped[:, 1]))
            cv2.circle(undistorted_frame, (marker_center_x, marker_center_y), 5, marker_color, -1)

            # Add ID label
            top_right = corners_reshaped[0]
            cv2.putText(undistorted_frame, f"ID: {ids[0]}", tuple(top_right),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

            # Extra info for designated marker
            if is_designated:
                # "DESIGNATED" label
                designated_pos = (top_right[0], top_right[1] + 25)
                cv2.putText(undistorted_frame, "DESIGNATED", designated_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Line from marker to center
                cv2.line(undistorted_frame, (marker_center_x, marker_center_y),
                        (center_x, center_y), (0, 255, 255), 2)

                # Display offset
                error_x = marker_center_x - center_x
                error_y = marker_center_y - center_y
                offset_text = f"Offset: ({error_x:+.0f}, {error_y:+.0f})"
                cv2.putText(undistorted_frame, offset_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display frame
    cv2.imshow("Capture", undistorted_frame)

    # Exit on 'q' or window close
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Capture", cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()