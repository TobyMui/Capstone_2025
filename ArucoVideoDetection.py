from cv2 import aruco
import numpy as np
import cv2


'''Init for aruco detection'''
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
detector_params = aruco.DetectorParameters();
detector = aruco.ArucoDetector(marker_dict, detector_params)

'''Variables for camera calibration'''
data = np.load("../calib_data/MultiMatrix.npz")
camera_matrix = data["camMatrix"]
dist_coeffs = data["distCoef"]

'''Init for video Capture'''
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    if not ret:
        break

    '''Apply Calibration'''
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    '''Aruco detection'''
    gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = detector.detectMarkers(gray_frame)

    if marker_corners:
        # Draw detected markers and display IDs
        aruco.drawDetectedMarkers(undistorted_frame, marker_corners, marker_IDs)

        for ids, corners in zip(marker_IDs, marker_corners):
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()

            # Display marker ID
            cv2.putText(
                undistorted_frame,
                f"ID: {ids[0]}",
                tuple(top_right),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.imshow("Capture",undistorted_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break



