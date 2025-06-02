import cv2
import numpy as np

# Load the ArUco dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Create detector parameters
detector_params = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

# Load the image
image = cv2.imread("content/ChosenOne.jpg")


# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


corners, ids, rejectedCandidates = detector.detectMarkers(gray)

# Draw detected markers
if ids is not None:
    # cv2.aruco.drawDetectedMarkers(image, corners, ids)

    # Then draw thicker borders manually around each detected marker
    for corner in corners:
        corner = corner[0].astype(int)  # Flatten the array for easier access
        cv2.polylines(image, [corner], isClosed=True, color=(0, 255, 0), thickness=5)  # Adjust thickness as needed
    for i, corner in enumerate(corners):
        # Calculate the position for the ID text (top-left corner of the marker)
        corner = corner[0].astype(int)
        position = tuple(corner[0])  # Use the first corner point as the position for the ID

        # Draw the ID with a larger font size and thickness
        cv2.putText(image, f"ID: {ids[i][0]}", position, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5, color=(255, 0, 0), thickness=3)

# Display the image
cv2.namedWindow("ArUco Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ArUco Detection", 800,800)
cv2.imshow("ArUco Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

