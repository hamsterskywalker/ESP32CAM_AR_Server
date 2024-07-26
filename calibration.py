import cv2
import numpy as np
import glob

# Define the dimensions of the checkerboard
CHECKERBOARD = (7, 7)  # Number of inner corners per a chessboard row and column
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create vectors to store 3D points for each checkerboard image
objpoints = []
# Create vectors to store 2D points for each checkerboard image
imgpoints = []

# Define the real world coordinates for points in the checkerboard
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Capture frames from ESP32-CAM stream
url = "http://192.168.219.94/stream"
cap = cv2.VideoCapture(url)

calibration_images = []
while len(calibration_images) < 20:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            calibration_images.append(frame)
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
            cv2.imshow('Calibration', frame)
            cv2.waitKey(500)
    else:
        print("Failed to retrieve frame")

cap.release()
cv2.destroyAllWindows()

for frame in calibration_images:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the camera calibration results
np.savetxt('cameraMatrix.txt', camera_matrix, delimiter=',')
np.savetxt('cameraDistortion.txt', dist_coeffs, delimiter=',')

print("Calibration complete. Camera matrix and distortion coefficients saved.")
