import cv2
import cv2.aruco as aruco
import numpy as np
from objloader_simple import *

# OpenCV ArUco marker detection setup
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
calib_path = ""
camera_matrix = np.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
camera_distortion = np.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')

url = "http://192.168.219.94/stream"
cap = cv2.VideoCapture(url)

# Load 3D model
obj = OBJ('./banana.obj', swapyz=True)

def draw_model(img, obj, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Draw a 3D model over a detected marker.
    """
    # Project the 3D points to the image plane
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = img.shape[:2]

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        dst, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts = np.int32(dst)

        cv2.fillConvexPoly(img, imgpts, (137, 27, 211))

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if np.all(ids is not None):
            for i in range(len(ids)):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, camera_distortion)
                aruco.drawDetectedMarkers(frame, corners)
                aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 0.1)
                
                draw_model(frame, obj, rvec, tvec, camera_matrix, camera_distortion)
                
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to retrieve frame")

cap.release()
cv2.destroyAllWindows()