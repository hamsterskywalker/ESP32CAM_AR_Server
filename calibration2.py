import cv2
import numpy as np
import pickle
import time

# ESP32-CAM의 스트리밍 URL (자신의 ESP32-CAM IP 주소로 변경해주세요)
url = "http://192.168.219.94/stream"

# 체스보드의 크기를 정의합니다. (가로 코너 수, 세로 코너 수)
chessboard_size = (9, 6)

# 3D 점 좌표를 준비합니다
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 모든 이미지에서 찾은 3D 점과 2D 점을 저장할 배열
objpoints = [] # 3D 점 (실제 세계의 점)
imgpoints = [] # 2D 점 (이미지 평면의 점)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(url)

# 캘리브레이션에 사용할 이미지 개수
num_images = 20
captured_images = 0

while captured_images < num_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너를 찾습니다
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 코너가 발견되면, 객체 점과 이미지 점을 저장합니다
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 코너를 그립니다
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        captured_images += 1
        print(f"Captured image {captured_images}/{num_images}")
        time.sleep(1)  # 1초 대기 (너무 비슷한 이미지를 캡처하지 않기 위해)

    cv2.imshow('ESP32-CAM Stream', frame)
    
    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 캘리브레이션을 수행합니다
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 결과를 출력합니다
print("Camera matrix:")
print(camera_matrix)
print("\nDistortion coefficients:")
print(dist_coeffs)

# 결과를 파일로 저장합니다
calibration_result = {
    "camera_matrix": camera_matrix,
    "dist_coeffs": dist_coeffs
}

with open('esp32cam_calibration.pkl', 'wb') as f:
    pickle.dump(calibration_result, f)

print("\nCalibration results saved to 'esp32cam_calibration.pkl'")