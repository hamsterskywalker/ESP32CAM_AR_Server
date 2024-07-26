import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pywavefront

# ESP32-CAM의 IP 주소와 포트
url = 'http://192.168.219.94/stream'

# 3D 객체 로드
obj = pywavefront.Wavefront('banana.obj', collect_faces=True)

# ArUco 마커 사전과 파라미터 설정
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters_create()

# OpenGL 초기화
def init_gl(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / height), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_DEPTH_TEST)

# 3D 객체 그리기
def draw_obj():
    glPushMatrix()
    glRotatef(180, 1, 0, 0)
    glScalef(0.1, 0.1, 0.1)  # 크기 조정
    for mesh in obj.mesh_list:
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for vertex_i in face:
                glVertex3f(*obj.vertices[vertex_i])
        glEnd()
    glPopMatrix()

# 메인 루프
def main():
    cap = cv2.VideoCapture(url)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ArUco 마커 감지
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
        
        if ids is not None:
            # 마커가 감지되면 3D 객체 그리기
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i in range(len(ids)):
                # 마커의 포즈 추정
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)
                
                # OpenGL로 3D 객체 그리기
                glPushMatrix()
                glTranslatef(tvec[0][0][0], tvec[0][0][1], -tvec[0][0][2])
                glRotatef(rvec[0][0][2] * 180 / np.pi, 0, 0, 1)
                glRotatef(rvec[0][0][1] * 180 / np.pi, 0, 1, 0)
                glRotatef(rvec[0][0][0] * 180 / np.pi, 1, 0, 0)
                draw_obj()
                glPopMatrix()
        
        cv2.imshow('ESP32-CAM Stream', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow("AR with ESP32-CAM")
    init_gl(800, 600)
    main()