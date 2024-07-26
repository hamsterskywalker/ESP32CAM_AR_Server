import cv2
import numpy as np
import requests
from requests.exceptions import RequestException

def get_frame(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        bytes = b''
        for chunk in response.iter_content(chunk_size=1024):
            bytes += chunk
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b+2]
                bytes = bytes[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                return frame
    except RequestException as e:
        print(f"Error getting frame: {e}")
        return None

# ESP32-CAM의 스트리밍 URL
url = "http://192.168.219.94:80/stream"

while True:
    frame = get_frame(url)
    if frame is not None:
        cv2.imshow('ESP32-CAM Stream', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()