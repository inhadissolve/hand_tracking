# filename: gesture_recognition_live.py

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# 모델 경로 지정 (자동 다운로드)
model_path = 'gesture_recognizer.task'

# GestureRecognizer 설정
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=lambda result, output_image, timestamp: print_result(result)
)

# 결과 출력 함수
def print_result(result):
    if result.gestures:
        gesture = result.gestures[0][0]
        print(f"[INFO] 인식된 수어: {gesture.category_name} ({gesture.score*100:.2f}%)")

# GestureRecognizer 인스턴스 생성
recognizer = GestureRecognizer.create_from_options(options)

# 웹캠 연결
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] 웹캠을 열 수 없습니다.")
    exit()

print("[INFO] 웹캠 연결 성공. 제스처 인식 시작...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 프레임을 읽을 수 없습니다.")
        break

    # BGR → RGB 변환
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # 타임스탬프
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    # 제스처 인식 실행
    recognizer.recognize_async(mp_image, timestamp)

    # 결과 출력용 화면
    cv2.imshow('Gesture Recognition (Press q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
recognizer.close()
