import cv2
import numpy as np
import mediapipe as mp
import joblib
import os
from tensorflow.keras.models import load_model
from collections import deque
from PIL import ImageFont, ImageDraw, Image  # 추가

# 한글 폰트 경로 설정 (윈도우 기준)
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 32)  # 크기 32 정도가 적당

# 👉 OpenCV 프레임에 한글을 그려주는 함수
def draw_korean_text(cv_img, text, position=(30, 50), color=(0, 255, 0)):
    # OpenCV → PIL 이미지로 변환
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_img)

    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)

    # 다시 OpenCV 포맷으로 변환
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# 현재 파일 기준으로 한 단계 상위 폴더로 이동
CURRENT_DIR = os.path.dirname(__file__)                   # hand_tracking/동적수어/동적수어
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # hand_tracking/동적수어

# 모델과 인코더 경로 설정
model_path = os.path.join(BASE_DIR, "dynamic_gesture_model_lstm.h5")
encoder_path = os.path.join(BASE_DIR, "label_encoder_lstm.pkl")

# 경로 확인 로그
print(f"[INFO] 모델 경로: {model_path}")
print(f"[INFO] 인코더 경로: {encoder_path}")

# 로딩 시도
model = load_model(model_path)
label_encoder = joblib.load(encoder_path)

# 🎯 설정
sequence_length = 30
confidence_threshold = 0.8
frame_queue = deque(maxlen=sequence_length)

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
print("[INFO] 실시간 예측을 시작합니다. ESC 키를 누르면 종료합니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        row = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
        frame_queue.append(row)

        if len(frame_queue) == sequence_length:
            input_data = np.array(frame_queue).reshape(1, sequence_length, -1)
            prediction = model.predict(input_data)[0]
            confidence = np.max(prediction)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

            if confidence >= confidence_threshold:
                frame = draw_korean_text(frame, f"예측: {predicted_label} ({confidence:.2f})",
                                         position=(30, 50), color=(0, 255, 0))
                print(f"[INFO] 예측: {predicted_label}, 확률: {confidence:.2f}")
            else:
                frame = draw_korean_text(frame, "예측: 없음", position=(30, 50), color=(0, 0, 255))

    else:
        frame = draw_korean_text(frame, "손이 두 개 감지되지 않았습니다", position=(30, 50), color=(0, 0, 255))

    cv2.imshow("Dynamic Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()