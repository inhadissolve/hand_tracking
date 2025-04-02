import cv2
import numpy as np
import pandas as pd
import joblib
import time
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# 모델 불러오기
model = joblib.load('gesture_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 폰트 경로 설정 (한글 표시용)
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 32)

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Feature 이름 정의
feature_names = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]

# 웹캠 연결
cap = cv2.VideoCapture(0)
print("[INFO] 실시간 제스처 예측 시작...")

# 출력 속도 조절용 변수
DELAY = 0.5
last_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 프레임을 읽을 수 없습니다.")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    label = "None"
    score = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # 예측 수행
            X_df = pd.DataFrame([landmarks], columns=feature_names)
            probs = model.predict_proba(X_df)[0]
            max_prob = np.max(probs)
            pred_label = model.predict(X_df)[0]
            label = label_encoder.inverse_transform([pred_label])[0] if max_prob > 0.8 else "None"
            score = max_prob if max_prob > 0.8 else 0.0

            # 랜드마크 시각화
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 텍스트 출력 간격 조절
    current_time = time.time()
    if current_time - last_time > DELAY:
        print(f"[INFO] 예측: {label}, 정확도: {score:.2f}")
        last_time = current_time

    # PIL로 변환 후 한글 텍스트 출력
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((30, 30), f"제스처: {label} ({score:.2f})", font=font, fill=(0, 0, 255))
    frame = np.array(img_pil)

    cv2.imshow("Gesture Prediction", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()