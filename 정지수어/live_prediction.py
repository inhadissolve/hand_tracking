# ✅ step4: live_prediction.py

import cv2
import numpy as np
import pandas as pd
import joblib
import os
import time
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ────────────────────── 설정 ──────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'gesture_model.pkl')
encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 32)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

feature_names = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]

# None으로 처리할 확률 임계값
MINIMUM_CONFIDENCE = 0.2

cap = cv2.VideoCapture(0)
print("[INFO] 실시간 제스처 예측 시작...")

DELAY = 0.5
last_time = 0

# ────────────────────── 예측 루프 ──────────────────────
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

            X_df = pd.DataFrame([landmarks], columns=feature_names)
            probs = model.predict_proba(X_df)[0]
            max_prob = np.max(probs)
            pred_index = np.argmax(probs)

            # 예측 확률이 충분히 높으면 라벨 출력, 너무 낮으면 None
            if max_prob >= MINIMUM_CONFIDENCE:
                label = label_encoder.inverse_transform([pred_index])[0]
                score = max_prob
            else:
                label = "None"
                score = 0.0

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    current_time = time.time()
    if current_time - last_time > DELAY:
        print(f"[INFO] 예측: {label}, 정확도: {score:.2f}")
        last_time = current_time

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((30, 30), f"제스처: {label} ({score:.2f})", font=font, fill=(0, 0, 255))
    frame = np.array(img_pil)

    cv2.imshow("Gesture Prediction", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()