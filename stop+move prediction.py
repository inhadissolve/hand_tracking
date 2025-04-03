import cv2
import numpy as np
import pandas as pd
import joblib
import os
import time
from collections import deque
from PIL import ImageFont, ImageDraw, Image
from tensorflow.keras.models import load_model
import mediapipe as mp

# ────────────────────────── 경로 설정 ──────────────────────────
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))  # hand_tracking
STATIC_DIR = os.path.join(CURRENT_FILE_DIR, "정지수어")
DYNAMIC_DIR = os.path.join(CURRENT_FILE_DIR, "동적수어")

STATIC_MODEL_PATH = os.path.join(STATIC_DIR, "gesture_model.pkl")
STATIC_ENCODER_PATH = os.path.join(STATIC_DIR, "label_encoder.pkl")

DYNAMIC_MODEL_PATH = os.path.join(DYNAMIC_DIR, "dynamic_gesture_model_lstm.h5")
DYNAMIC_ENCODER_PATH = os.path.join(DYNAMIC_DIR, "label_encoder_lstm.pkl")


# ────────────────────────── 모델 불러오기 ──────────────────────────
static_model = joblib.load(STATIC_MODEL_PATH)
static_encoder = joblib.load(STATIC_ENCODER_PATH)

dynamic_model = load_model(DYNAMIC_MODEL_PATH)
dynamic_encoder = joblib.load(DYNAMIC_ENCODER_PATH)

# ────────────────────────── 설정 ──────────────────────────
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 32)

sequence_length = 30
confidence_threshold = 0.8
frame_queue = deque(maxlen=sequence_length)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.6, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

feature_names = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]

# ────────────────────────── 예측 루프 ──────────────────────────
cap = cv2.VideoCapture(0)
print("[INFO] 실시간 정지+동적 수어 예측 시작")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    label_stop = "None"
    score_stop = 0.0

    label_move = "None"
    score_move = 0.0

    if results.multi_hand_landmarks:
        hands_count = len(results.multi_hand_landmarks)

        if hands_count == 1:
            # ─ 정지 수어 ─
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            df = pd.DataFrame([landmarks], columns=feature_names)
            probs = static_model.predict_proba(df)[0]
            max_prob = np.max(probs)

            if max_prob > confidence_threshold:
                pred = static_model.predict(df)[0]
                label_stop = static_encoder.inverse_transform([pred])[0]
                score_stop = max_prob

        elif hands_count == 2:
            # ─ 동적 수어 ─
            row = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
            frame_queue.append(row)

            if len(frame_queue) == sequence_length:
                input_data = np.array(frame_queue).reshape(1, sequence_length, -1)
                pred = dynamic_model.predict(input_data)[0]
                max_prob = np.max(pred)

                if max_prob > confidence_threshold:
                    label_move = dynamic_encoder.inverse_transform([np.argmax(pred)])[0]
                    score_move = max_prob

    # ────────────────────────── 한글 표시 ──────────────────────────
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    if label_move != "None":
        draw.text((30, 30), f"동적: {label_move} ({score_move:.2f})", font=font, fill=(0, 255, 0))
    elif label_stop != "None":
        draw.text((30, 30), f"정지: {label_stop} ({score_stop:.2f})", font=font, fill=(0, 0, 255))
    else:
        draw.text((30, 30), "인식 실패", font=font, fill=(255, 0, 0))

    frame = np.array(img_pil)

    cv2.imshow("정지 + 동적 수어 실시간 예측", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()