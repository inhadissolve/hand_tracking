import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model("sign_language_model.h5")

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 랜드마크 좌표 수집
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.append(landmark.x)
                    hand_data.append(landmark.y)
                    hand_data.append(landmark.z)

                # 데이터 모델 입력
                input_data = np.array([hand_data])
                prediction = model.predict(input_data)
                predicted_label = np.argmax(prediction)

                # 화면에 결과 출력
                cv2.putText(frame, f"Prediction: {predicted_label}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Translator', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()