import cv2
import csv
import os
import time
import numpy as np
import mediapipe as mp

# 사용자 입력 라벨
label = input("저장할 수어 이름을 입력하세요: ")

# 저장 폴더 설정
output_dir = "dynamic_dataset"
label_dir = os.path.join(output_dir, label)
os.makedirs(label_dir, exist_ok=True)

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 양손
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

# 랜드마크 추출 함수
def extract_landmarks(results):
    landmark_data = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmark_data.extend([lm.x, lm.y, lm.z])
    # 1손일 경우 63개 좌표 채우기
    while len(landmark_data) < 126:
        landmark_data.extend([0.0, 0.0, 0.0])
    return landmark_data

# CSV 저장 경로 생성
def get_csv_path():
    t = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(label_dir, f"{label}_{t}.csv")

# 웹캠 실행
cap = cv2.VideoCapture(0)
print("웹캠 실행됨. 's'를 누르면 2초간 저장 시작, ESC를 누르면 종료.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Press 's' to record, ESC to exit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print(f"[INFO] '{label}' 수어 저장 시작 (2초)...")
        collected_data = []
        start = time.time()

        while time.time() - start < 2:
            ret, frame = cap.read()
            if not ret:
                continue
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            landmarks = extract_landmarks(results)
            collected_data.append(landmarks)

            # 시각화
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Recording: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Press 's' to record, ESC to exit", frame)
            cv2.waitKey(1)

        # CSV 저장
        file_path = get_csv_path()
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(collected_data)

        print(f"[INFO] 저장 완료: {file_path}")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()