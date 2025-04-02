# ✅ step1: data_collection.py
import cv2
import csv
import os
import time
import mediapipe as mp

# MediaPipe 손 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# 저장할 CSV 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(BASE_DIR, 'hand_gesture_data.csv')

# 라벨 입력
label = input("저장할 라벨을 입력하세요 (예: ㄱ): ")

# CSV 헤더 작성
header = []
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']
header.append('label')

# 파일이 없다면 헤더 작성
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# 웹캠 시작
cap = cv2.VideoCapture(0)
print("[INFO] 웹캠 시작. 's' 키를 눌러 프레임 저장, 'q' 키로 종료")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1)

    if key == ord('s') and result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            data = []
            for lm in hand_landmarks.landmark:
                data += [lm.x, lm.y, lm.z]
            data.append(label)
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
            print(f"[INFO] 저장 완료 - 라벨: {label}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()