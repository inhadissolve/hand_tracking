# ✅ step1_collect_dynamic.py
# 동적 수어 수집 - 2손 인식, 라벨 기반 저장

import cv2
import os
import time
import pandas as pd
from datetime import datetime
import mediapipe as mp

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "동적수어", "dynamic_dataset")
os.makedirs(DATA_DIR, exist_ok=True)

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 1. 라벨 입력
label = input("저장할 수어 라벨을 입력하세요 (예: 안녕하세요): ")
label_dir = os.path.join(DATA_DIR, label)
os.makedirs(label_dir, exist_ok=True)

print("[INFO] 's' 키를 누르면 2초간 데이터를 수집합니다. ESC 키로 종료합니다.")

# 웹캠 열기
cap = cv2.VideoCapture(0)
recording = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[ERROR] 카메라를 읽을 수 없습니다.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if recording:
        sequence = []
        start_time = time.time()
        while time.time() - start_time < 2:
            success, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                row = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                sequence.append(row)

        if sequence:
            num_cols = len(sequence[0])
            columns = [f"{axis}{i}" for i in range(num_cols // 3) for axis in ['x', 'y', 'z']]
            df = pd.DataFrame(sequence, columns=columns)
            df['label'] = label
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{label}_{timestamp}.csv"
            df.to_csv(os.path.join(label_dir, filename), index=False)
            print(f"[INFO] 저장 완료: {os.path.join(label_dir, filename)}")
        else:
            print("[WARNING] 유효한 데이터가 없어 저장하지 않았습니다.")
        recording = False

    cv2.putText(frame, f"Label: {label}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Dynamic Sign Recorder", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("[INFO] 2초간 데이터 수집 시작...")
        recording = True
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()