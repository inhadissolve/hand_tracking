import cv2
import os
import time
import pandas as pd
from datetime import datetime
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정 (윈도우용 경로)
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.rcParams['axes.unicode_minus'] = False

# 예제 그래프
plt.figure(figsize=(8, 4))
plt.title("한글 표시 테스트")
plt.plot([1, 2, 3], [1, 4, 9], label='수어 인식률')
plt.xlabel("시간")
plt.ylabel("정확도 (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 2손
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 경로 설정
DATA_DIR = "dynamic_dataset"

# 1. 라벨 입력
label = input("저장할 수어 라벨을 입력하세요 (예: 안녕하세요): ")
label_dir = os.path.join(DATA_DIR, label)gi
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
        while time.time() - start_time < 2:  # 2초 수집
            success, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                row = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                sequence.append(row)
            # (손이 2개 아닐 경우 이 프레임은 건너뜀)

        if sequence:
            # 유연하게 열 수를 구성
            num_cols = len(sequence[0])
            columns = [f"{axis}{i}" for i in range(num_cols // 3) for axis in ['x', 'y', 'z']]
            df = pd.DataFrame(sequence, columns=columns)
            df["label"] = label
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{label}_{timestamp}.csv"
            df.to_csv(os.path.join(label_dir, filename), index=False)
            print(f"[INFO] 저장 완료: {os.path.join(label_dir, filename)}")
        else:
            print("[WARNING] 유효한 데이터가 없어 저장하지 않았습니다.")

        recording = False  # 상태 초기화

    # 프레임 보여주기
    cv2.putText(frame, f"Label: {label}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Dynamic Sign Recorder", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("[INFO] 2초간 데이터 수집 시작...")
        recording = True
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()