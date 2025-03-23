import cv2
import mediapipe as mp
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np

# Tkinter 파일 선택 창 (여러 이미지 선택)
def select_image_files():
    root = tk.Tk()
    root.withdraw()  # Tkinter 창 숨기기
    file_paths = filedialog.askopenfilenames(
        title="여러 이미지를 선택하세요",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")]
    )
    return file_paths

# 여러 이미지 파일 선택
image_paths = select_image_files()
if not image_paths:
    print("⚠️ 파일을 선택하지 않았습니다. 프로그램을 종료합니다.")
    exit()

# MediaPipe 손 랜드마크 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 사용자에게 수어 라벨 입력받기
gesture = input("수어 라벨을 입력하세요 (예: Hello, ThankYou, ILoveYou): ")

# ✅ CSV 파일 저장할 폴더가 없으면 생성
if not os.path.exists("sign_data"):
    os.makedirs("sign_data")

# 기존 데이터 불러오기 (파일이 없으면 빈 DataFrame 생성)
csv_path = f"sign_data/{gesture}_data.csv"
if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path)
else:
    df_existing = pd.DataFrame()

# ✅ 컬럼 개수 수정 (21개 랜드마크 * 3 + 파일명 + 라벨 = 65개)
columns = []
for i in range(21):
    columns.append(f"X{i}")
    columns.append(f"Y{i}")
    columns.append(f"Z{i}")
columns += ["Image", "Label"]

# 데이터 저장용 리스트
landmark_data = []
not_detected_images = []  # 손이 감지되지 않은 이미지 목록

# MediaPipe Hands 실행
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3  # 감지 확률 낮춤
) as hands:

    for image_path in image_paths:
        # 한글 파일 경로 문제 해결: imdecode 사용
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        # 이미지 로드 확인
        if image is None:
            print(f"⚠️ 이미지 로드 실패: {image_path}")
            not_detected_images.append(image_path)
            continue  # 다음 이미지로 이동

        # BGR → RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            print(f"⚠️ 손 랜드마크를 감지하지 못했습니다: {image_path}")
            not_detected_images.append(image_path)  # 감지되지 않은 이미지 저장
            continue  # 손이 감지되지 않은 이미지는 제외

        for hand_landmarks in results.multi_hand_landmarks:
            # 랜드마크 좌표 수집
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.append(landmark.x)
                hand_data.append(landmark.y)
                hand_data.append(landmark.z)

            hand_data.append(os.path.basename(image_path))  # 이미지 파일명 추가
            hand_data.append(gesture)  # 수어 라벨 추가

            # 데이터 길이 체크 후 추가
            if len(hand_data) == 65:  # 21개 랜드마크 * 3(X, Y, Z) + 파일명 + 라벨 = 65개 컬럼
                landmark_data.append(hand_data)
            else:
                print(f"❌ 데이터 길이 불일치 - 저장되지 않음: {image_path}")

cv2.destroyAllWindows()

# ✅ 데이터프레임 생성 오류 해결
try:
    df_new = pd.DataFrame(landmark_data, columns=columns)
except ValueError as e:
    print(f"⚠️ 데이터프레임 생성 중 오류 발생: {e}")
    print("📌 landmark_data 개수:", len(landmark_data))
    print("📌 landmark_data 첫 번째 샘플 크기:", len(landmark_data[0]) if landmark_data else "데이터 없음")
    print("📌 예상 컬럼 개수:", len(columns))
    exit()

# 기존 데이터와 합치기
if not df_existing.empty:
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_combined = df_new

# ✅ CSV 저장 (이제 오류 발생 안 함)
df_combined.to_csv(csv_path, index=False)

print(f"✅ 여러 이미지 데이터 저장 완료! 파일 경로: {csv_path}")

# ❌ 손이 감지되지 않은 이미지 목록 출력
if not_detected_images:
    print("\n⚠️ 손이 감지되지 않은 이미지 목록:")
    for img in not_detected_images:
        print(f"   - {img}")