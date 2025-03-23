import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Tkinter 파일 선택 창 (이미지 선택)
def select_image_files():
    root = tk.Tk()
    root.withdraw()  # Tkinter 창 숨기기
    file_paths = filedialog.askopenfilenames(
        title="이미지 파일을 선택하세요",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")]
    )
    return file_paths

# 이미지 파일 선택
image_paths = select_image_files()
if not image_paths:
    print("⚠️ 파일을 선택하지 않았습니다. 프로그램을 종료합니다.")
    exit()

# MediaPipe 손 랜드마크 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 데이터 저장용 리스트
landmark_data = []
labels = []

# 사용자에게 수어 라벨 입력 받기
gesture = input("수어 라벨을 입력하세요 (예: Hello, ThankYou, ILoveYou): ")

# MediaPipe Hands 실행
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    for image_path in image_paths:
        # 이미지 파일 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ 이미지 로드 실패: {image_path}")
            continue  # 이미지가 정상적으로 로드되지 않으면 건너뜁니다

        # BGR → RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 랜드마크 좌표 수집
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.append(landmark.x)
                    hand_data.append(landmark.y)
                    hand_data.append(landmark.z)

                # 수어 라벨 추가
                hand_data.append(gesture)

                # 데이터 저장
                landmark_data.append(hand_data[:-1])  # 손 제스처 데이터 (라벨 제외)
                labels.append(hand_data[-1])  # 수어 라벨

        else:
            print(f"⚠️ 손 랜드마크를 감지하지 못했습니다: {image_path}")
            continue  # 손이 감지되지 않은 이미지는 제외

# 데이터프레임으로 변환
columns = [f"X{i},Y{i},Z{i}" for i in range(21)]  # X, Y, Z 21개의 랜드마크 좌표
df = pd.DataFrame(landmark_data, columns=columns)

# CSV 파일 저장
csv_path = f"sign_data/{gesture}_data.csv"
df["Label"] = labels  # 라벨 추가
df.to_csv(csv_path, index=False)

print(f"✅ 데이터 저장 완료! 파일 경로: {csv_path}")