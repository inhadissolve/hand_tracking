# ✅ step2_preprocess_dynamic.py
# 수집된 동적 수어 CSV 파일 병합 및 저장

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "dynamic_dataset")  # 경로 수정!!
SAVE_PATH = os.path.join(BASE_DIR, "dynamic_gesture_data_cleaned.csv")

all_csv_files = []

# 폴더 내 라벨별 폴더 탐색
for label_folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, label_folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                all_csv_files.append(os.path.join(folder_path, file))

print(f"[INFO] 총 {len(all_csv_files)}개의 CSV 파일이 발견됨.\n")

merged_data = []

for csv_file in all_csv_files:
    df = pd.read_csv(csv_file)
    if 'label' in df.columns:
        merged_data.append(df)
    else:
        print(f"[WARNING] {csv_file} 에 'label' 컬럼이 없어 건너뜀.")

if merged_data:
    merged_df = pd.concat(merged_data, ignore_index=True)
    merged_df.to_csv(SAVE_PATH, index=False)
    print(f"[INFO] 전처리 완료! 저장 위치: {SAVE_PATH}")

    # 시각화
    plt.figure(figsize=(10, 5))
    merged_df['label'].value_counts().plot(kind='bar', color='skyblue')
    plt.title("라벨별 프레임 수")
    plt.xlabel("수어 라벨")
    plt.ylabel("프레임 수")
    plt.tight_layout()
    plt.show()

    # 🔤 라벨 목록 출력
    print(f"[INFO] 최종 병합된 라벨 목록: {merged_df['label'].unique()}")
else:
    print("[ERROR] 병합할 데이터가 없습니다. 경로 및 CSV 형식을 확인하세요.")