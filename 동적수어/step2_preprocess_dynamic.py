import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from matplotlib import font_manager, rc

# ✅ 한글 폰트 설정 (Windows 기준)
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 폴더 경로
dataset_dir = "../dynamic_dataset"
save_path = "../동적수어/dynamic_gesture_data_cleaned.csv"

all_csv_files = glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
print(f"[INFO] 총 {len(all_csv_files)}개의 CSV 파일이 발견됨.\n")

data = []

for csv_file in all_csv_files:
    try:
        df = pd.read_csv(csv_file)
        print(f"[DEBUG] {csv_file} 컬럼: {df.columns.tolist()}")

        if 'label' not in df.columns:
            print(f"[WARNING] {csv_file} 파일에 'label' 컬럼이 없습니다. 건너뜀.")
            continue

        data.append(df)
    except Exception as e:
        print(f"[ERROR] {csv_file} 파일 읽기 중 오류 발생: {e}")

if not data:
    print("[ERROR] 병합할 데이터가 없습니다. 경로를 확인하세요.")
    exit()

merged_df = pd.concat(data, ignore_index=True)

# ✅ 라벨별 프레임 수 시각화
print("\n[INFO] 라벨별 프레임 수:")
print(merged_df['label'].value_counts())

plt.figure(figsize=(8, 5))
merged_df['label'].value_counts().plot(kind='bar')
plt.title("수어 라벨별 프레임 수")
plt.xlabel("수어 라벨")
plt.ylabel("프레임 수")
plt.tight_layout()
plt.show()

# ✅ CSV 저장
merged_df.to_csv(save_path, index=False)
print(f"\n[INFO] 전처리 완료! 저장 위치: {save_path}")