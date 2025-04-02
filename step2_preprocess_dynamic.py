import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ✅ 동적 수어 CSV 파일 경로 패턴 (하위 디렉토리 포함)
csv_pattern = os.path.join("dynamic_dataset", "**", "*.csv")
all_files = glob.glob(csv_pattern, recursive=True)  # 🔥 하위 폴더까지 검색
print(f"[INFO] 총 {len(all_files)}개의 CSV 파일이 발견됨.")

data = []
for file in all_files:
    df = pd.read_csv(file)
    df['source_file'] = os.path.basename(file)  # 어떤 파일에서 왔는지 추적용
    data.append(df)

if not data:
    print("[ERROR] 병합할 데이터가 없습니다. 경로를 확인하세요.")
    exit()

# 2. 병합
merged_df = pd.concat(data, ignore_index=True)

# 3. 라벨 분포 확인
print("\n[INFO] 라벨별 프레임 수:")
print(merged_df['label'].value_counts())

# 4. 시각화
plt.figure(figsize=(10, 5))
merged_df['label'].value_counts().plot(kind='bar', color='salmon')
plt.title("라벨별 프레임 수 (동적 수어)")
plt.xlabel("수어 라벨")
plt.ylabel("프레임 수")
plt.tight_layout()
plt.grid(True)
plt.show()

# 5. 전처리된 데이터 저장
merged_df.to_csv('dynamic_gesture_data_cleaned.csv', index=False)
print("\n[INFO] 전처리된 데이터를 저장했습니다: dynamic_gesture_data_cleaned.csv")