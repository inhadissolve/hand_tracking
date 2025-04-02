# step2_preprocess_visualize.py

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

# Windows용 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지

# 1. CSV 불러오기
df = pd.read_csv('hand_gesture_data.csv', encoding='cp949')

# 2. 0값만 포함된 행 제거 (x0~z20까지 63개의 좌표값이 0이면 삭제)
coords = df.columns[:-1]  # label 제외한 모든 좌표 컬럼
df = df[~(df[coords] == 0).all(axis=1)]  # 모두 0인 행 제거

# 3. 라벨별 데이터 개수 출력
print("라벨별 데이터 개수:")
print(df['label'].value_counts())

# 4. 히스토그램 시각화
plt.figure(figsize=(10, 5))
df['label'].value_counts().plot(kind='bar', color='skyblue')
plt.title("수집된 라벨별 데이터 개수")
plt.xlabel("수어 라벨")
plt.ylabel("데이터 수")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. 전처리된 데이터 저장
df.to_csv('gesture_data_cleaned.csv', index=False)
print("[INFO] 전처리 완료. gesture_data_cleaned.csv로 저장됨")