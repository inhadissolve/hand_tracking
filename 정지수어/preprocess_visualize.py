# ✅ step2: preprocess_visualize.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'hand_gesture_data.csv')
df = pd.read_csv(file_path, encoding='cp949')

coords = df.columns[:-1]
df = df[~(df[coords] == 0).all(axis=1)]

print("라벨별 데이터 개수:")
print(df['label'].value_counts())

plt.figure(figsize=(10, 5))
df['label'].value_counts().plot(kind='bar', color='skyblue')
plt.title("수집된 라벨별 데이터 개수")
plt.xlabel("수어 라벨")
plt.ylabel("데이터 수")
plt.grid(True)
plt.tight_layout()
plt.show()

save_path = os.path.join(BASE_DIR, 'gesture_data_cleaned.csv')
df.to_csv(save_path, index=False)
print("[INFO] 전처리 완료. gesture_data_cleaned.csv로 저장됨")