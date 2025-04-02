import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# 📁 데이터 경로
data_path = os.path.join(os.path.dirname(__file__), "..", "동적수어", "dynamic_gesture_data_cleaned.csv")
data_path = os.path.abspath(data_path)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

df = pd.read_csv(data_path)
print(f"[INFO] 데이터 로드 완료: {df.shape}")

# 🧼 특성과 라벨 분리
X = df.drop(columns=['label']).values
y = df['label'].values

# 🧮 입력 형태 설정
frames_per_sample = 30
total_features = X.shape[1]
num_features = total_features // frames_per_sample  # 정확히 나눌 수 있는 만큼만 사용

# 너무 많은 feature 제거 (예: 127개 중 120개만 사용)
X = X[:, :frames_per_sample * num_features]
X = X.reshape(-1, frames_per_sample, num_features)
print(f"[INFO] 재구성된 X shape: {X.shape}, y shape: {y.shape}")

# 🔤 라벨 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)
print(f"[INFO] 라벨 인코딩 완료. 클래스 수: {len(le.classes_)}")

# 💾 라벨 인코더 저장 (폴더 없으면 생성)
save_dir = os.path.join(os.path.dirname(__file__), "..", "동적수어")
os.makedirs(save_dir, exist_ok=True)
joblib.dump(le, os.path.join(save_dir, "label_encoder_lstm.pkl"))

# 🧪 학습/검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

# 🧠 LSTM 모델 구성
model = Sequential()
model.add(LSTM(128, input_shape=(frames_per_sample, num_features)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 🏋️ 학습
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=16)

# 💾 모델 저장
model.save(os.path.join("동적수어", "dynamic_gesture_model_lstm.h5"))
print("[INFO] 모델 저장 완료!")