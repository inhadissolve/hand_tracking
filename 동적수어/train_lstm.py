import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# 📁 CSV 파일 경로
data_path = os.path.join(os.path.dirname(__file__), "..", "동적수어", "dynamic_gesture_data_cleaned.csv")
data_path = os.path.abspath(data_path)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ 데이터 파일이 존재하지 않습니다: {data_path}")

# 🧼 데이터 로드
df = pd.read_csv(data_path)
print(f"[INFO] 데이터 로드 완료: {df.shape}")

# 특성과 라벨 분리 (label 컬럼이 실제 존재하는지 확인)
if 'label' not in df.columns:
    raise KeyError(f"❌ 'label' 컬럼이 존재하지 않습니다. 컬럼 목록: {df.columns.tolist()}")

X = df.drop(columns=['label']).values
y = df['label'].values

# 🧮 입력 형태 재구성
frames_per_sample = 1
if X.shape[1] % frames_per_sample != 0:
    raise ValueError(f"❌ 총 feature 수({X.shape[1]})가 프레임 수({frames_per_sample})로 나누어 떨어지지 않습니다.")

num_features = X.shape[1] // frames_per_sample  # → 그대로 126
X = X.reshape(-1, frames_per_sample, num_features)  # → (N, 1, 126)

# 🔤 라벨 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)
print(f"[INFO] 라벨 인코딩 완료. 클래스 수: {len(le.classes_)}")

# 💾 라벨 인코더 저장
encoder_path = os.path.join(os.path.dirname(__file__), "..", "동적수어", "label_encoder_lstm.pkl")
os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
joblib.dump(le, encoder_path)

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

# 🏋️‍♀️ 모델 학습
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=16)

# 💾 모델 저장
model_path = os.path.join(os.path.dirname(__file__), "..", "동적수어", "dynamic_gesture_model_lstm.h5")
model.save(model_path)
print(f"[INFO] 모델 저장 완료: {model_path}")