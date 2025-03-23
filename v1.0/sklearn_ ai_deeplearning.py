import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 데이터 불러오기
df = pd.read_csv("sign_language_sequence_data.csv")

# 입력 데이터 (시간 순서가 있는 손 좌표)
X = df.iloc[:, 1:-1].values  # 시간과 라벨을 제외한 값
X = X.reshape((X.shape[0], 21, 3))  # LSTM이 학습할 수 있도록 (프레임, 손가락 수, 좌표) 형태로 변환

# 출력 데이터 (라벨)
y = pd.get_dummies(df.iloc[:, -1]).values  # 원-핫 인코딩

# 훈련 & 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(21, 3)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')  # 클래스 개수만큼 출력
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# 모델 저장
model.save("sign_language_sequence_model.h5")
print("AI 모델 학습 완료 및 저장됨!")