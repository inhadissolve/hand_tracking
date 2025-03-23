# import tensorflow as tf
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# # 데이터 불러오기
# df = pd.read_csv("sign_data/ㄱ_data.csv")
#
# # 데이터 개수 확인
# print(f"📊 데이터셋 크기: {df.shape}")
#
# # 입력 데이터 (랜드마크 좌표)
# X = df.iloc[:, :-2].values  # 마지막 두 개 열(Image, Label) 제외
# X = (X - 0.5) / 0.5  # 데이터 정규화 (-1 ~ 1 범위)
#
# # 출력 데이터 (라벨을 원-핫 인코딩)
# y = pd.get_dummies(df["Label"]).values  # 원-핫 인코딩
#
# # 샘플 개수 확인 후 분할 방식 조정
# num_samples = X.shape[0]
#
# if num_samples > 1:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# else:
#     X_train, X_test, y_train, y_test = X, X, y, y
#     print("⚠️ 샘플이 1개뿐이므로 train_test_split을 사용하지 않고 전체 데이터를 훈련에 사용합니다.")
#
# # ✅ Softmax or Sigmoid 자동 선택
# if y.shape[1] > 1:
#     activation_function = 'softmax'
#     loss_function = 'categorical_crossentropy'
# else:
#     activation_function = 'sigmoid'
#     loss_function = 'binary_crossentropy'
#
# # MLP 모델 정의
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(y.shape[1], activation=activation_function)  # softmax 또는 sigmoid 자동 적용
# ])
#
# # 모델 컴파일
# model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
#
# # 모델 학습
# model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))
#
# # ✅ 최신 Keras 저장 방식 사용
# model.save("sign_language_model.keras")  # .h5 → .keras 변경
# print("✅ 손 랜드마크 기반 AI 모델 학습 완료 및 저장됨!")

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 예시 데이터 (손 랜드마크 좌표와 라벨)
X = np.random.rand(100, 63)  # 랜덤 데이터 (100개의 샘플, 각 샘플은 63개의 랜드마크 좌표)
y = np.random.randint(0, 4, 100)  # 4개의 클래스(예: thumbs_down, victory, thumbs_up, pointing_up)

# 데이터 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 훈련 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')  # 4개의 클래스(thumbs_down, victory, thumbs_up, pointing_up)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# 모델 저장
model.save("gesture_model.h5")  # 모델을 'gesture_model.h5'로 저장
print("✅ 모델 학습 완료 및 저장됨!")
