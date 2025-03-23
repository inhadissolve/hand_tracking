# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# import pandas as pd
# import os
# from sklearn.preprocessing import StandardScaler
#
# # ✅ 학습된 모델 로드
# model = tf.keras.models.load_model("sign_language_model.keras")  # 최신 Keras 형식 사용
#
# # ✅ 여러 개의 제스처 라벨 불러오기
# gesture_labels = []
#
# # ✅ 'sign_data/' 폴더 내의 모든 CSV 파일을 불러와 라벨 리스트 생성
# data_folder = "sign_data"
# for file in os.listdir(data_folder):
#     if file.endswith("_data.csv"):  # 모든 학습된 CSV 파일 검색
#         df = pd.read_csv(os.path.join(data_folder, file))
#         unique_labels = df["Label"].unique()
#         gesture_labels.extend(unique_labels)
#
# # ✅ 중복 제거 후 정렬 (ㄱ~ㅎ 전체 포함하도록 정리)
# gesture_labels = sorted(list(set(gesture_labels)))
# print(f"✅ 불러온 제스처 라벨: {gesture_labels}")
#
# # ✅ MediaPipe 손 랜드마크 초기화
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
#
# # ✅ 웹캠 열기
# cap = cv2.VideoCapture(0)
#
# # ✅ StandardScaler 로드 (학습 데이터와 동일한 정규화 방식 사용)
# scaler = StandardScaler()
#
# # ✅ Temperature Scaling 값 (확률을 조정하기 위함)
# temperature = 2.5  # 클수록 신뢰도 조정이 강하게 적용됨
#
# # ✅ MediaPipe Hands 실행
# with mp_hands.Hands(
#         max_num_hands=1,
#         min_detection_confidence=0.6,
#         min_tracking_confidence=0.6) as hands:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # ✅ BGR → RGB 변환
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)
#
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # ✅ 손 랜드마크 그리기
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#                 # ✅ 랜드마크 좌표 수집 (상대 좌표 변환)
#                 hand_data = []
#                 for landmark in hand_landmarks.landmark:
#                     hand_data.append(landmark.x - hand_landmarks.landmark[0].x)  # 상대 좌표
#                     hand_data.append(landmark.y - hand_landmarks.landmark[0].y)
#                     hand_data.append(landmark.z)
#
#                 # ✅ 데이터 정규화 (StandardScaler 사용)
#                 hand_data = np.array(hand_data).reshape(1, -1)  # (1, 63)
#                 hand_data = scaler.fit_transform(hand_data)  # 학습 데이터와 동일한 방식으로 정규화
#
#                 # ✅ 랜덤 노이즈 추가 (과적합 방지)
#                 hand_data += np.random.normal(0, 0.01, hand_data.shape)
#
#                 # ✅ 모델 예측
#                 predictions = model.predict(hand_data)
#
#                 # ✅ Temperature Scaling 적용 (확률 조정)
#                 scaled_predictions = np.exp(predictions / temperature) / np.sum(np.exp(predictions / temperature))
#
#                 # ✅ 예측 확률 정렬
#                 sorted_indices = np.argsort(-scaled_predictions)[0]  # 내림차순 정렬
#                 max_index = sorted_indices[0]  # 가장 높은 확률을 가진 클래스
#                 second_best_index = sorted_indices[1] if len(sorted_indices) > 1 else None  # 두 번째 확률
#
#                 confidence = scaled_predictions[0, max_index] * 100  # 가장 높은 확률 (0~100%)
#                 second_confidence = scaled_predictions[0, second_best_index] * 100 if second_best_index is not None else 0
#
#                 # ✅ 특정 확률 이하라면 'Unknown' 처리
#                 if confidence < 85 or (second_best_index is not None and confidence - second_confidence < 10):
#                     predicted_label = "Unknown"
#                     predicted_label_korean = "알 수 없음"  # 한글 라벨 (콘솔 출력용)
#                 else:
#                     predicted_label = gesture_labels[max_index]
#                     predicted_label_korean = predicted_label  # 한글 라벨
#
#                 # ✅ 콘솔에 한글로 예측 결과 출력
#                 print(f"🔍 예측 결과: {predicted_label_korean}, 신뢰도: {confidence:.2f}%")
#
#                 # ✅ 웹캠 화면에서는 영어로 표시 (한글은 깨짐)
#                 cv2.putText(frame, f"Predicted: {predicted_label}", (10, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#                 # ✅ 확률 값도 표시
#                 cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 80),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         # ✅ 화면 출력
#         cv2.imshow('Sign Language Recognition', frame)
#
#         # ✅ ESC 키로 종료
#         if cv2.waitKey(5) & 0xFF == 27:  # ESC 키
#             break
#
# cap.release()
# cv2.destroyAllWindows()

import tensorflow as tf
import numpy as np
import pandas as pd

# 1. 모델 로드
model = tf.keras.models.load_model("gesture_model.h5")  # 학습된 모델 파일 경로

# 2. 모델 구조 확인
print("### 모델 구조 ###")
model.summary()  # 모델의 레이어, 파라미터 수, 출력 형태 등을 확인

# 3. 모델의 학습된 가중치 확인
print("\n### 모델의 학습된 가중치 ###")
weights = model.get_weights()
for i, weight in enumerate(weights):
    print(f"Layer {i} Weights shape: {weight.shape}")

# 4. 출력 레이어 활성화 함수 확인
print("\n### 출력 레이어 활성화 함수 ###")
output_layer = model.layers[-1]
print(f"Output layer activation function: {output_layer.activation}")

# 5. 학습 데이터의 라벨 확인
print("\n### 학습 데이터 라벨 확인 ###")
df = pd.read_csv('sign_data/gesture_data.csv')  # 학습에 사용된 데이터셋 경로
print(f"Unique Labels: {df['Label'].unique()}")  # 라벨 목록 확인

# 6. 테스트 데이터로 모델 예측 테스트
# 예시 랜덤 데이터 (손 랜드마크 좌표 형태로 가정)
test_data = np.random.rand(1, 63)  # 63개 랜드마크 좌표로 구성된 입력 데이터
test_data = (test_data - 0.5) / 0.5  # 정규화

# 예측 수행
predictions = model.predict(test_data)

# 예측된 라벨과 그에 대한 확률 출력
predicted_label = np.argmax(predictions, axis=1)  # 가장 높은 확률을 가진 클래스의 인덱스
confidence = np.max(predictions) * 100  # 예측 확률

print(f"\n### 예측 결과 ###")
print(f"Predicted label: {predicted_label}")
print(f"Confidence: {confidence:.2f}%")

