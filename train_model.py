# step3_train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. 전처리된 CSV 로드
df = pd.read_csv('gesture_data_cleaned.csv')

# 2. 라벨 인코딩 (문자 → 숫자)
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# 3. 입력(X)과 정답(y) 분리
X = df.drop(['label', 'label_encoded'], axis=1)
y = df['label_encoded']

# 4. 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 평가
y_pred = model.predict(X_val)
print("[INFO] 모델 정확도: {:.2f}%".format(accuracy_score(y_val, y_pred) * 100))
print("\n[INFO] 라벨별 정밀도 등 평가:\n")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# 7. 모델 및 라벨 인코더 저장
joblib.dump(model, 'gesture_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("[INFO] 학습된 모델 및 인코더 저장 완료!")