# ✅ step3: train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'gesture_data_cleaned.csv')
df = pd.read_csv(data_path)

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X = df.drop(['label', 'label_encoded'], axis=1)
y = df['label_encoded']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("[INFO] 모델 정확도: {:.2f}%".format(accuracy_score(y_val, y_pred) * 100))
print("\n[INFO] 라벨별 정밀도 등 평가:\n")
print(classification_report(y_val, y_pred, target_names=le.classes_))

model_path = os.path.join(BASE_DIR, 'gesture_model.pkl')
encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')
joblib.dump(model, model_path)
joblib.dump(le, encoder_path)
print("[INFO] 학습된 모델 및 인코더 저장 완료!")