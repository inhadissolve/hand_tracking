import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# CSV 불러오기
data = pd.read_csv("sign_data/ㄱ_data.csv")  # 수어 데이터 CSV 파일 경로

# 데이터 전처리
X = data.drop(columns=['label'])
y = data['label']

# 0이 포함된 행 제거 (정확한 학습을 위해)
X = X[~(X == 0).any(axis=1)]
y = y[X.index]  # 라벨도 인덱스에 맞게

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 평가
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 모델 저장
joblib.dump(clf, 'hand_gesture_model.pkl')
print("✅ 모델 저장 완료: hand_gesture_model.pkl")
