import pandas as pd
import os
from pandasgui import show

# 사용자에게 수어 라벨 입력받기
gesture = input("수어 라벨을 입력하세요 (예: Hello, ThankYou, ILoveYou): ")
csv_path = f"sign_data/{gesture}_data.csv"

# CSV 파일 확인
if not os.path.exists(csv_path):
    print("⚠️ 해당 수어 데이터 파일이 존재하지 않습니다.")
    exit()

# CSV 파일 불러오기
df = pd.read_csv(csv_path)

# GUI 창에서 CSV 파일 수정
print("📌 Pandas GUI 창을 열어 데이터를 수정하세요. 수정 후 닫으면 자동 저장됩니다.")
show(df)

# 수정된 데이터를 다시 저장
df.to_csv(csv_path, index=False)
print("✅ 데이터 수정이 완료되었습니다!")