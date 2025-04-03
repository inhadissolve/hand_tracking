# import pandas as pd
#
# # 1. csv 불러오기 (인코딩 지정)
# df = pd.read_csv('hand_gesture_data.csv', encoding='cp949')
#
# # 2. 삭제할 라벨 선택
# target_label = "ㅌ"
#
# # 3. 해당 라벨 제거
# df = df[df['label'] != target_label]
#
# # 4. 다시 저장 (같은 인코딩으로 저장)
# df.to_csv("hand_gesture_data.csv", index=False, encoding='cp949')
#
# print(f"[INFO] '{target_label}' 라벨 데이터 제거 완료!")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import os

# ───── 한글 폰트 설정 ─────
font_path = "C:/Windows/Fonts/malgun.ttf"  # 윈도우 기준
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.rcParams['axes.unicode_minus'] = False

# ───── 데이터 불러오기 ─────
file_path = "hand_gesture_data.csv"
df = pd.read_csv(file_path, encoding="cp949")

# ───── 삭제 대상 라벨 입력 ─────
target_label = input("시각화하고 싶은 라벨을 입력하세요 (예: ㅌ): ")

# ───── 해당 라벨 시각화 ─────
subset = df[df['label'] == target_label]

if subset.empty:
    print(f"[INFO] 라벨 '{target_label}'에 해당하는 데이터가 없습니다.")
else:
    print(f"[INFO] 라벨 '{target_label}'의 데이터 개수: {len(subset)}")

    # 히스토그램 (해당 라벨만)
    plt.figure(figsize=(8, 4))
    plt.hist(subset.index, bins=20, color='orange')
    plt.title(f"'{target_label}' 라벨 데이터 분포")
    plt.xlabel("데이터 인덱스")
    plt.ylabel("개수")
    plt.tight_layout()
    plt.show()

    # ───── 삭제 여부 확인 ─────
    confirm = input(f"[확인] 라벨 '{target_label}' 데이터를 삭제하시겠습니까? (y/n): ").lower()
    if confirm == 'y':
        df = df[df['label'] != target_label]
        df.to_csv(file_path, index=False, encoding="cp949")
        print(f"[INFO] '{target_label}' 라벨 데이터가 삭제되었습니다.")
    else:
        print("[INFO] 삭제를 취소했습니다.")
