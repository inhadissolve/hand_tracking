import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

# 모델 경로
model_path = 'gesture_recognizer.task'

# 설정값
DELAY_SECONDS = 0.5
IDLE_RESET_SECONDS = 2.0  # 2초 이상 새로운 단어 없으면 문장 출력
last_recognized_time = 0
last_gesture = None

# 누적 텍스트
sentence = []

# 제스처 인식기 설정
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=lambda result, output_image, timestamp_ms: on_result(result)
)

# 콜백 함수 정의
def on_result(result):
    global last_recognized_time, last_gesture, sentence
    current_time = time.time()

    if result.gestures:
        gesture = result.gestures[0][0].category_name

        if gesture != last_gesture:
            # 새로운 단어일 경우 누적
            sentence.append(gesture)
            print(f"[ADD] {gesture}")
            last_gesture = gesture
            last_recognized_time = current_time
    else:
        # 일정 시간 동안 단어 없으면 문장 출력
        if sentence and (current_time - last_recognized_time > IDLE_RESET_SECONDS):
            print("[FINAL SENTENCE] " + " ".join(sentence))
            sentence = []  # 문장 초기화
            last_gesture = None

# 제스처 인식기 초기화
print("[DEBUG] GestureRecognizer 초기화 중...")
recognizer = vision.GestureRecognizer.create_from_options(options)

# 웹캠 시작
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] 웹캠을 열 수 없습니다.")
    exit()

print("[INFO] 웹캠 연결 성공. 제스처 인식 시작...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 프레임을 읽을 수 없습니다.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(time.time() * 1000)

    recognizer.recognize_async(mp_image, timestamp_ms)

    # 누적된 문장을 화면에 표시
    current_sentence = " ".join(sentence[-5:])  # 최근 5단어만 표시
    cv2.putText(frame, current_sentence, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()