import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import gesture_recognizer
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizer, GestureRecognizerOptions
from mediapipe.framework.formats import image_format_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe import Image, ImageFormat


# ✅ 디버깅: 확인
print("[DEBUG] 필요한 모듈 import 완료")

# 모델 경로
model_path = 'gesture_recognizer.task'

# 속도 조절 변수
DELAY_SECONDS = 0.5
last_recognized_time = 0

# 인식 콜백 함수
def result_callback(result, output_image, timestamp_ms):
    global last_recognized_time, recognized_label
    current_time = time.time()

    if result.gestures:
        top_gesture = result.gestures[0][0]
        label = top_gesture.category_name
        score = top_gesture.score

        if current_time - last_recognized_time > DELAY_SECONDS:
            print(f"[RECOGNIZED] Gesture: {label}, Score: {score:.2f}")
            recognized_label = label
            last_recognized_time = current_time

# 옵션 설정
base_options = python.BaseOptions(model_asset_path=model_path)
options = GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback
)

print("[DEBUG] GestureRecognizerOptions 설정 완료")

# 제스처 인식기 초기화
recognizer = GestureRecognizer.create_from_options(options)

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] 웹캠을 열 수 없습니다.")
    exit()

print("[INFO] 웹캠 연결 성공. 제스처 인식 시작...")

recognized_label = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 프레임을 읽을 수 없습니다.")
        break

    # BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe 이미지 객체 생성
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

    # 현재 시간 (ms)
    timestamp_ms = int(time.time() * 1000)

    # 인식 요청
    recognizer.recognize_async(mp_image, timestamp_ms)

    # 화면에 텍스트 표시
    cv2.putText(frame, f"Gesture: {recognized_label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # 결과 출력
    cv2.imshow("Gesture Recognition", frame)

    # 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()