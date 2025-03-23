import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import time

# 초기화
last_print_time = 0
PRINT_INTERVAL = 1.0  # 초 단위 (예: 1초에 한 번 출력)

# 콜백 함수 수정
def gesture_callback(result, output_image, timestamp):
    global last_print_time
    current_time = time.time()
    if result.gestures:
        top_gesture = result.gestures[0][0].category_name
        if current_time - last_print_time > PRINT_INTERVAL:
            print(f"[제스처 인식] {top_gesture}")
            last_print_time = current_time

# 모델 경로
MODEL_PATH = 'gesture_recognizer.task'

# 웹캠 설정
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] 웹캠을 열 수 없습니다.")
    exit()

# 옵션 설정
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=gesture_callback
)

recognizer = vision.GestureRecognizer.create_from_options(options)

# 프레임 읽기 및 처리
print("[INFO] 제스처 인식 시작...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 프레임을 읽을 수 없습니다.")
        break

    # OpenCV의 BGR -> RGB 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe Image 형식으로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 현재 시간 (마이크로초 단위)
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)

    # 제스처 인식
    recognizer.recognize_async(mp_image, timestamp)

    # 화면 출력
    cv2.imshow("Gesture Recognition", frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료
cap.release()
cv2.destroyAllWindows()