# pip install opencv-python

import cv2
from ultralytics import YOLO

# 모델 파일 경로 확인
model = YOLO('./custom_train/yolov8n_rock_paper_scissors.pt')  # 모델 파일 경로를 정확히 설정하세요.
# model = YOLO('./yolov8n.pt')  # 필요에 따라 다른 경로 사용

# 모델 클래스 이름 출력
print(model.names)

# 웹캠 초기화
webcamera = cv2.VideoCapture(0)

# 웹캠 해상도 설정 (필요시 주석 해제)
# webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    success, frame = webcamera.read()
    if not success:
        print("웹캠에서 프레임을 읽어올 수 없습니다.")
        break

    # 객체 탐지 (일반적인 객체 탐지의 경우)
    results = model.predict(frame, classes=None, conf=0.4, imgsz=640)  # classes=None으로 모든 클래스 탐지

    # 객체 수 텍스트 추가
    cv2.putText(frame, f"Total: {len(results[0].boxes)}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # 탐지된 결과 시각화
    annotated_frame = results[0].plot()
    cv2.imshow("Live Camera", annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 리소스 해제
webcamera.release()
cv2.destroyAllWindows()
