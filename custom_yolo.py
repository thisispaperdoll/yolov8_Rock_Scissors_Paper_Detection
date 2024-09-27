from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Interactive Interface for YOLOv8")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")
    
# import streamlit as st
# import cv2
# from ultralytics import YOLO
# import numpy as np
# from PIL import Image
# from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam
# # from ultralytics import solutions

# # Pass a model as an argument
# # solutions.inference(model="./custom_train/yolov8n_rock_paper_scissors.pt")

# # 페이지 설정
# st.set_page_config(page_title="YOLO Model Demo", layout="centered")

# # 페이지 제목 및 설명
# st.title("🤸‍♀️YOLOv8")
# st.title("Rock✊, Scissors✌, Paper🖐 Detection")
# st.write("""
# YOLOv8 모델을 사용하여 가위, 바위, 보를 실시간으로 분류합니다.
# 이 모델은 Roboflow를 통해 학습된 데이터셋을 사용하며, 다음과 같은 과정을 거쳤습니다:
# - 데이터 수집 및 라벨링
# - 모델 학습 및 검증
# - 웹캠을 통한 실시간 객체 탐지
# """)

# # 수평으로 이미지 정렬
# col1, col2 = st.columns(2)

# # 각 컬럼에 이미지 추가
# with col1:
#     st.image("https://github.com/user-attachments/assets/cbc639f2-7055-419e-a94e-6e1fbe143b61", 
#              caption="Rock, Paper, Scissors Dataset", use_column_width=True)

# with col2:
#     st.image("https://github.com/user-attachments/assets/3840f1b6-06ce-401b-b242-1dc0e1dbf891", 
#              caption="YOLOv8 Model", use_column_width=True)

# # 모델 설명 추가
# st.write("""
# 위의 이미지는 학습된 데이터셋과 YOLOv8 모델의 구조를 나타냅니다.
# 실시간 웹캠에서 가위✌, 바위✊, 보🖐 중 하나를 내면
# yolov8 모델을 통해 분류할 수 있습니다 !✨
# """)

# # 웹캠 선택 및 설정
# st.header("웹캠을 통해 실시간 분류하기")
# camera_index = st.sidebar.selectbox("Select Camera Index", [0, 1, 2])

# # 모델 파일 경로 설정
# model = YOLO('./custom_train/yolov8n_rock_paper_scissors.pt')  # 모델 파일 경로

# # 모델 클래스 이름 출력
# st.sidebar.text("Model Classes:")
# st.sidebar.write(model.names)

# # 업로드된 이미지 분류 섹션
# st.sidebar.header("Image Upload for Classification")
# uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # 업로드된 이미지를 PIL로 열기
#     image = Image.open(uploaded_file)

#     # 이미지를 OpenCV 형식으로 변환
#     frame = np.array(image)
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#     # 업로드된 이미지 표시
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # 객체 탐지 (Rock, Paper, Scissors 클래스 탐지)
#     results = model.predict(frame, classes=[0, 1, 2], conf=0.4, imgsz=640)

#     # 탐지된 결과 시각화
#     annotated_frame = results[0].plot()

#     # BGR 이미지를 RGB로 변환 (OpenCV는 BGR 형식이므로, RGB 형식으로 변환 필요)
#     annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

#     # Streamlit을 통해 탐지된 이미지 표시
#     st.image(annotated_frame, caption="Detected Image", use_column_width=True)

#     # 감지된 객체 목록 표시
#     st.subheader("Detected Objects")
#     if len(results[0].boxes) > 0:
#         for box in results[0].boxes:
#             cls = int(box.cls[0])
#             label = model.names[cls]
#             confidence = box.conf[0]
#             st.write(f"Detected {label} with {confidence:.2f} confidence.")
#     else:
#         st.write("No objects detected.")

# # 웹캠 스트리밍
# st.header("Real-Time Classification with Webcam")

# # 웹캠 스트리밍 시작 및 중지 상태를 관리하는 변수
# if "streaming" not in st.session_state:
#     st.session_state.streaming = False

# # 웹캠 스트리밍 시작 버튼
# start_button = st.button("Start Webcam")
# if start_button:
#     st.session_state.streaming = True

# # 웹캠 스트리밍 중지 버튼
# stop_button = st.button("Stop Webcam")
# if stop_button:
#     st.session_state.streaming = False

# # 웹캠 스트리밍 상태에 따른 처리
# if st.session_state.streaming:
#     confidence = 0.25
#     infer_uploaded_webcam(confidence, model)

# #     stframe = st.empty()  # Streamlit에서 사용할 빈 이미지 프레임 설정

# #     # while st.session_state.streaming:
# #     # st.camera_input()을 사용하여 웹캠에서 프레임 캡처
# #     img_file_buffer = st.camera_input("Capture")

# #     if img_file_buffer is not None:
# #         # 이미지를 OpenCV 형식으로 변환
# #         bytes_data = img_file_buffer.getvalue()
# #         frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
# #         # frame = img_file_buffer.getvalue()
# #         # frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
# #         # frame = np.array(Image.open(img_file_buffer))
# #         # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# #         # 객체 탐지 (Rock, Paper, Scissors 클래스 탐지)
# #         results = model.predict(frame, classes=[0, 1, 2], conf=0.4, imgsz=640)

# #         # 탐지된 결과 시각화
# #         annotated_frame = results[0].plot()

# #         # BGR 이미지를 RGB로 변환 (OpenCV는 BGR 형식이므로, RGB 형식으로 변환 필요)
# #         annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

# #         # Streamlit을 통해 이미지 표시
# #         stframe.image(annotated_frame, channels="RGB")

# #     st.write("Webcam streaming stopped.")


