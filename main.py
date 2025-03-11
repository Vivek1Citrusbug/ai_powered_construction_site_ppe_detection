import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image
import uuid

st.set_page_config(page_title="AI-Powered PPE Detection", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .title {
        text-align: center;
        color: #4CAF50;
        font-size: 28px;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<p class='title'>AI-Powered Construction Site PPE Detection</p>",
    unsafe_allow_html=True,
)
st.write(
    "🚀 Upload multiple **images/videos**, and the app will detect **Personal Protective Equipment (PPE)**."
)

st.sidebar.title("🔍 Select YOLOv8 Model")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    [
        "YOLOv8-medium",
        "YOLOv8-small",
        "YOLOv8-nano",
    ],
)

MODEL_PATHS = {
    "YOLOv8-medium": "models/Yolov8m_without_aug.pt",
    "YOLOv8-small":"models/Yolov8s_without_aug.pt",
    "YOLOv8-nano": "models/Yolov8n_without_aug.pt",
}
model = YOLO(MODEL_PATHS[model_choice])

st.sidebar.title("📤 Upload Images or Videos")
uploaded_files = st.sidebar.file_uploader(
    "Upload multiple images/videos",
    type=["jpg", "jpeg", "png", "mp4"],
    accept_multiple_files=True,
)

use_webcam = st.sidebar.checkbox("📷 Enable Webcam for Live Detection")


def live_webcam_detection(model):
    """
    Detect webcame and give live object detection of PPE kit.
    """

    st.write("📷 **Live Webcam Object Detection**")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error(
                "❌ **Failed to access webcam.** Please check your camera settings."
            )
            break

        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]}: {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()


def process_image(image_path, model):
    """
    Take image as input and gives bounding box of PPE kit.
    """

    image = cv2.imread(image_path)
    results = model(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]}: {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return image


def process_video(video_path, model):
    """
    Take video clip as input, gives bounding box for detected PPE kit.
    """
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_video.name
    temp_video.close()

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    st.write("🔄 **Processing Video... Please wait.**")
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]}: {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        out.write(frame)
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_path


if use_webcam:
    live_webcam_detection(model)

elif uploaded_files:
    image_cols, video_cols = st.columns(2)

    for file in uploaded_files:
        file_extension = file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file_extension}"
        ) as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name

        if file_extension in ["jpg", "jpeg", "png"]:
            image = Image.open(file)

            with st.spinner("🛠️ **Processing Image...**"):
                processed_image = process_image(temp_path, model)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

            with image_cols:
                st.image(
                    processed_image,
                    caption="🔍 Processed Image",
                    use_container_width=True,
                )

        elif file_extension in ["mp4"]:

            with st.spinner("🛠️ **Processing Video...**"):
                output_video_path = process_video(temp_path, model)

            key = uuid.uuid4()

            with video_cols:
                st.video(output_video_path)
                st.download_button(
                    key=key,
                    label=f"📥 Download Processed Video {key}",
                    data=open(output_video_path, "rb").read(),
                    file_name=f"processed_video_{key}.mp4",
                    mime="video/mp4",
                )
else:
    st.info("📤 **Upload images or videos for object detection.**")

st.sidebar.markdown("---")
st.sidebar.info("👨‍💻 **Developed with Streamlit & YOLOv8**")
