import streamlit as st
from page_components import configure_ui, get_sidebar_options
from detect_service import PPEDetector
import tempfile
import cv2
from PIL import Image
import uuid

MODEL_PATHS = {
    "YOLOv8-medium": "models/Yolov8m_without_aug.pt",
    "YOLOv8-small": "models/Yolov8s_without_aug.pt",
    "YOLOv8-nano": "models/Yolov8n_without_aug.pt",
}

configure_ui()

model_choice, uploaded_files, use_webcam = get_sidebar_options()

detector = PPEDetector(MODEL_PATHS[model_choice])

if use_webcam:
    detector.process_webcam()

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
                processed_image = detector.process_image(temp_path)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

            with image_cols:
                st.image(
                    processed_image,
                    caption="🔍 Processed Image",
                    use_container_width=True,
                )

        elif file_extension in ["mp4"]:
            with st.spinner("🛠️ **Processing Video...**"):
                output_video_path = detector.process_video(temp_path)

            key = uuid.uuid4()

            with video_cols:
                st.video(output_video_path)
else:
    st.info("📤 **Upload images or videos for object detection.**")

st.sidebar.markdown("---")
st.sidebar.info("👨‍💻 **Developed with Streamlit & YOLOv8**")
