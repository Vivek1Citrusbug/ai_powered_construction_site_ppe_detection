import streamlit as st
import tempfile
import cv2
import uuid
from pathlib import Path
from PIL import Image
from page_components import configure_ui, get_sidebar_options
from detect_service import PPEDetector
import subprocess

MODEL_PATHS = {
    "YOLOv8-medium": "models/Yolov8m_without_aug.pt",
    "YOLOv8-small": "models/Yolov8s_without_aug.pt",
    "YOLOv8-nano": "models/Yolov8n_without_aug.pt",
}
SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png"}
SUPPORTED_VIDEO_FORMATS = {"mp4"}


def convert_to_h264(input_path, output_path):
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        output_path,
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save uploaded file to a temporary location.
    """

    file_extension = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{file_extension}"
    ) as temp_file:
        temp_file.write(uploaded_file.read())
        return Path(temp_file.name)


def process_and_display_image(detector: PPEDetector, image_path: Path):
    """
    Process and display an image using the detector.
    """

    image = Image.open(image_path)
    with st.spinner("🛠️ Processing Image..."):
        processed_image = detector.process_image(str(image_path))
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    st.image(processed_image, caption="🔍 Processed Image", use_container_width=True)


def process_and_display_video(detector: PPEDetector, video_path: Path):
    """
    Process and display a video using the detector.
    """

    with st.spinner("🛠️ Processing Video..."):
        output_video_path = detector.process_video(str(video_path))
        converted_video_path = f"dest/{uuid.uuid4()}.mp4"
        convert_to_h264(output_video_path, converted_video_path)
    
    st.video(str(converted_video_path))


def main():
    """
    Main execution function.
    """

    configure_ui()

    model_choice, uploaded_files, use_webcam = get_sidebar_options()

    # Initialize Detector
    detector = PPEDetector(MODEL_PATHS.get(model_choice, "YOLOv8-small"))

    if use_webcam:
        detector.process_webcam()

    elif uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            file_extension = file_path.suffix[1:].lower()

            if file_extension in SUPPORTED_IMAGE_FORMATS:
                process_and_display_image(detector, file_path)

            elif file_extension in SUPPORTED_VIDEO_FORMATS:
                process_and_display_video(detector, file_path)

            else:
                st.warning(f"⚠️ Unsupported file format: {file_extension}")

    else:
        st.info("📤 Upload images or videos for object detection.")

    st.sidebar.markdown("---")
    st.sidebar.info("👨‍💻 Developed with Streamlit & YOLOv8")


if __name__ == "__main__":
    main()
