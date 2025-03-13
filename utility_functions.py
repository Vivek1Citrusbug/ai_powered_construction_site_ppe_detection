import streamlit as st
import subprocess
import tempfile
import cv2
import uuid
from PIL import Image
from pathlib import Path
from detect_service import PPEDetector


def convert_to_h264(input_path, output_path):
    """
    Converting input file to h264 encoded formate.

    Reason:-
    streamlit html5 does not support mp4v,h264 codec. We need to manually convert input video file using ffmpeg.
    """

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
