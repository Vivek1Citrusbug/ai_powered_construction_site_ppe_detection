import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import streamlit as st

class PPEDetector:
    def __init__(self, model_path):
        """
        Initialize the PPE Detector with a given YOLO model
        """

        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        """
        Detect PPE objects in an image frame
        """
        
        results = self.model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{self.model.names[cls]}: {conf:.2f}"

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

        return frame

    def process_image(self, image_path):
        """
        Process an image for PPE detection
        """
        
        image = cv2.imread(image_path)
        return self.detect_objects(image)

    def process_video(self, video_path):
        """
        Process a video and detect PPE in each frame
        """
        
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"h264")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = temp_video.name
        temp_video.close()

        progress_bar = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = self.detect_objects(frame)
        
            out.write(processed_frame)
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)

        cap.release()
        out.release()
        return output_path
