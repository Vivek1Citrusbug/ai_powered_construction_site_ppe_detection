import cv2
from ultralytics import YOLO
import streamlit as st
import tempfile
import os
import subprocess

def process_image(image_path, model):
    image = cv2.imread(image_path)
    results = model(image)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]}: {conf:.2f}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    isColor=True
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_video.name  
    print("Outputpath### : ",output_path)
    temp_video.close() 

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height),isColor)
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
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()  

    print("Outputpath### 4534354: ",output_path)
    return output_path  

st.title("Object Detection using YOLOv8")
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    model = YOLO("C:/Users/vivek/VivekInternship/AI_powered_construction_site_ppe_detection/models/Yolov8n_without_aug.pt")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
        # print("/n Temp_path : ",temp_path)

    if uploaded_file.type.startswith("video"):
        st.write("Processing Video... ⏳")
        output_video_path = process_video(temp_path, model)
        print("/n Output_Video_path : ",output_video_path)
        st.write("### Processed Video 📹")
        st.video(output_video_path) 
    elif uploaded_file.type.startswith("image"):
        processed_image = process_image(temp_path, model)
        st.image(processed_image, caption="Processed Image", channels="BGR")
