- Command to run streamlit application
    -- streamlit run main.py --server.fileWatcherType none


# 🚀 Model Comparison: YOLOv8n vs YOLOv8s vs YOLOv8m

## 📌 Project Overview
This repository compares the performance of different YOLOv8 models (YOLOv8n, YOLOv8s, YOLOv8m) for PPE detection in construction sites. We trained each model on the same dataset and analyzed their performance based on **mAP, training loss, inference speed, and precision-recall curves**.

---

## ⚙️ Model Details
| Model  | Params | Input Size | Dataset | Epochs | Batch Size |
|--------|--------|-----------|---------|--------|------------|
| YOLOv8n | 3.2M  | 640x640   | PPE Dataset | 50 | 16 |
| YOLOv8s | 11.2M | 640x640   | PPE Dataset | 50 | 16 |
| YOLOv8m | 25.9M | 640x640   | PPE Dataset | 50 | 16 |

---

## 📈 Training Performance

### 🔥 **Loss Curves**
Here’s how the training and validation loss evolved over time for each model:

| YOLOv8n | YOLOv8s | YOLOv8m |
|---------|---------|---------|
| ![YOLOv8n Loss](assets/yolov8n_loss.png) | ![YOLOv8s Loss](assets/yolov8s_loss.png) | ![YOLOv8m Loss](assets/yolov8m_loss.png) |

---

### 🏆 **mAP (Mean Average Precision) Scores**
| Model  | mAP@50 | mAP@50-95 | Precision | Recall |
|--------|--------|----------|-----------|--------|
| YOLOv8n | 83.2% | 61.8% | 82.5% | 79.1% |
| YOLOv8s | 87.5% | 68.2% | 85.9% | 81.4% |
| YOLOv8m | 91.2% | 74.5% | 89.8% | 85.7% |

---

### ⏱ **Inference Speed**
| Model  | FPS (Frames per Second) |
|--------|----------------------|
| YOLOv8n | 110 FPS |
| YOLOv8s | 85 FPS |
| YOLOv8m | 60 FPS |

---

## 🔍 Key Observations
- **YOLOv8n** is the fastest but has lower accuracy.  
- **YOLOv8m** has the highest accuracy but runs slower.  
- **YOLOv8s** offers a balance between speed and accuracy.  

---

## 🛠️ Reproducing the Results
### 🚀 **Training the Model**
To train any model, use:
```bash
yolo train data=data.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=16
