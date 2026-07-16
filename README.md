# 🌊 Intelligent Underwater Object Detection Using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-GradCAM-red?style=for-the-badge&logo=pytorch)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Object%20Detection-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B?style=for-the-badge&logo=streamlit)

An AI-powered web application for underwater image enhancement and object detection using **YOLOv11**, **TensorFlow**, **Grad-CAM**, and **Streamlit**.

</div>

---

# 📖 Overview

Underwater images often suffer from poor visibility, low contrast, color distortion, and haze, making object detection challenging.

This project presents an intelligent underwater object detection system that enhances underwater images using a deep learning-based dehazing model before performing object detection with **YOLOv11**. To improve model transparency, **Grad-CAM** is integrated to visualize the regions influencing predictions.

The application provides an intuitive Streamlit interface with image enhancement, object detection, explainability, and an AI chatbot.

---

# ✨ Features

- 🌊 Underwater Image Dehazing
- 🎯 Underwater Object Detection using YOLOv11
- 🔥 Grad-CAM Explainability
- 🤖 Gemini AI Chatbot
- 🔐 User Authentication
- 📥 Download Enhanced Images
- ⚡ Automatic & Manual Dehazing Modes
- 🖥️ Interactive Streamlit Web Interface

---

# 🏗️ System Architecture

```
                 Underwater Image
                        │
                        ▼
             Image Quality Analysis
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
  Image Dehazing                  Original Image
         │                             │
         └──────────────┬──────────────┘
                        ▼
                 YOLOv11 Detection
                        │
                        ▼
              Bounding Box Prediction
                        │
                        ▼
              Grad-CAM Visualization
                        │
                        ▼
             Streamlit Web Application
                        │
                        ▼
               Gemini AI Assistant
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| YOLOv11 | Object Detection |
| TensorFlow | Image Dehazing |
| PyTorch | Grad-CAM |
| OpenCV | Image Processing |
| Streamlit | Web Application |
| SQLite | Authentication |
| Gemini API | AI Chatbot |

---

# 📂 Project Structure

```
Underwater-Object-Detection-Using-Deep-Learning
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks
│   ├── image_dehazing.ipynb
│   └── yolo11_gradcam.ipynb
│
├── model
│   └── README.md
│
├── trained_model
│   └── README.md
│
└── sample_images
```

---

# 📊 Dataset

The project is trained using an underwater object detection dataset containing marine organisms and underwater debris.

### Example Classes

- Fish
- Crab
- Eel
- Starfish
- Marine Plants
- ROV
- Plastic Bottles
- Plastic Bags
- Nets
- Ropes
- Containers
- Cups
- Wreckage
- Other Marine Debris

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/uttejkaramala/Underwater-Object-Detection-Using-Deep-Learning.git

cd Underwater-Object-Detection-Using-Deep-Learning
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📁 Required Models

This repository does **not** include the trained models due to GitHub size limitations.

Place the following files before running the application.

## YOLO Model

```
model/
└── best.pt
```

## TensorFlow Dehazing Model

```
trained_model/
├── saved_model.pb
└── variables/
    ├── variables.data-00000-of-00001
    └── variables.index
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

# 🚀 Workflow

1. Upload an underwater image.
2. Select Automatic or Manual Dehazing.
3. Enhance the image using the dehazing model.
4. Detect underwater objects using YOLOv11.
5. Visualize model attention with Grad-CAM.
6. View detected objects and confidence scores.
7. Download enhanced images or interact with the AI chatbot.

---



# 🔮 Future Improvements

- Real-time underwater video detection
- Marine species classification
- Object tracking
- Edge device deployment
- Autonomous underwater vehicle integration
- Multi-language chatbot support

---

# 👨‍💻 Author

**Uttej Karamala**

B.Tech in Artificial Intelligence

Java Full Stack Developer | Machine Learning Enthusiast

---
