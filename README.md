# 😊 Real-Time Face & Emotion Detection using YOLOv8 and ViT

This project combines YOLOv8 for **face detection** with a Vision Transformer (ViT) model for **emotion classification**. It runs in real-time using a webcam feed and annotates detected faces with their predicted emotions.

---

## 🔧 Features

- Real-time face detection using `YOLOv8`
- Emotion classification using Hugging Face model: [`trpakov/vit-face-expression`](https://huggingface.co/trpakov/vit-face-expression)
- Seamless integration between detection and classification
- Clean OpenCV-based visualization

---

## 📦 Requirements

Install the required Python packages:

```bash
pip install ultralytics opencv-python torch torchvision transformers huggingface_hub pillow
🔐 Hugging Face Token
This script uses a private YOLOv8 model hosted on Hugging Face. You'll need to provide an access token.

Get your token from: https://huggingface.co/settings/tokens

Replace this line in the code with your token:

python
Copy
Edit
HF_TOKEN = "***********************"
🧠 Models Used
👤 Face Detection
Model: arnabdhar/YOLOv8-Face-Detection

Platform: Hugging Face Hub

Usage: Downloads model.pt dynamically

😊 Emotion Classification
Model: trpakov/vit-face-expression

Framework: Hugging Face Transformers

🖥️ How It Works
Captures webcam video stream using OpenCV.

Detects faces using a YOLOv8 model.

Crops and preprocesses each face.

Uses a ViT-based classifier to detect the emotion.

Annotates each face with a bounding box and emotion label.

Example Output
On the webcam, each detected face will be highlighted with a green rectangle and the predicted emotion label above it, such as:

happy

angry

neutral

surprised
