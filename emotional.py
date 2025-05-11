from ultralytics import YOLO
import cv2
import torch
from huggingface_hub import hf_hub_download
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Load YOLOv8 face detection model
def load_face_model():
    HF_TOKEN = "***********************"  # Replace with your actual token
    face_model_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection",
        filename="model.pt",
        use_auth_token=HF_TOKEN
    )
    return YOLO(face_model_path)

# Load pre-trained emotion classification model from Hugging Face

from transformers import AutoModelForImageClassification, AutoFeatureExtractor

def load_emotion_classifier():
    model_name = "trpakov/vit-face-expression"  # Updated model
    model = AutoModelForImageClassification.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model.eval()
    return model, feature_extractor



# Preprocessing function for emotion classification
def preprocess_face(face, feature_extractor):
    face = Image.fromarray(face)
    inputs = feature_extractor(images=face, return_tensors="pt")
    return inputs

# Run face detection + emotion classification
def detect_emotion():
    face_model = load_face_model()
    emotion_model, feature_extractor = load_emotion_classifier()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = face_model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract face
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # Preprocess and classify emotion
                inputs = preprocess_face(face, feature_extractor)
                with torch.no_grad():
                    outputs = emotion_model(**inputs)
                    emotion_label = emotion_model.config.id2label[torch.argmax(outputs.logits).item()]

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_emotion()
