# Webcam
from ultralytics import YOLO
import torch
# Load a model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO("yolo11n-pose.pt")  # load an official model
model = model.to(device)  

# Predict with the model
try:
    # Try to predict with source=4
    results = model(source=4, show=True, conf=0.4)  # predict on an image
except Exception as e:
    # If an error occurs, print the error and use source=0 instead
    print(f"USB Camera not detected. Using webcam instead.")
    results = model(source=0, show=True, conf=0.4)  # predict on an image

