# Webcam
from ultralytics import YOLO
import torch
# Load a model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = YOLO("yolo11n-seg.pt")  # load an official model
# model = YOLO("yolo11s-seg.pt")  # load an official model
# model = YOLO("yolo11m-seg.pt")  # load an official model
# model = YOLO("yolo11l-seg.pt")  # load an official model
model = YOLO("yolo11x-seg.pt")  # load an official model
model = model.to(device)  

# Predict with the model
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
results = model(source=0, show=True, conf=0.4)  # predict on an image

