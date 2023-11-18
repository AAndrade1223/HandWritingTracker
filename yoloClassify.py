import torch
import ultralytics
from PIL import Image
from ultralytics import YOLO

# Model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='mnist', epochs=1, imgsz=32)

# Images
imgs = ['data/testImage/sample.jpg']  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)

