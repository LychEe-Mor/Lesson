# Lesson
Crack Detection and Recognition based on YOLOv8
%pip install ultralytics
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/kaggle/input/clack1/data.yaml", epochs=300, device=[0, 1], batch=16, workers=8, save_period=50)
