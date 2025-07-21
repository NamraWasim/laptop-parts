from ultralytics import YOLO

# Pre-trained YOLOv8 nano model (fast and small)
model = YOLO("yolov8n.pt")

# Train the model on your dataset
model.train(
    data="laptop-parts--1/data.yaml",  # Make sure this path is correct
    epochs=20,
    imgsz=640
)
