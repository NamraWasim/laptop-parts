from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction
results = model.predict(source="images.jpeg", save=False, conf=0.3)

# Load and display the result image with OpenCV
from PIL import Image
import numpy as np

# Convert YOLO result to numpy image
for r in results:
    img_array = r.plot()  # Draw bounding boxes
    img = Image.fromarray(img_array)
    img.show()  # Show image using PIL
