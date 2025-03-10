import torch
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import random

# Define paths
models = {
    "disease": "runs/detect/quadrant_enumeration_disease_train/weights/best.pt",
    "enumeration": "runs/detect/quadrant_enumeration_train/weights/best.pt",
    "quadrant": "runs/detect/quadrant_train/weights/best.pt",
}
input_dir = "xrays"
output_dir = "prediction/aggregate"
os.makedirs(output_dir, exist_ok=True)

# Load models
yolo_models = {name: YOLO(path) for name, path in models.items()}

# Get input images
image_paths = list(Path(input_dir).glob("*.png"))

# Function to generate distinct colors
def get_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

# Run inference and save predictions
for img_path in image_paths:
    img = cv2.imread(str(img_path))
    detections = []
    
    # Run disease model inference
    disease_results = yolo_models["disease"](str(img_path))[0]
    
    for box, cls in zip(disease_results.boxes.xyxy, disease_results.boxes.cls):  # Iterate through disease bounding boxes
        x1, y1, x2, y2 = map(int, box[:4])
        disease_label = yolo_models["disease"].names[int(cls)]  # Get disease class label
        
        # Run quadrant and enumeration models to get corresponding labels
        quadrant_results = yolo_models["quadrant"](str(img_path))[0]
        enumeration_results = yolo_models["enumeration"](str(img_path))[0]
        
        quadrant_label = yolo_models["quadrant"].names[int(quadrant_results.boxes.cls[0])] if len(quadrant_results.boxes.cls) > 0 else "?"
        enumeration_label = yolo_models["enumeration"].names[int(enumeration_results.boxes.cls[0])] if len(enumeration_results.boxes.cls) > 0 else "?"
        
        # Combine labels
        label = f"Q: {quadrant_label} N: {enumeration_label} D: {disease_label}"
        detections.append((x1, y1, x2, y2, label))
    
    # Assign unique colors to bounding boxes
    random.shuffle(detections)
    box_colors = [get_random_color() for _ in detections]
    
    # Draw bounding boxes with combined labels
    for (x1, y1, x2, y2, label), color in zip(detections, box_colors):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    output_file = Path(output_dir) / img_path.name
    cv2.imwrite(str(output_file), img)
    print(f"Saved: {output_file}")

print("Aggregate inference completed!")
