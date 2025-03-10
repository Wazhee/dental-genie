# Dental Genie

## Yolo Quadrant, Tooth number, and disease predictions
![val_0](https://github.com/user-attachments/assets/6db71d72-729c-46aa-9c74-e46ff893d51b)

## Yolo Aggrate Predictions
![val_0-2](https://github.com/user-attachments/assets/8612a1ca-f056-4285-ad5e-ae7a70338965)

## Get tooth bounding boxes
```python
disease_results = yolo_models["disease"](str(img_path))[0]
for box, cls in zip(disease_results.boxes.xyxy, disease_results.boxes.cls):  # Iterate through disease bounding boxes
        x1, y1, x2, y2 = map(int, box[:4])
        disease_label = yolo_models["disease"].names[int(cls)]  # Get disease class label
```

## Get quadrant and tooth number
```python
quadrant_results = yolo_models["quadrant"](str(img_path))[0]
enumeration_results = yolo_models["enumeration"](str(img_path))[0]
        
quadrant_label = yolo_models["quadrant"].names[int(quadrant_results.boxes.cls[0])] if len(quadrant_results.boxes.cls) > 0 else "?"
enumeration_label = yolo_models["enumeration"].names[int(enumeration_results.boxes.cls[0])] if len(enumeration_results.boxes.cls) > 0 else "?"

label = f"Q: {quadrant_label} N: {enumeration_label} D: {disease_label}"
```

## Draw single bounding box with labels
```python
for (x1, y1, x2, y2, label), color in zip(detections, box_colors):
  cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
  cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
```
