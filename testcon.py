import json
import os

# Load COCO annotations
with open('coco_annotations.json', 'r') as f:
    coco_data = json.load(f)

# Map COCO category IDs to YOLO class IDs
category_id_to_yolo_class = {
    # categories_1 (Quadrants)
    0: 0,  # Quadrant 1
    1: 1,  # Quadrant 2
    2: 2,  # Quadrant 3
    3: 3,  # Quadrant 4
    # categories_2 (Teeth)
    0: 4,  # Tooth 1
    1: 5,  # Tooth 2
    2: 6,  # Tooth 3
    3: 7,  # Tooth 4
    4: 8,  # Tooth 5
    5: 9,  # Tooth 6
    6: 10, # Tooth 7
    7: 11, # Tooth 8
    # categories_3 (Diseases)
    0: 12, # Impacted
    1: 13, # Caries
    2: 14, # Periapical Lesion
    3: 15  # Deep Caries
}

# Specify the output folder path
output_folder = r"C:\Users\destr\Desktop\Study\dental_genie\valy"  # Your output directory

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image
for image_info in coco_data['images']:
    image_id = image_info['id']
    image_width = image_info['width']
    image_height = image_info['height']
    yolo_annotations = []

    # Find all annotations for this image
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            # Check if all required category IDs exist
            if 'category_id_1' not in ann or 'category_id_2' not in ann or 'category_id_3' not in ann:
                print(f"Warning: Missing category_id in annotation {ann['id']}. Skipping this annotation.")
                print(ann)  # Print the problematic annotation for debugging
                continue  # Skip this annotation

            # Get category IDs, with default values if missing
            category_id_1 = ann.get('category_id_1', 0)  # Default to 0 if missing
            category_id_2 = ann.get('category_id_2', 0)  # Default to 0 if missing
            category_id_3 = ann.get('category_id_3', 0)  # Default to 0 if missing

            # Check if category IDs are valid
            if (category_id_1 not in category_id_to_yolo_class or
                category_id_2 not in category_id_to_yolo_class or
                category_id_3 not in category_id_to_yolo_class):
                print(f"Warning: Unknown category_id in annotation {ann['id']}. Skipping this annotation.")
                continue  # Skip this annotation

            # Get bounding box coordinates
            x_min, y_min, width, height = ann['bbox']

            # Convert to YOLO format
            x_center = (x_min + width / 2) / image_width
            y_center = (y_min + height / 2) / image_height
            norm_width = width / image_width
            norm_height = height / image_height

            # Add YOLO annotation with quadrant, tooth, and disease IDs separated by spaces
            yolo_annotations.append(f"{category_id_1} {category_id_2} {category_id_3} {x_center} {y_center} {norm_width} {norm_height}")

    # Save YOLO annotations to a .txt file in the output folder
    output_file_path = os.path.join(output_folder, f"{image_info['file_name'].split('.')[0]}.txt")
    with open(output_file_path, 'w') as f:
        f.write("\n".join(yolo_annotations))

print(f"YOLO annotations saved to: {output_folder}")