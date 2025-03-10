import json
import os

# Define the category mapping
category_mapping = {
    # category_1: Quadrants
    "quadrant_1": 0,
    "quadrant_2": 1,
    "quadrant_3": 2,
    "quadrant_4": 3,
    # category_2: Teeth
    "tooth_1": 0,
    "tooth_2": 1,
    "tooth_3": 2,
    "tooth_4": 3,
    "tooth_5": 4,
    "tooth_6": 5,
    "tooth_7": 6,
    "tooth_8": 7,
    # category_3: Dental Issues
    "Impacted": 0,
    "Caries": 1,
    "Periapical Lesion": 2,
    "Deep Caries": 3,
}

def coco_to_yolo(coco_json_path, output_dir, category_mapping):
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check which category levels exist in the COCO data
    # Handle both 'categories' and 'categories_1' for the first dataset
    has_categories_1 = 'categories_1' in coco_data or 'categories' in coco_data
    has_categories_2 = 'categories_2' in coco_data
    has_categories_3 = 'categories_3' in coco_data

    # Process each image
    for img_info in coco_data['images']:
        img_id = img_info['id']
        anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        yolo_lines = []
        for ann in anns:
            # Get category IDs for each level (if they exist)
            # Handle both 'category_id' and 'category_id_1' for datasets with only category_1
            category_id_1 = ann.get('category_id_1') or ann.get('category_id')
            category_id_2 = ann.get('category_id_2')
            category_id_3 = ann.get('category_id_3')

            # Map category IDs to names (if the corresponding categories exist)
            category_name_1 = None
            category_name_2 = None
            category_name_3 = None

            if has_categories_1 and category_id_1 is not None:
                # Use 'categories_1' if it exists, otherwise use 'categories'
                categories_1 = coco_data.get('categories_1', coco_data.get('categories', []))
                category_name_1 = next((cat['name'] for cat in categories_1 if cat['id'] == category_id_1), None)
                if category_name_1 is not None:
                    category_name_1 = f"quadrant_{category_name_1}"  # Prefix for quadrants

            if has_categories_2 and category_id_2 is not None:
                category_name_2 = next((cat['name'] for cat in coco_data['categories_2'] if cat['id'] == category_id_2), None)
                if category_name_2 is not None:
                    category_name_2 = f"tooth_{category_name_2}"  # Prefix for teeth

            if has_categories_3 and category_id_3 is not None:
                category_name_3 = next((cat['name'] for cat in coco_data['categories_3'] if cat['id'] == category_id_3), None)

            # Combine category names (if needed) or use the most specific one
            category_name = category_name_3 or category_name_2 or category_name_1

            if not category_name:
                print(f"Warning: No valid category found for annotation {ann['id']}. Skipping.")
                continue

            # Map category name to YOLO class ID
            class_id = category_mapping.get(str(category_name), -1)  # Ensure category_name is a string

            if class_id == -1:
                print(f"Warning: Category '{category_name}' not found in mapping. Skipping annotation.")
                continue

            # Convert COCO bbox [x_min, y_min, width, height] to YOLO format [x_center, y_center, width, height]
            x_min, y_min, width, height = ann['bbox']
            x_center = (x_min + width / 2) / img_info['width']
            y_center = (y_min + height / 2) / img_info['height']
            width_norm = width / img_info['width']
            height_norm = height / img_info['height']

            yolo_lines.append(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}")

        # Save YOLO annotation file
        image_file_name = img_info['file_name']
        output_file_name = os.path.splitext(image_file_name)[0] + ".txt"  # Replace extension with .txt
        output_path = os.path.join(output_dir, output_file_name)
        with open(output_path, 'w') as f:
            f.write("\n".join(yolo_lines))

# Example usage
coco_json_path = r"C:\Users\destr\Desktop\Study\data\training_data\quadrant_enumeration_disease\train_quadrant_enumeration_disease.json"
output_dir = r"C:\Users\destr\Desktop\Study\data\training_data\quadrant_enumeration_disease\Yolo_New"
coco_to_yolo(coco_json_path, output_dir, category_mapping)