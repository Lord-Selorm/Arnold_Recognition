import os
import shutil

mapping = {
    "Apple___Apple_scab": "C:/Users/DEVICES/.gemini/antigravity/brain/db9f08f4-2abf-4f5b-b03e-93b730db9b23/apple_scab_1_1767875339226.png",
    "Apple___healthy": "C:/Users/DEVICES/.gemini/antigravity/brain/db9f08f4-2abf-4f5b-b03e-93b730db9b23/apple_healthy_1_1767875359959.png",
    "Tomato___Early_blight": "C:/Users/DEVICES/.gemini/antigravity/brain/db9f08f4-2abf-4f5b-b03e-93b730db9b23/tomato_early_blight_1_1767875375369.png",
    "Tomato___healthy": "C:/Users/DEVICES/.gemini/antigravity/brain/db9f08f4-2abf-4f5b-b03e-93b730db9b23/tomato_healthy_1_1767875392736.png",
    "Corn___Common_rust": "C:/Users/DEVICES/.gemini/antigravity/brain/db9f08f4-2abf-4f5b-b03e-93b730db9b23/corn_rust_1_1767875409670.png",
    "Corn___healthy": "C:/Users/DEVICES/.gemini/antigravity/brain/db9f08f4-2abf-4f5b-b03e-93b730db9b23/corn_healthy_1_1767875427025.png"
}

data_dir = 'data'

for class_name, src_path in mapping.items():
    class_path = os.path.join(data_dir, class_name)
    # Remove old noise images
    if os.path.exists(class_path):
        for f in os.listdir(class_path):
            os.remove(os.path.join(class_path, f))
    else:
        os.makedirs(class_path)
    
    # Copy the real image 10 times to have a "dataset"
    for i in range(10):
        dest_path = os.path.join(class_path, f"real_{i}.png")
        shutil.copy(src_path, dest_path)

print("Status: Distributed real images to data directories.")
