import os
import shutil
import numpy as np
from PIL import Image

def generate_synthetic_data():
    base_dir = "data"
    
    # Same classes as in download_data.py
    classes = [
        "Apple___Apple_scab",
        "Apple___healthy", 
        "Tomato___Early_blight",
        "Tomato___healthy",
        "Corn___Common_rust",
        "Corn___healthy"
    ]
    
    print("🎨 Generating synthetic dataset...")
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    for class_name in classes:
        class_path = os.path.join(base_dir, class_name)
        os.makedirs(class_path, exist_ok=True)
        
        print(f"Generating images for {class_name}...")
        
        # Generate 10 dummy images per class
        for i in range(10):
            # Create random RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Save
            img.save(os.path.join(class_path, f"synthetic_{i}.jpg"))
            
    print("\n✅ Synthetic data generation complete!")
    print(f"Created {len(classes) * 10} images across {len(classes)} classes.")

if __name__ == "__main__":
    generate_synthetic_data()
