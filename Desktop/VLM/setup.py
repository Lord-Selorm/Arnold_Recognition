"""
Setup script for Plant Disease Detection System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def download_dataset():
    """Download PlantVillage dataset (placeholder)"""
    print("\nDataset Setup:")
    print("To train the model, you need to download the PlantVillage dataset:")
    print("1. Visit: https://github.com/spMohanty/PlantVillage-Dataset")
    print("2. Download the dataset")
    print("3. Extract images to the 'data/' directory")
    print("4. Organize images by class in subdirectories")
    
    # Create data directory structure
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    print("✓ Data directories created")

def main():
    print("🌱 Plant Disease Detection System Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Dataset setup
    download_dataset()
    
    print("\n✓ Setup completed!")
    print("\nNext steps:")
    print("1. Download and organize the dataset")
    print("2. Run training: python src/train.py --data_path data/ --epochs 50")
    print("3. Start web app: python app.py")
    print("4. Open browser to: http://localhost:5000")

if __name__ == "__main__":
    main()
