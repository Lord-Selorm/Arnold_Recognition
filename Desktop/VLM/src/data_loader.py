import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

class PlantDiseaseDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    """Data augmentation and preprocessing transforms"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_data_from_directory(data_dir):
    """Load data from directory structure"""
    image_paths = []
    labels = []
    class_names = []
    
    # Get class names from subdirectories
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            class_idx = len(class_names) - 1
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_path, img_name))
                    labels.append(class_idx)
    
    return image_paths, labels, class_names

def create_data_loaders(data_dir, batch_size=32, test_size=0.2, val_size=0.1):
    """Create train, validation, and test data loaders"""
    
    # Load data
    image_paths, labels, class_names = load_data_from_directory(data_dir)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=test_size + val_size, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=42, stratify=y_temp
    )
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(X_train, y_train, train_transform)
    val_dataset = PlantDiseaseDataset(X_val, y_val, val_transform)
    test_dataset = PlantDiseaseDataset(X_test, y_test, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, class_names

if __name__ == "__main__":
    # Test data loading
    data_dir = "../data"
    if os.path.exists(data_dir):
        train_loader, val_loader, test_loader, class_names = create_data_loaders(data_dir)
        print(f"Number of classes: {len(class_names)}")
        print(f"Class names: {class_names}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
    else:
        print(f"Data directory {data_dir} not found.")
