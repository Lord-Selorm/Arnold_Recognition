import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import argparse
from tqdm import tqdm
import json

from model import get_model
from data_loader import create_data_loaders

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, class_names, device, lr=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        
        # Label Smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        print("Unfreezing backbone for fine-tuning...")
        for param in self.model.parameters():
            param.requires_grad = True
        # Lower learning rate for fine-tuning
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader, desc="Training")):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs):
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Optional: Unfreeze at middle of training
            if epoch == epochs // 3:
                self.unfreeze_backbone()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), '../models/best_model.pth')
                print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
        
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Classification report
        report = classification_report(all_labels, all_preds, target_names=self.class_names)
        print("\nClassification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Clinical Confusion Matrix - PlantGuard Pro')
        plt.ylabel('Ground Truth')
        plt.xlabel('AI Diagnosis')
        plt.tight_layout()
        plt.savefig('../models/confusion_matrix_premium.png')
        plt.show()
        
        return report, cm
    
    def plot_training_history(self):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='#1b4d3e')
        plt.plot(self.val_losses, label='Validation Loss', color='#d4af37')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', color='#1b4d3e')
        plt.plot(self.val_accuracies, label='Validation Accuracy', color='#d4af37')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../models/training_history_premium.png')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model')
    parser.add_argument('--data_path', type=str, default='../data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        args.data_path, args.batch_size
    )
    
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    # Create model
    model = get_model(num_classes=len(class_names))
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, class_names, device, lr=args.lr)
    
    # Train model
    print("Starting specialized training...")
    trainer.train(args.epochs)
    
    # Evaluate model
    print("Performing final evaluation...")
    trainer.evaluate()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save class names
    with open('../models/class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    print("Advanced training completed for PlantGuard Pro!")

if __name__ == "__main__":
    main()
