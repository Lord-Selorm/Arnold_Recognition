import torch
import torch.nn as nn
import torchvision.models as models

class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseClassifier, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.base_model = models.efficientnet_b0(pretrained=True)
        
        # Freeze early layers
        for param in self.base_model.features.parameters():
            param.requires_grad = False
            
        # Modify classifier
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)
    
    def unfreeze_layers(self, num_layers=5):
        """Unfreeze last few layers for fine-tuning"""
        layers = list(self.base_model.features.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

def get_model(num_classes=38):
    """Factory function to create model"""
    model = PlantDiseaseClassifier(num_classes)
    return model

if __name__ == "__main__":
    # Test model
    model = get_model()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
