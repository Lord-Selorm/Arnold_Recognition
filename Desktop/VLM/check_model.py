import torch

model_path = 'models/best_model.pth'
try:
    state_dict = torch.load(model_path, map_location='cpu')
    # Look for the last linear layer's bias or weight to find num_classes
    # In EfficientNet-B0 classifier, it's usually the last layer.
    # Based on src/model.py: self.base_model.classifier[4] (index 4 of the Sequential)
    classifier_weight_key = 'base_model.classifier.4.weight'
    if classifier_weight_key in state_dict:
        num_classes = state_dict[classifier_weight_key].shape[0]
        print(f"Model num_classes: {num_classes}")
    else:
        # Try to find any key ending in .bias or .weight that looks like the final layer
        for key in reversed(list(state_dict.keys())):
            if 'weight' in key:
                num_classes = state_dict[key].shape[0]
                print(f"Likely num_classes from {key}: {num_classes}")
                break
except Exception as e:
    print(f"Error: {e}")
