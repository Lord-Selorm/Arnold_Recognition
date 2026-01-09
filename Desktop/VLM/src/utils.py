import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json

def load_model(model_path, num_classes):
    """Load trained model from checkpoint"""
    try:
        from .model import get_model
    except ImportError:
        from model import get_model
    
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def load_class_names(json_path):
    """Load class names from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def preprocess_image(image_path):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_disease(model, image_tensor, class_names, device='cpu'):
    """Make prediction on preprocessed image"""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        top3_predictions = []
        for i in range(3):
            class_name = class_names[top3_indices[0][i].item()]
            prob = top3_probs[0][i].item() * 100
            top3_predictions.append((class_name, prob))
    
    return predicted_class, confidence_score, top3_predictions

def get_disease_info(disease_name):
    """Get information about the disease and treatment recommendations"""
    disease_info = {
        'Apple___Apple_scab': {
            'description': 'Apple scab is a serious fungal disease caused by Venturia inaequalis. It causes olive-green to black velvety spots on leaves and fruit, which can lead to premature leaf drop and distorted fruit.',
            'treatment': [
                'Remove and destroy infected fallen leaves and fruit to reduce inoculum.',
                'Apply preventive fungicides (e.g., captan, sulfur, or myclobutanil) starting at bud break.',
                'Prune trees to improve air circulation and sunlight penetration.',
                'Consider planting scab-resistant varieties like "Liberty" or "Enterprise".'
            ],
            'severity': 'Moderate'
        },
        'Tomato___Early_blight': {
            'description': 'Early blight, caused by Alternaria solani, typically starts on lower leaves as small brown spots with concentric rings (target-like pattern). It can cause significant defoliation.',
            'treatment': [
                'Immediately remove infected lower leaves to prevent upward spread.',
                'Apply copper-based fungicides or mancozeb every 7-10 days during wet weather.',
                'Mulch around plants to prevent soil-borne spores from splashing onto leaves.',
                'Practice crop rotation (avoid planting tomatoes or potatoes in the same spot for 3 years).'
            ],
            'severity': 'High'
        },
        'Corn___Common_rust': {
            'description': 'Common rust (Puccinia sorghi) is characterized by cinnamon-brown, elongated pustules on both upper and lower leaf surfaces. It thrives in cool, moist conditions.',
            'treatment': [
                'Fungicides like strobilurins or triazoles are effective if applied at the first sign of pustules.',
                'Plant resistant hybrids which are highly effective at controlling this disease.',
                'Avoid overhead irrigation to reduce leaf wetness duration.',
                'Control weed hosts that might harbor the fungus.'
            ],
            'severity': 'Moderate'
        },
        'healthy': {
            'description': 'The plant leaf appears healthy and vibrant. No signs of infection, nutrient deficiency, or pest damage were detected.',
            'treatment': [
                'Maintain regular watering and fertilization schedules.',
                'Continue monitoring leaves regularly for any changes.',
                'Ensure the plant receives adequate sunlight and ventilation.',
                'Practice preventive hygiene by cleaning gardening tools.'
            ],
            'severity': 'None'
        }
    }
    
    # Default info if disease not found
    default_info = {
        'description': f'Analysis detected potential signs of {disease_name}. A more detailed visual inspection is recommended.',
        'treatment': [
            'Monitor the plant closely for progression of symptoms.',
            'Isolate the plant if possible to prevent potential spread.',
            'Consult with a local agricultural extension office or professional agronomist.',
            'Maintain optimal growing conditions to support plant immunity.'
        ],
        'severity': 'Unknown'
    }
    
    # Try exact match first
    if disease_name in disease_info:
        return disease_info[disease_name]

    # Try to find matching disease (case-insensitive)
    for key, info in disease_info.items():
        if key.lower() in disease_name.lower():
            return info
    
    return default_info

def format_prediction_results(predicted_class, confidence, top3_predictions):
    """Format prediction results for display"""
    disease_info = get_disease_info(predicted_class)
    
    results = {
        'prediction': predicted_class,
        'confidence': round(confidence, 2),
        'severity': disease_info['severity'],
        'description': disease_info['description'],
        'treatment': disease_info['treatment'],
        'top3': [{'class': pred[0], 'confidence': round(pred[1], 2)} for pred in top3_predictions]
    }
    
    return results

if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test model loading (if model exists)
    try:
        model = load_model('../models/best_model.pth', 6)
        print("Model loaded successfully!")
    except:
        print("Model not found. Train the model first.")
    
    # Test class names loading (if file exists)
    try:
        class_names = load_class_names('../models/class_names.json')
        print(f"Loaded {len(class_names)} class names")
    except:
        print("Class names file not found.")
