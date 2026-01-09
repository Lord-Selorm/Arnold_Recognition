# Plant Disease Detection System

A computer vision project that uses deep learning to automatically detect plant diseases from leaf images.

## Project Overview

This system helps farmers identify plant diseases early by analyzing images of plant leaves. It uses convolutional neural networks (CNNs) to classify various plant diseases with high accuracy.

## Features

- **Multi-class disease classification** for different plant types
- **Real-time prediction** through web interface
- **Model training** with data augmentation
- **Performance metrics** and visualization
- **Mobile-friendly** interface for field use

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python train.py --data_path data/ --epochs 50 --batch_size 32
```

### Running the Web App
```bash
python app.py
```

## Project Structure
```
├── data/              # Dataset directory
├── models/            # Trained models
├── src/               # Source code
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── app.py             # Flask web application
└── requirements.txt   # Dependencies
```

## Dataset

The system uses the PlantVillage dataset which contains images of healthy and diseased plant leaves across multiple crop types.

## Model Architecture

- **Base Model**: EfficientNet-B0 (pre-trained on ImageNet)
- **Custom Layers**: Fully connected layers for disease classification
- **Input Size**: 224x224 RGB images
- **Output**: Probability scores for each disease class

## Performance

- **Accuracy**: ~95% on test set
- **Inference Time**: <100ms per image
- **Model Size**: ~20MB

## Future Improvements

- Integration with mobile apps
- Real-time camera feed analysis
- Disease severity assessment
- Treatment recommendations
