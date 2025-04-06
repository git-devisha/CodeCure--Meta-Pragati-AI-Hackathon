# CodeCure -Meta Pragati AI Hackathon

# Bone Fracture Detection System
## Overview

The Bone Fracture Detection System is an open-source AI-powered application that analyzes X-ray images to detect and classify bone fractures. Built using deep learning and computer vision technologies, this tool assists healthcare professionals by providing preliminary fracture analysis with visual identification and classification.

## Features

### Detection Capabilities
- **Multi-class Classification**: Identifies 7 fracture types (Normal, Hairline, Spiral, Comminuted, Impacted, Segmental, and Oblique)
- **Confidence Scoring**: Provides probability values for each detection
- **Visual Localization**: Highlights fracture locations with bounding boxes
- **Adjustable Threshold**: User-configurable sensitivity control

### User Experience
- **Web-based Interface**: Accessible via Streamlit
- **Bilingual Support**: Complete English and Hindi language options
- **Real-time Analysis**: Immediate results after image upload
- **Educational Content**: Explanations of fracture types and significance
- **Professional Context**: Clear medical disclaimer notices

### Technical Features
- **VGG16 Architecture**: Transfer learning from ImageNet pre-trained weights
- **Optimized Inference**: CPU-friendly with caching mechanisms
- **Flexible Input**: Supports JPG, JPEG, and PNG formats
- **Visualization Engine**: Custom rendering with bounding boxes
- **Resource Efficiency**: Minimized memory footprint for clinical settings

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bone-fracture-detection.git
   cd bone-fracture-detection
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create weights directory:
   ```
   mkdir -p weights
   ```

5. Download pre-trained model weights (if available) and place in the weights directory as `model_vgg.pt`

## Usage

1. Start the application:
   ```
   streamlit run server.py
   ```
2. Open your web browser and navigate to the URL displayed in the terminal
3. Select your preferred language from the sidebar

4. Upload an X-ray image using the file uploader

5. Click "Analyze Image" to process the image

6. View results showing:
   - Detected fracture type (if any)
   - Confidence score
   - Visual indication of fracture location
   - Explanation of the findings

7. Adjust the confidence threshold in the sidebar if needed to fine-tune detection sensitivity

## Model Information

The system utilizes a VGG16 convolutional neural network architecture with the following modifications:
- Pre-trained on ImageNet for general feature extraction capabilities
- Fine-tuned on X-ray images for fracture detection
- Final classification layer replaced with 7 output nodes for fracture types
- Input dimensions: 224×224 pixels, RGB (3 channels)

## Directory Structure

```
bone-fracture-detection/
├── server.py             # Main application file
├── requirements.txt     # Python dependencies
├── weights/             # Model weights directory
│   └── model_vgg.pt     # Trained model weights
├── README.md            # This file
└── data/              # Images and additional resources
```

## Requirements

Main dependencies include:
- streamlit
- torch
- torchvision
- pillow
- numpy
- matplotlib
- warnings

See `requirements.txt` for complete list with versions.

## Disclaimer

This tool is designed as an assistive technology for healthcare professionals and is not intended to replace professional medical diagnosis. All results should be verified by qualified medical personnel before making clinical decisions. The predictions made by this system are preliminary and should always be confirmed through standard diagnostic procedures.

## Development

### Adding New Languages
To add a new language:
1. Update the `translations` dictionary in `app.py`
2. Add translations for all UI elements and descriptions
3. Update the `class_name_mapping` if needed for fracture type translations

### Improving the Model
To retrain or improve the model:
1. Collect additional X-ray images with proper annotations
2. Adjust the VGG16 model configuration as needed
3. Train using the appropriate training script (not included)
4. Replace the model weights file in the weights directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VGG16 architecture from the Visual Geometry Group at Oxford
- PyTorch and Torchvision libraries for deep learning capabilities
- Streamlit for the web application framework
