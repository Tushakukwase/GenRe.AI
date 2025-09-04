# 🔍 Real vs AI Image Detector

A deep learning project that distinguishes between real photographs and AI-generated images using Transfer Learning with EfficientNetB0.

## 🎯 Project Overview

This project implements a binary classification model that can detect whether an uploaded image is:
- **Real Image**: A photograph taken with a camera
- **AI-Generated Image**: An image created by AI models (DALL-E, Midjourney, Stable Diffusion, etc.)

The model uses **Transfer Learning** with **EfficientNetB0** as the base architecture, enhanced with custom dense layers for optimal performance.

## 🏗️ Architecture

- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Custom Layers**: Global Average Pooling + Dense layers with Dropout
- **Input Size**: 224x224 pixels (RGB)
- **Output**: Binary classification (Real vs AI-Generated)
- **Framework**: TensorFlow/Keras

## 📁 Project Structure

```
real-vs-ai-detector/
├── dataset/
│   ├── real/          # Real images for training
│   └── ai/            # AI-generated images for training
├── model/
│   └── model.h5       # Trained model (generated after training)
├── app.py             # Streamlit web application
├── train.py           # Training script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Sufficient disk space for dataset and model

### Step 1: Clone or Download

Download this project to your local machine.

### Step 2: Install Dependencies

```bash
cd real-vs-ai-detector
pip install -r requirements.txt
```

### Step 3: Prepare Dataset

1. Create the dataset structure:
   ```bash
   mkdir -p dataset/real dataset/ai
   ```

2. Add your images:
   - Place **real photographs** in `dataset/real/`
   - Place **AI-generated images** in `dataset/ai/`
   
   **Supported formats**: PNG, JPG, JPEG, BMP, TIFF

3. **Important**: Ensure you have a balanced dataset with sufficient images in both classes for better training results.

## 🎓 Training the Model

### Step 1: Dataset Preparation

Make sure you have images in both `dataset/real/` and `dataset/ai/` folders.

### Step 2: Run Training

```bash
# Basic training with auto model selection
python train.py

# Specify a specific model
python train.py --model efficientnet
python train.py --model resnet
python train.py --model mobilenet
python train.py --model custom

# Customize training parameters
python train.py --model resnet --epochs 100 --batch-size 16
```

The training script will:
- Load and preprocess your dataset
- Build the EfficientNetB0 model with custom layers
- Train the model using transfer learning
- Apply fine-tuning for better performance
- Save the trained model as `model/model.h5`
- Generate training metrics and visualizations

### Training Features

- **Data Augmentation**: Rotation, zoom, shift, flip for better generalization
- **Transfer Learning**: Uses pre-trained EfficientNetB0 weights
- **Fine-tuning**: Unfreezes last layers for domain-specific learning
- **Early Stopping**: Prevents overfitting
- **Learning Rate Reduction**: Adaptive learning rate scheduling
- **Validation Split**: 20% of data used for validation

### Expected Output

After successful training, you'll see:
- Training progress and metrics
- Confusion matrix visualization
- Training history plots
- Model saved to `model/model.h5`

## 🌐 Running the Web Application

### Step 1: Ensure Model is Trained

Make sure you have `model/model.h5` file (generated after training).

### Step 2: Launch Streamlit App

```bash
streamlit run app.py
```

### Step 3: Access the Application

Open your web browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`).

## 📱 Using the Web App

1. **Upload Image**: Click "Browse files" to select an image
2. **View Results**: See the prediction (Real vs AI-Generated) with confidence score
3. **Analysis**: View detailed analysis including:
   - Image details (size, format, mode)
   - Prediction confidence
   - Probability distribution
   - Confidence interpretation

## 🔧 Customization

### Model Parameters

You can modify training parameters in `train.py`:

```python
# Image size
img_size = (224, 224)

# Batch size
batch_size = 32

# Training epochs
epochs = 50

# Learning rates
initial_lr = 0.001
fine_tune_lr = 1e-5
```

### Data Augmentation

Adjust augmentation parameters in the `ImageDataGenerator`:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # Rotation range in degrees
    width_shift_range=0.2,  # Width shift range
    height_shift_range=0.2, # Height shift range
    shear_range=0.2,        # Shear range
    zoom_range=0.2,         # Zoom range
    horizontal_flip=True,   # Horizontal flip
    fill_mode='nearest'
)
```

## 📊 Performance Metrics

The model provides comprehensive evaluation metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate for AI-generated images
- **Recall**: Sensitivity for detecting AI-generated images
- **Confusion Matrix**: Detailed classification results
- **Training History**: Loss and accuracy curves

## 🐛 Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Solution: Train the model first using `python train.py`

2. **"Dataset is empty" error**
   - Solution: Ensure you have images in both `dataset/real/` and `dataset/ai/` folders

3. **Memory issues during training**
   - Solution: Reduce batch size in `train.py`

4. **CUDA/GPU errors**
   - Solution: Install CPU-only TensorFlow: `pip install tensorflow-cpu`

5. **Import errors**
   - Solution: Ensure all dependencies are installed: `pip install -r requirements.txt`

6. **Model loading errors (shape mismatch, etc.)**
   - Solution: Use alternative models or custom CNN:
     ```bash
     python train.py --model resnet      # Use ResNet50
     python train.py --model mobilenet   # Use MobileNetV2
     python train.py --model custom      # Use custom CNN from scratch
     ```
   - The system will automatically try different models in AUTO mode

### Performance Tips

- Use GPU if available for faster training
- Ensure balanced dataset (similar number of images in each class)
- Use high-quality images for better results
- Consider increasing training epochs for better accuracy

## 🔬 Technical Details

### Model Architecture

```
Input (224x224x3)
    ↓
EfficientNetB0 (frozen)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Dense (512, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (256, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (1, Sigmoid)
    ↓
Output (Binary)
```

### Training Strategy

1. **Phase 1**: Train with frozen EfficientNetB0 base
2. **Phase 2**: Fine-tune last 30 layers with lower learning rate
3. **Regularization**: Dropout layers prevent overfitting
4. **Optimization**: Adam optimizer with adaptive learning rate

## 📈 Future Enhancements

- [ ] Support for video analysis
- [ ] Real-time webcam detection
- [ ] Batch image processing
- [ ] API endpoint for integration
- [ ] Mobile app development
- [ ] Additional AI generation models detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- TensorFlow/Keras team for the excellent deep learning framework
- Streamlit for the beautiful web app framework
- EfficientNet paper authors for the base architecture
- Open source community for various libraries and tools

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the error messages in the terminal
3. Ensure all dependencies are correctly installed
4. Verify your dataset structure

---

**Happy Detecting! 🎉**

*Built with ❤️ using TensorFlow, Keras, and Streamlit*
#   G e n R e . A I  
 