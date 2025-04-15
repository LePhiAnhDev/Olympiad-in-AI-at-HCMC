# 🍄 Mushroom Classification with Vision Transformers (ViT)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project for accurately classifying mushroom species using Vision Transformers (ViT). This model leverages state-of-the-art attention mechanisms to achieve high accuracy in mushroom classification.

<div align="center">
  <img src="https://raw.githubusercontent.com/user/mushroom-classification/main/assets/mushroom_banner.jpg" alt="Mushroom Classification" width="80%">
</div>

## 📋 Table of Contents

- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Directory Structure](#-directory-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Advanced Usage](#-advanced-usage)
- [Implementation Details](#-implementation-details)
- [Training Process](#-training-process)
- [Data Augmentation Techniques](#-data-augmentation-techniques)
- [Performance Metrics](#-performance-metrics)
- [License](#-license)

## ✨ Features

- **Vision Transformer (ViT) Architecture**: Implements the state-of-the-art attention-based image classification
- **Multiple Model Sizes**: Configurable model complexity (tiny, small, base, large)
- **Advanced Data Augmentation**: Enhances training data with various transformations
  - Basic (rotation, flipping, brightness/contrast adjustments)
  - Advanced (MixUp, CutMix, elastic transformations)
- **Cross-Validation**: K-fold cross-validation for robust model evaluation
- **Performance Optimization**:
  - Mixed precision training
  - Early stopping and learning rate scheduling
  - Test-time augmentation (TTA)
- **Model Ensemble**: Support for ensemble predictions
- **Comprehensive Evaluation**: Detailed metrics and performance analysis

## 🧠 Model Architecture

The implemented Vision Transformer (ViT) architecture processes images by:

1. **Patch Extraction**: Divides input images into fixed-size patches
2. **Patch Encoding**: Maps patches to token embeddings with positional information
3. **Transformer Encoder**: Processes tokens through multiple transformer blocks with:
   - Multi-head self-attention mechanism
   - Layer normalization
   - MLP blocks with dropout
4. **Classification Head**: Final layers for mushroom species prediction

<div align="center">
  <img src="https://raw.githubusercontent.com/user/mushroom-classification/main/assets/vit_architecture.png" alt="Vision Transformer Architecture" width="80%">
</div>

### Model Size Configurations

| Size | Patch Size | Heads | Transformer Layers | Projection Dim | MLP Units | Dropout Rate |
|------|------------|-------|-------------------|----------------|-----------|--------------|
| Tiny | 8          | 4     | 4                 | 64             | [128, 64] | 0.1          |
| Small| 4          | 6     | 5                 | 128            | [256, 128]| 0.1          |
| Base | 4          | 8     | 6                 | 48             | [96, 48]  | 0.1          |
| Large| 4          | 12    | 8                 | 256            | [512, 256]| 0.2          |

## 📂 Directory Structure

```
mushroom-classification/
├── main.py                  # Main script for training and evaluation
├── mark_labeled.py          # Script to create labeled dataset CSV
├── requirements.txt         # Python dependencies
├── train/                   # Training data directory
│   ├── bao ngu xam trang/   # Oyster mushroom images (Class 1)
│   ├── dui ga baby/         # Chicken leg mushroom images (Class 2)
│   ├── linh chi trang/      # White lingzhi mushroom images (Class 3)
│   └── nam mo/              # Wood ear mushroom images (Class 0)
├── test/                    # Test data directory
│   └── *.jpg                # Test images
└── csv/                     # Generated CSV files
    └── mushroom_labels.csv  # Labels mapping file
```

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/user/mushroom-classification.git
   cd mushroom-classification
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare your dataset:
   - Place training images in the `train` directory (create subdirectories for each mushroom type)
   - Place test images in the `test` directory

## 🚀 Quick Start

### Step 1: Generate Labels File

First, generate the label mapping CSV file:

```bash
python mark_labeled.py
```

This script scans the `train` directory structure and creates a CSV file mapping image IDs to their respective class labels.

### Step 2: Train the Model

Run the main script with default parameters:

```bash
python main.py
```

### Step 3: View Results

After training completes, the following files will be generated:
- `config.json`: Configuration parameters used for training
- `mushroom_model_best.h5`: The trained model weights
- `mushroom_predictions.csv`: Predictions for the test dataset

## 🔧 Advanced Usage

The training script supports various command-line arguments:

```bash
python main.py --model_size base --batch_size 32 --epochs 50 --img_size 32 --learning_rate 1e-4
```

### Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--seed` | Random seed for reproducibility | 42 |
| `--img_size` | Image size for processing | 32 |
| `--batch_size` | Batch size for training | 32 |
| `--epochs` | Number of training epochs | 50 |
| `--learning_rate` | Initial learning rate | 1e-4 |
| `--val_split` | Validation split ratio | 0.2 |
| `--model_size` | Model size (tiny/small/base/large) | base |
| `--use_mixed_precision` | Enable mixed precision training | False |
| `--k_fold` | Number of folds for cross-validation (0 to disable) | 0 |
| `--augment` | Enable data augmentation | True |
| `--tta` | Use test-time augmentation | False |
| `--ensemble` | Number of models for ensemble | 1 |

### Examples

#### Training with Cross-Validation

```bash
python main.py --k_fold 5 --model_size small
```

#### Using Test-Time Augmentation

```bash
python main.py --tta --img_size 64
```

#### Training with Mixed Precision

```bash
python main.py --use_mixed_precision --model_size large
```

## 💻 Implementation Details

### Mushroom Classes

The model classifies mushrooms into 4 types:

| Class ID | Vietnamese Name | English Name |
|----------|-----------------|--------------|
| 0 | Nấm mỡ | Wood ear mushroom |
| 1 | Nấm bào ngư xám trắng | Grey oyster mushroom |
| 2 | Nấm đùi gà baby | Baby king oyster mushroom |
| 3 | Nấm linh chi trắng | White lingzhi mushroom |

### Core Components

1. **Data Processing**:
   - `load_labels()`: Loads the CSV mapping file
   - `preprocess_image()`: Resizes and normalizes images
   - `load_and_augment_data_enhanced()`: Creates the augmented dataset

2. **Model Architecture**:
   - `Patches`: Custom layer for image patch extraction
   - `PatchEncoder`: Adds positional embeddings to patches
   - `build_vit_model()`: Constructs the Vision Transformer model

3. **Training Process**:
   - `train_model()`: Handles the main training loop with callbacks
   - `train_kfold()`: Implements k-fold cross-validation
   - `evaluate_model()`: Calculates comprehensive metrics

4. **Prediction**:
   - `predict_test_dataset()`: Generates predictions for test images
   - `apply_test_time_augmentation()`: Improves predictions with multiple augmentations

## 🏋️ Training Process

The training process follows these steps:

1. **Data Preparation**:
   - Load images from the training directory
   - Preprocess to consistent size and normalize
   - Apply augmentation techniques if enabled

2. **Model Configuration**:
   - Configure Vision Transformer parameters based on selected size
   - Set up optimizers and learning rate schedules

3. **Training Loop**:
   - Train with early stopping to prevent overfitting
   - Reduce learning rate on plateau to improve convergence
   - Save the best performing model based on validation accuracy

4. **Evaluation**:
   - Calculate comprehensive metrics (accuracy, precision, recall, F1-score)
   - Generate confusion matrix to analyze class performance

5. **Prediction**:
   - Apply the trained model to test images
   - Use test-time augmentation if enabled
   - Save predictions to CSV format

## 🔄 Data Augmentation Techniques

### Basic Augmentations

- **Geometric Transformations**:
  - 90°, 180°, 270° rotations
  - Horizontal and vertical flips
  - Random zoom and cropping

- **Color and Intensity Adjustments**:
  - Brightness enhancement/reduction
  - Contrast adjustment
  - Random noise addition

### Advanced Augmentations

- **MixUp**: Blends two images with a random weight
- **CutMix**: Replaces a random patch from one image with another
- **Elastic Transformations**: Applies non-rigid deformations to simulate natural variations

## 📊 Performance Metrics

The model's performance is evaluated using:

- **Accuracy**: Overall correct predictions
- **Precision**: Ability to avoid false positives
- **Recall**: Ability to find all positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions by class

## 📑 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <p>Created with ❤️ for mushroom classification research</p>
</div>
