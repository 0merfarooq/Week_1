#  Garbage Classification using CNN (Deep Learning Project)

---

## Problem Statement

Improper waste management is one of the most persistent environmental challenges faced by modern society. Every year, millions of tons of waste are generated, and a large portion ends up in landfills due to ineffective segregation at the source. Manual waste sorting is not only time-consuming and costly but also exposes workers to unhygienic and hazardous conditions.

To address these challenges, this project focuses on developing an **AI-powered garbage classification system** capable of identifying and categorizing different types of waste using **Convolutional Neural Networks (CNNs)**. The goal is to automate the process of waste segregation based on image inputs, enabling machines to distinguish between recyclable, non-recyclable, and biodegradable materials with high accuracy.

By introducing an automated image-based classification model, we aim to:

- Enhance the **efficiency and accuracy** of waste segregation systems.
- **Reduce human effort** and associated health risks in waste-handling operations.
- **Promote environmental sustainability** by facilitating better recycling and disposal practices.
- Provide a scalable foundation for **smart waste management systems**, such as intelligent dustbins, sorting belts, and recycling plant automation.

In essence, this project demonstrates how artificial intelligence — specifically deep learning through CNNs — can be applied to solve real-world environmental problems by making waste management smarter, faster, and more sustainable.

---

##  Dataset Overview

**Name & Source:**

- Dataset: *Garbage Classification v2* ([Kaggle Link](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data))
- License: **MIT License**

**Dataset Statistics & Composition:**

- Total images: **19,762**
- Number of classes: **10**
- Classes: `battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`
- Image resolution: varies, standardized to 224×224 during preprocessing
- Split: **85% Training**, **15% Validation**
- Dataset integrity: 3 corrupt images detected and removed during preprocessing

**How this dataset suits the project:**\
This dataset provides a balanced representation of real-world garbage images across various materials. It’s ideal for training a CNN model to classify waste effectively for environmental automation tasks.

---

##  Project Architecture

###  Overall Workflow

1. **Data Acquisition**
   - Dataset sourced from Kaggle and extracted locally.
2. **Data Preprocessing**
   - Resized all images to **224×224×3** and normalized pixel values.
   - Split dataset into **train (85%)** and **validation (15%)** sets.
   - Removed 3 corrupt images automatically.
3. **Model Design**
   - CNN architecture based on **EfficientNetB0** (transfer learning).
   - Custom dense layers added for classification across 10 classes.
4. **Training Phase**
   - Trained classifier head for 15 epochs until reaching target accuracy.
   - Early stopping and model checkpoint callbacks used.
5. **Evaluation & Deployment**
   - Best model exported as `.keras` file and label map saved as `.json`.

###  Conceptual Flow Diagram

Dataset  →  Data Cleaning & Augmentation  →  CNN Model (EfficientNetB0)  →  Training  →  Validation  →  Model Export

---

##  CNN Model Architecture

### Overview

The model leverages **EfficientNetB0** as a pre-trained feature extractor, with custom layers added for the classification task.

| Layer Type              | Output Shape  | Parameters | Description                        |
| ----------------------- | ------------- | ---------- | ---------------------------------- |
| Input                   | (224, 224, 3) | 0          | Input layer for RGB images         |
| Data Augmentation       | (224, 224, 3) | 0          | Random flips, rotations, and zooms |
| EfficientNetB0 (Frozen) | (7, 7, 1280)  | 4,049,571  | Pretrained feature extractor       |
| Global Average Pooling  | (1280)        | 0          | Reduces feature maps to vector     |
| BatchNormalization      | (1280)        | 5,120      | Stabilizes training                |
| Dropout(0.3)            | (1280)        | 0          | Prevents overfitting               |
| Dense (256, ReLU)       | (256)         | 327,936    | Fully connected layer              |
| BatchNormalization      | (256)         | 1,024      | Normalization                      |
| Dropout(0.3)            | (256)         | 0          | Regularization                     |
| Dense (128, ReLU)       | (128)         | 32,896     | Hidden dense layer                 |
| Dropout(0.3)            | (128)         | 0          | Regularization                     |
| Dense (10, Softmax)     | (10)          | 1,290      | Output layer for 10 classes        |

**Total parameters:** 4,417,837
**Trainable parameters:** 365,194
**Non-trainable parameters:** 4,052,643

---

## Model Training and Performance

###  Hardware and Setup

- **Python Version:** 3.12.12
- **TensorFlow Version:** 2.19.0
- **GPU:** NVIDIA Tesla (enabled)

###  Training Configuration

- **Train/Validation Split:** 85/15
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 32
- **Learning Rate:** 1e-3 (adaptive)
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

###  Training Progress

- **Phase 1:** Training classifier head for 15 epochs
- **Best Validation Accuracy:** **90.23%**
- **Final Model Evaluation:**
  - Accuracy: **0.9023 (90.23%)**
  - Validation Accuracy: **0.9339 (93.39%)**
  - Loss: 1.1065

### ✅ Training Outputs

🎯 New best: 0.9023 (90.23%)
✅ TARGET REACHED! Val accuracy: 0.9023
🏆 FINAL VALIDATION ACCURACY: 90.23%
✅ MODEL SAVED: /content/exports/garbage_classifier_final.keras
✅ LABEL MAP SAVED: /content/exports/label_map.json

---

## 📂 Repository Structure

📁 Garbage-Classification-CNN/
├── /data/                  # Dataset folder
├── /exports/               # Saved models and label map
│   ├── garbage_classifier_final.keras
│   └── label_map.json
├── /notebooks/             # Colab or Jupyter notebooks
├── model_training.py       # Main training script
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── LICENSE

---

## Installation and Setup

###  Requirements

- Python ≥ 3.10
- TensorFlow ≥ 2.12
- NumPy, Pandas, Matplotlib, Pillow

### Installation

# Clone the repository
git clone https://github.com/<your-username>/garbage-classification-cnn.git
cd garbage-classification-cnn

# Install dependencies
pip install -r requirements.txt

### Usage

# Train model
python model_training.py

# Evaluate trained model
python evaluate_model.py

# Run inference
python predict.py --image path/to/image.jpg

---

## Advanced Usage

- Integrate model into **web or mobile apps** for real-time waste classification.
- Deploy model on **Edge devices (Raspberry Pi, Jetson Nano)**.
- Extend dataset or fine-tune using **EfficientNetV2** or **Vision Transformers (ViT)** for improved performance.

---

## Technical Summary

- **Model Framework:** TensorFlow /Keras
- **Architecture:** EfficientNetB0 + Custom Dense Layers
- **Dataset:** Garbage Classification v2 (Kaggle)
- **Accuracy:** ~90%
- **Purpose:** Automating environmental waste segregation using AI

---

## Conclusion

This project demonstrates the potential of deep learning to tackle sustainability challenges. By leveraging CNNs for garbage classification, we can contribute toward smarter waste management and a cleaner environment.

---

**Author:** Mohammed Omer Farooq
**License:** MIT
