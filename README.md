# 🐾 Animal Species Classification — Deep Learning Assignment
### CT118-3-3 ODL | Optimisation and Deep Learning | Asia Pacific University (APU)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.13-red?logo=keras)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project applies deep learning to classify **15 animal species** from images using two architectures:

1. **Custom Convolutional Neural Network (CNN)** —> built from scratch
2. **ResNet50 Transfer Learning** —> pretrained on ImageNet, fine-tuned for animal classification

Both models are optimised using **Keras Tuner** (Hyperband for CNN, Bayesian Optimisation for ResNet50) and evaluated on a held-out test set.

---

## 🐘 Animal Classes

| | | | | |
|--|--|--|--|--|
| 🪲 Beetle | 🦋 Butterfly | 🐱 Cat | 🐄 Cow | 🐶 Dog |
| 🐘 Elephant | 🦍 Gorilla | 🦛 Hippo | 🦎 Lizard | 🐒 Monkey |
| 🐭 Mouse | 🐼 Panda | 🕷️ Spider | 🐯 Tiger | 🦓 Zebra |

---

## 📁 Project Structure

```
ODL-Animal-Classification/
│
├── notebooks/
│   ├── A1_Task1_Dataset_Algorithm_Selection.ipynb   # Dataset justification & literature review
│   ├── A1_Task2_EDA_Data_Preparation.ipynb          # EDA, preprocessing & data pipeline
│   ├── A2_Task1_Model_Building.ipynb                # CNN & ResNet50 training
│   └── A2_Task2_Model_Tuning.ipynb                  # Hyperparameter optimisation
│
├── models/
│   ├── best_tuned_cnn.keras                         # Best tuned CNN model
│   └── best_tuned_resnet.keras                      # Best tuned ResNet50 model
│
├── history/
│   ├── history_cnn.pkl                              # CNN training history
│   ├── history_resnet.pkl                           # ResNet50 training history
│   ├── history_tuned_cnn.pkl                        # Tuned CNN history
│   └── history_tuned_resnet.pkl                     # Tuned ResNet50 history
│
├── dataset/                                         # Download from Kaggle (not included)
│   ├── Training Data/
│   │   └── Training Data/
│   │       ├── Beetle/
│   │       ├── Butterfly/
│   │       └── ... (15 classes)
│   ├── Validation Data/
│   │   └── Validation Data/
│   └── Testing Data/
│       └── Testing Data/
│
├── requirements.txt                                 # Python dependencies
├── .gitignore                                       # Files excluded from git
└── README.md                                        # This file
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ODL-Animal-Classification.git
cd ODL-Animal-Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from Kaggle and place in the `dataset/` folder:
```
https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset
```

### 4. Update dataset path
In each notebook, update the path variable:
```python
# Local
DATASET_PATH = r'path/to/your/dataset'

# Kaggle
DATASET_PATH = '/kaggle/input/animal-image-classification-dataset'
```

### 5. Run notebooks in order
```
1. A1_Task1_Dataset_Algorithm_Selection.ipynb
2. A1_Task2_EDA_Data_Preparation.ipynb
3. A2_Task1_Model_Building.ipynb
4. A2_Task2_Model_Tuning.ipynb
```

---

## 🧠 Model Architectures

### Model 1 — Custom CNN
```
Input (224×224×3)
    ↓
Conv Block 1: Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout
    ↓
Conv Block 2: Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout
    ↓
Conv Block 3: Conv2D(128) → BatchNorm → ReLU → Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout
    ↓
Conv Block 4: Conv2D(256) → BatchNorm → ReLU → Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout
    ↓
GlobalAveragePooling2D
    ↓
Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(15, softmax)
```

### Model 2 — ResNet50 Transfer Learning
```
Input (224×224×3)
    ↓
ResNet50 (pretrained ImageNet — Phase 1: frozen, Phase 2: last 30 layers unfrozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu) → BatchNorm → Dropout(0.4)
    ↓
Dense(15, softmax)
```

---

## 📄 License

This project is for academic purposes only.  
Dataset credit: [Utkarsh Saxena on Kaggle](https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset)
