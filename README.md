# 🧠 Skin Disease Detection (HAM10000 + Deep Learning)

This project builds a deep learning model to detect and classify **skin diseases** using the **HAM10000 dataset**, a large collection of dermatoscopic images.

---

## 🚀 Features

* 🧠 Deep learning-based skin lesion classification
* 📊 Multi-class classification of 7 skin disease categories
* 🖼️ Image preprocessing and normalization
* ⚡ GPU acceleration support (CUDA)
* 📈 Evaluation using accuracy and classification metrics

---

## 📂 Dataset

**Dataset used:**
👉 HAM10000 ("Human Against Machine with 10000 training images")

The dataset contains **10,000+ dermatoscopic images** of common pigmented skin lesions.

### 📁 Structure

```id="6l3k5v"
HAM10000/
│── HAM10000_images_part_1/
│── HAM10000_images_part_2/
│── HAM10000_metadata.csv
```

### 📌 Classes

| Label | Description                   |
| ----- | ----------------------------- |
| akiec | Actinic keratoses             |
| bcc   | Basal cell carcinoma          |
| bkl   | Benign keratosis-like lesions |
| df    | Dermatofibroma                |
| nv    | Melanocytic nevi              |
| mel   | Melanoma                      |
| vasc  | Vascular lesions              |

---

## ⚙️ Installation

### 1. Create Environment

```bash id="p7v9xz"
conda create -n skin_detect python=3.10 -y
conda activate skin_detect
```

### 2. Install Dependencies

```bash id="0db23w"
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow
```

---

## 🧠 Model Architecture

* Model: **Convolutional Neural Network (CNN)**
* Designed for multi-class classification

```python id="4k5h2u"
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64*X*X, 128),
    nn.ReLU(),
    nn.Linear(128, 7)
)
```

---

## 🏃 Training

Run training script:

```bash id="e1o7bx"
python train.py
```

### Default Settings

* Batch size: 16
* Epochs: 10–20
* Optimizer: Adam
* Learning rate: 0.001

---

## 📊 Evaluation

* Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

* 
<img width="800" height="600" alt="RAPIDS_CODE_CONFUSION_MATRIX" src="https://github.com/user-attachments/assets/59a6dfb0-9d3f-4062-b775-36e1863decef" />
<img width="640" height="480" alt="ROC_CURVE_RAPIDS" src="https://github.com/user-attachments/assets/5c6aca88-5985-4ba5-81e6-e12cba1bb602" />
<img width="640" height="480" alt="LOSS_CURVE_RAPIDS" src="https://github.com/user-attachments/assets/7a6c92a6-2d79-4f9f-8a32-6fd890b9c3d3" />
<img width="640" height="480" alt="ACCURACY_VS_EPOCH_RAPIDS" src="https://github.com/user-attachments/assets/2a8751e8-c9ac-4eb1-8fad-1df90bef7d84" />

---

## 🧠 Key Learnings

* Medical datasets are often **imbalanced across classes**
* Data augmentation improves performance significantly
* CNNs are effective for dermatological image classification
* Preprocessing plays a crucial role in model accuracy

---

## ⚠️ Hardware Requirements

* GPU recommended (4GB+ VRAM)
* CPU supported (slower training)

---

## 📈 Future Improvements

* 🔥 Transfer learning (EfficientNet / ResNet)
* 🔥 Grad-CAM visualization
* 🔥 Web app deployment (Streamlit)
* 🔥 Dataset balancing techniques

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 📜 License

This project is for educational and research purposes.

