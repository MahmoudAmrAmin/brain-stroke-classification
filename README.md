<div align="center">

# 🧠 Brain Stroke Classification

**Deep learning–based binary classification of brain CT/MRI scans (Stroke vs. Normal) using transfer learning.**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DenseNet121-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow.svg)]()

[Overview](#-overview) •
[Dataset](#-dataset) •
[Methodology](#-methodology) •
[Results](#-results) •
[Roadmap](#-roadmap) •
[Getting Started](#-getting-started)

</div>

---

## 📌 Overview

**Brain Stroke Classification** is a computer vision project that classifies brain scan images into **Stroke** or **Normal** categories using deep convolutional neural networks and transfer learning.

Early and accurate stroke detection is critical — stroke is one of the leading causes of death and long-term disability worldwide, and timely diagnosis directly impacts patient outcomes. This project explores how modern CNN architectures, originally trained on ImageNet, can be adapted to the medical imaging domain to support faster, assistive screening.

The project is being developed **iteratively**: it starts with a DenseNet121 baseline and is designed to expand into a full benchmark comparing multiple architectures, with a strong emphasis on rigorous evaluation (not just accuracy, but metrics that actually matter in a clinical context — recall, specificity, and AUC).

> ⚠️ **Disclaimer:** This project is for research and educational purposes only. It is **not** a certified medical diagnostic tool and should not be used for real clinical decision-making.

---

## 🗂 Dataset

- **Classes:** `Normal`, `Stroke` (binary classification)
- **Structure:** Pre-split into `Train` / `Test` directories, organized in `ImageFolder` format
- **Preprocessing:** Images resized to `224×224`, converted to 3-channel format for compatibility with ImageNet-pretrained backbones
- **Split strategy:** Stratified train/validation split (80/20) to preserve class balance across sets

```
data/
├── Train/
│   ├── Normal/
│   └── Stroke/
└── Test/
    ├── Normal/
    └── Stroke/
```

---

## ⚙️ Methodology

| Component | Details |
|---|---|
| **Baseline architecture** | DenseNet121 (ImageNet pretrained) |
| **Fine-tuning strategy** | Frozen convolutional backbone + custom fully connected classifier head |
| **Classifier head** | `1024 → 256 → 64 → 1` with ReLU activations + Sigmoid output |
| **Loss function** | Binary Cross-Entropy (BCE) |
| **Optimizer** | Adam |
| **LR schedule** | ReduceLROnPlateau |
| **Regularization** | Early stopping + best-checkpoint saving on validation loss |
| **Evaluation metrics** | Accuracy, Precision, Recall, Specificity, Sensitivity, Confusion Matrix, Classification Report |

### Project Structure

```
brain-stroke-classification/
├── data/                   # Train/Test image data (not tracked in git)
├── models/                 # Saved checkpoints, training curves
├── src/
│   └── notebook.ipynb      # Main experimentation notebook
├── requirements.txt
└── README.md
```

---

## 📊 Results

Current baseline results (DenseNet121, first iteration):

| Metric | Normal | Stroke |
|---|---|---|
| Precision | 0.53 | 0.86 |
| Recall | 0.98 | 0.14 |
| F1-score | 0.69 | 0.25 |

**Overall Test Accuracy:** ~56%

### 🔍 Key Finding

Baseline experimentation surfaced a **train/validation preprocessing inconsistency** (mismatched normalization between training and evaluation transforms), which is the primary driver behind the current recall gap on the `Stroke` class. This has been identified and is the top priority fix in the roadmap below — it's a great example of why rigorous evaluation matters as much as model choice in medical imaging pipelines.

Training/validation loss curves and confusion matrices are available in `models/`.

---

## 🛣 Roadmap

This project is under **active iteration**. Planned improvements, in priority order:

- [ ] **Fix train/eval preprocessing consistency** — align normalization across train, validation, and test transforms
- [ ] **Data augmentation** — add rotation, flipping, and zoom to the training pipeline to reduce overfitting
- [ ] **Regularization** — introduce Dropout in the classifier head; explore progressive unfreezing of backbone layers
- [ ] **Threshold & class-imbalance tuning** — evaluate class-weighted loss and optimal decision threshold via ROC/AUC analysis
- [ ] **Architecture benchmarking** — compare DenseNet121 against DenseNet169/201, ResNet50, and VGG16 under identical conditions
- [ ] **Explainability** — add Grad-CAM visualizations to highlight regions driving model predictions
- [ ] **Model serving** — expose the best model via a FastAPI inference endpoint
- [ ] **Interactive demo** — build a Streamlit app for uploading a scan and viewing predictions + Grad-CAM overlay
- [ ] **Experiment tracking** — integrate MLflow or Weights & Biases for systematic run comparison

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.10
```

### Installation

```bash
git clone https://github.com/MahmoudAmrAmin/brain-stroke-classification.git
cd brain-stroke-classification
pip install -r requirements.txt
```

### Usage

1. Place your dataset under `data/Train/` and `data/Test/`, split into `Normal/` and `Stroke/` subfolders.
2. Open and run `src/notebook.ipynb` to train and evaluate the model.
3. Trained checkpoints and evaluation plots are saved automatically to `models/`.

---

## 🧰 Tech Stack

`Python` · `PyTorch` · `torchvision` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn`

---

## 🤝 Contributing

Contributions, issue reports, and suggestions are welcome. Feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Mahmoud Amr Amin**
AI / ML Engineer — NLP 
[GitHub](https://github.com/MahmoudAmrAmin)