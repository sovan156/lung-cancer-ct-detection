# Lung Cancer CT Detection

An AI-powered lung cancer detection system that analyzes CT scan images and classifies them as `normal` or `cancer` using a deep learning model.

## Overview

This project is built to support early lung cancer detection using medical image processing and deep learning. It includes dataset preparation, model training, inference, Grad-CAM explainability, and a Streamlit-based application for interactive predictions.

## Project Summary

Lung cancer remains one of the leading causes of cancer-related deaths. Early and accurate detection from CT scans can improve patient outcomes. This repository uses Transfer Learning with MobileNetV2 to classify CT images into:

- `normal`
- `cancer`

## Features

- CT image preprocessing and dataset setup
- Transfer learning model training with TensorFlow / Keras
- Model checkpointing and training history visualization
- Single-image prediction and class probabilities
- Grad-CAM visualization for explainability
- Streamlit app for interactive inference
- PDF report generation

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model training and inference |
| OpenCV | Image preprocessing and augmentation |
| NumPy | Numerical operations |
| Pandas | Data handling and metadata management |
| Matplotlib | Training plots and charts |
| Seaborn | Confusion matrix visualization |
| Plotly | Interactive charts in Streamlit |
| Streamlit | Web application interface |
| Pillow | Image loading and processing |
| ReportLab | PDF report generation |
| Joblib | Model metadata persistence |
| scikit-learn | Evaluation metrics and reporting |
| tqdm | Progress bars during preprocessing and training |

## Project Structure

```
lung-cancer-ct-detection/
├── app.py
├── gradcam.py
├── predict.py
├── preprocessing.py
├── report_generator.py
├── requirements.txt
├── setup_dataset.py
├── train_model.py
├── README.md
├── model/
│   └── best_model.h5
├── test/
│   ├── cancer/
│   └── normal/
└── dataset/ (created by setup_dataset.py)
    ├── train/
    │   ├── cancer/
    │   └── normal/
    └── test/
        ├── cancer/
        └── normal/
```

> Note: `dataset/` is created and populated by `setup_dataset.py` from the training images.

## Installation

1. Clone this repository.
2. Create and activate a Python environment.
3. Install dependencies.

```bash
git clone <repository-url>
cd lung-cancer-ct-detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Prepare the dataset

```bash
python setup_dataset.py
```

### Train the model

```bash
python train_model.py
```

### Run the Streamlit application

```bash
streamlit run app.py
```

### Predict a single image

```bash
python predict.py --image path/to/image.png
```

### Generate Grad-CAM visualization

```bash
python gradcam.py --image path/to/image.png
```

## Dataset

The project uses CT scan images organized by class. The model expects the following directory layout after dataset setup:

- `dataset/train/cancer/`
- `dataset/train/normal/`
- `dataset/test/cancer/`
- `dataset/test/normal/`

The included `test/` folder can be used for quick evaluation of sample images.

## Model

The trained model is saved at:

- `model/best_model.h5`

The model is built using Transfer Learning with MobileNetV2 and fine-tuned for binary lung-cancer classification.

## Evaluation

Model performance is tracked with:

- Accuracy
- Classification report
- Confusion matrix
- Training / validation loss and accuracy plots

## License

This project is intended for educational and research purposes only. Add your preferred license here, for example `MIT License`.
