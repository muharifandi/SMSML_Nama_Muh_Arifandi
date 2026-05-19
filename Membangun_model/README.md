# MLFlow Image Classification

Deep Learning image classification project using TensorFlow, MLflow, and DagsHub for experiment tracking and model management.

---

# Project Overview

This project implements an image classification model using Convolutional Neural Network (CNN) architecture with TensorFlow/Keras.

The project also integrates:

- MLflow Tracking
- MLflow Artifact Logging
- Hyperparameter Tuning
- DagsHub Online Tracking
- Manual Logging for Advanced Criteria

Dataset used:

- Intel Image Classification Dataset

---

# Project Structure

```text
Membangun_model/
│
├── intel_image_preprocessing/
│   ├── train/
│   ├── val/
│   └── test/
│
├── model/
│   ├── MLmodel
│   └── model.keras
│
├── modelling.py
├── modelling_tuning.py
│
├── classification_report.txt
├── confusion_matrix.png
├── training_history.png
│
├── screenshoot_dashboard.jpeg
├── screenshoot_model_artifak.jpeg
├── screenshoot_dashboard_dagshub.jpeg
├── screenshoot_model_artifak_dagshub.jpeg
│
├── requirements.txt
├── DagsHub.txt
└── README.md
```

---

# Technologies Used

| Technology       | Description               |
| ---------------- | ------------------------- |
| Python           | Main programming language |
| TensorFlow/Keras | Deep learning framework   |
| MLflow           | Experiment tracking       |
| DagsHub          | Online MLflow tracking    |
| Matplotlib       | Visualization             |
| Scikit-Learn     | Evaluation metrics        |

---

# Model Architecture

The model uses CNN architecture with:

- Conv2D
- MaxPooling2D
- Dropout
- Dense Layer
- Softmax Output

---

# MLflow Integration

This project implements:

- MLflow autologging in `modelling.py`
- Manual logging in `modelling_tuning.py`
- Hyperparameter tuning
- Artifact logging
- Online experiment tracking with DagsHub

---

# MLflow Model Artifact

MLflow automatically stores model artifacts in the following structure:

```text
model/
├── MLmodel
├── conda.yaml
├── python_env.yaml
├── requirements.txt
└── data/
    └── model.keras
```

These artifacts ensure model reproducibility and environment consistency.

---

# Additional Logged Artifacts

The project logs additional artifacts including:

- confusion_matrix.png
- classification_report.txt
- training_history.png
- model_summary.txt

---

# Hyperparameters

| Parameter     | Value |
| ------------- | ----- |
| Epochs        | 5     |
| Batch Size    | 32    |
| Learning Rate | 0.001 |
| Dropout Rate  | 0.3   |

---

# Submission Screenshots

This project includes the following screenshots:

- screenshoot_dashboard.jpeg
- screenshoot_model_artifak.jpeg
- screenshoot_dashboard_dagshub.jpeg
- screenshoot_model_artifak_dagshub.jpeg

These screenshots demonstrate:

- MLflow tracking
- Metrics logging
- Artifact storage
- DagsHub online synchronization

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run MLflow Server

```bash
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlartifacts \
--host 127.0.0.1 \
--port 5000
```

---

## Run Training

Basic:

```bash
python modelling.py
```

Advanced/Tuning:

```bash
python modelling_tuning.py
```

---

# MLflow Dashboard

Open:

```text
http://127.0.0.1:5000
```

---

# DagsHub Repository

DagsHub tracking URL is available inside:

```text
DagsHub.txt
```

---

# Evaluation Result

The model successfully performs image classification with good validation accuracy and tracked experiment history using MLflow and DagsHub.

---

# Author

Muh Arifandi
