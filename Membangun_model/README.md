# Intel Image Classification with MLflow and DagsHub

## Project Overview

Project ini merupakan implementasi machine learning lifecycle menggunakan:

- TensorFlow
- MLflow
- DagsHub
- CNN (Convolutional Neural Network)

Tujuan utama project:

- melakukan training image classification,
- melakukan experiment tracking,
- menyimpan metrics,
- menyimpan artifact,
- melakukan online experiment tracking menggunakan DagsHub.

Dataset yang digunakan merupakan dataset preprocessing hasil Kriteria 1.

---

# Project Structure

```text
Membangun_model/
│
├── modelling.py
├── modelling_tuning.py
├── intel_image_preprocessing/
│   ├── train/
│   ├── val/
│   └── test/
│
├── cnn_model.keras
├── training_history.png
├── confusion_matrix.png
├── classification_report.txt
├── model_summary.txt
│
├── screenshot_dashboard.jpg
├── screenshot_artifak.jpg
│
├── requirements.txt
└── DagsHub.txt
```

---

# Dataset Information

Dataset:

- Intel Image Classification Dataset

Class:

- buildings
- forest
- glacier
- mountain
- sea
- street

Dataset telah melalui:

- preprocessing,
- splitting,
- normalization,
- optimization.

---

# Machine Learning Workflow

```mermaid
graph TD

A[Dataset Preprocessing] --> B[Load Dataset]

B --> C[Normalization]

C --> D[Dataset Optimization]

D --> E[Build CNN Model]

E --> F[Model Training]

F --> G[Model Evaluation]

G --> H[MLflow Logging]

H --> I[DagsHub Online Tracking]

I --> J[Artifact Storage]
```

---

# CNN Architecture Flow

```mermaid
graph TD

A[Input Image 150x150x3]

A --> B[Conv2D 32 Filters]

B --> C[MaxPooling2D]

C --> D[Conv2D 64 Filters]

D --> E[MaxPooling2D]

E --> F[Flatten Layer]

F --> G[Dense 128]

G --> H[Dropout 0.3]

H --> I[Dense Softmax 6 Class]

I --> J[Prediction Output]
```

---

# MLflow Tracking Flow

```mermaid
graph TD

A[TensorFlow Training]

A --> B[MLflow Tracking]

B --> C[Log Parameters]

B --> D[Log Metrics]

B --> E[Log Artifacts]

B --> F[Log Model]

F --> G[DagsHub Online Tracking]
```

---

# modelling.py Workflow

```mermaid
graph TD

A[Load Dataset]

A --> B[Build CNN]

B --> C[Compile Model]

C --> D[Train Model]

D --> E[Evaluate Model]

E --> F[MLflow Autolog]
```

---

# modelling_tuning.py Workflow

```mermaid
graph TD

A[Initialize DagsHub]

A --> B[Load Dataset]

B --> C[Normalize Dataset]

C --> D[Dataset Optimization]

D --> E[Set Hyperparameter]

E --> F[Build CNN]

F --> G[Training Model]

G --> H[Evaluation]

H --> I[Manual Logging]

I --> J[Artifact Logging]

J --> K[Save Model]

K --> L[DagsHub Tracking]
```

---

# Technologies Used

| Technology   | Description                |
| ------------ | -------------------------- |
| TensorFlow   | Deep Learning Framework    |
| MLflow       | Experiment Tracking        |
| DagsHub      | Online Experiment Tracking |
| Matplotlib   | Visualization              |
| Scikit-Learn | Evaluation Metrics         |
| NumPy        | Numerical Computation      |

---

# Hyperparameter Configuration

| Hyperparameter | Value   |
| -------------- | ------- |
| Image Size     | 150x150 |
| Batch Size     | 32      |
| Learning Rate  | 0.001   |
| Dropout Rate   | 0.3     |
| Epoch          | 5       |

---

# Metrics Logged

MLflow dan DagsHub menyimpan:

- accuracy
- val_accuracy
- loss
- val_loss
- test_accuracy
- test_loss

---

# Artifacts Logged

Project ini menyimpan artifact berikut:

- training_history.png
- confusion_matrix.png
- classification_report.txt
- model_summary.txt
- cnn_model.keras

---

# DagsHub Repository

Repository DagsHub:

https://dagshub.com/arif76440/MLFlow-Image-Classification

---

# How to Run Project

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Login DagsHub

```bash
dagshub login
```

---

## 3. Run MLflow UI

```bash
mlflow ui
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## 4. Run Training

### modelling.py

```bash
python modelling.py
```

### modelling_tuning.py

```bash
python modelling_tuning.py
```

---

# MLflow Dashboard

MLflow dashboard digunakan untuk:

- melihat metrics,
- melihat parameters,
- melihat artifact,
- melihat experiment tracking.

---

# DagsHub Integration

DagsHub digunakan untuk:

- online experiment tracking,
- menyimpan experiment history,
- monitoring metrics,
- menyimpan artifact online.

---

# Evaluation Result

Model berhasil melakukan:

- image classification,
- experiment tracking,
- artifact logging,
- online monitoring menggunakan DagsHub.

---

# Author

Nama:
Muh Arifandi

Project:
Machine Learning Lifecycle with MLflow and DagsHub
