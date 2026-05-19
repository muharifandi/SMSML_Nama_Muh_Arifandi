# ============================================================
# 1. IMPORT LIBRARY
# ============================================================

import os
import tensorflow as tf

import mlflow
import mlflow.tensorflow

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.utils import (
    image_dataset_from_directory
)


# ============================================================
# 2. SET TRACKING URI & EXPERIMENT
# ============================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Intel_Image_Classification")


# ============================================================
# 3. KONFIGURASI DATASET
# ============================================================

DATASET_DIR = "intel_image_preprocessing"

IMG_SIZE = (150, 150)

BATCH_SIZE = 32

EPOCHS = 5

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 4. LOAD DATASET
# ============================================================

train_dataset = image_dataset_from_directory(
    f"{DATASET_DIR}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = image_dataset_from_directory(
    f"{DATASET_DIR}/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_dataset = image_dataset_from_directory(
    f"{DATASET_DIR}/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 5. NORMALISASI DATASET
# ============================================================

normalization_layer = tf.keras.layers.Rescaling(
    1./255
)

train_dataset = train_dataset.map(
    lambda x, y: (
        normalization_layer(x),
        y
    )
)

val_dataset = val_dataset.map(
    lambda x, y: (
        normalization_layer(x),
        y
    )
)

test_dataset = test_dataset.map(
    lambda x, y: (
        normalization_layer(x),
        y
    )
)


# ============================================================
# 6. OPTIMASI DATASET
# ============================================================

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 7. MEMBANGUN MODEL CNN
# ============================================================

model = Sequential([

    Input(
        shape=(150,150,3)
    ),

    Conv2D(
        32,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),

    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        6,
        activation="softmax"
    )
])


# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 9. AKTIFKAN AUTOLOG
# ============================================================

mlflow.tensorflow.autolog()


# ============================================================
# 10. TRAINING MODEL
# ============================================================

with mlflow.start_run():

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS
    )


    # ========================================================
    # 11. EVALUASI MODEL
    # ========================================================

    test_loss, test_accuracy = model.evaluate(
        test_dataset
    )

    print("Test Accuracy :", test_accuracy)


    # ========================================================
    # 12. LOG METRIC TAMBAHAN
    # ========================================================

    mlflow.log_metric(
        "test_accuracy",
        test_accuracy
    )

    mlflow.log_metric(
        "test_loss",
        test_loss
    )


    # ========================================================
    # 13. LOG MODEL
    # (otomatis buat folder model berisi MLmodel,
    #  conda.yaml, python_env.yaml, requirements.txt)
    # ========================================================

    mlflow.tensorflow.log_model(
        model,
        artifact_path="model"
    )


print(
    "Training selesai dan berhasil dicatat di MLflow"
)