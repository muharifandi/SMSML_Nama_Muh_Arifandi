# ============================================================
# 1. IMPORT LIBRARY
# ============================================================

import os
import dagshub
import mlflow
import mlflow.tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

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
# 2. INISIALISASI DAGSHUB
# ============================================================

dagshub.init(
    repo_owner="arif76440",
    repo_name="MLFlow-Image-Classification",
    mlflow=True
)


# ============================================================
# 3. SET EXPERIMENT
# ============================================================

mlflow.set_experiment(
    "Intel_Image_Classification"
)


# ============================================================
# 4. KONFIGURASI DATASET
# ============================================================

DATASET_DIR = "intel_image_preprocessing"

IMG_SIZE = (150,150)

BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 5. LOAD DATASET
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
# 6. CLASS NAMES
# ============================================================

class_names = train_dataset.class_names


# ============================================================
# 7. NORMALISASI
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
# 8. OPTIMASI DATASET
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
# 9. HYPERPARAMETER
# ============================================================

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.3

EPOCHS = 5


# ============================================================
# 10. START RUN
# ============================================================

with mlflow.start_run():

    # ========================================================
    # 11. LOG PARAMETER
    # ========================================================

    mlflow.log_param(
        "learning_rate",
        LEARNING_RATE
    )

    mlflow.log_param(
        "batch_size",
        BATCH_SIZE
    )

    mlflow.log_param(
        "dropout_rate",
        DROPOUT_RATE
    )

    mlflow.log_param(
        "epochs",
        EPOCHS
    )


    # ========================================================
    # 12. MODEL CNN
    # ========================================================

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

        Dropout(
            DROPOUT_RATE
        ),

        Dense(
            6,
            activation="softmax"
        )
    ])


    # ========================================================
    # 13. MODEL SUMMARY
    # ========================================================

    with open(
        "model_summary.txt",
        "w"
    ) as file:

        model.summary(
            print_fn=lambda x:
            file.write(x + "\n")
        )

    mlflow.log_artifact(
        "model_summary.txt"
    )


    # ========================================================
    # 14. COMPILE MODEL
    # ========================================================

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )


    # ========================================================
    # 15. TRAIN MODEL
    # ========================================================

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS
    )


    # ========================================================
    # 16. EVALUASI
    # ========================================================

    test_loss, test_accuracy = model.evaluate(
        test_dataset
    )


    # ========================================================
    # 17. LOG METRIC
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
    # 18. TRAINING HISTORY
    # ========================================================

    plt.figure(figsize=(8,6))

    plt.plot(
        history.history["accuracy"],
        label="train_accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="val_accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training History")

    plt.legend()

    plt.savefig(
        "training_history.png"
    )

    plt.close()

    mlflow.log_artifact(
        "training_history.png"
    )


    # ========================================================
    # 19. PREDIKSI
    # ========================================================

    y_true = np.concatenate([
        y for x, y in test_dataset
    ])

    y_pred = np.argmax(
        model.predict(test_dataset),
        axis=1
    )


    # ========================================================
    # 20. CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(8,8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(ax=ax)

    plt.title("Confusion Matrix")

    plt.savefig(
        "confusion_matrix.png"
    )

    plt.close()

    mlflow.log_artifact(
        "confusion_matrix.png"
    )


    # ========================================================
    # 21. CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    with open(
        "classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    mlflow.log_artifact(
        "classification_report.txt"
    )


    # ========================================================
    # 22. SAVE MODEL
    # ========================================================

    os.makedirs("model", exist_ok=True)

    model.save("model/model.keras")


    # ========================================================
    # 23. FILE TAMBAHAN MODEL
    # ========================================================

    with open("model/MLmodel", "w") as f:
        f.write("TensorFlow CNN Model")


    # ========================================================
    # 24. LOG MODEL ARTIFACT
    # ========================================================

    mlflow.log_artifact("model/model.keras")

    mlflow.log_artifact("model/MLmodel")

    # ========================================================
    # 25. LOG MODEL KE MLFLOW
    # ========================================================

    mlflow.tensorflow.log_model(
        model,
        artifact_path="model"
    )

    # Simpan run_id untuk keperluan CI
    run_id = mlflow.active_run().info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print(f"Run ID: {run_id}")


print(
    "Training dan logging berhasil"
)