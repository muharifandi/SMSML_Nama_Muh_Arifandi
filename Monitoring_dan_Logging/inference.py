from flask import Flask, request, jsonify, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

import tensorflow as tf
import numpy as np
from PIL import Image
import time
import psutil
import os

app = Flask(__name__)

# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    "../Membangun_model/cnn_model.keras"
)

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

# ==========================================
# PROMETHEUS METRICS
# ==========================================

REQUEST_COUNT = Counter(
    "request_count",
    "Total API Requests"
)

PREDICTION_COUNT = Counter(
    "prediction_count",
    "Total Predictions"
)

ERROR_COUNT = Counter(
    "error_count",
    "Total Errors"
)

LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

CPU_USAGE = Gauge(
    "cpu_usage_percent",
    "CPU Usage"
)

MEMORY_USAGE = Gauge(
    "memory_usage_percent",
    "Memory Usage"
)

DISK_USAGE = Gauge(
    "disk_usage_percent",
    "Disk Usage"
)

PROCESS_MEMORY = Gauge(
    "process_memory_mb",
    "Process Memory Usage"
)

MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Model load time"
)

PREDICTION_CONFIDENCE = Gauge(
    "prediction_confidence",
    "Prediction confidence"
)

# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():
    return jsonify({
        "message": "ML Monitoring API Running"
    })

# ==========================================
# METRICS
# ==========================================

@app.route("/metrics")
def metrics():

    CPU_USAGE.set(
        psutil.cpu_percent()
    )

    MEMORY_USAGE.set(
        psutil.virtual_memory().percent
    )

    DISK_USAGE.set(
        psutil.disk_usage("/").percent
    )

    process = psutil.Process(os.getpid())

    PROCESS_MEMORY.set(
        process.memory_info().rss / 1024 / 1024
    )

    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )

# ==========================================
# PREDICT
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    start_time = time.time()

    try:

        REQUEST_COUNT.inc()

        if "file" not in request.files:
            ERROR_COUNT.inc()

            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        image = Image.open(file).convert("RGB")
        image = image.resize((150, 150))

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        prediction = model.predict(img_array)

        class_index = np.argmax(prediction)

        confidence = float(
            np.max(prediction)
        )

        predicted_class = CLASS_NAMES[class_index]

        PREDICTION_COUNT.inc()

        PREDICTION_CONFIDENCE.set(
            confidence
        )

        latency = time.time() - start_time

        LATENCY.observe(latency)

        return jsonify({
            "class": predicted_class,
            "confidence": confidence,
            "latency": latency
        })

    except Exception as e:

        ERROR_COUNT.inc()

        return jsonify({
            "error": str(e)
        }), 500

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )