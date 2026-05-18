# Smart Machine Learning Submission Project

## Final Submission — Sistem Machine Learning Terapan

Nama: Muh Arifandi  
Kategori: Image Classification  
Level Submission: Advanced

---

# Deskripsi Proyek

Proyek ini merupakan implementasi sistem machine learning end-to-end untuk klasifikasi gambar menggunakan TensorFlow dan MLflow. Sistem mencakup:

- Data preprocessing
- Training dan hyperparameter tuning
- Experiment tracking menggunakan MLflow
- Continuous Integration (CI/CD)
- Docker containerization
- Monitoring menggunakan Prometheus
- Visualisasi monitoring menggunakan Grafana
- Alerting system menggunakan Grafana Alerting

Proyek dikembangkan untuk memenuhi seluruh kriteria submission Dicoding Sistem Machine Learning Terapan mulai dari Kriteria 1 hingga Kriteria 4.

---

# Struktur Submission

```text
SMSML_Nama_Muh_Arifandi/
│
├── Membangun_model/
│   ├── intel_image_preprocessing/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── requirements.txt
│   ├── screenshot_dashboard.jpeg
│   ├── screenshot_artifak.jpeg
│   ├── training_history.png
│   ├── DagsHub.txt
│   └── README.md
│
├── Monitoring_dan_Logging/
│   ├── bukti_serving/
│   ├── bukti_monitoring_prometheus/
│   ├── bukti_monitoring_grafana/
│   ├── bukti_alerting_grafana/
│   ├── prometheus.yml
│   ├── prometheus_exporter.py
│   ├── inference.py
│   └── docker-compose.yml
│
├── Workflow-CI.txt
└── Eksperimen_SML_Muh_Arifandi.txt
```

---

# Kriteria 1 — Membangun Model Machine Learning

## Tujuan

Membangun model machine learning untuk melakukan klasifikasi gambar menggunakan TensorFlow.

## Implementasi

Pada tahap ini dilakukan:

- Pengumpulan dataset gambar
- Data preprocessing
- Data augmentation
- Training model CNN
- Evaluasi model
- Penyimpanan model

## Teknologi yang Digunakan

- Python
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn

## File Utama

| File                       | Deskripsi                    |
| -------------------------- | ---------------------------- |
| modelling.py               | Training model utama         |
| intel_image_preprocessing/ | Folder preprocessing dataset |
| training_history.png       | Visualisasi training         |

## Hasil

Model berhasil melakukan klasifikasi gambar dengan performa yang baik dan mampu digunakan untuk inference.

---

# Kriteria 2 — Experiment Tracking dengan MLflow

## Tujuan

Melakukan experiment tracking untuk mencatat:

- Parameter training
- Metrics model
- Artifacts model
- Versioning eksperimen

## Implementasi

MLflow digunakan untuk:

- Logging parameter
- Logging metrics
- Logging artifacts
- Menyimpan model hasil training

## Fitur yang Digunakan

- mlflow.start_run()
- mlflow.log_param()
- mlflow.log_metric()
- mlflow.log_artifact()
- mlflow.tensorflow.log_model()

## File dan Bukti

| File                      | Deskripsi                |
| ------------------------- | ------------------------ |
| screenshot_dashboard.jpeg | Dashboard MLflow         |
| screenshot_artifak.jpeg   | Bukti artifact tersimpan |
| DagsHub.txt               | Link DagsHub repository  |

## Hasil

Eksperimen berhasil tercatat dan seluruh artifact model berhasil disimpan menggunakan MLflow.

---

# Kriteria 3 — CI/CD dan Docker Deployment

## Tujuan

Membangun workflow otomatis untuk:

- Training model
- Upload artifacts
- Build Docker image
- Push Docker image ke Docker Hub

## Implementasi CI/CD

Workflow GitHub Actions digunakan untuk:

1. Checkout repository
2. Setup Python
3. Install dependencies
4. Run MLflow project
5. Upload artifacts
6. Build Docker image
7. Login Docker Hub
8. Push Docker image

## Teknologi

- GitHub Actions
- Docker
- Docker Hub

## Docker Deployment

Model di-containerize menggunakan Docker sehingga dapat dijalankan secara portable.

## File dan Bukti

| File               | Deskripsi                |
| ------------------ | ------------------------ |
| Workflow-CI.txt    | Informasi workflow CI/CD |
| docker-compose.yml | Konfigurasi container    |

## Hasil

CI/CD pipeline berhasil berjalan otomatis dan Docker image berhasil dipush ke Docker Hub.

---

# Kriteria 4 — Monitoring dan Logging

## Tujuan

Membangun sistem monitoring untuk memantau performa model machine learning secara real-time.

## Implementasi Monitoring

Monitoring dilakukan menggunakan:

- Prometheus
- Grafana
- Grafana Alerting

## Metrics yang Dipantau

Sistem memantau berbagai metrics seperti:

1. Total API Requests
2. Total Predictions
3. Total Errors
4. Prediction Latency
5. CPU Usage
6. Memory Usage
7. Disk Usage
8. Process Memory
9. Prediction Confidence
10. Model Load Time

## Prometheus

Prometheus digunakan untuk:

- Scraping metrics
- Menyimpan metrics time-series
- Query metrics monitoring

### File

| File                   | Deskripsi               |
| ---------------------- | ----------------------- |
| prometheus.yml         | Konfigurasi Prometheus  |
| prometheus_exporter.py | Exporter custom metrics |

### Bukti

Folder:

```text
bukti_monitoring_prometheus/
```

berisi:

- Query metrics Prometheus
- Status target UP
- Monitoring metrics berjalan

---

## Grafana Dashboard

Grafana digunakan untuk visualisasi monitoring.

### Dashboard Menampilkan

- API request monitoring
- Prediction monitoring
- Error monitoring
- CPU monitoring
- Memory monitoring
- Disk monitoring
- Latency monitoring
- Confidence monitoring

### Bukti

Folder:

```text
bukti_monitoring_grafana/
```

berisi screenshot dashboard monitoring.

---

## Grafana Alerting

Grafana Alerting digunakan untuk membuat sistem notifikasi ketika metrics melebihi threshold.

### Alert Rules

1. CPU Usage High
2. Memory Usage High
3. Disk Usage High

### Contact Point

Grafana contact point digunakan untuk konfigurasi notifikasi alert.

### Bukti

Folder:

```text
bukti_alerting_grafana/
```

berisi:

- Rules alert
- Contact point
- Notification configuration
- Alert firing status

---

# Serving Model

Model berhasil di-serving menggunakan inference API.

## Fitur API

- Endpoint prediksi
- Response JSON
- Confidence prediction
- Monitoring integration

## File

| File         | Deskripsi           |
| ------------ | ------------------- |
| inference.py | API inference model |

## Bukti

Folder:

```text
bukti_serving/
```

berisi:

- API running
- Predict success
- Response JSON

---

# Teknologi yang Digunakan

| Teknologi      | Fungsi                   |
| -------------- | ------------------------ |
| Python         | Bahasa pemrograman utama |
| TensorFlow     | Deep learning framework  |
| MLflow         | Experiment tracking      |
| DagsHub        | Remote tracking MLflow   |
| Docker         | Containerization         |
| GitHub Actions | CI/CD pipeline           |
| Prometheus     | Monitoring metrics       |
| Grafana        | Visualisasi monitoring   |
| Flask/FastAPI  | Serving API              |

---

# Kesimpulan

Proyek berhasil memenuhi seluruh kriteria submission Sistem Machine Learning Terapan, meliputi:

- Kriteria 1: Membangun model machine learning
- Kriteria 2: Experiment tracking menggunakan MLflow
- Kriteria 3: CI/CD dan Docker deployment
- Kriteria 4: Monitoring dan logging menggunakan Prometheus dan Grafana

Sistem monitoring berhasil menampilkan lebih dari 10 metrics berbeda dan memiliki 3 alerting rules aktif sehingga memenuhi kategori Advanced.

---

# Author

Muh Arifandi
