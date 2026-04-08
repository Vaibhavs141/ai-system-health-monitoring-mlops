# 🖥️ AI-Based System Health Monitoring & Failure Prediction (MLOps)

An end-to-end **MLOps-based machine learning system** that predicts system health status (`Healthy`, `Warning`, `Critical`) using real-time system metrics.

This project demonstrates a **complete production-style ML lifecycle** including data versioning, pipeline orchestration, experiment tracking, deployment, monitoring, CI/CD, and a user-facing interface.

---

## 🚀 Problem Statement

Traditional system monitoring is **reactive** — issues are detected only after failure.

This project introduces **predictive monitoring**, enabling early detection of system risks using:

- CPU usage
- Memory usage
- Disk usage
- Temperature
- Network traffic
- Error count
- Response time

---

## 🎯 Project Objective

To build a **production-ready ML system** that:

- Predicts system health state
- Estimates failure probability
- Enables proactive system maintenance
- Demonstrates real-world MLOps practices

---

## 🏗️ System Architecture

```
User (Streamlit UI)
        ↓
FastAPI (/predict)
        ↓
ML Model (Best Model)
        ↓
Prediction Output
        ↓
Prometheus (/metrics monitoring)

Training Pipeline:
DVC → Prefect → MLflow → Model Selection
```

---

## 🧰 Tech Stack

### 🔹 Core ML & Data
- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest)

### 🔹 MLOps Tools

| Tool | Purpose |
|------|---------|
| **Git + GitHub** | Code versioning |
| **DVC** | Data & pipeline versioning |
| **Prefect** | Pipeline orchestration |
| **MLflow** | Experiment tracking |
| **FastAPI** | Model serving |
| **Docker** | Containerization |
| **GitHub Actions** | CI/CD |
| **Prometheus** | Monitoring |

### 🔹 Frontend
- **Streamlit** → Interactive UI

---

## 📊 Dataset

- Synthetic dataset with realistic system behavior
- Balanced classification: `Healthy`, `Warning`, `Critical`

### Features

| Feature | Description |
|---------|-------------|
| `cpu_usage` | CPU utilization percentage |
| `memory_usage` | RAM usage percentage |
| `temperature` | System temperature |
| `voltage` | Power supply voltage |
| `disk_usage` | Disk utilization percentage |
| `fan_speed` | Cooling fan RPM |
| `network_traffic` | Network I/O volume |
| `error_count` | Number of logged errors |
| `response_time` | API/service response time |

### Engineered Features

- Thermal stress
- Memory pressure
- Power instability
- Cooling efficiency

---

## 🤖 Model Development

### Models Used

| Model | Purpose |
|-------|---------|
| Logistic Regression | Baseline & Final Model |
| Random Forest | Tuned Model |

### Evaluation Metrics

- Accuracy
- Weighted F1 Score
- Confusion Matrix

### Final Model

- ✅ Logistic Regression selected
- 🎯 ~97% accuracy

---

## 🔁 ML Pipeline (Prefect + DVC)

Pipeline stages:

1. Data ingestion
2. Data validation
3. Data preprocessing
4. Feature engineering
5. Train / validation / test split
6. Model training
7. Hyperparameter tuning
8. Model evaluation

### Run Pipeline

```bash
python -m pipeline.training_pipeline
```

### Using DVC

```bash
dvc repro
```

---

## 📈 Experiment Tracking (MLflow)

Tracks:
- Model parameters
- Evaluation metrics
- Artifacts

### Run MLflow UI

```bash
mlflow ui
```

> Open `http://localhost:5000` in your browser.

---

## 🌐 API Deployment (FastAPI)

### Run API

```bash
uvicorn api.app:app --reload
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict system health |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

---

## 🖥️ Frontend (Streamlit UI)

### Run UI

```bash
streamlit run frontend/app.py
```

### Features

- Interactive input sliders
- Real-time prediction
- Probability visualization
- Clean UI for demo

---

## 🐳 Docker Deployment

### Build

```bash
docker build -t system-health-api .
```

### Run

```bash
docker run -p 8000:8000 system-health-api
```

---

## ⚙️ CI/CD (GitHub Actions)

Automated pipeline includes:

- Linting (`flake8`)
- Unit testing (`pytest`)
- Docker build

**Triggered on:** `push` and `pull_request`

---

## 📊 Monitoring (Prometheus)

Prometheus scrapes the `/metrics` endpoint exposed by the FastAPI app.

### Key Metrics

| Metric | Description |
|--------|-------------|
| `api_requests_total` | Total number of API requests |
| `api_request_latency_seconds` | Request latency histogram |
| `api_prediction_total` | Predictions per class label |
| `model_failure_probability` | Predicted failure probability |
| `model_confidence` | Model confidence score |

### Run Prometheus

```bash
prometheus --config.file=monitoring/prometheus.yml
```

### Open Prometheus UI

```
http://localhost:9090
```

---

## 🔍 Observability

### Metrics (Prometheus)
- Request count
- Latency
- Prediction distribution
- Model confidence

### Logs
- Request inputs
- Prediction outputs
- Errors

### Tracing (Simulated)
- Request IDs for tracking execution flow

---

## 📂 Project Structure

```
project/
│
├── data/                  # Raw and processed datasets
├── src/                   # Core source modules
├── pipeline/              # Prefect training pipeline
├── api/                   # FastAPI application
├── frontend/              # Streamlit UI
├── monitoring/            # Prometheus config
├── models/                # Saved model artifacts
├── tests/                 # Unit and integration tests
│
├── dvc.yaml               # DVC pipeline definition
├── params.yaml            # Pipeline parameters
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## ▶️ How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run training pipeline

```bash
python -m pipeline.training_pipeline
```

### 3. Start API server

```bash
uvicorn api.app:app --reload
```

### 4. Start frontend

```bash
streamlit run frontend/app.py
```

### 5. Start monitoring

```bash
prometheus --config.file=monitoring/prometheus.yml
```

---

## 🎓 Key Learnings

- End-to-end MLOps pipeline design
- Data versioning with DVC
- Pipeline orchestration with Prefect
- Experiment tracking with MLflow
- API deployment with FastAPI
- Containerization with Docker
- CI/CD with GitHub Actions
- Production monitoring with Prometheus

---

## 💡 Future Improvements

- [ ] Grafana dashboards for visualization
- [ ] Real drift detection with Evidently AI
- [ ] Cloud deployment (AWS / GCP / Azure)
- [ ] Alerting system integration
- [ ] Real-world production datasets

---

## ⭐ Star This Project

If you found this project useful, please consider giving it a ⭐ on GitHub — it helps others discover it!