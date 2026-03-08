<div align="center">

# 🚀 CallCenterAI – Intelligent Customer Ticket Classification

<p>
  <img src="https://img.shields.io/badge/MLOps-Pipeline-blue" />
  <img src="https://img.shields.io/badge/Python-3.11-green" />
  <img src="https://img.shields.io/badge/FastAPI-0.104-lightblue" />
  <img src="https://img.shields.io/badge/Docker-Containers-orange" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<p>
  <img src="https://img.shields.io/badge/Tests-23%20passed-brightgreen" />
  <img src="https://img.shields.io/badge/Coverage-92%25-success" />
  <img src="https://img.shields.io/badge/Version-1.0.0-blueviolet" />
</p>

<p>
  <b>Production-Ready MLOps System</b> · Dual NLP Architecture · Real-Time Intelligence
</p>

<p>
  <a href="https://github.com/nessmattcash/CallCenterAI/stargazers">
    <img src="https://img.shields.io/github/stars/nessmattcash/CallCenterAI?style=social" />
  </a>
  <a href="https://github.com/nessmattcash/CallCenterAI/issues">
    <img src="https://img.shields.io/github/issues/nessmattcash/CallCenterAI" />
  </a>
</p>

> **Dual-model NLP system** that classifies customer service tickets in real time using a smart routing agent — combining TF-IDF+SVM speed with Transformer multilingual accuracy, wrapped in a full MLOps pipeline.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [ML Models & Performance](#ml-models--performance)
- [AI Agent Intelligence](#ai-agent-intelligence)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Monitoring & Observability](#monitoring--observability)
- [CI/CD Pipeline](#cicd-pipeline)
- [Project Structure](#project-structure)
- [Development Guide](#development-guide)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

CallCenterAI is a production-grade MLOps system that automatically categorizes customer support tickets into 8 categories (Hardware, Access, HR Support, Purchase, Administrative, Storage, Internal Project, Miscellaneous).

**Key capabilities:**
- **Dual-model routing** — TF-IDF+SVM for fast English queries, DistilBERT for multilingual/complex text
- **Smart agent** — selects the right model per request based on language, length, and complexity
- **PII scrubbing** — auto-removes emails, phones, CIN numbers, and names before classification
- **Full MLOps stack** — MLflow tracking, DVC versioning, Prometheus metrics, Grafana dashboards
- **CI/CD** — GitHub Actions with Trivy security scanning and Docker publishing

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Client Request (HTTP/JSON)              │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │     AI Agent :8000    │
          │  PII Scrubbing        │
          │  Language Detection   │
          │  Intelligent Routing  │
          └──────┬────────┬───────┘
                 │        │
    ┌────────────▼─┐  ┌───▼──────────────┐
    │ TF-IDF+SVM   │  │  Transformer     │
    │ :8010        │  │  :8020           │
    │ Fast · 10ms  │  │  Multilingual    │
    │ English      │  │  45-55ms         │
    └──────────────┘  └──────────────────┘
                 │        │
          ┌──────▼────────▼───────┐
          │      MLOps Stack      │
          │  MLflow :5000         │
          │  Prometheus :9090     │
          │  Grafana :3000        │
          └───────────────────────┘
```

**Service map:**

| Service | Port | Responsibility |
|---|---|---|
| AI Agent | 8000 | Routing, PII scrubbing, language detection |
| TF-IDF Service | 8010 | Fast classification — simple English cases |
| Transformer Service | 8020 | Advanced multilingual classification |
| MLflow | 5000 | Experiment tracking & model registry |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards & visualization |

---

## ML Models & Performance

### Comparison

| Metric | TF-IDF + SVM | DistilBERT Transformer |
|---|---|---|
| Accuracy | **94.8%** | 92.3% |
| F1-Score | **94.7%** | 91.8% |
| Inference Speed | ⚡ 10–15ms | 45–55ms |
| Memory | 120MB | 890MB |
| Multilingual | ❌ Limited | ✅ Excellent |
| Complex Context | ⚠️ Basic | ✅ Advanced |
| Training Time | 2 min | 45 min |

### Per-category F1 scores

| Category | TF-IDF | Transformer | Winner |
|---|---|---|---|
| Hardware | 0.96 | 0.93 | TF-IDF |
| Access | 0.94 | 0.95 | Transformer |
| HR Support | 0.91 | 0.89 | TF-IDF |
| Purchase | 0.93 | 0.90 | TF-IDF |
| Administrative | 0.90 | 0.92 | Transformer |
| Storage | 0.94 | 0.93 | TF-IDF |
| Internal Project | 0.89 | 0.91 | Transformer |
| Miscellaneous | 0.92 | 0.91 | Tie |

---

## AI Agent Intelligence

### Routing Logic

```python
class IntelligentRouter:
    def select_model(self, text: str, language: str, word_count: int) -> str:
        rules = [
            # Short English tickets → TF-IDF (fast)
            (lambda: language == "en" and word_count <= 8,         "TF-IDF"),
            # Low complexity → TF-IDF
            (lambda: self.estimate_complexity(text) < 0.3,         "TF-IDF"),
            # Non-English → Transformer
            (lambda: language in ["fr", "ar", "es"],               "Transformer"),
            # Long or technical text → Transformer
            (lambda: word_count > 15 or self.has_technical_terms(text), "Transformer"),
            # Default fallback
            (lambda: True,                                          "Transformer"),
        ]
        for condition, model in rules:
            if condition():
                return model
```

### PII Scrubbing

```python
class PIIScrubber:
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
        'cin':   r'\b\d{8}\b',   # Tunisian CIN
        'name':  r'\b(Mr|Ms|Mrs|Dr)\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
    }

    def scrub(self, text: str) -> tuple[str, dict]:
        detected = {}
        scrubbed = text
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                detected[pii_type] = matches
                for i, _ in enumerate(matches):
                    scrubbed = re.sub(pattern, f'[{pii_type.upper()}_{i}]', scrubbed, count=1)
        return scrubbed, detected
```

---

## Quick Start

### Requirements

- Docker 20.10+ and Docker Compose 2.0+
- Python 3.11+ (for local development only)

### Deploy in one command

```bash
git clone https://github.com/nessmattcash/CallCenterAI.git
cd CallCenterAI
docker-compose up -d --build
```

### Verify all services are running

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Expected output:
```
NAMES                         STATUS         PORTS
callcenterai-grafana-1        Up             0.0.0.0:3000->3000/tcp
callcenterai-agent-1          Up             0.0.0.0:8000->8000/tcp
callcenterai-prometheus-1     Up             0.0.0.0:9090->9090/tcp
callcenterai-tfidf-1          Up             0.0.0.0:8010->8010/tcp
callcenterai-transformer-1    Up             0.0.0.0:8020->8020/tcp
callcenterai-mlflow-1         Up             0.0.0.0:5000->5000/tcp
```

### Health check

```bash
curl -s http://localhost:8000/health | python -m json.tool
curl -s http://localhost:8010/health | python -m json.tool
curl -s http://localhost:8020/health | python -m json.tool
```

### Access dashboards

| Interface | URL | Credentials |
|---|---|---|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| MLflow UI | http://localhost:5000 | — |
| Agent API docs | http://localhost:8000/docs | — |

---

## API Reference

### POST `/classify` — Main endpoint

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "My laptop screen is broken. Need replacement ASAP! Contact: john@company.com"}'
```

Response:

```json
{
  "success": true,
  "category": "Hardware",
  "confidence": 0.9945,
  "model_used": "TF-IDF + SVM",
  "model_choice_reason": "Short English (9 words)",
  "pii_scrubbed": true,
  "pii_details": {
    "emails": ["john@company.com"],
    "phones": [],
    "cins": [],
    "names": []
  },
  "detected_language": "en",
  "processing_time_ms": 65.2,
  "timestamp": "2025-12-15T18:13:36.123Z"
}
```

### POST `/classify` — French input (Transformer routing)

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "J'\''ai des problèmes d'\''accès au système de facturation en ligne"}'
```

### POST `/batch_classify` — Multiple tickets

```bash
curl -X POST http://localhost:8000/batch_classify \
  -H "Content-Type: application/json" \
  -d '{
    "tickets": [
      {"id": "1", "text": "Monitor not working"},
      {"id": "2", "text": "Need password reset"},
      {"id": "3", "text": "Problème avec le serveur"}
    ]
  }'
```

### Direct model access

```bash
# TF-IDF service directly
curl -X POST http://localhost:8010/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Printer not working"}'

# Transformer service directly
curl -X POST http://localhost:8020/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Problème réseau critique"}'

# Prometheus metrics
curl http://localhost:8000/metrics | grep http_requests_total
```

---

## Monitoring & Observability

### Prometheus metrics exposed per service

| Metric | Description |
|---|---|
| `http_requests_total` | Total request count by status |
| `http_request_duration_seconds` | Latency histogram |
| `model_predictions_total` | Prediction count per model |
| `prediction_confidence` | Confidence score distribution |
| `pii_detection_total` | PII items detected by type |

### Alert rules (`monitoring/prometheus/alerts.yml`)

```yaml
groups:
  - name: callcenterai_alerts
    rules:

      - alert: HighErrorRate
        expr: rate(http_requests_total{status="500"}[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on {{ $labels.service }}"

      - alert: HighModelLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency above 1s on {{ $labels.service }}"

      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
```

### Grafana dashboards

Four pre-configured dashboards:
- **System Health** — uptime, resource usage, service status
- **Model Performance** — accuracy, F1, confusion matrix
- **Language Analytics** — distribution by language and category
- **Inference Metrics** — P95/P99 latency, throughput, error rates

---

## CI/CD Pipeline

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Lint
        run: |
          black --check src tests
          flake8 src tests
          isort --check-only src tests

      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Security scan (Bandit)
        run: bandit -r src

      - name: Trivy filesystem scan
        run: trivy fs --severity HIGH,CRITICAL .

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker images
        run: |
          docker build -f docker/Dockerfile.agent       -t callcenterai/agent:latest .
          docker build -f docker/Dockerfile.tfidf       -t callcenterai/tfidf:latest .
          docker build -f docker/Dockerfile.transformer -t callcenterai/transformer:latest .

      - name: Scan images with Trivy
        run: |
          trivy image callcenterai/agent:latest
          trivy image callcenterai/tfidf:latest
          trivy image callcenterai/transformer:latest

      - name: Push to Docker Hub
        if: github.ref == 'refs/heads/main'
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker push callcenterai/agent:latest
          docker push callcenterai/tfidf:latest
          docker push callcenterai/transformer:latest
```

---

## Project Structure

```
CallCenterAI/
├── .github/
│   └── workflows/
│       ├── ci-cd.yml               # Main CI/CD pipeline
│       ├── security-scan.yml       # Trivy + Bandit scans
│       └── docker-publish.yml      # Docker Hub publishing
│
├── docker/
│   ├── Dockerfile.agent            # AI Agent service
│   ├── Dockerfile.tfidf            # TF-IDF+SVM service
│   ├── Dockerfile.transformer      # DistilBERT service
│   └── prometheus.yml              # Prometheus scrape config
│
├── src/
│   ├── agent/
│   │   ├── api.py                  # FastAPI application
│   │   ├── router.py               # Model selection logic
│   │   ├── pii_scrubber.py         # PII detection & removal
│   │   └── language_detector.py    # Language identification
│   │
│   ├── tfidf_service/
│   │   ├── api.py                  # REST API endpoints
│   │   ├── model.py                # TF-IDF+SVM wrapper
│   │   └── metrics.py              # Prometheus metrics
│   │
│   ├── transformer_service/
│   │   ├── api.py                  # FastAPI application
│   │   ├── model.py                # HuggingFace model wrapper
│   │   └── tokenizer.py            # Text preprocessing
│   │
│   ├── training/
│   │   ├── train_tfidf.py          # TF-IDF pipeline training
│   │   ├── train_transformer.py    # Transformer fine-tuning
│   │   └── data_preprocessor.py   # Data cleaning
│   │
│   └── utils/
│       ├── mlflow_client.py        # MLflow integration
│       ├── dvc_utils.py            # DVC helpers
│       └── config.py               # Configuration management
│
├── tests/
│   ├── test_agent.py               # Agent routing & PII tests
│   ├── test_tfidf.py               # TF-IDF service tests
│   ├── test_transformer.py         # Transformer service tests
│   ├── test_integration.py         # End-to-end service tests
│   ├── test_training.py            # Training pipeline tests
│   ├── test_data.py                # Data validation tests
│   └── conftest.py                 # Shared pytest fixtures
│
├── data/
│   ├── raw/                        # Original Kaggle dataset
│   ├── processed/                  # Cleaned & split data
│   └── external/                   # External resources
│
├── models/
│   ├── tfidf_svm.pkl               # Trained TF-IDF+SVM pipeline
│   ├── label_encoder.pkl           # Category label mappings
│   └── transformer/                # Fine-tuned DistilBERT
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer.json
│
├── monitoring/
│   ├── grafana/
│   │   └── dashboard.json          # Grafana dashboard definition
│   └── prometheus/
│       ├── alerts.yml              # Alert rules
│       └── recording_rules.yml     # Recording rules
│
├── scripts/
│   ├── deploy.sh                   # Deployment helper
│   ├── health_check.sh             # Service health check
│   └── benchmark.py               # Performance benchmarking
│
├── docker-compose.yaml             # Full stack orchestration
├── dvc.yaml                        # Data pipeline definition
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Dev + test dependencies
├── pyproject.toml                  # Python project config
├── .pre-commit-config.yaml         # Git hooks
└── README.md
```

---

## Development Guide

### Local setup (without Docker)

```bash
# Clone
git clone https://github.com/nessmattcash/CallCenterAI.git
cd CallCenterAI

# Virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Pull data via DVC
dvc pull

# Run services locally
uvicorn src.agent.api:app          --reload --port 8000
uvicorn src.tfidf_service.api:app  --reload --port 8010
uvicorn src.transformer_service.api:app --reload --port 8020
```

### Train models

```bash
# Train TF-IDF + SVM
python src/training/train_tfidf.py \
  --data-path data/processed/train.csv \
  --model-save-path models/tfidf_svm.pkl \
  --mlflow-tracking-uri http://localhost:5000

# Fine-tune DistilBERT
python src/training/train_transformer.py \
  --model-name distilbert-base-multilingual-cased \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5

# Log experiments to MLflow
python train_and_log.py
```

---

## Testing

```bash
# Run full test suite with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific suites
pytest tests/test_agent.py -v
pytest tests/test_integration.py -v

# Skip slow tests
pytest -m "not slow"

# Integration tests only
pytest -m "integration"
```

**Test results:**

| File | Tests | Status |
|---|---|---|
| test_agent.py | 5 | ✅ Passed |
| test_tfidf.py | 4 | ✅ Passed |
| test_transformer.py | 3 | ✅ Passed |
| test_integration.py | 2 | ✅ Passed |
| test_training.py | 6 | ✅ Passed |
| test_data.py | 3 | ✅ Passed |
| **Total** | **23** | **92% coverage** |

---

## Troubleshooting

**Port already in use:**
```bash
# Find process using port 8000
sudo lsof -i :8000          # Linux/Mac
netstat -ano | findstr :8000 # Windows
sudo kill -9 <PID>
```

**Docker build failures:**
```bash
docker system prune -a
docker-compose build --no-cache
```

**sklearn version mismatch warning:**
```bash
# Retrain models with current version
python train_and_log.py --retrain
# Or pin version
pip install scikit-learn==1.6.1
```

**Services can't communicate:**
```bash
# Test internal DNS
docker exec callcenterai-agent-1 curl -s http://tfidf:8010/health
docker exec callcenterai-agent-1 nslookup transformer
```

**Scale for production:**
```bash
# Scale specific services
docker-compose up -d --scale tfidf=3 --scale transformer=2
```

---

## License

MIT License — see [LICENSE](LICENSE) file.

---

<div align="center">

Built by [Aziz Mehdi](https://nessmattcash.github.io/) — Final Year Engineering Student at ESPRIT · PFE @ EY Ernst & Young

⭐ Star this repo if you found it useful

</div>
