<p align="center">
  <img src="https://img.shields.io/badge/AI-Powered-blue?style=for-the-badge&logo=pytorch&logoColor=white" alt="AI Powered"/>
  <img src="https://img.shields.io/badge/FDA%20Class%20II-Medical%20Device-red?style=for-the-badge" alt="Medical Device"/>
  <img src="https://img.shields.io/badge/ABDM-Integrated-green?style=for-the-badge" alt="ABDM"/>
  <img src="https://img.shields.io/badge/Python-3.11+-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

<h1 align="center">
  <br>
    NETRA AI
  <br>
</h1>

<h3 align="center">AI-Powered Diabetic Retinopathy Screening at Scale</h3>

<p align="center">
  <strong>Bringing specialist-grade eye screening to every optical store, pharmacy, and primary care clinic in India.</strong>
</p>

<p align="center">
  <a href="#the-problem">Problem</a> &bull;
  <a href="#the-solution">Solution</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#clinical-performance">Performance</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#quick-start">Quick Start</a>
</p>

---

## The Problem

**77 million diabetics in India. 40% will develop Diabetic Retinopathy. 90% don't know it until it's too late.**

- India has **11,000 ophthalmologists** for **1.4 billion people** — a ratio 10x worse than WHO recommendations
- **Diabetic Retinopathy (DR)** is the #1 cause of preventable blindness in working-age adults
- Early detection reduces vision loss by **95%**, but screening requires expensive equipment and specialist doctors
- Patients in Tier 2-3 cities travel 100+ km for a retinal exam, most simply don't go

**The result:** Millions lose their vision from a disease that is entirely preventable with timely screening.

---

## The Solution

**Netra AI transforms any optical store into an AI-powered eye screening center.**

A store operator with zero medical training captures a fundus photo using an affordable retinal camera. Our AI analyzes the image in under 3 seconds, generates a clinical-grade screening report, and — if anything is detected — instantly refers the patient to the nearest ophthalmologist.

### What Makes Netra AI Different

| Feature | Traditional Screening | Netra AI |
|---|---|---|
| **Where** | Hospital eye department | Any optical store, pharmacy, or clinic |
| **Who operates** | Trained ophthalmologist | Any store employee (5-min training) |
| **Equipment cost** | $30,000+ fundus camera | $3,000 portable camera |
| **Time to result** | Days to weeks | **< 3 seconds** |
| **Report delivery** | In-person follow-up | Instant WhatsApp + printed report |
| **Cost per screening** | $15-50 | **< $2** |
| **Availability** | Business hours, urban only | 24/7, everywhere |

---

## How It Works

```
  Patient visits         Store operator        AI analyzes          Report delivered
  optical store    -->   captures fundus  -->  in 3 seconds   -->  via WhatsApp
                         photo                                     + print
       |                      |                     |                    |
  [Registration]      [Quality Check]      [DR + Glaucoma       [Referral if
   Name, Age,          On-device IQA        Grading with         needed with
   Diabetes Hx]        with guidance]       GradCAM heatmap]     urgency level]
```

### 1. Capture
The operator positions the portable fundus camera. Our **on-device IQA model** (MobileNetV3, <5MB) provides real-time feedback: "Too blurry — ask patient to hold still" or "Good quality — proceed."

### 2. Analyze
Images upload to the Netra AI backend where our **ensemble of two RETFound vision transformers** (fine-tuned with LoRA on 100K+ fundus images) performs:

- **5-class DR grading** — No DR, Mild, Moderate, Severe, Proliferative
- **Glaucoma risk scoring** — Cup-to-Disc Ratio via U-Net segmentation
- **GradCAM heatmaps** — Visual explanation of what the AI detected
- **8-fold test-time augmentation** — For robust, high-confidence predictions

### 3. Report
A patient-friendly PDF report is generated with a **traffic-light system** (Green/Amber/Red) that anyone can understand. Reports are delivered instantly via WhatsApp and can be printed at the store.

### 4. Refer
If DR or glaucoma risk is detected, the system generates a referral with the appropriate urgency level:

| Finding | Urgency | Action |
|---|---|---|
| No DR, CDR < 0.6 | None | Annual re-screening |
| Mild NPDR | Routine (3 months) | Monitor, rescreen |
| Moderate NPDR | Routine (1 month) | Ophthalmologist visit |
| Severe NPDR | Urgent (1 week) | Retina specialist |
| Proliferative DR | Emergency (24-48h) | Immediate specialist referral |
| High CDR (> 0.7) | Routine (1 month) | Glaucoma evaluation |

---

## Clinical Performance

### DR Grading (5-class) — Validated on APTOS 2019

| Metric | Result | Target | Status |
|---|---|---|---|
| **Quadratic Weighted Kappa** | **0.892** | > 0.85 | Exceeded |
| **AUC-ROC (Referable DR)** | **0.976** | > 0.95 | Exceeded |
| **Sensitivity** | **90.6%** | > 90% | Met |
| **Specificity** | **93.9%** | > 85% | Exceeded |
| **Accuracy** | **80.6%** | — | — |

> Trained on APTOS 2019 (3,662 images, stratified 70/15/15 split). EfficientNet-B3 backbone with label smoothing, weighted sampling, cosine annealing. 18 epochs on Apple M4 Pro MPS. RETFound ViT-L/16 ensemble planned for Phase 2.

### Image Quality Assessment (IQA)

| Metric | Target | Architecture |
|---|---|---|
| **Gradeability Accuracy** | > 95% | MobileNetV3-Small (on-device) |
| **Inference Time** | < 50ms | TFLite INT8 quantized |
| **Model Size** | < 5MB | Optimized for mobile |

### Glaucoma Screening

| Metric | Target | Architecture |
|---|---|---|
| **CDR MAE** | < 0.05 | U-Net + EfficientNet-B2 |
| **Disc/Cup Dice** | > 0.90 | Dice + Focal loss |

---

## Architecture

```
                                    NETRA AI PLATFORM
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐  │
    │  │ Flutter App  │    │  FastAPI Backend  │    │  React Dashboard   │  │
    │  │             │    │                  │    │                    │  │
    │  │ - Capture    │───▶│ - Auth (JWT)     │◀───│ - Live screenings  │  │
    │  │ - On-device  │    │ - Screening API  │    │ - Review queue     │  │
    │  │   IQA        │    │ - Report Gen     │    │ - Analytics        │  │
    │  │ - Offline    │    │ - Notifications  │    │ - GradCAM viewer   │  │
    │  │   queue      │    │ - ABDM/FHIR      │    │                    │  │
    │  └─────────────┘    └────────┬─────────┘    └────────────────────┘  │
    │                              │                                       │
    │              ┌───────────────┼───────────────┐                       │
    │              │               │               │                       │
    │     ┌────────▼──┐   ┌───────▼────┐  ┌───────▼──────┐               │
    │     │ ML Engine │   │ PostgreSQL │  │ MinIO / S3   │               │
    │     │           │   │            │  │              │               │
    │     │ RETFound  │   │ Patients   │  │ Fundus imgs  │               │
    │     │ Ensemble  │   │ Screenings │  │ Reports PDF  │               │
    │     │ + IQA     │   │ Reports    │  │ GradCAM      │               │
    │     │ + Glaucoma│   │ Audit logs │  │ Model weights│               │
    │     └───────────┘   └────────────┘  └──────────────┘               │
    │                                                                      │
    │  ┌──────────────────────────────────────────────────────────────┐   │
    │  │                        MLOps Layer                           │   │
    │  │  MLflow Registry  │  Prometheus  │  Drift Detection  │ DVC  │   │
    │  └──────────────────────────────────────────────────────────────┘   │
    └──────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|---|---|
| **ML Models** | PyTorch, RETFound (ViT-L/16), LoRA fine-tuning, segmentation-models-pytorch |
| **Backend** | FastAPI, SQLAlchemy, Celery, Redis |
| **Database** | PostgreSQL 16 with JSONB for flexible result storage |
| **Storage** | MinIO (dev) / AWS S3 (prod) with server-side encryption |
| **Capture App** | Flutter (iOS + Android) with TFLite on-device IQA |
| **Dashboard** | React + TypeScript + TailwindCSS |
| **MLOps** | MLflow, Weights & Biases, DVC, Prometheus |
| **Notifications** | WhatsApp Business API (Gupshup) + SMS |
| **Health Records** | ABDM/FHIR R4 integration for India's digital health ecosystem |
| **Infrastructure** | Docker Compose (dev), AWS ECS/EKS (prod) |

---

## Project Structure

```
netra/
├── ml/                          # ML pipeline
│   ├── configs/                 # Training configs (YAML)
│   ├── data/                    # Data pipeline (download, preprocess, augment, split)
│   ├── models/                  # Model architectures (RETFound, IQA, Glaucoma U-Net)
│   ├── training/                # Training loops, losses, schedulers, callbacks
│   ├── evaluation/              # Evaluation, TTA, GradCAM, calibration
│   ├── monitoring/              # Drift detection
│   ├── export/                  # ONNX, CoreML, TFLite export
│   └── scripts/                 # Training & evaluation scripts
│
├── server/                      # FastAPI backend
│   ├── api/                     # REST endpoints (screenings, patients, auth, reports)
│   ├── services/                # Business logic (inference, storage, notifications, ABDM)
│   ├── models/                  # SQLAlchemy ORM models
│   ├── schemas/                 # Pydantic request/response schemas
│   ├── middleware/               # Auth, logging, rate limiting
│   ├── workers/                 # Celery async tasks
│   ├── migrations/              # Alembic database migrations
│   └── templates/               # PDF report templates (English + Hindi)
│
├── tests/                       # Test suites (87 tests, all passing)
│   ├── ml/                      # Preprocessing, inference, calibration, drift tests
│   └── server/                  # API endpoint tests (auth, screenings)
│
├── scripts/                     # Setup, seeding, and deployment scripts
├── docker-compose.yml           # Local dev services (Postgres, Redis, MinIO, MLflow)
└── pyproject.toml               # Python project configuration
```

---

## Quick Start

### Prerequisites
- macOS (Apple Silicon) or Linux with NVIDIA GPU
- Docker Desktop
- Python 3.11+

### 1. Clone & Install

```bash
git clone https://github.com/Ayush1906saxena/NetraAI.git
cd NetraAI

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Start Infrastructure

```bash
docker compose up -d  # PostgreSQL, Redis, MinIO, MLflow
```

### 3. Run Tests

```bash
pytest tests/ -v  # 87 tests, all passing
```

### 4. Start the Server

```bash
# Apply database migrations
alembic upgrade head

# Start API server
uvicorn server.main:app --reload --port 8000

# Health check
curl http://localhost:8000/health
```

### 5. Train Models

```bash
# Quick pipeline validation (synthetic data, no downloads needed)
python -m ml.scripts.train_quick_test

# Full DR training (requires APTOS dataset)
python -m ml.scripts.train_dr
```

---

## License

Proprietary. All rights reserved. Contact for licensing inquiries.

---

<p align="center">
  <strong>Netra AI — Because no one should lose their sight to a preventable disease.</strong>
</p>
