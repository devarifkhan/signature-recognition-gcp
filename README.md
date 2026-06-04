# Signature Recognition

#### Language and Libraries

<p>
<a><img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" alt="python"/></a>
<a><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="pytorch"/></a>
<a><img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="fastapi"/></a>
<a><img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy"/></a>
<a><img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="opencv"/></a>
<a><img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)" alt="docker"/></a>
<a><img src="https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white" alt="gcp"/></a>
</p>

## Problem Statement

Signature recognition involves building a system that can automatically distinguish between **genuine** signatures (signed by the actual person) and **forged** signatures (imitated by someone else). The system must work robustly in the presence of noise and natural variation in handwriting style.

## Solution

A FastAPI-based web application that serves a ResNet-34 deep learning model fine-tuned for binary classification — **genuine vs forged**. The system supports two environments:

- **Dev mode** — runs locally, uses data from the `data/` folder
- **Prod mode** — runs on GCP, pulls dataset from Google Cloud Storage

## Dataset

**CEDAR Signature Database** — an offline signature verification dataset.

- 55 individuals, each with 24 genuine signatures → 1,320 genuine total
- 1,320 forged signatures (others imitating each person's signature)
- Scanned at 300 dpi grayscale, preprocessed with noise removal and slant normalization
- Folder structure: `001/` = genuine, `001_forg/` = forged

## Model

**ResNet-34** (pretrained on ImageNet, fine-tuned for binary classification)

- 34 layers: 33 convolutional + 1 fully connected
- Final FC layer replaced: outputs 2 classes (genuine / forged)
- Residual connections allow deeper training without degradation

## Project Structure

```
├── app.py                      # FastAPI application (dev + prod)
├── main.py                     # Run training pipeline from CLI
├── config/config.yaml          # Dataset, training hyperparameters
├── .env                        # ENV_MODE=dev or prod
├── data/                       # Local dataset (dev only)
│   └── signature-recog/
│       ├── 001/                # Genuine signatures
│       ├── 001_forg/           # Forged signatures
│       └── ...
├── model/                      # Saved trained model
│   └── signature_model.pkl
├── templates/
│   └── index.html              # Web UI
├── src/
│   ├── components/
│   │   ├── data_ingestion.py   # Load from GCS (prod) or local (dev)
│   │   ├── data_transformation.py  # Dataset, transforms, DataLoaders
│   │   └── model_trainer.py    # ResNet-34 training loop
│   ├── pipeline/
│   │   └── training.py         # Orchestrates all pipeline steps
│   ├── entity/
│   │   ├── config_entity.py    # Config dataclasses
│   │   └── artifact_entity.py  # Artifact dataclasses
│   ├── configurations/
│   │   └── gcloud_syncer.py    # gsutil wrapper for GCS
│   ├── constants/__init__.py   # Paths, device, env mode
│   ├── utils/main_utils.py     # save/load object, YAML reader
│   ├── logger/__init__.py      # Timestamped file logging
│   └── exception/__init__.py   # Custom exception with line info
├── Dockerfile                  # Production container (GCP)
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI with interactive feature cards |
| GET | `/health` | API status + model readiness check |
| POST | `/train` | Run full training pipeline |
| POST | `/predict` | Upload signature image → genuine / forged + confidence |
| GET | `/docs` | Swagger UI (dev mode only) |

## How to Run (Dev)

### Step 1 — Clone the repository
```bash
git clone <repository-url>
cd signature-recognition-gcp
```

### Step 2 — Create and activate virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

> Windows users: install **Microsoft Visual C++ Redistributable 2022 (x64)** before installing torch.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 4 — Add dataset
Place the CEDAR dataset inside the `data/` folder:
```
data/
└── signature-recog/
    ├── 001/
    ├── 001_forg/
    └── ...
```

### Step 5 — Configure environment
In `.env`, set:
```
ENV_MODE=dev
```

### Step 6 — Run the server
```bash
python app.py
```

Open `http://127.0.0.1:8000` in your browser. Click **Train** to train the model, then **Predict** to classify a signature image.

## How to Run (Prod / GCP)

### Step 1 — Configure environment
In `.env`, set:
```
ENV_MODE=prod
```
Make sure your GCS bucket (`sig-recog`) contains `dataset.zip`.

### Step 2 — Authenticate GCloud
```bash
gcloud auth login
gcloud auth application-default login
```

### Step 3 — Build and run Docker container
```bash
docker build -t signature-api .
docker run -d -p 8080:8080 signature-api
```

Open `http://0.0.0.0:8080`. Swagger docs are disabled in prod mode.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | PyTorch, ResNet-34 |
| API | FastAPI, Uvicorn |
| UI | Bootstrap 5, Vanilla JS |
| Storage | Google Cloud Storage |
| Container | Docker |
| Infrastructure | GCP (Cloud Run / Compute Engine) |

## Infrastructure Required (Prod)

1. Google Cloud Storage — dataset and model storage
2. Google Compute Engine or Cloud Run — server hosting
3. Google Artifact Registry — Docker image registry
