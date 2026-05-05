# Automated Model Robustness & Fairness Evaluator

A comprehensive MLOps audit platform that ingests machine learning models and datasets to perform **robustness testing**, **fairness bias detection**, **data drift analysis** (PSI + KS test), **prediction confidence analysis**, **feature explainability (SHAP)**, and **RICE-prioritized action planning** — all surfaced through an interactive Streamlit dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Capabilities](#key-capabilities)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Features](#features)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Setup & Running](#setup--running)
- [Understanding the Dashboard](#understanding-the-dashboard)
- [Demo Files](#demo-files)

---

## Project Overview

The **Automated Model Robustness & Fairness Evaluator** is an end-to-end MLOps tool focused entirely on model auditing. It helps data scientists and ML engineers understand how reliable, fair, and stable their models are — before those models are trusted in production.

You provide a trained scikit-learn model (`.pkl`) and a dataset (`.csv` or `.json`). The system runs it through a multi-stage evaluation pipeline and produces an interactive report with a health grade, prioritized action items, and specific remediation advice.

---

## Key Capabilities

### The 6 Evaluation Pillars

| # | Pillar | What It Does |
|---|--------|-------------|
| 1 | **Robustness Testing** | Injects Gaussian noise, simulates adversarial feature shifts, applies boundary stress tests, and runs k-fold cross-validation to measure model stability under real-world conditions. |
| 2 | **Prediction Confidence** | Uses `predict_proba` to measure mean confidence, minimum confidence, and the fraction of low-confidence predictions (< 60%) — flagging models that are uncertain even on clean data. |
| 3 | **Fairness & Bias Detection** | Analyzes predictions across demographic groups using Demographic Parity Difference and Equalized Odds Difference. Computes per-group positive prediction rates. |
| 4 | **Data Drift Analysis** | Combines **Population Stability Index (PSI)** and the **Kolmogorov-Smirnov (KS) test** — a feature is only flagged as drifted when **both** signals agree, significantly reducing false positives. |
| 5 | **Feature Explainability** | Uses SHAP (TreeExplainer with generic fallback) to determine which features most influence the model's decisions globally. |
| 6 | **RICE Priority Engine** | Translates all raw metrics into actionable business intelligence — scoring every detected issue with the RICE framework (Reach × Impact × Confidence / Effort) and providing specific remediation steps. |

---

## Architecture

```
Automated Model Robustness & Fairness Evaluator/
│
├── backend/                        FastAPI REST API + evaluation pipeline
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints.py        POST /evaluate, GET /status, GET /evaluations,
│   │   │                           DELETE /evaluations/{id}, GET /compare
│   │   ├── core/
│   │   │   ├── config.py           Centralized settings (overridable via .env)
│   │   │   ├── database.py         SQLAlchemy engine + session factory
│   │   │   └── logging_config.py   Rotating file + console logger
│   │   ├── models/
│   │   │   └── eval_report.py      SQLAlchemy ORM model (EvaluationTask)
│   │   ├── schemas/
│   │   │   └── eval_report.py      Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── evaluator.py        Pipeline orchestrator
│   │   │   ├── ml_loaders.py       Model/data loading, NaN imputation,
│   │   │   │                       categorical encoding, target validation
│   │   │   ├── robustness.py       Noise, adversarial shift, boundary, CV,
│   │   │   │                       ablation, confidence analysis
│   │   │   ├── fairness.py         Classification & regression fairness
│   │   │   ├── explainability.py   SHAP global feature importance
│   │   │   ├── drift_detection.py  PSI + KS two-sample test per feature
│   │   │   ├── rice_engine.py      RICE scoring (9 issue types)
│   │   │   └── llm_reporter.py     Audit report generator with A–F grade
│   │   └── main.py                 FastAPI app + DB-aware health check
│   ├── scripts/
│   │   └── generate_demo.py        Generates demo model & biased dataset
│   ├── uploads/                    Temporary file storage (auto-cleaned)
│   └── requirements.txt
│
├── frontend/                       Streamlit dashboard
│   ├── app.py                      3 pages: Evaluate | History | Compare
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

---

## How It Works

### Step 1 — Upload & Configure (Streamlit Sidebar)

Upload a `.pkl` model and a `.csv`/`.json` dataset. The sidebar instantly previews the first 5 rows and lists all column names as dropdowns:

- **Target Column** — select which column the model predicts (defaults to last column if left blank)
- **Sensitive Attribute** — select the demographic column to test for bias (e.g. `Gender`, `Race`, `AgeGroup`)
- **Task Type** — `Classification` or `Regression`

### Step 2 — Ingestion & Preprocessing (Backend: `ml_loaders.py`)

Before any evaluation runs, the backend:
1. Validates the target column exists (raises a clear error if not)
2. Drops fully empty columns and duplicate rows
3. Imputes missing numeric values with the **column median**
4. Imputes missing categorical values with the **column mode**
5. Label-encodes all string/category columns so sklearn models can consume them

This means real-world messy datasets work out of the box without manual cleaning.

### Step 3 — Parallel Evaluation Pipeline (`evaluator.py`)

The orchestrator sends the cleaned data to six evaluation engines simultaneously. Each engine is wrapped in an independent `try/except` — a failure in one (e.g. SHAP timing out) does not crash the others.

| Engine | Key Metrics |
|--------|------------|
| **Robustness** | Noise stability %, adversarial resilience %, boundary resilience %, CV mean ± std, feature ablation impact |
| **Confidence** | Mean confidence, min confidence, low-confidence fraction (classifiers only) |
| **Fairness** | Demographic Parity Difference, Equalized Odds Difference, per-group positive rates |
| **SHAP** | Top-15 features by mean absolute SHAP value |
| **Drift** | PSI per feature, KS statistic + p-value per feature, combined severity classification |
| **RICE** | 9 issue types scored and ranked by business priority |

### Step 4 — RICE Prioritization (`rice_engine.py`)

Nine issue types are evaluated and scored:

1. **Model Sensitive to Noise** — stability < 90%
2. **Low Adversarial Resilience** — resilience < 85%
3. **Boundary Stress Vulnerability** — boundary resilience < 85%
4. **Cross-Validation Instability** — CV std > 0.05
5. **Low Prediction Confidence** — > 30% of predictions below 60% confidence
6. **Demographic Parity Bias** — |DPD| > 0.1
7. **Equalized Odds Violation** — |EOD| > 0.1
8. **Sensitive Attribute Is Top SHAP Feature** — protected attribute ranks #1 by SHAP importance *(new)*
9. **Data Distribution Drift** — PSI + KS both exceed thresholds

### Step 5 — Health Grade & Report (`llm_reporter.py`)

A composite score (0–100) is computed from all pillars with weighted contributions:
- Robustness: 40%
- Fairness: 30%
- CV Stability: 20%
- Drift: 10%

The score maps to a letter grade:

| Score | Grade |
|-------|-------|
| ≥ 90 | **A** |
| ≥ 80 | **B** |
| ≥ 65 | **C** |
| ≥ 50 | **D** |
| < 50 | **F** |

### Step 6 — Storage & Presentation

The complete evaluation package is persisted to SQLite. The Streamlit dashboard renders the grade badge, metrics, RICE table, SHAP chart, drift chart, CV chart, ablation chart, confidence metrics, and group demographics chart.

---

## Features

| Category | Capabilities |
|----------|-------------|
| **Data Preprocessing** | NaN imputation (median/mode), categorical encoding, duplicate removal, target column validation |
| **Robustness** | Gaussian noise injection, adversarial feature shift (1.5σ), boundary stress testing, k-fold CV stability, feature ablation |
| **Confidence** | `predict_proba`-based mean/min confidence, low-confidence fraction, confidence threshold flagging |
| **Fairness** | Demographic Parity Difference, Equalized Odds Difference, per-group positive prediction rates, regression group-disparity |
| **Drift Detection** | PSI per feature, KS two-sample test per feature, dual-signal flagging (both must agree), severity by % of drifted features |
| **Explainability** | SHAP global feature importance (TreeExplainer with generic fallback), top 15 features |
| **RICE Engine** | 9 rule types, severity levels (critical / warning / info), per-issue remediation advice, SHAP-fairness cross-check |
| **Reporting** | Composite health grade (A–F), PDF & Markdown export |
| **History & Compare** | List all evaluations, delete, radar-chart comparison of up to 5 models |
| **Dashboard** | Target column shown, dataset preview, KS test results table, confidence analysis section |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/evaluate/` | Submit model + dataset for evaluation |
| `GET` | `/api/v1/status/{id}` | Poll evaluation status and retrieve full results |
| `GET` | `/api/v1/evaluations/` | List all evaluations, newest first (paginated) |
| `DELETE` | `/api/v1/evaluations/{id}` | Delete an evaluation record |
| `GET` | `/api/v1/compare/?ids=1,2,3` | Compare up to 5 evaluations side-by-side |
| `GET` | `/health` | DB-aware health check — returns `"degraded"` if database is unreachable |

### POST `/api/v1/evaluate/` — Form Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_file` | File | ✅ | Trained scikit-learn model (`.pkl`) |
| `dataset_file` | File | ✅ | Evaluation dataset (`.csv` or `.json`) |
| `model_name` | string | ✅ | Display name for the model |
| `dataset_name` | string | ✅ | Display name for the dataset |
| `task_type` | string | ✅ | `"classification"` or `"regression"` |
| `target_column` | string | ⬜ | Column to predict. Falls back to last column if empty. |
| `sensitive_attr` | string | ⬜ | Column for fairness analysis (e.g. `Gender`) |

---

## Configuration

All settings live in `backend/app/core/config.py` and can be overridden via a `.env` file in `backend/`:

```env
# backend/.env  (example)
DATABASE_URL=sqlite:///./evaluator.db
MAX_FILE_SIZE_MB=200
CV_FOLDS=5
NOISE_LEVEL=0.1
PSI_DRIFT_THRESHOLD=0.2
KS_DRIFT_THRESHOLD=0.05
SHAP_SAMPLE_SIZE=100
LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:8501"]
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./evaluator.db` | Database connection string |
| `MAX_FILE_SIZE_MB` | `200` | Max upload file size |
| `CV_FOLDS` | `5` | Cross-validation fold count |
| `NOISE_LEVEL` | `0.1` | Gaussian noise level (fraction of std) |
| `PSI_DRIFT_THRESHOLD` | `0.2` | PSI threshold for drift flagging |
| `KS_DRIFT_THRESHOLD` | `0.05` | KS test p-value threshold for drift flagging |
| `SHAP_SAMPLE_SIZE` | `100` | Max rows sampled for SHAP computation |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins — lock this in production |

---

## Setup & Running

### Prerequisites

- Python 3.10+
- A virtual environment (recommended)

### 1. Create & Activate Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 3. (Optional) Generate Demo Model & Dataset

Generates a RandomForest classifier trained on synthetic HR attrition data with injected demographic bias — ready to test all evaluation pillars:

```bash
python backend/scripts/generate_demo.py
```

This creates `demo_model.pkl` and `demo_dataset.csv` in the project root.

### 4. Run the Application

Open **two separate terminals** with the virtual environment activated:

**Terminal 1 — Backend:**
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | API root |
| `http://localhost:8000/docs` | Interactive Swagger UI |
| `http://localhost:8000/health` | DB-aware health check |

**Terminal 2 — Frontend:**
```bash
cd frontend
streamlit run app.py
```

| URL | Description |
|-----|-------------|
| `http://localhost:8501` | Streamlit dashboard |

> **Development note:** Add `--reload` to the backend command during development so Uvicorn auto-restarts on file changes: `python -m uvicorn app.main:app --reload`

---

## Understanding the Dashboard

### New Evaluation Page

**Sidebar:**
- Upload your `.pkl` model and `.csv`/`.json` dataset
- A **live 5-row dataset preview** appears immediately after upload
- **Target Column** is a dropdown populated from your dataset's actual columns (no more guessing)
- **Sensitive Attribute** is also a dropdown — select the column to test for demographic bias
- Select Task Type (`Classification` / `Regression`) and click **Run Evaluation**

**Results Dashboard (after evaluation completes):**

| Section | What It Shows |
|---------|--------------|
| **Health Grade Badge** | A–F letter grade + composite score out of 100 |
| **Top Metrics Row** | Noise Stability %, Adversarial Resilience %, Fairness Score %, Boundary Resilience % |
| **Prediction Confidence** | Mean confidence, min confidence, % of low-confidence predictions (classifiers only) |
| **RICE Priority Table** | All detected issues ranked by business priority with severity badges and remediation steps |
| **Feature Importance (SHAP)** | Horizontal bar chart of top-15 global SHAP values |
| **AI Audit Report** | Plain-English executive summary with grade explanation and mitigation strategies — exportable as `.md` or `.pdf` |
| **Data Drift Analysis** | PSI bar chart per feature with threshold line + collapsible KS test results table |
| **Cross-Validation Stability** | Per-fold score bar chart with mean line |
| **Feature Ablation** | Performance drop when each feature is removed |
| **Dataset Demographics** | Distribution of target by sensitive attribute (if provided) |

### History Page

Lists all past evaluations with status, robustness %, fairness %, duration, and the target column used. Supports **View Full Results** (loads any completed evaluation back into the dashboard) and **Delete**.

### Compare Page

Select 2–5 completed evaluations to compare side-by-side:
- **Radar chart** across 5 dimensions: Stability, Adversarial Resilience, Boundary Resilience, Fairness, CV Mean
- **Comparison table** including target column, mean prediction confidence, and duration

---

## Demo Files

Run `python backend/scripts/generate_demo.py` to generate:

- **`demo_model.pkl`** — A RandomForest classifier trained on synthetic HR attrition data
- **`demo_dataset.csv`** — 1,500 rows with injected demographic bias for testing the full fairness pipeline

**Suggested demo settings:**
- Task Type: `Classification`
- Target Column: `Attrition` (or whichever the last column is)
- Sensitive Attribute: `Gender` (or the injected bias column)

> These files are excluded from git via `.gitignore`. Re-generate them locally with the script above.