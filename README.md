# 🛡️ Credit Card Fraud Detection — Production MLOps Platform

> A full end-to-end MLOps system built on **zero budget** using only open-source tools and free tiers.  
> Every component a real ML engineer uses in production — data versioning, feature store, experiment tracking, model registry, A/B testing, drift detection, CI/CD — all wired together and live.



---

## 🔗 Live Links

| Platform | Link | What's there |
|---|---|---|
| 🧪 Experiment Tracking | [DagsHub MLflow Dashboard](https://dagshub.com/Hruthikck28/credit-card-fraud-mlops/experiments) | 10+ Optuna runs, ROC-AUC scores, hyperparameter comparisons |
| 🤗 Live Demo | [Hugging Face Spaces](https://huggingface.co/spaces/hruthikck/credit-card-fraud-mlops) | Interactive fraud prediction UI |
| 📦 Data Versioning | [DagsHub DVC Remote](https://dagshub.com/Hruthikck28/credit-card-fraud-mlops) | Dataset tracked with DVC, 150MB creditcard.csv versioned |

---

## 🏗️ System Architecture

This project implements **7 production MLOps layers** — the same stack used by ML teams at real companies, rebuilt entirely for free.

```
Raw CSV (284K transactions)
       │
       ▼
┌─────────────────────┐
│  Great Expectations  │  ← Data quality gates (nulls, value ranges, schema)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│   DVC + DagsHub     │  ← Dataset versioning, reproducible data lineage
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│   Feast Feature      │  ← Parquet offline store + Redis online store
│      Store           │     Zero training-serving skew by design
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Optuna + MLflow     │  ← Hyperparameter tuning, 10+ experiments logged
│  (hosted DagsHub)   │     to hosted cloud dashboard
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Model Registry      │  ← Best run auto-promoted to Staging via MLflowClient
│  (fraud-predictor)  │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  FastAPI + Feast     │  ← Loads model from Registry, fetches features from
│  Serving API         │     Redis at prediction time. No skew.
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  A/B Router          │  ← 80/20 traffic split, 2-proportion Z-test for
│  + Evidently Drift   │     statistical significance + HTML drift report
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  GitHub Actions      │  ← CI pipeline: spins up Redis, runs pytest on
│  CI/CD               │     FastAPI endpoint on every push to main
└─────────────────────┘
```

<img width="1440" height="811" alt="Screenshot 2026-05-20 at 12 25 58 AM" src="https://github.com/user-attachments/assets/dc5ec0a8-2d8f-4c85-987d-09a6eb26fff2" />


---

## 🧰 Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Data Versioning | DVC + DagsHub | Git for data — every dataset version reproducible |
| Data Validation | Great Expectations | Schema gates block bad data before training |
| Feature Store | Feast (local) | Eliminates training-serving skew via Parquet/Redis |
| Experiment Tracking | MLflow on DagsHub | Hosted cloud dashboard, shareable experiment URLs |
| Hyperparameter Tuning | Optuna | Bayesian search over 10 trials, logged to MLflow |
| Model Registry | MLflow Model Registry | `fraud-predictor` model promoted to Staging |
| Model Serving | FastAPI | Loads model from Registry, features from Redis |
| A/B Testing | Custom ABRouter + statsmodels | Deterministic hash routing, Z-test significance |
| Drift Detection | Evidently | HTML drift report comparing reference vs live data |
| Monitoring Infra | Grafana + Prometheus | Docker Compose stack ready |
| CI/CD | GitHub Actions | Runs pytest with ephemeral Redis on every push |
| Infrastructure | Docker Compose | 5-service stack (Airflow, Postgres, Redis, Grafana, Prometheus) |

**Cost: ₹0 / $0.00** — 100% open source + free tiers.

---

## 📁 Project Structure

```
credit-card-fraud-mlops/
│
├── src/
│   ├── data/
│   │   ├── validate.py          # Great Expectations quality gates
│   │   └── prep_feast.py        # Parquet prep for Feast offline store
│   │
│   ├── models/
│   │   ├── train.py             # Optuna tuning — 10 trials → DagsHub MLflow
│   │   ├── prod_train.py        # Production training with best params
│   │   └── register.py          # Auto-promotes best run to Staging
│   │
│   ├── serving/
│   │   ├── api.py               # FastAPI endpoint — loads model + Feast features
│   │   └── ab_router.py         # 80/20 traffic split + statistical significance test
│   │
│   └── monitoring/
│       └── check_drift.py       # Evidently drift report generator
│
├── feature_repo/
│   ├── fraud_features.py        # Feast FeatureView definition
│   ├── feature_store.yaml       # Feast config (Redis online + Parquet offline)
│   └── data/
│       └── creditcard_features.parquet   # Offline feature store (76MB)
│
├── data/
│   └── raw/
│       └── creditcard.csv.dvc   # DVC pointer — actual file on DagsHub remote
│
├── tests/
│   └── test_api.py              # FastAPI test with mocked Feast store
│
├── .github/
│   └── workflows/
│       └── mlops_ci.yml         # CI/CD: Redis service + pytest on push
│
├── docker-compose.yml           # Airflow, Postgres, Redis, Grafana, Prometheus
├── drift_report.html            # Evidently drift report output
└── .dvc/config                  # DagsHub DVC remote config
```

---

## ⚙️ How to Run Locally

### 1. Clone & Setup

```bash
git clone https://github.com/Hruthikck28/credit-card-fraud-mlops.git
cd credit-card-fraud-mlops
pip install -r requirements.txt
```

### 2. Pull the Data

```bash
# Set up your DagsHub token in .env
echo "DAGSHUB_TOKEN=your_token_here" > .env

# Pull dataset from DagsHub DVC remote
dvc pull
```

### 3. Start the Infrastructure

```bash
docker-compose up -d
# Starts: Redis (6379), Postgres (5432), Grafana (3000), Prometheus (9091), Airflow (8080)
```

### 4. Run the Pipeline

```bash
# Step 1 — Validate data quality
python src/data/validate.py

# Step 2 — Prep features for Feast
python src/data/prep_feast.py

# Step 3 — Apply Feast feature definitions
cd feature_repo && feast apply && feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
cd ..

# Step 4 — Tune hyperparameters (logs 10 runs to DagsHub)
python src/models/train.py

# Step 5 — Train production model & register to Staging
python src/models/prod_train.py

# Step 6 — Start the prediction API
uvicorn src.serving.api:app --reload

# Step 7 — Generate drift report
python src/monitoring/check_drift.py
# Opens drift_report.html
```

### 5. Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 42}'
```

```json
{
  "transaction_id": 42,
  "fraud_probability": 0.0312,
  "is_fraud": false
}
```

### 6. Run Tests

```bash
pytest tests/test_api.py -v
```

---

## 🧪 Experiments on DagsHub

<img width="1440" height="811" alt="Screenshot 2026-05-20 at 12 27 17 AM" src="https://github.com/user-attachments/assets/7f172899-67d0-4e7a-8ff1-935487460e8f" />


The `train.py` script runs **Optuna Bayesian hyperparameter search** over `GradientBoostingClassifier`, logging every trial to the hosted DagsHub MLflow server:

```python
params = {
    "n_estimators": trial.suggest_int("n_estimators", 50, 150),
    "max_depth": trial.suggest_int("max_depth", 3, 6),
    "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
}
```

Best run params used in production: `n_estimators=55`, `max_depth=5`, `learning_rate=0.153`

---

## 🏪 Feature Store (Feast)

<img width="832" height="114" alt="Screenshot 2026-05-20 at 12 41 38 AM" src="https://github.com/user-attachments/assets/55798d51-6f0b-4682-b5ef-c988cf096ad8" />

The feature store eliminates **training-serving skew** — the single most common bug in production ML systems:

- **Offline store** (Parquet) — used during training via `prep_feast.py`
- **Online store** (Redis) — used during inference via `api.py`
- Same `fraud_features` FeatureView used by both paths → guaranteed identical features

```yaml
# feature_store.yaml
online_store:
  type: redis
  connection_string: localhost:6379
offline_store:
  type: file   # Parquet — free, no cloud needed
```

---

## 🔀 A/B Testing & Statistical Significance

<img width="662" height="146" alt="Screenshot 2026-05-20 at 12 34 44 AM" src="https://github.com/user-attachments/assets/e4434df6-b6d9-4d87-9312-6929b7f895c1" />




`ab_router.py` implements a production-grade A/B routing system:

- **80% traffic → Champion model**, **20% → Challenger**
- Hash-based deterministic routing (same transaction always goes to same variant)
- **2-proportion Z-test** to determine when the challenger is statistically better
- Simulated 5,000 transactions: Challenger shows +8% uplift, p < 0.05 → safe to promote

```python
def get_variant(self, transaction_id: str) -> str:
    h = int(hashlib.md5(f"fraud_exp_1:{transaction_id}".encode()).hexdigest(), 16)
    return "challenger" if h % 100 < 20 else "champion"
```

---

## 📊 Data Drift Detection
<img width="1036" height="606" alt="Screenshot 2026-05-20 at 12 35 27 AM" src="https://github.com/user-attachments/assets/c6aae9e5-9840-4139-aa28-fe67b71cfdab" />

`check_drift.py` uses **Evidently** to generate a full HTML monitoring report comparing:

- **Reference data** — original training distribution
- **Current data** — simulated production data with `Amount × 5.0` spike (realistic fraud pattern shift)

```bash
python src/monitoring/check_drift.py
# → drift_report.html  (opens in browser, shows per-feature drift)
```

---

## 🔄 CI/CD Pipeline

<img width="1433" height="798" alt="Screenshot 2026-05-20 at 12 36 57 AM" src="https://github.com/user-attachments/assets/bad02a3f-5fad-4518-a2f6-c215bbfe4a98" />

Every push to `main` triggers `.github/workflows/mlops_ci.yml`:

1. Spins up an **ephemeral Redis container** as a GitHub Actions service
2. Installs all dependencies (FastAPI, MLflow, Feast, scikit-learn...)
3. Runs `pytest tests/test_api.py` — tests the `/predict` endpoint with a mocked Feast store
4. Fails the build if the API breaks

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - 6379:6379
```

---

## 📦 Data Versioning with DVC



The 150MB `creditcard.csv` dataset is tracked with DVC and stored on DagsHub's free 10GB remote:

```bash
dvc add data/raw/creditcard.csv
dvc push  # → uploads to https://dagshub.com/Hruthikck28/credit-card-fraud-mlops.dvc
```

The `.dvc` pointer file is committed to Git — anyone who clones this repo can `dvc pull` to get the exact same dataset version.

---

## 🐳 Infrastructure (Docker Compose)

```bash
docker-compose up -d
```

Starts 5 services locally:

| Service | Port | Purpose |
|---|---|---|
| Airflow | 8080 | Pipeline orchestration |
| PostgreSQL | 5432 | Feature store metadata |
| Redis | 6379 | Feast online store |
| Grafana | 3000 | Monitoring dashboard |
| Prometheus | 9091 | Metrics scraping |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Algorithm | GradientBoostingClassifier |
| ROC-AUC | See DagsHub experiments |
| Dataset | Kaggle Credit Card Fraud (284K transactions, 0.17% fraud) |
| Training set | 5,492 samples (all 492 frauds + 5,000 sampled legit) |
| Features used | Amount, V1, V2, V3, V4, V5 |

---

## 💡 What Makes This Different

Most "MLOps projects" on GitHub are just a Jupyter notebook with `mlflow.log_metric()` added. This project implements the **full production pipeline**:

| Feature | Most GitHub projects | This project |
|---|---|---|
| Data versioning | ❌ | ✅ DVC + DagsHub remote |
| Data validation | ❌ | ✅ Great Expectations quality gates |
| Feature store | ❌ | ✅ Feast with Redis online store |
| Hosted experiment tracking | ❌ | ✅ DagsHub MLflow (public URL) |
| Hyperparameter tuning | Sometimes | ✅ Optuna Bayesian search |
| Model registry + staging | ❌ | ✅ MLflow Model Registry |
| API loads from registry | ❌ | ✅ FastAPI pulls from DagsHub Staging |
| A/B testing | ❌ | ✅ Hash router + Z-test significance |
| Drift detection | ❌ | ✅ Evidently HTML report |
| CI/CD with real services | ❌ | ✅ GitHub Actions + Redis service |
| Docker infra | ❌ | ✅ 5-service Compose stack |

---

## 🙏 Built With

- [DagsHub](https://dagshub.com) — Free hosted MLflow + DVC remote
- [MLflow](https://mlflow.org) — Experiment tracking & model registry
- [Feast](https://feast.dev) — Feature store
- [Evidently](https://www.evidentlyai.com) — Data drift monitoring
- [Optuna](https://optuna.org) — Hyperparameter optimization
- [FastAPI](https://fastapi.tiangolo.com) — Model serving
- [DVC](https://dvc.org) — Data version control
- [Great Expectations](https://greatexpectations.io) — Data validation
- [Docker Compose](https://docs.docker.com/compose/) — Local infrastructure

---
*Developed by Hruthik charan*

*Built as part of a zero-budget MLOps learning project. Every tool here is production-grade and free.*
