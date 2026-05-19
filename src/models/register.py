import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import dagshub

print("Loading secure environment variables...")
load_dotenv()

# Explicit Auth
os.environ["MLFLOW_TRACKING_USERNAME"] = "Hruthikck28"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("DAGSHUB_TOKEN")

print("Initializing DagsHub connection...")
dagshub.init(repo_owner='Hruthikck28', repo_name='credit-card-fraud-mlops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Hruthikck28/credit-card-fraud-mlops.mlflow")

client = MlflowClient()
experiment_name = "fraud-tuning-v3"

print("Searching for the best Optuna run...")
# 2. Find the experiment ID
experiment = client.get_experiment_by_name(experiment_name)

# 3. Query MLflow to get the run with the highest ROC-AUC score
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"],
    max_results=1
)

best_run = runs[0]
best_run_id = best_run.info.run_id
best_auc = best_run.data.metrics['roc_auc']

print(f"Found Best Run ID: {best_run_id} with ROC-AUC: {best_auc:.4f}")

# 4. Register this specific model in the MLflow Model Registry
model_name = "fraud-predictor"
model_uri = f"runs:/{best_run_id}/model"

print(f"Registering model as '{model_name}'...")
mv = mlflow.register_model(model_uri, model_name)

# 5. Promote the model to Staging
print("Promoting model version to 'Staging'...")
client.transition_model_version_stage(
    name=model_name,
    version=mv.version,
    stage="Staging"
)

print("✅ Phase 5 Complete! Model is staged and ready for serving.")