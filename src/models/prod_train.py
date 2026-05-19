import mlflow
import pandas as pd
import os
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

print("Loading environment variables...")
load_dotenv()

# 1. Strict MLflow Auth (No dagshub.init allowed!)
os.environ["MLFLOW_TRACKING_USERNAME"] = "Hruthikck28"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

# 2. Strict Tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Hruthikck28/credit-card-fraud-mlops.mlflow")
mlflow.set_experiment("fraud-production")

print("Loading data...")
df = pd.read_parquet("feature_repo/data/creditcard_features.parquet")

# Downsample for speed
fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0].sample(n=5000, random_state=42)
balanced_df = pd.concat([fraud, legit])

X = balanced_df[['Amount', 'V1', 'V2', 'V3', 'V4', 'V5']]
y = balanced_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training production model with best Optuna parameters...")
# Using the exact best params from your screenshot!
params = {'n_estimators': 55, 'max_depth': 5, 'learning_rate': 0.153}

with mlflow.start_run(run_name="production-candidate") as run:
    mlflow.log_params(params)
    
    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    mlflow.log_metric("roc_auc", auc)
    
    print(f"Model trained! ROC-AUC: {auc:.4f}")
    print("Logging and Registering model directly to DagsHub cloud...")
    
    # THE MAGIC FIX: registered_model_name links the file and registry instantly
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="fraud-predictor"
    )
    
print("Promoting model to Staging...")
client = MlflowClient()
client.transition_model_version_stage(
    name="fraud-predictor",
    version=model_info.registered_model_version,
    stage="Staging"
)

print("✅ MASTER SUCCESS! Model is officially in Staging on DagsHub!")