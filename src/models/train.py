import mlflow
import optuna
import pandas as pd
import os
import dagshub
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score

# 1. Load the hidden secrets
load_dotenv()

# 2. Explicit Auth (Crucial for the artifact uploader)
os.environ["MLFLOW_TRACKING_USERNAME"] = "Hruthikck28"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN") 
os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("DAGSHUB_TOKEN")

# 3. Setup connection
dagshub.init(repo_owner='Hruthikck28', repo_name='credit-card-fraud-mlops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Hruthikck28/credit-card-fraud-mlops.mlflow")

# Use a fresh experiment to guarantee a clean artifact path
mlflow.set_experiment("fraud-tuning-v3")

print("Loading data...")
# Read the offline Parquet file we made in Phase 3
df = pd.read_parquet("feature_repo/data/creditcard_features.parquet")

# 2. Downsample for speed (keeps all 492 frauds, samples 5000 legit transactions)
fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0].sample(n=5000, random_state=42)
balanced_df = pd.concat([fraud, legit])

X = balanced_df[['Amount', 'V1', 'V2', 'V3', 'V4', 'V5']]
y = balanced_df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    # 3. Let Optuna guess the best hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
    }

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        # 4. Train the model
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 5. Evaluate
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)

        # Log metrics and the model itself to DagsHub
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

    return auc

print("Starting hyperparameter tuning with Optuna...")
# Run 10 experiments to find the best model
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print(f"✅ Tuning complete! Best ROC-AUC Score: {study.best_value:.4f}")
print("Go check your DagsHub MLflow dashboard!")