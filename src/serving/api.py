from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import os
from dotenv import load_dotenv
from feast import FeatureStore

# 1. Load auth securely
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = "Hruthikck28"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
mlflow.set_tracking_uri("https://dagshub.com/Hruthikck28/credit-card-fraud-mlops.mlflow")

app = FastAPI(title="Fraud Detection API")

# 2. Connect to the local Feast Feature Store (Redis)
store = FeatureStore(repo_path="feature_repo")

# 3. Download the model from DagsHub Staging
print("Downloading model from DagsHub Registry (Staging)...")
model = mlflow.sklearn.load_model("models:/fraud-predictor/Staging")
print("Model loaded and ready for predictions!")

class PredictRequest(BaseModel):
    transaction_id: int

@app.post("/predict")
def predict(request: PredictRequest):
    # 4. Ask Feast to fetch the real-time features from Redis
    feature_vector = store.get_online_features(
        features=[
            "fraud_features:Amount",
            "fraud_features:V1",
            "fraud_features:V2",
            "fraud_features:V3",
            "fraud_features:V4",
            "fraud_features:V5",
        ],
        entity_rows=[{"transaction_id": request.transaction_id}]
    ).to_df()

    # Clean up the column names for the model
    feature_vector = feature_vector[['Amount', 'V1', 'V2', 'V3', 'V4', 'V5']]

    # 5. Make the prediction
    prob = model.predict_proba(feature_vector)[0][1]

    return {
        "transaction_id": request.transaction_id,
        "fraud_probability": round(prob, 4),
        "is_fraud": bool(prob > 0.5)
    }