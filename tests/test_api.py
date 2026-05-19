from fastapi.testclient import TestClient
from src.serving.api import app
import pandas as pd
from unittest.mock import patch

client = TestClient(app)

# Intercept the call to the Feast Feature Store
@patch("src.serving.api.store.get_online_features")
def test_predict_endpoint(mock_get_features):
    
    # 1. Create a fake response with dummy data
    class MockFeastResponse:
        def to_df(self):
            return pd.DataFrame({
                "Amount": [150.0],
                "V1": [0.1],
                "V2": [-0.5],
                "V3": [1.2],
                "V4": [-0.8],
                "V5": [0.3]
            })
            
    # 2. Tell our interceptor to return the fake data
    mock_get_features.return_value = MockFeastResponse()

    # 3. Run the test!
    response = client.post(
        "/predict",
        json={"transaction_id": 10}
    )
    
    # 4. Verify it works
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data