from fastapi.testclient import TestClient
from src.serving.api import app

client = TestClient(app)

def test_predict_endpoint():
    # Send a fake transaction ID to your API
    response = client.post(
        "/predict",
        json={"transaction_id": 10}
    )
    
    # Check that the API didn't crash (200 OK)
    assert response.status_code == 200
    
    # Check that it returned a fraud probability
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data