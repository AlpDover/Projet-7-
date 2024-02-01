import json
import pytest
from api_p7 import app  # Update with the correct name of your Flask app file

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to the credit score prediction API" in response.data

def test_predict_endpoint_valid_request(client):
    data = {
        "features": [
            {
                "feature1": 0.1,
                "feature2": 0.5,
                # Add other features as needed
            }
        ]
    }

    response = client.post('/predict', json=data)
    assert response.status_code == 200

    result = json.loads(response.data)
    print("API Response:", result)  # Add this line for debugging
    assert "predictions_binary" in result

def test_predict_endpoint_invalid_request(client):
    # Invalid request without 'features'
    invalid_data = {"invalid_key": "invalid_value"}

    response = client.post('/predict', json=invalid_data)
    assert response.status_code == 200  # Adjust based on your expected behavior

    result = json.loads(response.data)
    assert "error" in result

# Add more test cases as needed