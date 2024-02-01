import json
import pytest
from api_p7 import app  # Update with the correct name of your Flask app file
import pickle
import requests

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to the credit score prediction API" in response.data

def test_predict_endpoint_valid_request(client):

    with open('test_results.pkl','rb') as data:
        data = pickle.load(data)
    data = data.reset_index(drop=True)

    selected_client_data = data.sample(n=1).drop(columns=['SK_ID_CURR'])
    api_data = {'features': [selected_client_data.to_dict(orient='records')[0]]}

    api_url = 'http://127.0.0.1:5000/predict'
    #response = client.post('/predict', json=api_data)
    response = requests.post(api_url, json=api_data)
    assert response.status_code == 200

    #result = json.loads(response.data)
    result = response.json()
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