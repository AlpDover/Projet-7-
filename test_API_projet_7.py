import pytest
from flask_testing import TestCase
from API_projet_7 import app, main  # Import the 'main' function from API_projet_7

class TestAPI(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        return app

    def test_predict_endpoint(self):
        # Mock input data for testing
        df_predict_cleaned = main(debug=True)

        # Assuming you have df_predict_cleaned defined somewhere in your test script
        test_data = {
            'test_data': df_predict_cleaned.to_dict(orient='records')
        }

        # Send a POST request to the /predict endpoint
        response = self.client.post('/predict', json=test_data)

        # Assert the response status code
        assert response.status_code == 200

        # Assert the response content or structure as needed
        assert 'predictions' in response.json()