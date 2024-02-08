from flask import Flask, request, jsonify
import pickle
import pandas as pd
import xgboost


app = Flask(__name__)

# Load the scaler and model
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the test results data
with open('test_results.pkl', 'rb') as file:
    test_results_data = pickle.load(file)

@app.route('/')
def home_page():
    return "Welcome to the credit score prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Use the loaded test results data for prediction
        features = request.json['features']

        # Convert features to DataFrame
        features_df = pd.DataFrame(features)

        # Drop 'SK_ID_CURR' if it exists
        if 'SK_ID_CURR' in features_df.columns:
            features_df = features_df.drop(columns=['SK_ID_CURR'])

        # Preprocess the features using the loaded scaler
        features_scaled = scaler.transform(features_df)

        # Make the prediction using the loaded model
        prediction_proba = model.predict_proba(features_scaled)

        # Apply threshold to predicted probabilities
        threshold = 0.45
        predictions_binary = (prediction_proba[:, 1] > threshold).astype(int)

        # Create a DataFrame with the binary predictions
        y_test = pd.DataFrame(predictions_binary, columns=['TARGET'])

        # Return the binary predictions as a JSON response
        return jsonify({'predictions_binary': y_test['TARGET'].tolist()[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)