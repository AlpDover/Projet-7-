import streamlit as st
import pandas as pd
import requests
import json

# Streamlit UI
st.title("Credit Score Prediction Dashboard")

# Input client ID
client_id = st.text_input("Enter Client ID:")
if not client_id:
    st.warning("Please enter a Client ID.")
    st.stop()

# Load test_results_data (assuming it's a DataFrame with SK_ID_CURR column)
# Adjust the path based on your file structure
test_results_data = pd.read_pickle('test_results.pkl')

# Check if the entered Client ID is valid
if client_id not in test_results_data['SK_ID_CURR'].astype(str).values:
    st.warning(f"Client ID '{client_id}' not found in the dataset.")
    st.stop()

# Extract features for the selected client
selected_client_data = test_results_data[test_results_data['SK_ID_CURR'].astype(str) == client_id].drop(columns=['SK_ID_CURR'])

# Display selected client's features
st.subheader(f"Features for Client ID {client_id}")
st.write(selected_client_data)

# Prepare data for API request
api_data = {'features': [selected_client_data.to_dict(orient='records')[0]]}

# Make a request to the API
api_url = 'http://127.0.0.1:5000/predict'  # Update with the correct URL where your Flask app is running
response = requests.post(api_url, json=api_data)

# Check if the request was successful
if response.status_code == 200:
    prediction_result = response.json()
    st.subheader("Prediction Result:")
    st.write(prediction_result)
else:
    st.error(f"API request failed with status code: {response.status_code}")