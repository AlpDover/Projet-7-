from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

#On lit le pickle du modèle ainsi que le pickle scaler
#model = pickle.load(open('model.pkl', 'rb'))

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    
@app.route('/')
def home_page():
    return "Welcome to the credit score prediction api"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict_proba(data['features'])
    return jsonify(prediction.tolist())


if __name__ == '__main__':
    app.run(debug=True)