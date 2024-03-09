import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pickle
import shap
import streamlit.components.v1 as components


# Load the dataset
@st.cache
def load_data():
    df = pd.read_parquet('client_data.parquet')
    #df['SK_ID_CURR'] = df.index  # Assuming the index contains SK_ID_CURR
    return df

# Function to generate descriptive statistics and visualization
def compare_clients(selected_variable, selected_client_value, dataset, pred_result):

    
    # Check if there is data for the selected client
    # if selected_client_data.empty:
    #      st.warning("No data found for the selected client.")
    #      return
    
    dataset_filtered = dataset[dataset['TARGET']==pred_result[0]]
    selected_client_value = selected_client_value[selected_variable].values[0]

    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=dataset_filtered, x=selected_variable, kde=True, color='blue', alpha=0.5, label='Overall Dataset', ax=ax)
    ax.axvline(x=selected_client_value, color='red', linestyle='--', label='Selected Client')
    ax.set_title(f'Distribution of {selected_variable}')
    ax.set_xlabel(selected_variable)
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

def credit_score_gauge(score):
    # Color gradient from red to yellow to green
    colors = ['#00FF00', '#FFFF00','#FF0000']  # Red, Yellow, Green
    thresholds = [0, 0.5, 1]
    # Interpolate color based on score
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", list(zip(thresholds, colors)))
    norm = mcolors.Normalize(vmin=0, vmax=1)
    #color = cmap(norm(score))
    # Plot gauge
    fig, ax = plt.subplots(figsize=(6, 0.5))  # Reduced height to accommodate lower text
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    # Draw color gradient as colorbar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 0.5])
    # Draw tick marks and labels
    for i, threshold in enumerate(thresholds):
        ax.plot([threshold, threshold], [0.45, 0.5], color='black')
        ax.text(threshold, 0.55, str(threshold), fontsize=12, ha='center', va='bottom', color='black')
    # Draw dotted line at 0.5 threshold with legend
    ax.plot([0.5, 0.5], [0, 0.5], linestyle='--', color='black', label='Threshold')
    # Draw prediction indicator with legend
    ax.plot([score, score], [0, 0.5], color='black', linewidth=2, label='Client score')
    # Draw score below with the same color as the prediction indicator
    ax.text(score, -0.7, f'{score:.2f}', fontsize=14, ha='center', va='bottom', color='black')
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), fancybox=True, shadow=True, ncol=2)
    st.pyplot(fig, clear_figure=True)

# Streamlit UI
def main():


    st.title("Credit Score Prediction and Client Data Comparison Dashboard")
    
    # Load the dataset
    dataset = load_data()
    

    
    # Input client ID
    client_id = st.text_input("Enter Client ID:")
    if not client_id:
        st.warning("Please enter a Client ID.")
        st.stop()
    
    
    # Load test_results_data (assuming it's a DataFrame with SK_ID_CURR column)
    # Adjust the path based on your file structure
    test_results_data = pd.read_pickle('test_results.pkl')

    # Generate descriptive statistics and visualization
    
    
    # Prepare data for API request
    selected_client_data = test_results_data[test_results_data['SK_ID_CURR'].astype(str) == client_id].drop(columns=['SK_ID_CURR'])
    selected_client_data_viz = selected_client_data[['CODE_GENDER', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'PAYMENT_RATE']]

    st.subheader(f"Features for Client ID {client_id}")
    st.write(selected_client_data_viz)

    api_data = {'features': [selected_client_data.to_dict(orient='records')[0]]}
   

    # Make a request to the API
    api_url = 'http://127.0.0.1:5000/predict'  # Update with the correct URL where your Flask app is running
    response = requests.post(api_url, json=api_data)

# Check if the request was successful
    if response.status_code == 200:
        prediction_result = response.json()

        # Extract prediction and probability
        prediction = prediction_result.get('predictions_binary')
        probability = prediction_result.get('prediction_proba', [[0, 0]])[0][1]  # Probability of positive class (1)


        st.subheader("Prediction Result:")
        st.write(f"Credit Score: {prediction}")
        st.write(f"Probability: {probability * 100:.2f}%")  # Display probability without rounding

        # Interpretation
        interpretation = "High Risk" if probability >= 0.5 else "Low Risk"
        st.write(f"Interpretation: {interpretation}")




    else:
        st.error(f"API request failed with status code: {response.status_code}")

    
    credit_score_gauge(probability)

    #print(dataset.columns)

    # Dropdown for variable selection
    selected_variable = st.selectbox("Select Variable:", ['CODE_GENDER', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'PAYMENT_RATE'])

    compare_clients(selected_variable, selected_client_data_viz, dataset, prediction)

   

    # # Dropdown for variable selection
    selected_variable_1 = st.selectbox("Select Variable 1:", ['DAYS_EMPLOYED', 'DAYS_BIRTH', 'PAYMENT_RATE', 'CODE_GENDER'])
    selected_variable_2 = st.selectbox("Select Variable 2:", ['DAYS_EMPLOYED', 'DAYS_BIRTH', 'PAYMENT_RATE', 'CODE_GENDER'])

    # À placer au début de votre script Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Scatter plot for selected variables
    plt.figure(figsize=(8, 6))

    # Plot all data points
    sns.scatterplot(data=dataset, x=selected_variable_1, y=selected_variable_2)

    # Highlight the selected client
    selected_client_data_int = test_results_data[test_results_data['SK_ID_CURR'] == int(client_id)]
    plt.scatter(selected_client_data_int[selected_variable_1], selected_client_data_int[selected_variable_2], color='orange', label='Selected Client')

    plt.title(f'Scatter Plot of {selected_variable_1} and {selected_variable_2}')
    plt.xlabel(selected_variable_1)
    plt.ylabel(selected_variable_2)
    plt.legend()
    st.pyplot()

     # Load the XGBoost model from the pickle file
    with open('model.pkl', 'rb') as f:
        xgb_regressor = pickle.load(f)

    # Calculate SHAP values
    explainer = shap.Explainer(xgb_regressor)
    shap_values = explainer.shap_values(selected_client_data)

    # Plot global feature importance
    feature_importances = xgb_regressor.feature_importances_
    feature_names = test_results_data.drop(columns=['SK_ID_CURR']).columns.tolist()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    top_10_features = importance_df.head(10)
    
    st.subheader("Top 10 Features Importance (Global)")
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(top_10_features['Feature'], top_10_features['Importance'], color='skyblue')
    ax.set_xlabel('Feature Names')
    ax.set_ylabel('Importance Score')
    ax.set_title('Top 10 Features Importance (Global)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

    # Plot local feature importance using SHAP values

    # Calculate SHAP values
    explainer = shap.Explainer(xgb_regressor)
    shap_values = explainer.shap_values(selected_client_data)

    # Plot local SHAP values
    st.subheader("Local SHAP Values")
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
    # Streamlit webpage

    shap.initjs()
    force_plot_html = shap.force_plot(base_value=explainer.expected_value, shap_values=shap_values, features=selected_client_data.columns)
    st_shap(force_plot_html, 400)

if __name__ == "__main__":
    main()