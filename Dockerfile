# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files
COPY app.py .
COPY dashboard_1.py .
COPY scaler.pkl .
COPY model.pkl .
COPY test_results.pkl .
COPY client_data.parquet .

# Install dependencies
RUN pip install --no-cache-dir \
    Flask==3.0.0 \
    pandas==2.0.2 \
    numpy==1.24.3 \
    requests==2.31.0 \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    shap==0.43.0 \
    scikit-learn==1.3.1 \
    streamlit==1.30.0 \
    xgboost==2.0.0 \
    ipython==8.14.0

# Expose the ports for both the Flask API and the Streamlit app
EXPOSE 5000
EXPOSE 8501

# Command to run both applications
CMD ["bash", "-c", "python app.py & streamlit run dashboard_1.py"]