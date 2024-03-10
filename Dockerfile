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
    xgboost==2.0.0 

# Expose the ports for both the Flask API and the Streamlit app
EXPOSE 5000
EXPOSE 8501

# Command to run both applications
CMD ["bash", "-c", "python app.py"]