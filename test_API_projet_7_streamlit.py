import streamlit as st
import requests

def main():
    st.title("API Testing with Streamlit")

    # Input form for user to enter test data
    st.header("Enter Test Data")
    train_data = st.text_area("Enter train_data (JSON format):", '{"feature1": value1, "feature2": value2}')
    test_data = st.text_area("Enter test_data (JSON format):", '{"feature1": value1, "feature2": value2}')

    if st.button("Test API"):
        # Convert user input to JSON
        try:
            train_data = json.loads(train_data)
            test_data = json.loads(test_data)
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please enter valid JSON for train_data and test_data.")
            return

        # Make a POST request to the API
        response = requests.post("http://127.0.0.1:5000/predict", json={"train_data": train_data, "test_data": test_data})

        # Display API response
        st.header("API Response")
        st.json(response.json())

if __name__ == "__main__":
    main()