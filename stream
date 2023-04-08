import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.joblib")

# Create a function to get user input
def get_user_input():
    st.sidebar.header("Enter the COVID-19 data:")
    new_cases = st.sidebar.number_input("New cases:")
    new_deaths = st.sidebar.number_input("New deaths:")
    new_tests = st.sidebar.number_input("New tests:")
    data = {"New cases": new_cases,
            "New deaths": new_deaths,
            "New tests": new_tests}
    features = pd.DataFrame(data, index=[0])
    return features

# Create a function to make predictions
def predict_cases(features):
    predictions = model.predict(features)
    return predictions

# Create the Streamlit app
def main():
    st.title("COVID-19 Cases Prediction")
    st.write("This app predicts the number of new COVID-19 cases based on the input data.")
    features = get_user_input()
    predictions = predict_cases(features)
    st.write("## Predictions:")
    st.write("The predicted number of new cases is:", predictions[0])

if __name__ == "__main__":
    main()
