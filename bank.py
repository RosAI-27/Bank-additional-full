# app.py
import streamlit as st
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load('bank_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title("Bank Marketing Campaign Prediction")

# Input fields (simplified example - add all features here!)
age = st.number_input("Age", min_value=18, max_value=100)
duration = st.number_input("Duration (seconds)", min_value=0)
# ... Add more input fields for other features ...

# Predict button
if st.button("Predict Subscription"):
    # Create a DataFrame with all input features (match training data!)
    # NOTE: You must include ALL features used in the model here.
    input_data = pd.DataFrame([[age, duration]], columns=['age', 'duration'])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Show result
    st.success("Prediction: " + ("Yes" if prediction == 1 else "No"))