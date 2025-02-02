import streamlit as st
import joblib
import pandas as pd
import gdown

# Download the model file from Google Drive
file_id = "11lJWYbmvcxdU1gyfDTuw4VT2n_z5qjvV" 
url = f"https://drive.google.com/uc?id={file_id}"
output = "bank_model.pkl"
gdown.download(url, output, quiet=False)

# Load the model
model = joblib.load(output)

# Load the scaler (if needed)
scaler = joblib.load('scaler.pkl')  # Ensure scaler.pkl is in your repo

# Streamlit app
st.title("Bank Marketing Campaign Prediction")

# Input fields (simplified example)
age = st.number_input("Age", min_value=18, max_value=100)
duration = st.number_input("Duration (seconds)", min_value=0)
# ... Add more input fields for other features ...

# Predict button
if st.button("Predict Subscription"):
    # Create a DataFrame with all input features
    input_data = pd.DataFrame([[age, duration]], columns=['age', 'duration'])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Show result
    st.success("Prediction: " + ("Yes" if prediction == 1 else "No"))

