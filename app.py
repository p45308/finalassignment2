

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
try:
    loaded_model = joblib.load('svc_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    st.success("Model and scaler loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    loaded_model = None
    loaded_scaler = None

# Define the features expected by the model after preprocessing
# This order is crucial and must match the order of columns in X_train after encoding and scaling
feature_order = [
    'person_age', 'person_income', 'person_emp_length', 'loan_grade', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length',
    'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
    'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
    'loan_intent_PERSONAL', 'loan_intent_VENTURE'
]

# Create the Streamlit app
st.title("Credit Risk Prediction App")

st.write("Enter the details below to predict credit risk.")

# Get user input for each feature
person_age = st.number_input("Person Age", min_value=0)
person_income = st.number_input("Person Income", min_value=0)
person_emp_length = st.number_input("Person Employment Length (Years)", min_value=0.0)
loan_grade = st.selectbox("Loan Grade (0-6)", options=[0, 1, 2, 3, 4, 5, 6]) # Assuming loan_grade is label encoded 0-6
loan_amnt = st.number_input("Loan Amount", min_value=0)
loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0)
loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0)
cb_person_default_on_file = st.selectbox("Default on File (Y/N)", options=['N', 'Y'])
cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0)
person_home_ownership = st.selectbox("Home Ownership", options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
loan_intent = st.selectbox("Loan Intent", options=['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])


# Function to preprocess input data
def preprocess_input(data):
    # Create a DataFrame from input
    input_df = pd.DataFrame([data])

    # Apply the same encoding steps as in the notebook

    # One-Hot Encoding for 'person_home_ownership' and 'loan_intent'
    input_df = pd.get_dummies(input_df, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

    # Binary Mapping for 'cb_person_default_on_file'
    input_df['cb_person_default_on_file'] = input_df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

    # Ensure all columns present during training are present in the input DataFrame
    # and reorder columns to match the training data
    for col in feature_order:
        if col not in input_df.columns:
            input_df[col] = 0 # Add missing one-hot encoded columns with value 0

    input_df = input_df[feature_order] # Reorder columns

    # Scale the input data
    scaled_input = loaded_scaler.transform(input_df)

    return scaled_input

# Make prediction when button is clicked
if st.button("Predict Credit Risk"):
    if loaded_model is not None and loaded_scaler is not None:
        input_data = {
            'person_age': person_age,
            'person_income': person_income,
            'person_emp_length': person_emp_length,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_default_on_file': cb_person_default_on_file,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'person_home_ownership': person_home_ownership,
            'loan_intent': loan_intent
        }

        processed_input = preprocess_input(input_data)

        prediction = loaded_model.predict(processed_input)

        if prediction[0] == 1:
            st.error("Prediction: High Credit Risk")
        else:
            st.success("Prediction: Low Credit Risk")
    else:
        st.warning("Model or scaler not loaded. Cannot make prediction.")
