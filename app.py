
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Employee Salary Category Prediction")
st.write("This app predicts whether an employee earns more than 50K or not.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
education_num = st.slider("Education Level (Number)", 1, 16, 10)
hours_per_week = st.slider("Hours per week", 1, 100, 40)
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)

# Encoded categorical inputs
workclass = st.selectbox("Workclass", [0,1,2,3,4,5,6])
marital_status = st.selectbox("Marital Status", [0,1,2,3,4,5])
occupation = st.selectbox("Occupation", [0,1,2,3,4,5,6,7,8,9,10,11,12])
relationship = st.selectbox("Relationship", [0,1,2,3,4,5])
race = st.selectbox("Race", [0,1,2,3,4])
sex = st.selectbox("Sex", [0,1])
native_country = st.selectbox("Native Country", [0,1,2,3,4,5,6,7,8,9,10])

# Make prediction
if st.button("Predict"):
    input_data = np.array([[age, workclass, education_num, marital_status, 
                            occupation, relationship, race, sex, capital_gain, 
                            capital_loss, hours_per_week, native_country]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income: {result}")
