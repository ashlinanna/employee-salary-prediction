
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set page config
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# Inject background CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #e6f0ff;
        background-size: cover;
    }
    .title-container {
        text-align: center;
        padding: 30px;
    }
    .predict-title {
        font-size: 36px;
        font-weight: bold;
        color: #003366;
    }
    .result-slide {
        text-align: center;
        padding: 80px;
        background-color: #cce0ff;
        border-radius: 20px;
        font-size: 32px;
        font-weight: bold;
        color: #003366;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'app_stage' not in st.session_state:
    st.session_state.app_stage = 'welcome'
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Welcome page
if st.session_state.app_stage == 'welcome':
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    st.markdown('<p class="predict-title">Employee Salary Category Prediction</p>', unsafe_allow_html=True)
    st.image("https://cdn.pixabay.com/photo/2016/11/29/04/17/meeting-1869514_1280.jpg", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Predict Salary"):
        st.session_state.app_stage = 'form'

# Input form
elif st.session_state.app_stage == 'form':
    st.header("Enter Employee Details")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    education_num = st.slider("Education Level (1-16)", 1, 16, 10)
    hours_per_week = st.slider("Hours per week", 1, 100, 40)
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)

    # Dropdowns with readable labels
    workclass_map = {
        "Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2, "Federal-gov": 3,
        "Local-gov": 4, "State-gov": 5, "Without-pay": 6
    }
    marital_status_map = {
        "Married-civ-spouse": 0, "Divorced": 1, "Never-married": 2,
        "Separated": 3, "Widowed": 4, "Married-spouse-absent": 5
    }
    occupation_map = {
        "Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3,
        "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6,
        "Machine-op-inspct": 7, "Adm-clerical": 8
