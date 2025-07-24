
import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .stApp {
            background-color: #e6f2ff;
            padding: 2rem;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #003366;
        }
        .sub-title {
            text-align: center;
            font-size: 1.2rem;
            color: #1a1a1a;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">👩‍💻 Employee Salary Category Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict whether an employee earns more than $50K</div>', unsafe_allow_html=True)

# Display professional images
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/1995/1995574.png", caption="HR Expert", use_container_width=True)
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", caption="Data Scientist", use_container_width=True)
with col3:
    st.image("https://cdn-icons-png.flaticon.com/512/1053/1053244.png", caption="Salary Analyst", use_container_width=True)
st.markdown("---")

# Button to reveal form
show_form = st.button("🔍 Predict Salary Category")

if show_form:
    model = joblib.load("salary_model.pkl")
    scaler = joblib.load("scaler.pkl")

    st.markdown("## 📋 Enter Employee Details")

    # Input fields
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    education_num = st.slider("Education Level (Number)", 1, 16, 10)
    hours_per_week = st.slider("Hours per week", 1, 100, 40)
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)

    workclass = st.selectbox("Workclass", [0,1,2,3,4,5,6])
    marital_status = st.selectbox("Marital Status", [0,1,2,3,4,5])
    occupation = st.selectbox("Occupation", [0,1,2,3,4,5,6,7,8,9,10,11,12])
    relationship = st.selectbox("Relationship", [0,1,2,3,4,5])
    race = st.selectbox("Race", [0,1,2,3,4])
    sex = st.selectbox("Sex", [0,1])
    native_country = st.selectbox("Native Country", [0,1,2,3,4,5,6,7,8,9,10])

    # Predict button
    if st.button("Submit"):
        input_data = np.array([[age, workclass, education_num, marital_status,
                                occupation, relationship, race, sex, capital_gain,
                                capital_loss, hours_per_week, native_country]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        result = ">50K" if prediction == 1 else "<=50K"
        st.success(f"💰 Predicted Income: {result}")
