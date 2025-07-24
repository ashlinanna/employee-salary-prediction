
import streamlit as st
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# Show image and title on load
st.image(
    "https://cdn.pixabay.com/photo/2017/08/10/07/32/business-2616748_1280.jpg",
    use_container_width=True
)

st.markdown(
    "<h1 style='text-align: center; color: #2c3e50;'>Employee Salary Category Prediction</h1>",
    unsafe_allow_html=True
)
st.write("This app predicts whether an employee earns more than $50K per year based on demographic and work-related inputs.")

# Initialize session state
if "show_form" not in st.session_state:
    st.session_state.show_form = False

# Predict button to toggle input form
if st.button("🔍 Predict Salary Category"):
    st.session_state.show_form = True

# Show input form
if st.session_state.show_form:
    # Load model and scaler
    model = joblib.load("salary_model.pkl")
    scaler = joblib.load("scaler.pkl")

    st.markdown("## 📋 Enter Employee Details")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    education_num = st.slider("Education Level (Number)", 1, 16, 10)
    hours_per_week = st.slider("Hours per week", 1, 100, 40)
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)

    # Encoded categorical fields
    workclass = st.selectbox("Workclass", [0, 1, 2, 3, 4, 5, 6])
    marital_status = st.selectbox("Marital Status", [0, 1, 2, 3, 4, 5])
    occupation = st.selectbox("Occupation", list(range(13)))
    relationship = st.selectbox("Relationship", list(range(6)))
    race = st.selectbox("Race", list(range(5)))
    sex = st.selectbox("Sex", [0, 1])
    native_country = st.selectbox("Native Country", list(range(11)))

    if st.button("Submit"):
        input_data = np.array([[age, workclass, education_num, marital_status,
                                occupation, relationship, race, sex, capital_gain,
                                capital_loss, hours_per_week, native_country]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        result = ">50K" if prediction == 1 else "<=50K"
        st.success(f"💰 Predicted Income: {result}")
