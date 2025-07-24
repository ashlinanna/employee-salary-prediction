
import streamlit as st
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# Show banner image
st.image(
    "https://images.unsplash.com/photo-1600880292203-757bb62b4baf?auto=format&fit=crop&w=1450&q=80",
    caption="Empowering Professionals with AI",
    use_container_width=True
)

# Title and description
st.markdown(
    "<h1 style='text-align: center; color: #2c3e50;'>Employee Salary Category Prediction</h1>",
    unsafe_allow_html=True
)
st.write("This app predicts whether an employee earns more than $50K per year based on demographic and work-related inputs.")

# Session state to control form display
if "show_form" not in st.session_state:
    st.session_state.show_form = False

# Button to reveal form
if st.button("🔍 Predict Salary Category"):
    st.session_state.show_form = True

# If form is shown
if st.session_state.show_form:
    st.markdown("## 📋 Enter Employee Details")

    # Load model and scaler
    model = joblib.load("salary_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Input fields
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    education_num = st.slider("Education Level (Number)", 1, 16, 10)
    hours_per_week = st.slider("Hours per week", 1, 100, 40)
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)

    # Encoded categorical inputs
    workclass = st.selectbox("Workclass", [0, 1, 2, 3, 4, 5, 6])
    marital_status = st.selectbox("Marital Status", [0, 1, 2, 3, 4, 5])
    occupation = st.selectbox("Occupation", list(range(13)))
    relationship = st.selectbox("Relationship", list(range(6)))
    race = st.selectbox("Race", list(range(5)))
    sex = st.selectbox("Sex", [0, 1])
    native_country = st.selectbox("Native Country", list(range(11)))

    # Predict
    if st.button("Submit"):
        input_data = np.array([[age, workclass, education_num, marital_status,
                                occupation, relationship, race, sex,
                                capital_gain, capital_loss, hours_per_week, native_country]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        result = ">50K" if prediction == 1 else "<=50K"
        st.success(f"💰 Predicted Income: {result}")
