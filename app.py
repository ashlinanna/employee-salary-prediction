
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
if 'show_form' not in st.session_state:
    st.session_state.show_form = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Title page
if not st.session_state.show_form:
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    st.markdown('<p class="predict-title">Employee Salary Category Prediction</p>', unsafe_allow_html=True)
    st.image("https://cdn.pixabay.com/photo/2016/11/29/04/17/meeting-1869514_1280.jpg", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Predict Salary"):
        st.session_state.show_form = True
        st.experimental_rerun()

# Prediction form
else:
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
        "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9,
        "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12
    }
    relationship_map = {
        "Wife": 0, "Own-child": 1, "Husband": 2, "Not-in-family": 3, "Other-relative": 4, "Unmarried": 5
    }
    race_map = {
        "White": 0, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 2, "Other": 3, "Black": 4
    }
    sex_map = {"Female": 0, "Male": 1}
    country_map = {
        "United-States": 0, "India": 1, "Philippines": 2, "Germany": 3, "Canada": 4,
        "England": 5, "China": 6, "Mexico": 7, "Italy": 8, "Japan": 9, "Other": 10
    }

    workclass = st.selectbox("Workclass", list(workclass_map.keys()))
    marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
    occupation = st.selectbox("Occupation", list(occupation_map.keys()))
    relationship = st.selectbox("Relationship", list(relationship_map.keys()))
    race = st.selectbox("Race", list(race_map.keys()))
    sex = st.selectbox("Sex", list(sex_map.keys()))
    native_country = st.selectbox("Native Country", list(country_map.keys()))

    if st.button("Submit"):
        input_data = np.array([[age,
                                workclass_map[workclass],
                                education_num,
                                marital_status_map[marital_status],
                                occupation_map[occupation],
                                relationship_map[relationship],
                                race_map[race],
                                sex_map[sex],
                                capital_gain,
                                capital_loss,
                                hours_per_week,
                                country_map[native_country]]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.session_state.prediction = prediction
        st.experimental_rerun()

# Show result in separate slide
if st.session_state.prediction is not None:
    income_result = ">50K" if st.session_state.prediction == 1 else "<=50K"
    st.markdown('<div class="result-slide">', unsafe_allow_html=True)
    st.write(f"🎯 **Predicted Income Category:** {income_result}")
    st.markdown('</div>', unsafe_allow_html=True)
