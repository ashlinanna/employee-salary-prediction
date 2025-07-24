
import streamlit as st
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# ---- Custom CSS for styling ----
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
    }
    .main {
        background-color: #f0f8ff;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Show top image ----
st.image(
    "https://images.unsplash.com/photo-1600880292203-757bb62b4baf?auto=format&fit=crop&w=1450&q=80",
    caption="Empowering Professionals with AI",
    use_container_width=True
)

st.markdown("<h1>Employee Salary Category Prediction</h1>", unsafe_allow_html=True)
st.write("Predict whether an employee earns more than $50K/year based on input details.")

# ---- Load model and scaler ----
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")

# Mapping dictionaries for dropdowns
workclass_dict = {
    "Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2,
    "Federal-gov": 3, "Local-gov": 4, "State-gov": 5, "Without-pay": 6
}
marital_dict = {
    "Married-civ-spouse": 0, "Divorced": 1, "Never-married": 2,
    "Separated": 3, "Widowed": 4, "Married-spouse-absent": 5
}
occupation_dict = {
    "Tech-support": 0, "Craft-repair": 1, "Other-service": 2,
    "Sales": 3, "Exec-managerial": 4, "Prof-specialty": 5,
    "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 8,
    "Farming-fishing": 9, "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12
}
relationship_dict = {
    "Wife": 0, "Own-child": 1, "Husband": 2, "Not-in-family": 3, "Other-relative": 4, "Unmarried": 5
}
race_dict = {
    "White": 0, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 2,
    "Other": 3, "Black": 4
}
sex_dict = {"Female": 0, "Male": 1}
country_dict = {
    "United-States": 0, "India": 1, "Mexico": 2, "Philippines": 3,
    "Germany": 4, "Canada": 5, "England": 6, "China": 7,
    "Cuba": 8, "Jamaica": 9, "Italy": 10
}

# Session state to control form display
if "show_form" not in st.session_state:
    st.session_state.show_form = False

if st.button("🔍 Predict Salary Category"):
    st.session_state.show_form = True

if st.session_state.show_form:
    st.markdown("## 📋 Enter Employee Details")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    education_num = st.slider("Education Level (Number)", 1, 16, 10)
    hours_per_week = st.slider("Hours per week", 1, 100, 40)
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)

    # Dropdowns with labels
    workclass = st.selectbox("Workclass", list(workclass_dict.keys()))
    marital_status = st.selectbox("Marital Status", list(marital_dict.keys()))
    occupation = st.selectbox("Occupation", list(occupation_dict.keys()))
    relationship = st.selectbox("Relationship", list(relationship_dict.keys()))
    race = st.selectbox("Race", list(race_dict.keys()))
    sex = st.selectbox("Sex", list(sex_dict.keys()))
    native_country = st.selectbox("Native Country", list(country_dict.keys()))

    if st.button("Submit"):
        input_data = np.array([[
            age,
            workclass_dict[workclass],
            education_num,
            marital_dict[marital_status],
            occupation_dict[occupation],
            relationship_dict[relationship],
            race_dict[race],
            sex_dict[sex],
            capital_gain,
            capital_loss,
            hours_per_week,
            country_dict[native_country]
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        result = ">50K" if prediction == 1 else "<=50K"
        st.success(f"💰 Predicted Income: {result}")
