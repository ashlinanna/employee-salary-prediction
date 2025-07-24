
import streamlit as st
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

# Custom CSS for background and styling
st.markdown("""
    <style>
        body {
            background-color: #e6f2ff;
        }
        .title-text {
            text-align: center;
            color: #003366;
        }
        .predict-button button {
            background-color: #003366;
            color: white;
            font-weight: bold;
        }
        .big-result {
            font-size: 36px;
            color: white;
            background-color: #0059b3;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")

# Show professional image and title
st.image("https://cdn.pixabay.com/photo/2016/03/26/13/09/man-1282232_960_720.jpg", use_container_width=True)
st.markdown("<h1 class='title-text'>Employee Salary Category Prediction</h1>", unsafe_allow_html=True)

# Hide input form until button is clicked
if "show_form" not in st.session_state:
    st.session_state["show_form"] = False

if not st.session_state["show_form"]:
    if st.button("Predict"):
        st.session_state["show_form"] = True
        st.experimental_rerun()
else:
    st.subheader("Enter Employee Details")

    # Input fields
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    education_num = st.slider("Education Level (Number)", 1, 16, 10)
    hours_per_week = st.slider("Hours per week", 1, 100, 40)
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)

    # Dropdown mappings
    workclass_options = {
        "Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2, "Federal-gov": 3,
        "Local-gov": 4, "State-gov": 5, "Without-pay": 6
    }
    marital_status_options = {
        "Never-married": 0, "Married-civ-spouse": 1, "Divorced": 2,
        "Separated": 3, "Widowed": 4, "Married-spouse-absent": 5
    }
    occupation_options = {
        "Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3,
        "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6,
        "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9,
        "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12
    }
    relationship_options = {
        "Wife": 0, "Own-child": 1, "Husband": 2, "Not-in-family": 3,
        "Other-relative": 4, "Unmarried": 5
    }
    race_options = {
        "White": 0, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 2,
        "Other": 3, "Black": 4
    }
    sex_options = {"Female": 0, "Male": 1}
    native_country_options = {
        "United-States": 0, "India": 1, "Mexico": 2, "Philippines": 3,
        "Germany": 4, "Canada": 5, "England": 6, "China": 7, "Cuba": 8,
        "Iran": 9, "Italy": 10
    }

    # Categorical dropdowns
    workclass = st.selectbox("Workclass", list(workclass_options.keys()))
    marital_status = st.selectbox("Marital Status", list(marital_status_options.keys()))
    occupation = st.selectbox("Occupation", list(occupation_options.keys()))
    relationship = st.selectbox("Relationship", list(relationship_options.keys()))
    race = st.selectbox("Race", list(race_options.keys()))
    sex = st.selectbox("Sex", list(sex_options.keys()))
    native_country = st.selectbox("Native Country", list(native_country_options.keys()))

    # Predict button
    if st.button("Submit Prediction"):
        input_data = np.array([[
            age,
            workclass_options[workclass],
            education_num,
            marital_status_options[marital_status],
            occupation_options[occupation],
            relationship_options[relationship],
            race_options[race],
            sex_options[sex],
            capital_gain,
            capital_loss,
            hours_per_week,
            native_country_options[native_country]
        ]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        result = ">50K" if prediction == 1 else "<=50K"

        st.markdown(f"<div class='big-result'>Predicted Income: {result}</div>", unsafe_allow_html=True)
