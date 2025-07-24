
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")

# Mapping dictionaries for dropdowns
workclass_options = {
    'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2,
    'Federal-gov': 3, 'Local-gov': 4, 'State-gov': 5, 'Without-pay': 6
}
marital_status_options = {
    'Married-civ-spouse': 0, 'Divorced': 1, 'Never-married': 2,
    'Separated': 3, 'Widowed': 4, 'Married-spouse-absent': 5
}
occupation_options = {
    'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2, 'Sales': 3,
    'Exec-managerial': 4, 'Prof-specialty': 5, 'Handlers-cleaners': 6,
    'Machine-op-inspct': 7, 'Adm-clerical': 8, 'Farming-fishing': 9,
    'Transport-moving': 10, 'Priv-house-serv': 11, 'Protective-serv': 12
}
relationship_options = {
    'Wife': 0, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3, 'Other-relative': 4, 'Unmarried': 5
}
race_options = {
    'White': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2,
    'Other': 3, 'Black': 4
}
sex_options = {'Female': 0, 'Male': 1}
native_country_options = {
    'United-States': 0, 'India': 1, 'Philippines': 2, 'Germany': 3, 'Mexico': 4,
    'Canada': 5, 'England': 6, 'China': 7, 'Italy': 8, 'Japan': 9, 'Other': 10
}

# Set background color using markdown
st.markdown("""
    <style>
    .main {
        background-color: #e3f2fd;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #0d47a1;'>👩‍💻 Employee Salary Category Prediction</h1>", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.header("Enter Employee Details")


age = st.sidebar.number_input("Age", 18, 100, 30)
education_num = st.sidebar.slider("Education Level (1-16)", 1, 16, 10)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)

workclass = st.sidebar.selectbox("Workclass", list(workclass_options.keys()))
marital_status = st.sidebar.selectbox("Marital Status", list(marital_status_options.keys()))
occupation = st.sidebar.selectbox("Occupation", list(occupation_options.keys()))
relationship = st.sidebar.selectbox("Relationship", list(relationship_options.keys()))
race = st.sidebar.selectbox("Race", list(race_options.keys()))
sex = st.sidebar.selectbox("Sex", list(sex_options.keys()))
native_country = st.sidebar.selectbox("Native Country", list(native_country_options.keys()))


workclass_val = workclass_options[workclass]
marital_status_val = marital_status_options[marital_status]
occupation_val = occupation_options[occupation]
relationship_val = relationship_options[relationship]
race_val = race_options[race]
sex_val = sex_options[sex]
native_country_val = native_country_options[native_country]

if st.button("Predict"):
    input_data = np.array([[age, workclass_val, education_num, marital_status_val,
                            occupation_val, relationship_val, race_val, sex_val,
                            capital_gain, capital_loss, hours_per_week, native_country_val]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = ">50K" if prediction == 1 else "<=50K"

    st.markdown("---")
    st.success(f"🎯 Predicted Income: **{result}**")
    if prediction == 1:
        st.balloons()

st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ by <b>Ashlin Anna</b> | Internship Project 2025</p>", unsafe_allow_html=True)
