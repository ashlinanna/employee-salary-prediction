
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🧠 Employee Salary Category Prediction")
st.write("This app predicts whether an employee earns more than 50K or not based on personal and job details.")

# Input fields (numerical)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
education_num = st.slider("Education Level (1 = Least, 16 = Highest)", 1, 16, 10)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)

# Dictionaries for label mapping
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
    'White': 0, 'Black': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'Other': 4
}
sex_options = {'Male': 0, 'Female': 1}
native_country_options = {
    'United-States': 0, 'Mexico': 1, 'Philippines': 2, 'Germany': 3, 'Canada': 4,
    'India': 5, 'England': 6, 'Cuba': 7, 'Jamaica': 8, 'China': 9, 'Other': 10
}

# Show readable dropdowns
workclass = st.selectbox("Workclass", list(workclass_options.keys()))
marital_status = st.selectbox("Marital Status", list(marital_status_options.keys()))
occupation = st.selectbox("Occupation", list(occupation_options.keys()))
relationship = st.selectbox("Relationship", list(relationship_options.keys()))
race = st.selectbox("Race", list(race_options.keys()))
sex = st.selectbox("Sex", list(sex_options.keys()))
native_country = st.selectbox("Native Country", list(native_country_options.keys()))

# Map selected text to encoded values
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

# Prediction button
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"💰 Predicted Income: **{result}**")
