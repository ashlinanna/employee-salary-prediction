
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set Streamlit page settings
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# CSS for background and styling
st.markdown(
    '''
    <style>
        body {
            background-color: #e0f0ff;
        }
        .title-container {
            text-align: center;
        }
        .result-container {
            text-align: center;
            font-size: 30px;
            color: white;
            background-color: #007bff;
            padding: 40px;
            border-radius: 15px;
            margin-top: 30px;
        }
    </style>
    ''',
    unsafe_allow_html=True
)

# Control flow with session state
if "show_form" not in st.session_state:
    st.session_state.show_form = False
if "prediction_counts" not in st.session_state:
    st.session_state.prediction_counts = {">50K": 0, "<=50K": 0}

# Title page with image
if not st.session_state.show_form:
    st.markdown('<div class="title-container"><h1>Employee Salary Category Prediction</h1></div>', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1629904853893-c2c8981a1dc5", use_container_width=True)
    if st.button("Predict"):
        st.session_state.show_form = True
        st.rerun()

# Input form and prediction
else:
    model = joblib.load("salary_model.pkl")
    scaler = joblib.load("scaler.pkl")

    st.header("Enter Employee Details")

    age = st.number_input("Age", 18, 100, value=30)
    education_num = st.slider("Education Level (1 = Low, 16 = High)", 1, 16, 10)
    hours_per_week = st.slider("Hours per week", 1, 100, 40)
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)

    workclass_options = {
        'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2,
        'Federal-gov': 3, 'Local-gov': 4, 'State-gov': 5, 'Without-pay': 6
    }
    marital_status_options = {
        'Married-civ-spouse': 0, 'Divorced': 1, 'Never-married': 2,
        'Separated': 3, 'Widowed': 4, 'Married-spouse-absent': 5
    }
    occupation_options = {
        'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2,
        'Sales': 3, 'Exec-managerial': 4, 'Prof-specialty': 5, 'Handlers-cleaners': 6,
        'Machine-op-inspct': 7, 'Adm-clerical': 8, 'Farming-fishing': 9,
        'Transport-moving': 10, 'Priv-house-serv': 11, 'Protective-serv': 12
    }
    relationship_options = {
        'Wife': 0, 'Own-child': 1, 'Husband': 2,
        'Not-in-family': 3, 'Other-relative': 4, 'Unmarried': 5
    }
    race_options = {
        'White': 0, 'Black': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'Other': 4
    }
    sex_options = {'Male': 0, 'Female': 1}
    country_options = {
        'United-States': 0, 'Mexico': 1, 'Philippines': 2,
        'Germany': 3, 'Canada': 4, 'India': 5, 'England': 6,
        'Cuba': 7, 'Jamaica': 8, 'South': 9, 'China': 10
    }

    workclass = st.selectbox("Workclass", list(workclass_options.keys()))
    marital_status = st.selectbox("Marital Status", list(marital_status_options.keys()))
    occupation = st.selectbox("Occupation", list(occupation_options.keys()))
    relationship = st.selectbox("Relationship", list(relationship_options.keys()))
    race = st.selectbox("Race", list(race_options.keys()))
    sex = st.selectbox("Sex", list(sex_options.keys()))
    native_country = st.selectbox("Native Country", list(country_options.keys()))

    if st.button("Submit & Predict"):
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
            country_options[native_country]
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        result = ">50K" if prediction == 1 else "<=50K"

        # Show result
        st.markdown(f'<div class="result-container">Predicted Income: {result}</div>', unsafe_allow_html=True)

        # Update pie chart state
        st.session_state.prediction_counts[result] += 1

        # Pie Chart
        st.subheader("📊 Prediction Distribution (This Session)")
        labels = list(st.session_state.prediction_counts.keys())
        sizes = list(st.session_state.prediction_counts.values())

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
