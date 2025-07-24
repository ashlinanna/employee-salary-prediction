# 👩‍💻 Employee Salary Prediction using Machine Learning

This project predicts whether an employee earns more than $50K per year using a dataset containing demographic and work-related information.  
It uses machine learning (Random Forest Classifier) and is deployed using **Streamlit** as a web application.

---

## 🚀 Features

- Data Preprocessing (Missing values, Encoding)
- Feature Scaling
- Model Training (Random Forest)
- Accuracy Evaluation
- Streamlit Web App Interface
- Exported model and scaler (`salary_model.pkl`, `scaler.pkl`)

---

## 📂 Folder Structure

```
├── app.py                                 # Streamlit web app
├── salary_model.pkl                       # Trained machine learning model
├── scaler.pkl                             # Scaler used to standardize data
├── EmployeeSalaryPrediction_Training.ipynb  # Model training notebook
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

---

## 📊 Dataset

The dataset is based on the **UCI Adult Census Income dataset**.  
It includes features such as:
- Age, Education, Occupation, Hours per week, Gender, Marital Status, etc.

**Target Column**:  
`income` → `<=50K` or `>50K`

---

## 🛠️ How to Run the Streamlit App

1. Clone this repository or download the ZIP  
2. Make sure `app.py`, `salary_model.pkl`, and `scaler.pkl` are in the same folder  
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

---

## ✅ Author

**Ashlin Anna**  
Internship Project | 2025
