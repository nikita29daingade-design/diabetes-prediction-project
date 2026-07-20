# app.py

import streamlit as st
import pickle
import numpy as np


# =========================
# Load Model
# =========================

model = pickle.load(open("best_model.pkl","rb"))

# model = pickle.load(open("final_model.pkl"))
# =========================
# Page Config-
# =========================
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# =========================
# Title
# =========================
st.title("🩺 Diabetes Prediction System")
st.markdown("### Enter patient details to predict diabetes")

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("Input Features")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 1, 100, 25)
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
smoking_history = st.sidebar.selectbox(
    "Smoking History",
    ["never", "former", "not current", "current", "ever"]
)

bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
hba1c = st.sidebar.slider("HbA1c Level", 3.0, 15.0, 5.5)
glucose = st.sidebar.slider("Blood Glucose Level", 50, 300, 100)

# =========================
# Manual Encoding (same as training)
# =========================
gender_map = {"Male": 1, "Female": 0}
smoking_map = {
    'never': 0,
    'former': 1,
    'not current': 2,
    'current': 3,
    'ever': 2
}

gender = gender_map[gender]
smoking_history = smoking_map[smoking_history]

# =========================
# Prediction Button
# =========================
if st.button("🔍 Predict"):

    import pandas as pd

    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],   
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose]
    })

    prediction = model.predict(input_data)[0]
    # prediction = model.predict(input_data)[0]

    # Get probability
    probability = model.predict_proba(input_data)[0][1]
    # =========================
    # Output Section
    # =========================
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ No Diabetes Detected")

    st.write(f"🧠 Probability of Diabetes: {round(probability * 100, 2)}%")


