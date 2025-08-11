import streamlit as st
import numpy as np
import joblib
import sklearn


def main():
    # Load the trained model and scaler
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Page title
    st.title("🩺 Diabetes Prediction App")
    st.write("Enter patient details to predict whether they are diabetic.")

    # Input fields for features
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=79)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0, format="%.1f")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    Age = st.number_input("Age", min_value=1, max_value=120, value=33)

    # Prediction button
    if st.button("Predict"):
        # Prepare input for prediction
        user_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        user_data_scaled = scaler.transform(user_data)

        prediction = model.predict(user_data_scaled)[0]
        prediction_proba = model.predict_proba(user_data_scaled)[0][prediction]

        if prediction == 1:
            st.error(f"⚠️ The model predicts **Diabetic** with probability {prediction_proba:.2f}")
        else:
            st.success(f"✅ The model predicts **Not Diabetic** with probability {prediction_proba:.2f}")

if __name__ == "__main__":
    main()