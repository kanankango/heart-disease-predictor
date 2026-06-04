import streamlit as st
import requests

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀")

st.title("🫀 Heart Disease Risk Predictor")
st.write("Fill in the patient details below to get a prediction.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG (0-2)", options=[0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment (0-2)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thal (1=normal, 2=fixed, 3=reversible)", options=[1, 2, 3])

if st.button("🔍 Predict"):
    payload = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            confidence = result["confidence"]
            if prediction == "Heart Disease":
                st.error(f"⚠️ Prediction: **{prediction}**")
            else:
                st.success(f"✅ Prediction: **{prediction}**")
            st.metric("Confidence", f"{round(confidence * 100, 1)}%")
        else:
            st.warning("⚠️ Please check your inputs and try again.")
    except Exception:
        st.error("❌ Could not connect to the prediction service. Please make sure the API is running.")