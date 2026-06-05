import streamlit as st
import requests

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .block-container { padding: 2rem 3rem; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #e63946, #c1121f);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border: none;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .stButton>button:hover { background: #c1121f; }
    div[data-testid="metric-container"] {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #2d3250;
    }
    .section-header {
        color: #e63946;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
        <span style='font-size:48px;'>🫀</span>
        <h1 style='color:white; margin:0; font-size:2.5rem;'>Heart Disease Risk Predictor</h1>
        <p style='color:#aaa; font-size:1rem; margin-top:0.5rem;'>
            Enter patient clinical data below to assess cardiovascular risk using AI
        </p>
    </div>
    <hr style='border-color:#2d3250; margin-bottom:2rem;'>
""", unsafe_allow_html=True)

# Form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="section-header">👤 Patient Info</p>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        fbs = st.selectbox("Fasting Blood Sugar > 120?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG (0-2)", options=[0, 1, 2])

    with col2:
        st.markdown('<p class="section-header">❤️ Heart Metrics</p>', unsafe_allow_html=True)
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
        trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    with col3:
        st.markdown('<p class="section-header">🔬 Clinical Details</p>', unsafe_allow_html=True)
        cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
        exang = st.selectbox("Exercise Induced Angina?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        slope = st.selectbox("Slope of ST Segment (0-2)", options=[0, 1, 2])
        ca = st.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thal", options=[1, 2, 3], format_func=lambda x: {1:"Normal", 2:"Fixed Defect", 3:"Reversible"}[x])

    submitted = st.form_submit_button("🔍 Analyse Risk")

# Result
if submitted:
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

            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns([1,2,1])
            with col_b:
                if prediction == "Heart Disease":
                    st.error(f"⚠️  **{prediction} Detected**")
                    risk_color = "#e63946"
                else:
                    st.success(f"✅  **{prediction}**")
                    risk_color = "#2dc653"

                st.markdown(f"""
                    <div style='background:#1e2130; border-radius:12px; padding:1.5rem; 
                    text-align:center; border: 1px solid {risk_color}; margin-top:1rem;'>
                        <p style='color:#aaa; margin:0; font-size:14px;'>Model Confidence</p>
                        <p style='color:{risk_color}; font-size:3rem; font-weight:bold; margin:0;'>
                            {round(confidence * 100, 1)}%
                        </p>
                        <p style='color:#aaa; font-size:12px; margin-top:0.5rem;'>
                            Based on Random Forest Classifier — 98.54% training accuracy
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    except Exception:
        st.error("❌ Cannot connect to API. Make sure `uvicorn api:app --reload` is running.")

# Footer
st.markdown("""
    <hr style='border-color:#2d3250; margin-top:3rem;'>
    <p style='text-align:center; color:#555; font-size:12px;'>
        UCI Heart Disease Dataset · Random Forest Model · FastAPI Backend
    </p>
""", unsafe_allow_html=True)