import streamlit as st
import pickle
import pandas as pd

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

model = pickle.load(open('model.pkl', 'rb'))

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #0d0d0d !important; }
    .block-container { padding: 2rem 4rem !important; }
    header[data-testid="stHeader"] { background: #0d0d0d !important; }
    header[data-testid="stHeader"] button { color: white !important; background: transparent !important; }
    header[data-testid="stHeader"] svg { fill: white !important; stroke: white !important; }
    #MainMenu, footer { visibility: hidden !important; }
    .stDeployButton { visibility: hidden !important; }
    input, textarea {
        background-color: #1a1a2e !important;
        color: white !important;
        border: 1px solid #3f3f5a !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #1a1a2e !important;
        border: 1px solid #3f3f5a !important;
        color: white !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] span { color: white !important; }
    div[role="listbox"] { background-color: #1a1a2e !important; }
    div[role="option"] { background-color: #1a1a2e !important; color: white !important; }
    div[role="option"]:hover { background-color: #7c3aed !important; }
    label, p { color: #a0a0b0 !important; font-size: 13px !important; }
    .stFormSubmitButton > button {
        background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        padding: 0.8rem !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
        letter-spacing: 1px !important;
        margin-top: 1.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1.5rem 0;'>
        <div style='font-size:56px;'>🫀</div>
        <h1 style='margin:0.3rem 0; font-size:2.8rem; font-weight:700;
            background: linear-gradient(90deg, #a855f7, #ec4899);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Heart Disease Risk Predictor
        </h1>
        <p style='color:#6b7280; font-size:0.9rem; letter-spacing:2px; margin:0;'>
            AI-POWERED CARDIOVASCULAR RISK ASSESSMENT
        </p>
    </div>
    <div style='height:1px; background:linear-gradient(90deg, transparent, #3f3f5a, transparent); margin-bottom:2rem;'></div>
""", unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
            <div style='background:#111122; border:1px solid #2a2a3d; 
            border-radius:10px; padding:0.7rem 1rem; margin-bottom:1rem;'>
            <span style='color:#8888aa; font-size:11px; font-weight:600; letter-spacing:2px;'>
            👤 PATIENT INFO</span></div>
        """, unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        fbs = st.selectbox("Fasting Blood Sugar > 120?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG (0-2)", options=[0, 1, 2])

    with col2:
        st.markdown("""
            <div style='background:#111122; border:1px solid #2a2a3d; 
            border-radius:10px; padding:0.7rem 1rem; margin-bottom:1rem;'>
            <span style='color:#8888aa; font-size:11px; font-weight:600; letter-spacing:2px;'>
            ❤️ HEART METRICS</span></div>
        """, unsafe_allow_html=True)
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
        trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    with col3:
        st.markdown("""
            <div style='background:#111122; border:1px solid #2a2a3d; 
            border-radius:10px; padding:0.7rem 1rem; margin-bottom:1rem;'>
            <span style='color:#8888aa; font-size:11px; font-weight:600; letter-spacing:2px;'>
            🔬 CLINICAL DETAILS</span></div>
        """, unsafe_allow_html=True)
        cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
        exang = st.selectbox("Exercise Induced Angina?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        slope = st.selectbox("Slope of ST Segment (0-2)", options=[0, 1, 2])
        ca = st.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thal", options=[1, 2, 3], format_func=lambda x: {1:"Normal", 2:"Fixed Defect", 3:"Reversible"}[x])

    submitted = st.form_submit_button("⚡ ANALYSE CARDIOVASCULAR RISK")

if submitted:
    age_chol_risk = age * chol / 1000
    bp_category = 0 if trestbps < 120 else (1 if trestbps < 140 else 2)
    heart_rate_reserve = thalach - (220 - age)
    chol_per_age = chol / age
    st_index = oldpeak * (slope + 1)

    input_dict = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal,
        'age_chol_risk': age_chol_risk, 'bp_category': bp_category,
        'heart_rate_reserve': heart_rate_reserve,
        'chol_per_age': chol_per_age, 'st_index': st_index
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][prediction]
    result = "Heart Disease" if prediction == 1 else "No Heart Disease"
    color = "#ef4444" if prediction == 1 else "#22c55e"

    st.markdown("<br>", unsafe_allow_html=True)

    # Dribbble style cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
            <div style='background:#111827; border-radius:16px; padding:1.2rem 1.5rem;
            border:1px solid #1f2937;'>
                <p style='color:#6b7280; font-size:10px; margin:0; font-weight:600; letter-spacing:1.5px;'>DIAGNOSIS</p>
                <p style='color:{color}; font-size:1.1rem; font-weight:700; margin:0.5rem 0 0 0;'>
                    {"⚠️ Positive" if prediction==1 else "✅ Negative"}
                </p>
                <span style='background:{"#450a0a" if prediction==1 else "#052e16"}; color:{color}; 
                font-size:10px; padding:2px 8px; border-radius:20px; font-weight:600;'>
                    {"High Risk" if prediction==1 else "Low Risk"}</span>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div style='background:#111827; border-radius:16px; padding:1.2rem 1.5rem;
            border:1px solid #1f2937;'>
                <p style='color:#6b7280; font-size:10px; margin:0; font-weight:600; letter-spacing:1.5px;'>CONFIDENCE</p>
                <p style='color:#a855f7; font-size:2rem; font-weight:800; margin:0.5rem 0 0.3rem 0;'>
                    {round(confidence*100, 1)}%
                </p>
                <div style='height:4px; background:#1f2937; border-radius:99px;'>
                    <div style='height:4px; width:{round(confidence*100)}%; background:#a855f7; border-radius:99px;'></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        chol_status = "Normal" if chol < 200 else ("Borderline" if chol < 240 else "High")
        chol_color = "#22c55e" if chol < 200 else ("#f59e0b" if chol < 240 else "#ef4444")
        st.markdown(f"""
            <div style='background:#111827; border-radius:16px; padding:1.2rem 1.5rem;
            border:1px solid #1f2937;'>
                <p style='color:#6b7280; font-size:10px; margin:0; font-weight:600; letter-spacing:1.5px;'>CHOLESTEROL</p>
                <p style='color:#f59e0b; font-size:2rem; font-weight:800; margin:0.5rem 0 0 0;'>
                    {chol}<span style='font-size:11px; color:#6b7280; margin-left:3px;'>mg/dl</span>
                </p>
                <span style='color:{chol_color}; font-size:10px; font-weight:600;'>{chol_status}</span>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div style='background:#111827; border-radius:16px; padding:1.2rem 1.5rem;
            border:1px solid #1f2937;'>
                <p style='color:#6b7280; font-size:10px; margin:0; font-weight:600; letter-spacing:1.5px;'>HEART RATE</p>
                <p style='color:#ec4899; font-size:2rem; font-weight:800; margin:0.5rem 0 0 0;'>
                    {thalach}<span style='font-size:11px; color:#6b7280; margin-left:3px;'>bpm</span>
                </p>
                <span style='color:#6b7280; font-size:10px;'>Max Heart Rate</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_b, _ = st.columns([1, 2, 1])
    with col_b:
        st.markdown(f"""
            <div style='background:#111827; border-radius:20px; padding:2rem;
            text-align:center; border:1px solid {color}33; border-top:4px solid {color};'>
                <p style='color:#6b7280; margin:0; font-size:11px; letter-spacing:3px; font-weight:600;'>
                    OVERALL RISK SCORE</p>
                <p style='color:{color}; font-size:4rem; font-weight:800; margin:0.5rem 0;'>
                    {round(confidence*100, 1)}%
                </p>
                <div style='height:8px; background:#1f2937; border-radius:99px; margin:1rem 0;'>
                    <div style='height:8px; width:{round(confidence*100)}%;
                        background:linear-gradient(90deg, {color}66, {color});
                        border-radius:99px;'></div>
                </div>
                <p style='color:#9ca3af; font-size:13px; margin:0;'>
                    {"⚠️ High risk detected. Please consult a cardiologist." if prediction==1 else "✅ Low risk. Keep maintaining a healthy lifestyle!"}
                </p>
                <p style='color:#374151; font-size:11px; margin:0.8rem 0 0 0; letter-spacing:1px;'>
                    GRADIENT BOOSTING · 98.54% ACCURACY · 5 ENGINEERED FEATURES
                </p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("""
    <div style='height:1px; background:linear-gradient(90deg, transparent, #3f3f5a, transparent); margin-top:3rem;'></div>
    <p style='text-align:center; color:#374151; font-size:11px; margin-top:1rem; letter-spacing:2px;'>
        UCI HEART DISEASE DATASET · GRADIENT BOOSTING · STREAMLIT
    </p>
""", unsafe_allow_html=True)