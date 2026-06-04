import pickle
import sqlite3
import logging
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd

# 1. Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 2. FastAPI app banao
app = FastAPI(title="Heart Disease Predictor API")

# 3. Model load karo (sirf ek baar — server start hone par)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

logger.info("Service started successfully!")

# 4. Database setup
def init_db():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_features TEXT,
            prediction TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 5. Input validation model
class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# 6. Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# 7. Predict endpoint
@app.post("/predict")
def predict(data: HeartInput):
    logger.info(f"POST /predict called at {datetime.now()}")

    try:
        # Custom features banao
        age_chol_risk = data.age * data.chol / 1000

        if data.trestbps < 120:
            bp_category = 0
        elif data.trestbps < 140:
            bp_category = 1
        else:
            bp_category = 2

        # Input dataframe banao
        input_dict = {
            'age': data.age,
            'sex': data.sex,
            'cp': data.cp,
            'trestbps': data.trestbps,
            'chol': data.chol,
            'fbs': data.fbs,
            'restecg': data.restecg,
            'thalach': data.thalach,
            'exang': data.exang,
            'oldpeak': data.oldpeak,
            'slope': data.slope,
            'ca': data.ca,
            'thal': data.thal,
            'age_chol_risk': age_chol_risk,
            'bp_category': bp_category
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_columns]

        # Prediction karo
        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][prediction]

        result = "Heart Disease" if prediction == 1 else "No Heart Disease"

        # Database mein save karo
        try:
            conn = sqlite3.connect('predictions.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (timestamp, input_features, prediction, confidence)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                json.dumps(input_dict),
                result,
                round(float(confidence), 4)
            ))
            conn.commit()
            conn.close()
            logger.info("Prediction saved to database")
        except Exception as db_error:
            logger.error(f"Database error: {db_error}")

        return {
            "prediction": result,
            "confidence": round(float(confidence), 2)
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))