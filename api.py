import os
import pickle
import sqlite3
import logging
import json
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd


# ==========================================
# 1. Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================
# 2. Logging setup
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ==========================================
# 3. FastAPI app
# ==========================================

app = FastAPI(
    title="Heart Disease Predictor API"
)


# ==========================================
# 4. Load trained model
# ==========================================

try:
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "columns.pkl"), "rb") as f:
        feature_columns = pickle.load(f)

    logger.info("Model and feature columns loaded successfully")

except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise


# ==========================================
# 5. Database setup
# ==========================================

DB_PATH = os.path.join(BASE_DIR, "predictions.db")


def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_features TEXT,
                prediction TEXT,
                confidence REAL
            )
        """)

        conn.commit()
        conn.close()

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Database initialization error: {e}")


init_db()


# ==========================================
# 6. Input validation model
# ==========================================

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


# ==========================================
# 7. Root endpoint
# ==========================================

@app.get("/")
def root():
    return {
        "message": "Heart Disease Predictor API is running"
    }


# ==========================================
# 8. Health endpoint
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ==========================================
# 9. Prediction endpoint
# ==========================================

@app.post("/predict")
def predict(data: HeartInput):

    logger.info("POST /predict called")

    try:

        # ------------------------------------------
        # Custom features
        # ------------------------------------------

        age_chol_risk = data.age * data.chol / 1000

        heart_rate_reserve = (
            data.thalach - (220 - data.age)
        )

        chol_per_age = data.chol / data.age

        st_index = data.oldpeak * (data.slope + 1)


        # ------------------------------------------
        # Blood pressure category
        # ------------------------------------------

        if data.trestbps < 120:
            bp_category = 0

        elif data.trestbps < 140:
            bp_category = 1

        else:
            bp_category = 2


        # ------------------------------------------
        # Create input dictionary
        # ------------------------------------------

        input_dict = {
            "age": data.age,
            "sex": data.sex,
            "cp": data.cp,
            "trestbps": data.trestbps,
            "chol": data.chol,
            "fbs": data.fbs,
            "restecg": data.restecg,
            "thalach": data.thalach,
            "exang": data.exang,
            "oldpeak": data.oldpeak,
            "slope": data.slope,
            "ca": data.ca,
            "thal": data.thal,

            "heart_rate_reserve": heart_rate_reserve,
            "chol_per_age": chol_per_age,
            "st_index": st_index,
            "age_chol_risk": age_chol_risk,
            "bp_category": bp_category
        }


        # ------------------------------------------
        # Create DataFrame
        # ------------------------------------------

        input_df = pd.DataFrame([input_dict])


        # Make sure columns match training columns
        input_df = input_df[feature_columns]


        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        prediction = model.predict(input_df)[0]

        confidence = model.predict_proba(input_df)[0][prediction]


        # ------------------------------------------
        # Result
        # ------------------------------------------

        result = (
            "Heart Disease"
            if prediction == 1
            else "No Heart Disease"
        )


        # ------------------------------------------
        # Save prediction to database
        # ------------------------------------------

        try:

            conn = sqlite3.connect(DB_PATH)

            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO predictions
                (
                    timestamp,
                    input_features,
                    prediction,
                    confidence
                )
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                json.dumps(input_dict),
                result,
                round(float(confidence), 4)
            ))

            conn.commit()
            conn.close()

            logger.info("Prediction saved to database")

        except Exception as db_error:

            logger.error(
                f"Database error: {db_error}"
            )


        # ------------------------------------------
        # Return response
        # ------------------------------------------

        return {
            "prediction": result,
            "confidence": round(float(confidence), 2)
        }


    except Exception as e:

        logger.error(
            f"Prediction error: {e}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )