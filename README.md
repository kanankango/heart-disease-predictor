# Heart Disease Prediction — ML API Project

## Project Overview
A machine learning web application that predicts heart disease risk using a Random Forest classifier trained on the UCI Heart Disease dataset.

Two custom engineered features were created:
- **age_chol_risk**: Age × Cholesterol ÷ 1000 — combines two major clinical risk factors into a single score (mirrors real medical risk assessment logic)
- **bp_category**: Blood pressure binned into Normal / Elevated / High — follows standard clinical thresholds used by doctors

Model Accuracy: **98.54%**

---

## Setup — Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 1 — Train the Model

```bash
python train_model.py
```

This creates `model.pkl` and `columns.pkl` in the project folder.

---

## Step 2 — Start the API

```bash
uvicorn api:app --reload
```

API runs at: `http://localhost:8000`

---

## Step 3 — Start the Web App

Open a new terminal, activate venv, then:

```bash
streamlit run app.py
```

App runs at: `http://localhost:8501`

---

## API Endpoints

### GET /health
```bash
curl http://localhost:8000/health
```
Response: `{"status": "ok"}`

### POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":160,"exang":0,"oldpeak":1.5,"slope":1,"ca":0,"thal":2}'
```
Response: `{"prediction": "Heart Disease", "confidence": 0.87}`

---

## Files
| File | Purpose |
|------|---------|
| train_model.py | Trains and saves the ML model |
| api.py | FastAPI backend serving predictions |
| app.py | Streamlit frontend UI |
| model.pkl | Saved trained model |
| columns.pkl | Saved feature column names |
| predictions.db | SQLite database of all predictions |
| app.log | API request logs |