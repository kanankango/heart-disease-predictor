import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 1. Dataset load karo
df = pd.read_csv('heart.csv')

# 2. Custom Feature 1: age_chol_risk
# Age aur cholesterol dono heart disease ke major factors hain
# Inhe multiply karke ek combined risk score banaya
df['age_chol_risk'] = df['age'] * df['chol'] / 1000

# 3. Custom Feature 2: bp_category
# Blood pressure ko doctors teen categories mein dekhte hain
# Normal, Elevated, aur High - yahi clinical logic use kiya hai
def categorize_bp(bp):
    if bp < 120:
        return 0  # Normal
    elif bp < 140:
        return 1  # Elevated
    else:
        return 2  # High

df['bp_category'] = df['trestbps'].apply(categorize_bp)

# 4. Features select karo
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal', 'age_chol_risk', 'bp_category']

X = df[features]
y = df['target']

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Model train karo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Accuracy check karo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 8. Model save karo
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 9. Feature names save karo (API ko zaroorat padegi)
with open('columns.pkl', 'wb') as f:
    pickle.dump(features, f)

print("Model saved successfully!")
print("Columns saved successfully!")