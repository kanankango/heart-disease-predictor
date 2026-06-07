import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle

# 1. Dataset load karo
df = pd.read_csv('heart.csv')

# 2. Original Custom Features
df['age_chol_risk'] = df['age'] * df['chol'] / 1000

def categorize_bp(bp):
    if bp < 120:
        return 0
    elif bp < 140:
        return 1
    else:
        return 2

df['bp_category'] = df['trestbps'].apply(categorize_bp)

# 3. NEW Custom Features
# Heart Rate Reserve — max heart rate minus resting BP difference
# Clinically used to assess cardiovascular fitness
df['heart_rate_reserve'] = df['thalach'] - (220 - df['age'])

# Cholesterol per age ratio
# Older patients with high cholesterol = higher risk
df['chol_per_age'] = df['chol'] / df['age']

# ST index — oldpeak normalized by slope
# Measures severity of ST depression relative to slope
df['st_index'] = df['oldpeak'] * (df['slope'] + 1)

print("New features created!")
print(f"heart_rate_reserve sample: {df['heart_rate_reserve'].head(3).values}")
print(f"chol_per_age sample: {df['chol_per_age'].head(3).values}")
print(f"st_index sample: {df['st_index'].head(3).values}")

# 4. Features select karo
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal',
            'age_chol_risk', 'bp_category',
            'heart_rate_reserve', 'chol_per_age', 'st_index']

X = df[features]
y = df['target']

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Better model — Gradient Boosting
print("\nTraining Gradient Boosting model...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_accuracy = accuracy_score(y_test, gb_model.predict(X_test))
print(f"Gradient Boosting Accuracy: {gb_accuracy * 100:.2f}%")

# 7. Random Forest bhi try karo
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# 8. Best model choose karo
if gb_accuracy >= rf_accuracy:
    best_model = gb_model
    print(f"\nBest Model: Gradient Boosting ({gb_accuracy * 100:.2f}%)")
else:
    best_model = rf_model
    print(f"\nBest Model: Random Forest ({rf_accuracy * 100:.2f}%)")

# 9. Save karo
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(features, f)

print("\nModel saved successfully!")
print("Columns saved successfully!")