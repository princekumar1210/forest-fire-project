"""
fire_risk_model.py
Trains a simple ML model to predict forest fire risk based on weather data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# ---------- STEP 1: Load dataset ----------
DATA_PATH = "../data/forestfires.csv"  # keep this file here
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Download UCI Forest Fire Dataset and place it in /data/ as forestfires.csv")

df = pd.read_csv(DATA_PATH)

# ---------- STEP 2: Select useful columns ----------
features = ["temp", "RH", "wind", "rain"]   # temperature, humidity, wind speed, rainfall
df = df[features + ["area"]]                # area = burned area

# Convert to binary classification: fire risk (1 = fire, 0 = no fire)
df["fire_risk"] = df["area"].apply(lambda x: 1 if x > 0 else 0)
X = df[features]
y = df["fire_risk"]

# ---------- STEP 3: Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- STEP 4: Train Model ----------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------- STEP 5: Evaluate ----------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Fire Risk Model Accuracy: {acc*100:.2f}%")

# ---------- STEP 6: Save Model ----------
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/fire_risk_model.pkl")
print("💾 Model saved to models/fire_risk_model.pkl")
