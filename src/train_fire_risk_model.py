"""
train_fire_risk_model.py
- Trains ML model (RandomForest) on UCI Forest Fire Dataset
- Predicts fire risk (High / Low)
- Saves trained model to models/fire_risk_model.pkl
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "forestfires.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded. Rows:", len(df))
print(df.head())

# 🔹 Basic cleaning
df = df[['temp', 'RH', 'wind', 'rain', 'area']]  # select key columns
df['area'] = df['area'].apply(lambda x: 1 if x > 0 else 0)  # fire happened? 1=yes,0=no
df.rename(columns={'RH': 'humidity'}, inplace=True)

X = df[['temp', 'humidity', 'wind', 'rain']]
y = df['area']

# 🔹 Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 Train model
print("🚀 Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔹 Save model
model_path = os.path.join(MODEL_DIR, "fire_risk_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved to:", model_path)
