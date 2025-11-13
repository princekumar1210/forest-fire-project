"""
fire_risk_mapping.py
- Generates random coordinates with temp, humidity, wind, rain
- Uses trained RandomForest model to predict fire risk
- Visualizes risk as heatmap using matplotlib
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---- CONFIG ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fire_risk_model.pkl")

# ---- Load model ----
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("✅ Model loaded:", MODEL_PATH)

# ---- Create mock data (simulate coordinates & weather) ----
np.random.seed(42)
num_points = 100
data = pd.DataFrame({
    "lat": np.random.uniform(10, 30, num_points),
    "lon": np.random.uniform(70, 90, num_points),
    "temp": np.random.uniform(15, 45, num_points),
    "humidity": np.random.uniform(10, 90, num_points),
    "wind": np.random.uniform(0, 10, num_points),
    "rain": np.random.uniform(0, 5, num_points),
})

# ---- Predict risk ----
features = data[["temp", "humidity", "wind", "rain"]]
data["risk"] = model.predict(features)

print("📊 Predicted fire risk for mock coordinates:")
print(data.head())

# ---- Plot heatmap ----
plt.figure(figsize=(7,6))
plt.scatter(data["lon"], data["lat"],
            c=data["risk"], cmap="coolwarm", s=80, edgecolors='k')
plt.colorbar(label="Fire Risk (0=Low, 1=High)")
plt.title("Simulated Fire Risk Mapping")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.tight_layout()

output_path = os.path.join(PROJECT_ROOT, "models", "risk_map.png")
plt.savefig(output_path)
plt.show()
print("✅ Heatmap saved to:", output_path)
