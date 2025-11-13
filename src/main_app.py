"""
Final Integrated Forest Fire Detection Dashboard
- CNN Image Classifier
- ML Weather Fire Risk Predictor
- Fire Risk Map Viewer
- Simulated SMS/Email Alerts
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import joblib
import subprocess
import cv2
import os
import sys
print(sys.executable)

# --------------------- Load Models ---------------------
CNN_MODEL_PATH = "../models/fire_cnn_best.keras"
ML_MODEL_PATH = "../models/fire_risk_model.pkl"
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
fire_risk_model = joblib.load(ML_MODEL_PATH)

# --------------------- Tkinter UI ---------------------
root = tk.Tk()
root.title("🔥 Forest Fire Detection Dashboard")
root.geometry("800x750")
root.config(bg="#181818")

tk.Label(root, text="🔥 Forest Fire Monitoring & Alert System", bg="#181818",
         fg="white", font=("Arial", 20, "bold")).pack(pady=10)

img_label = tk.Label(root, bg="#222")
img_label.pack(pady=10)
result_label = tk.Label(root, text="Select an image for fire detection...",
                        bg="#181818", fg="lightgray", font=("Arial", 14))
result_label.pack(pady=10)

# --------------------- NEW IMAGE FIRE DETECTION BUTTON AND CODE ---------------------
fire_detected = False
def detect_fire_image():
    global fire_detected
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return
    # Show preview
    img = Image.open(file_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    # Predict (EXACT ML backend)
    img_cv = cv2.imread(file_path)
    if img_cv is None:
        messagebox.showerror("Error", "Could not read image.")
        return
    img_cv = cv2.resize(img_cv, (128, 128))
    img_cv = img_cv.astype("float32") / 255.0
    img_cv = np.expand_dims(img_cv, axis=0)
    pred = cnn_model.predict(img_cv)[0][0]
    if pred > 0.5:
        fire_detected = True
        result_label.config(text=f"🔥 Fire Detected! (Confidence: {pred:.2f})", fg="red")
    else:
        fire_detected = False
        result_label.config(text=f"✅ No Fire (Confidence: {1-pred:.2f})", fg="green")

# --------------------- WEATHER RISK PREDICTION (UNCHANGED) ---------------------
tk.Label(root, text="🌦️ Weather Inputs (for Fire Risk Prediction)",
         bg="#181818", fg="white", font=("Arial", 16, "bold")).pack(pady=10)
weather_frame = tk.Frame(root, bg="#181818")
weather_frame.pack(pady=5)
labels = ["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Rain (mm)"]
entries = []
for i, lbl in enumerate(labels):
    tk.Label(weather_frame, text=lbl, bg="#181818", fg="lightgray", font=("Arial", 12)).grid(row=i, column=0, sticky="w", pady=5)
    entry = tk.Entry(weather_frame, width=10, font=("Arial", 12))
    entry.grid(row=i, column=1, padx=10)
    entries.append(entry)
result_weather = tk.Label(root, text="", bg="#181818", fg="lightgray", font=("Arial", 14))
result_weather.pack(pady=10)
def predict_weather_risk():
    try:
        temp = float(entries[0].get())
        humidity = float(entries[1].get())
        wind = float(entries[2].get())
        rain = float(entries[3].get())
        X_new = np.array([[temp, humidity, wind, rain]])
        pred = fire_risk_model.predict(X_new)[0]
        if pred == 1:
            result_weather.config(text="⚠️ High Fire Risk (Weather)", fg="red")
            trigger_alert()
        else:
            result_weather.config(text="✅ Low Fire Risk (Weather)", fg="green")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values")

def trigger_alert():
    if fire_detected:
        alert_msg = "🚨 ALERT: Fire detected in image and high weather risk!\n(SMS/Email would be sent to authorities)"
    else:
        alert_msg = "⚠️ ALERT: High fire risk detected based on weather!\n(SMS/Email would be sent to monitoring team)"
    messagebox.showwarning("Alert Triggered", alert_msg)
    print(alert_msg)  # console log

# --------------------- NEW SHOW HEATMAP BUTTON CODE ---------------------




def show_heatmap_result():
    try:
        script_path = os.path.abspath("show_heatmap.py")
        result = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            messagebox.showerror("Heatmap Error", result.stderr)
        else:
            messagebox.showinfo("Heatmap", "Fire risk heatmap generated and opened in browser.")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate or open heatmap:\n{e}")











# --------------------- BUTTONS ---------------------
btn_frame = tk.Frame(root, bg="#181818")
btn_frame.pack(pady=20)

tk.Button(btn_frame, text="Detect Fire", command=detect_fire_image,
          bg="#444", fg="white", font=("Arial", 12, "bold"),
          padx=15, pady=8).grid(row=0, column=0, padx=10)

tk.Button(btn_frame, text="Predict Fire Risk", command=predict_weather_risk,
          bg="#444", fg="white", font=("Arial", 12, "bold"),
          padx=15, pady=8).grid(row=0, column=1, padx=10)

tk.Button(btn_frame, text="Heatmap", command=show_heatmap_result,
          bg="#444", fg="white", font=("Arial", 12, "bold"),
          padx=15, pady=8).grid(row=0, column=2, padx=10)

tk.Button(btn_frame, text="Exit", command=root.destroy,
          bg="#aa3333", fg="white", font=("Arial", 12, "bold"),
          padx=15, pady=8).grid(row=0, column=3, padx=10)

root.mainloop()
