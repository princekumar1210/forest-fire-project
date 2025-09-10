import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# Paths
BASE_DIR = r"C:\Users\Prince\Desktop\forest-fire-project\data\images_raw"
FIRE_DIR = os.path.join(BASE_DIR, "fire")
NONFIRE_DIR = os.path.join(BASE_DIR, "nonfire")

# Parameters
IMG_SIZE = 128  # resize images
data = []
labels = []

# Load fire images
for img in os.listdir(FIRE_DIR):
    try:
        img_path = os.path.join(FIRE_DIR, img)
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data.append(img_array)
        labels.append(1)  # fire = 1
    except:
        pass

# Load nonfire images
for img in os.listdir(NONFIRE_DIR):
    try:
        img_path = os.path.join(NONFIRE_DIR, img)
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data.append(img_array)
        labels.append(0)  # nonfire = 0
    except:
        pass

# Convert to numpy
data = np.array(data) / 255.0  # normalize
labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Save datasets
os.makedirs(r"C:\Users\Prince\Desktop\forest-fire-project\data\processed", exist_ok=True)
joblib.dump((X_train, y_train), r"C:\Users\Prince\Desktop\forest-fire-project\data\processed\train.pkl")
joblib.dump((X_test, y_test), r"C:\Users\Prince\Desktop\forest-fire-project\data\processed\test.pkl")

print("✅ Images preprocessed and saved in data/processed/")
