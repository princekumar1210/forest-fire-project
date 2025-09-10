import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
RAW_DIR = os.path.join(BASE_DIR, "data", "images_raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Fire/Non-fire folders
FIRE_DIR = os.path.join(RAW_DIR, "fire")
NONFIRE_DIR = os.path.join(RAW_DIR, "non_fire")

IMG_SIZE = 128  # resize images

def load_images_from_folder(folder, label):
    data = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append((img, label))
        except Exception as e:
            print(f"⚠️ Could not load {filename}: {e}")
    return data

print("🔄 Loading fire images...")
fire_data = load_images_from_folder(FIRE_DIR, 1)

print("🔄 Loading non-fire images...")
nonfire_data = load_images_from_folder(NONFIRE_DIR, 0)

dataset = fire_data + nonfire_data
print(f"✅ Total images loaded: {len(dataset)}")

# Separate features and labels
X, y = zip(*dataset)
X = np.array(X) / 255.0  # normalize
y = np.array(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")

# Save to processed folder
with open(os.path.join(PROCESSED_DIR, "train.pkl"), "wb") as f:
    pickle.dump((X_train, y_train), f)

with open(os.path.join(PROCESSED_DIR, "test.pkl"), "wb") as f:
    pickle.dump((X_test, y_test), f)

print("✅ Dataset saved in processed/ folder")
