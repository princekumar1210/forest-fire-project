import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Paths
TRAIN_PATH = r"C:\Users\Prince\Desktop\forest-fire-project\data\processed\train.pkl"
TEST_PATH = r"C:\Users\Prince\Desktop\forest-fire-project\data\processed\test.pkl"
MODEL_PATH = r"C:\Users\Prince\Desktop\forest-fire-project\models\fire_cnn.h5"

# Load dataset
print("🔄 Loading dataset...")
X_train, y_train = joblib.load(TRAIN_PATH)
X_test, y_test = joblib.load(TEST_PATH)

print(f"✅ Dataset loaded: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary classification
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
print("🚀 Training started...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"📊 Test Accuracy: {acc*100:.2f}%")

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
