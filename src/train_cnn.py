import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load data
print("🔄 Loading processed dataset...")
with open(os.path.join(PROCESSED_DIR, "train.pkl"), "rb") as f:
    X_train, y_train = pickle.load(f)

with open(os.path.join(PROCESSED_DIR, "test.pkl"), "rb") as f:
    X_test, y_test = pickle.load(f)

print(f"✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Callbacks
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "fire_cnn_best.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Training
print("🚀 Training started...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop]
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"📊 Final Test Accuracy: {acc*100:.2f}%")

# Save final model
final_model_path = os.path.join(MODEL_DIR, "fire_cnn_final.keras")
model.save(final_model_path)
print(f"✅ Model saved to {final_model_path}")
