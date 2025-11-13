import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fire_cnn_best.keras")

# Load model
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

IMG_SIZE = 128  # should match training size

def predict_image(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0][0]
    if pred > 0.5:
        print(f"🔥 Fire Detected! (Confidence: {pred:.2f})")
    else:
        print(f"✅ No Fire (Confidence: {1 - pred:.2f})")

if __name__ == "__main__":
    # Example usage
    test_image = input("Enter path of image to test: ").strip()
    print("Shape:", img_array.shape)
    print("Max pixel value:", np.max(img_array))
    print("Min pixel value:", np.min(img_array))
    print("Sample pixels:", img_array[0, :2, :2, :])



    if os.path.exists(test_image):
        predict_image(test_image)
    else:
        print("⚠️ File not found, please give a valid path.")
