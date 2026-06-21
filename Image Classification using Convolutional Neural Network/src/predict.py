import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_image_classifier.keras"
LABELS_PATH = BASE_DIR / "models" / "class_names.txt"
IMAGE_SIZE = (224, 224)


def load_class_names():
    if not LABELS_PATH.exists():
        raise FileNotFoundError("Class names file not found. Train the model first.")

    return LABELS_PATH.read_text(encoding="utf-8").splitlines()


def predict_image(image_path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    image = load_img(image_path, target_size=IMAGE_SIZE)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    return class_names[predicted_index], confidence


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    label, confidence = predict_image(image_path)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
