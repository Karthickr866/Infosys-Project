import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, target_size)
    image_array = img_to_array(image_resized) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_class(model, image_path, class_names, target_size=(64, 64)):
    processed_image = preprocess_image(image_path, target_size)
    prediction = model.predict(processed_image)
    return class_names[np.argmax(prediction)]
