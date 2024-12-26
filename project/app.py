import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import logging

# Suppress Streamlit warnings
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Define class names manually
class_names = ['Akshay Kumar', 'Amitabh Bachchan', 'Prabhas', 'Vijay']  # Add more class names if necessary

# Define the path to the model file
model_path = r"C:\\Users\\Karthik\\Desktop\\project\\model.h5"

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    """
    Load the trained TensorFlow model.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# Function to preprocess the image
def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess the uploaded image to match the model's input requirements.
    """
    image = image.resize(target_size)  # Resize the image
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Streamlit app interface
st.title("Image Classification with Streamlit")
st.write("""
Upload an image, and the model will predict the class.
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying...")

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])  # Get the index of the highest probability
    predicted_class = class_names[predicted_class_index]  # Map to class name
    confidence = np.max(predictions)  # Get confidence score

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
