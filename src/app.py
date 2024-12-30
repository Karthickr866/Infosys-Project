import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

# Configure the page layout
st.set_page_config(page_title="Image Recognition For Next Era", layout="wide")

# Add custom CSS for styling
st.markdown(
    """
    <style>
    /* Background styling */
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        background-image: url('https://www.transparenttextures.com/patterns/black-linen.png'); /* Replace with your preferred image */
        background-size: cover;
    }

    /* Header styling */
    .header {
        background: linear-gradient(to right, #6a11cb, #2575fc); /* Purple to blue gradient */
        color: white;
        text-align: center;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
    }
    .header h1 {
        font-size: 36px;
        font-weight: bold;
        margin: 0;
    }
    .header h2 {
        font-size: 20px;
        font-weight: 300;
        margin: 0;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4caf50;
        color: white;
        font-size: 16px;
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }

    /* Prediction output styling */
    .results {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        margin-top: 20px;
    }
    .results h3 {
        color: #4caf50;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App header
st.markdown(
    '<div class="header"><h1>Image Recognition For Next Era</h1><h2>By Ankush Raut</h2></div>',
    unsafe_allow_html=True,
)

# Load the trained model
model_path = r"C:\Users\ankus\Downloads\final_model.h5"
try:
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define class names
class_names = ['Vijay', 'Amitabh Bachan', 'Prabhas', 'Akshay Kumar']

# Preprocess the uploaded image
def preprocess_image(image, target_size=(64, 64)):
    image_resized = cv2.resize(image, target_size)  # Resize the image
    image_array = img_to_array(image_resized) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Error: Could not process the image. Please upload a valid file.")
        else:
            # Display the uploaded image (in its original color format)
            st.image(image[:, :, ::-1], caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            processed_image = preprocess_image(image)

            # Predict the class
            prediction = model.predict(processed_image)
            predicted_class = class_names[np.argmax(prediction)]
            predicted_confidence = np.max(prediction) * 100

            # Display the results
            st.markdown('<div class="results">', unsafe_allow_html=True)
            st.write(f"<h3>Predicted Person: {predicted_class}</h3>", unsafe_allow_html=True)
            st.write(f"<h3>Confidence: {predicted_confidence:.2f}%</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
else:
    st.info("Please upload an image to start prediction.")
