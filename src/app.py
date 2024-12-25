import os
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Print current working directory (optional, for debugging)
print("Current Working Directory:", os.getcwd())

# Configure the page layout to wide
st.set_page_config(layout="wide", page_title="Advanced Image Recognition")

# Load your trained model
model = load_model("/Users/navyasrivanga/1/project/models/face_recognition_model_final.h5")  # Replace with your model's filename

# Class labels as provided
class_labels = ['Prabhas', 'Amitabh Bachchan', 'vijay', 'Akshay Kumar']  # Update with your classes

# After loading your model, print its input shape
print("Model Input Shape:", model.input_shape)

# Define class images with proper paths
class_images = {
    'Prabhas': '/Users/navyasrivanga/1/project/data/datasets/raw/My dataset/Prabhas/1.jpg',
    'Amitabh Bachchan': '/Users/navyasrivanga/1/project/data/datasets/raw/My dataset/Amitabh Bachchan/1.jpeg',
    'vijay': '/Users/navyasrivanga/1/project/data/datasets/raw/My dataset/vijay/0a41b0bd215617fae3bca036e804ac26.jpg',
    'Akshay Kumar': '/Users/navyasrivanga/1/project/data/datasets/raw/My dataset/Akshay Kumar/0a0df79221b794470d2e416986ca812d.jpg',
}

# Function to preprocess the input image
def preprocess_image(image, target_size=(64, 64)):  # Change to model's input size
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to the target size (64x64)
    resized_image = cv2.resize(gray_image, target_size)
    
    # Equalize and blur for better contrast and quality
    equalized_image = cv2.equalizeHist(resized_image)
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    
    # Normalize the image values to be between 0 and 1
    normalized_image = blurred_image / 255.0
    
    # Standardize image by removing mean and scaling by standard deviation
    mean, std = cv2.meanStdDev(normalized_image)
    standardized_image = (normalized_image - mean[0][0]) / (std[0][0] + 1e-8)
    
    # Resize again to the target size (64x64) to match model input size
    final_image = cv2.resize(standardized_image, target_size)
    
    # Expand dimensions to fit the model's input shape (height, width, channels)
    final_image = np.expand_dims(final_image, axis=-1)  # Add channel dimension (grayscale)
    final_image = np.repeat(final_image, 3, axis=-1)  # Convert grayscale to RGB (3 channels)
    
    # Add batch dimension (1, 64, 64, 3)
    return np.expand_dims(final_image, axis=0)  # Now (1, 64, 64, 3)


# Add custom styles for the Streamlit app
st.markdown(
    """
    <style>
        body {
            background-color: black;  /* Change background to black */
        }
        .header {
            background-color: black;
            padding: 20px;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: black;  /* Set button color to dark purple */
            color: white;
            border: none;
            border-radius: 20px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            width: 500px;
        }
        .stButton>button:hover {
            background-color: #5a1a89;  /* Slightly lighter purple on hover */
        }
        .class-images {
            display: flex;
            justify-content: space-evenly;
            padding: 10px;
            gap: 20px;
            border-radius: 20px;
        }
        .class-images img {
            width: 250px;
            height: 250px;
            object-fit: cover;
            border-radius: 20px;
        }
        .output-section {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }
        .output-image img {
            width: 250px;
            height: 250px;
            object-fit: cover;
        }
        .results-box {
            background-color: black;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="header">ADVANCED IMAGE RECOGNITION FOR NEXT ERA</div>', unsafe_allow_html=True)

# Split the screen into two halves for uploading image and displaying result
left_column, right_column = st.columns([1, 1])

# Left column: Upload image, "Predict" button, and results
with left_column:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    predict_button = st.button("Predict")

    if uploaded_file is not None:
        st.write("Image upload is successful. Please predict.")

    if predict_button and uploaded_file is not None:
        # Convert the uploaded file to a byte array and decode only once
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Corrected decoding

        # Preprocess the image (resize, normalize, etc.)
        preprocessed_image = preprocess_image(image)

        # Predict the class
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)
        prediction_prob = prediction[0][predicted_class] * 100

        # Display results and the image in the right column
        with right_column:
            st.markdown('<div class="output-section">', unsafe_allow_html=True)
            # Display the image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.markdown('<div class="output-image">', unsafe_allow_html=True)
            st.image(image_rgb, caption="Uploaded Image (Prediction)")
            st.markdown('</div>', unsafe_allow_html=True)

            # Display the results
            st.markdown('<div class="results-box">', unsafe_allow_html=True)
            st.subheader("Results")
            st.write(f"**Predicted Class:** {class_labels[predicted_class]}")
            st.write(f"**Confidence:** {prediction_prob:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Class References Section
st.subheader("Class References")
class_images_container = st.container()
with class_images_container:
    cols = st.columns(4)
    for idx, (class_name, image_path) in enumerate(class_images.items()):
        with cols[idx]:
            class_image = cv2.imread(image_path)
            if class_image is not None:
                class_image_rgb = cv2.cvtColor(class_image, cv2.COLOR_BGR2RGB)
                resized_class_image = cv2.resize(class_image_rgb, (250, 250))  # Ensure consistent size
                st.image(resized_class_image, caption=class_name, use_container_width=True)  # Display fixed size class images
