def preprocess_image(image_path, target_size=(64, 64)):

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Resize the image
    image_resized = cv2.resize(image, target_size)

    # Convert to array and normalize pixel values
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    return image_array

# Path to the image you want to predict
image_path = "/content/Infosys DataSet/Prabhas/Prabhas_1.jpg"  # Replace with your test image path

try:
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Predict the class
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]  # Map index to class name

    # Print the result
    print(f"The model predicts this image belongs to: {predicted_class}")
except Exception as e:
    print(f"Error: {e}")
