#Gray Scale Conversion display
# Define the dataset path
dataset_path = '/content/Infosys DataSet'

# Preprocessing function
def preprocess_image(image):
    # Resize the image to a fixed size (e.g., 128x128)
    resized_image = cv2.resize(image, (128, 128))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve contrast
    equalized_image = cv2.equalizeHist(gray_image)

    # Normalize the image to have pixel values between 0 and 1
    normalized_image = equalized_image / 255.0

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(np.uint8(blurred_image * 255), 100, 200)

    # Stack the grayscale, blurred, and edges as channels
    preprocessed_image = np.stack([normalized_image, blurred_image, edges], axis=-1)

    return preprocessed_image

# Dictionary to store preprocessed images by class
preprocessed_data = {}

# Process each subfolder (class)
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):  # Ensure it's a directory
        print(f"Processing images for class: {class_name}")
        preprocessed_images = []

        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            if os.path.isfile(file_path):  # Ensure it's a file
                try:
                    # Read the image
                    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    if image is not None:
                        # Apply preprocessing
                        preprocessed_image = preprocess_image(image)
                        preprocessed_images.append(preprocessed_image)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        # Store preprocessed images for this class
        preprocessed_data[class_name] = np.array(preprocessed_images)
        print(f"Preprocessed {len(preprocessed_images)} images for class '{class_name}'.\n")

# Check the results
for class_name, images in preprocessed_data.items():
    print(f"Class '{class_name}': {len(images)} preprocessed images.")

