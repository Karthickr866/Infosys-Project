#Display
# Preprocessing function (same as in your dataset code)
def preprocess_image(image):
    resized_image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    normalized_image = equalized_image / 255.0
    blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)
    edges = cv2.Canny(np.uint8(blurred_image * 255), 100, 200)
    preprocessed_image = np.stack([normalized_image, blurred_image, edges], axis=-1)
    return preprocessed_image

def display_preprocessed_image_from_path(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at '{image_path}'.")
        return

    try:
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Unable to read the image at '{image_path}'.")
            return

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Plot the different channels
        plt.figure(figsize=(12, 4))

        # Grayscale channel
        plt.subplot(1, 3, 1)
        plt.imshow(preprocessed_image[:, :, 0], cmap='gray')
        plt.title("Grayscale")
        plt.axis('off')

        # Blurred channel
        plt.subplot(1, 3, 2)
        plt.imshow(preprocessed_image[:, :, 1], cmap='gray')
        plt.title("Blurred")
        plt.axis('off')

        # Edges channel
        plt.subplot(1, 3, 3)
        plt.imshow(preprocessed_image[:, :, 2], cmap='gray')
        plt.title("Edges")
        plt.axis('off')

        plt.suptitle(f"Preprocessed Image: {os.path.basename(image_path)}", fontsize=14)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")