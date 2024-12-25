import matplotlib.pyplot as plt
import cv2

def visualize_preprocessed_images(preprocessed_data):
    plt.figure(figsize=(15, 10))
    for idx, (class_name, images) in enumerate(preprocessed_data.items()):
        if len(images) > 0:
            preprocessed_image = images[0]
            plt.subplot(2, len(preprocessed_data) // 2 + 1, idx + 1)
            if preprocessed_image.ndim == 3:
                plt.imshow(preprocessed_image[:, :, 0], cmap='gray')
            else:
                plt.imshow(preprocessed_image, cmap='gray')
            plt.title(f"Class: {class_name}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()

def show_prediction_results(example_image, processed_image, predicted_class):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.imshow(processed_image[:, :, 2], cmap='gray')
    plt.axis("off")

    plt.show()
