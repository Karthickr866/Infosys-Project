import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

# Paths for dataset and augmented data
base_folder = '/content/Infosys DataSet'  # Path to original dataset
augmented_data_path = '/content/augmented_dataset'  # Path to store both original and augmented images

# Create output directory if it doesn't exist
os.makedirs(augmented_data_path, exist_ok=True)

# Create ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Target image count per class
target_image_count = 500

# Iterate through each class (folder) in the dataset
for class_name in os.listdir(base_folder):
    class_path = os.path.join(base_folder, class_name)
    if os.path.isdir(class_path):  # Only process folders

        print(f"\nProcessing class: {class_name}")

        # Create a new directory for this class in the augmented dataset
        save_path = os.path.join(augmented_data_path, class_name)
        os.makedirs(save_path, exist_ok=True)

        # Copy original images to the new directory and prepare for augmentation
        images = []
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            if os.path.isfile(img_path):  # Ensure it's an image file
                # Copy original image to the augmented folder
                shutil.copy(img_path, save_path)

                # Read and preprocess the image for augmentation
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.resize(image, (224, 224))  # Resize to 224x224
                    images.append(image)

        # Count the number of original images
        num_original = len(images)
        print(f"Original images: {num_original}")

        # Calculate the number of augmented images needed
        needed_images = target_image_count - num_original

        if needed_images > 0:
            print(f"Augmenting {needed_images} images for class: {class_name}")

            # Convert images to numpy array and normalize them
            images_array = np.array(images).astype('float32') / 255.0

            # Generate augmented images
            total_generated = 0
            for batch in datagen.flow(images_array, batch_size=1, save_to_dir=save_path,
                                      save_prefix=class_name, save_format='jpg'):
                total_generated += 1
                if total_generated >= needed_images:
                    break
            print(f"Augmented images generated: {total_generated}")
        else:
            print(f"No augmentation needed for class: {class_name} (already has {num_original} images).")

print("\nData augmentation completed for all classes!")
