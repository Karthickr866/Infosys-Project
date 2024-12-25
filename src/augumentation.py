import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Augmentation function
def augment_images(class_name, images, save_path, augment_count=250):
    datagen = ImageDataGenerator(
        rotation_range=30,      # Random rotations
        width_shift_range=0.2,  # Horizontal shifts
        height_shift_range=0.2, # Vertical shifts
        shear_range=0.2,        # Shearing transformations
        zoom_range=0.2,         # Zoom in/out
        horizontal_flip=True,   # Random horizontal flipping
        brightness_range=[0.8, 1.2], # Random brightness changes
        fill_mode='nearest'     # Filling pixels outside boundaries
    )
    
    # Convert the images to numpy array
    images_array = np.array(images)
    images_array = images_array.astype('float32') / 255.0  # Normalize
    
    # Create directory for augmented images
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Generate augmented images
    total_generated = 0
    for image in datagen.flow(images_array, batch_size=1, save_to_dir=save_path,
                              save_prefix=class_name, save_format='jpg'):
        total_generated += 1
        if total_generated >= augment_count:
            break

# Augment each class
augmented_data_path = "./augmented_dataset"
for class_name, images in data.items():
    save_path = os.path.join(augmented_data_path, class_name)
    augment_images(class_name, images, save_path, augment_count=300)