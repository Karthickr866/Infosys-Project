import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(dataset_path, target_size=(224, 224), augment=False, augment_count=250):
    valid_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    data = {}
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            images = []
            for file in os.listdir(class_path):
                if file.lower().endswith(valid_formats):
                    file_path = os.path.join(class_path, file)
                    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    if image is not None:
                        image = cv2.resize(image, target_size)
                        images.append(image)

            if augment:
                save_path = os.path.join(dataset_path, f"augmented/{class_name}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                augment_images(class_name, images, save_path, datagen, augment_count)

            data[class_name] = np.array(images)
    return data

def augment_images(class_name, images, save_path, datagen, augment_count):
    images_array = np.array(images).astype('float32') / 255.0
    total_generated = 0
    for image in datagen.flow(images_array, batch_size=1, save_to_dir=save_path, save_prefix=class_name, save_format='jpg'):
        total_generated += 1
        if total_generated >= augment_count:
            break
