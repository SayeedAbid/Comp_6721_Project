import cv2
import numpy as np
import os
import random

def crop_image(image, x, y, width, height):
    return image[y:y+height, x:x+width]

def add_noise(image):
    gauss = np.random.normal(0, 1, image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
    return cv2.add(image, gauss)

# Path to your dataset
dataset_path = 'Main_Dataset/neutral'
augmented_path = 'augmented/neutral'

# Create the augmented dataset directory if it doesn't exist
if not os.path.exists(augmented_path):
    os.makedirs(augmented_path)

# Process each image in the dataset
for image_name in os.listdir(dataset_path):
    # Skip non-image files
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    image_path = os.path.join(dataset_path, image_name)
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is not None:
        # Define the suffix for new images
        suffix = os.path.splitext(image_name)[0]

        # Crop parameters
        height, width, _ = image.shape
        crop_x, crop_y = random.randint(0, width // 4), random.randint(0, height // 4)
        crop_width, crop_height = width // 2, height // 2

        # Cropped image
        cropped_image = crop_image(image, crop_x, crop_y, crop_width, crop_height)
        cropped_image_name = f'{suffix}_cropped.png'
        cropped_image_path = os.path.join(augmented_path, cropped_image_name)
        cv2.imwrite(cropped_image_path, cropped_image)

        # Noisy image
        noisy_image = add_noise(image)
        noisy_image_name = f'{suffix}_noisy.png'
        noisy_image_path = os.path.join(augmented_path, noisy_image_name)
        cv2.imwrite(noisy_image_path, noisy_image)

print("Augmentation completed.")
