import os
import cv2
import numpy as np
import random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.datasets import load_sample_image
​
​
######################################## Data Labeling ###########################################
​
dataset_path = "AI Project Final Dataset/test - original"  # Replace with the actual path of the dataset
label_mapping = {
    "angry": 1,
    "bored": 2,
    "engaged": 3,
    "neutral": 4,
}
​
for class_name, label in label_mapping.items():
    class_path = os.path.join(dataset_path, class_name)
    for i, image_name in enumerate(os.listdir(class_path)):
        # Rename the image with a label prefix (e.g., "0_image.jpg")
        new_name = f"{label}_{i}.{image_name.split('.')[-1]}"
        os.rename(os.path.join(class_path, image_name), os.path.join(class_path, new_name))
​
​
​
######################################### Data Resizing ###########################################
​
dataset_path = "AI Project Final Dataset/test - original"  # Replace with the actual path of the dataset
target_width = 48
target_height = 48
​
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        img = cv2.imread(image_path)
​
        # Resize the image to the target dimensions
        img_resized = cv2.resize(img, (target_width, target_height))
​
        # Save the resized image, overwriting the original
        cv2.imwrite(image_path, img_resized)
​
​
##################################### GrayScaling ############################################
​
input_dir = 'New folder'  # Replace with the actual path 
output_dir = 'augment'    # Replace with the actual path 
​
# Function to convert an image to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
​
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
​
# Iterate through the classes in your dataset
for class_folder in os.listdir(input_dir):
    if not os.path.isdir(os.path.join(input_dir, class_folder)):
        continue
​
    class_output_dir = os.path.join(output_dir, class_folder)
    os.makedirs(class_output_dir, exist_ok=True)
​
    # Iterate through the images in the class folder
    for filename in os.listdir(os.path.join(input_dir, class_folder)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_dir, class_folder, filename)
            image = cv2.imread(img_path)
​
            # Convert the image to grayscale
            grayscale_image = convert_to_grayscale(image)
​
            # Save the grayscale image in the output directory
            output_path = os.path.join(class_output_dir, filename)
            cv2.imwrite(output_path, grayscale_image)
​
​
​
​
####################################### Brightness and Rotation ########################################
​
dataset_root = 'Dataset\Source_data'    # Replace with the actual path
output_dir = 'Dataset\Augmented_data'   # Replace with the actual path
​
# Function to perform data augmentation (brightness correction and rotation)
def augment_image(image):
    # Random brightness correction
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.random.uniform(0.7, 1.3)
    image[:, :, 2] = image[:, :, 2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
​
    # Random rotation
    angle = random.randint(0, 360)
    h, w, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    return image
​
# Perform data augmentation and save augmented images
for class_folder in os.listdir(input_dir):
    if not os.path.isdir(os.path.join(input_dir, class_folder)):
        continue
​
    output_class_dir = os.path.join(output_dir, class_folder)
    os.makedirs(output_class_dir, exist_ok=True)
​
    for filename in os.listdir(os.path.join(input_dir, class_folder)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_dir, class_folder, filename)
            image = cv2.imread(img_path)
            augmented_image = augment_image(image)
            output_path = os.path.join(output_class_dir, filename)
            cv2.imwrite(output_path, augmented_image)