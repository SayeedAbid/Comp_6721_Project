import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

# load dataset
def load_dataset(dataset_path):
    class_labels = os.listdir(dataset_path)
    data = []
    for class_label in class_labels:
        class_path = os.path.join(dataset_path, class_label)
        image_files = os.listdir(class_path)
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            data.append({"image_path": image_path, "class": class_label})
    return data


dataset_path = 'Dataset'
dataset = load_dataset(dataset_path)

# random samples from dataset
def get_random_samples(data, num_samples=25):
    random_samples = random.sample(data, num_samples)
    return random_samples

# Class Distribution
def plot_class_distribution(data):
    class_labels, class_counts = np.unique([sample["class"] for sample in data], return_counts=True)
    plt.bar(class_labels, class_counts)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution")
    plt.xticks(class_labels)
    plt.show()

#Sample Images
def plot_sample_images(data, num_samples=25, rows=5, cols=5):
    random_samples = get_random_samples(data, num_samples)
    fig, axes = plt.subplots(rows, cols, figsize=(8.5, 11))  # Letter-sized page
    for i, ax in enumerate(axes.ravel()):
        sample = random_samples[i]
        image_path = sample["image_path"]
        image = Image.open(image_path)
        ax.imshow(image)
        ax.set_title(f"Class: {sample['class']}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Pixel Intensity Distribution
def plot_pixel_intensity_distribution(data, num_samples=25):
    random_samples = get_random_samples(data, num_samples)
    fig, axes = plt.subplots(1, 1, figsize=(8, 5)) 
    all_pixels = []
    
    for i in range(num_samples):
        image_path = random_samples[i]["image_path"]
        image = Image.open(image_path)
        image = np.array(image)
        
        if image.shape[-1] == 3:
            # (RGB) image
            for channel, color in enumerate(['r', 'g', 'b']):
                channel_pixels = image[:, :, channel].ravel()
                all_pixels.extend(channel_pixels)
                plt.hist(channel_pixels, bins=256, range=(0, 256), color=color, alpha=0.5,  )#label=f'Channel {channel}'
        else:
            # grayscale
            intensity_pixels = image.ravel()
            all_pixels.extend(intensity_pixels)
            plt.hist(intensity_pixels, bins=256, range=(0, 256), color='gray', alpha=0.5, label='Intensity')

    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Pixel Intensity Distribution")
    plt.legend()
    plt.show()




# Usage
plot_class_distribution(dataset)
plot_sample_images(dataset)
plot_pixel_intensity_distribution(dataset)
