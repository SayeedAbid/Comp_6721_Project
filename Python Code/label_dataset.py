import os

# Define the directory where your classes' folders are located
data_dir = 'Dataset'  # Update this path to the location of your dataset

# Go through each class directory
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        # Get all the image files in the directory
        images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Sort images to keep the renaming consistent
        images.sort()
        # Rename each image file
        for idx, image in enumerate(images):
            # Define the new filename
            new_filename = f"{class_name}_{idx+1}.jpg"  # Indexing starts at 1
            # Get the current image path
            image_path = os.path.join(class_dir, image)
            # Define the new image path
            new_image_path = os.path.join(class_dir, new_filename)
            # Rename the file
            os.rename(image_path, new_image_path)
            print(f"Renamed {image_path} to {new_image_path}")
