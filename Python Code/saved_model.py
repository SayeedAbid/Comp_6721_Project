import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from smallcnn import SmallCNN  # Import your VGG16 model class
from PIL import Image
# Load the model
model = SmallCNN()  # Initialize the model
model.load_state_dict(torch.load('model_cnn.pth'))
model.eval()  # Set to evaluation mode

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Assuming the dataset is in the same format as during training
data_dir = 'Dataset'
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Process the dataset
#for inputs, labels in data_loader:
#    outputs = model(inputs)
    # Process outputs (like getting predictions, etc.)


def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    print("Input image shape:", image.shape)  # Debug input shape

    output = model(image)
    print("Output shape:", output.shape)  # Debug output shape

    _, predicted = torch.max(output[0], 0)  # Get prediction for the first image
    class_index = predicted.item()
    return dataset.classes[class_index]  # Map index to class name




# Example usage
image_path = 'Dataset/angry/angry_98.jpg'#Dataset\angry\angry_1.jpg
predicted_class = classify_image(image_path)
print(f'The image is classified as: {predicted_class}')
