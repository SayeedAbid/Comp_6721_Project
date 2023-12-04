import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


class SmallCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SmallCNN, self).__init__()
        # Convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, padding=1)
        # Convolutional layer (sees 16x16x16 tensor after pooling)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding=1)
        # Convolutional layer (sees 8x8x32 tensor after pooling)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Linear layer (64 * 4 * 4 = 1024) kernal3*3
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        #self.fc1 = nn.Linear(64 * 2 * 2, 512)
        #self.fc1 = nn.Linear(64 * 1 * 1, 512)
        # Linear layer (512 -> num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        # Dropout layer (p=0.5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #print("x1",x.shape)
        
        x = x.view(-1, 64 * 4 * 4)
        #x = x.view(-1, 64 * 2 * 2)
        #x = x.view(-1, 64 * 1 * 1)
        #print("x2",x.shape)
        # Add dropout layer
        x = self.dropout(x)
        # Add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        #print("x3",x.shape)
        # Add dropout layer
        x = self.dropout(x)
        # Add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        #print("x4",x.shape)
        return x
    

def load_data(data_dir, batch_size=64, train_split=0.7, valid_split=0.15):
# Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # Assume 'dataset' is your complete dataset loaded using ImageFolder
    # dataset_size = len(dataset)
    # train_size = int(0.7 * dataset_size)
    # valid_size = int(0.15 * dataset_size)
    # test_size = dataset_size - (train_size + valid_size)

    # Split the dataset into training, validation, and testing sets
    # train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
    valid_dataset = datasets.ImageFolder(root='dataset/validation', transform=transform)
    test_dataset = datasets.ImageFolder(root='Bias_test/female', transform=transform)

    # Create data loaders for each set
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
# Number of epochs to train the model
    # Instantiate the CNN
    model = SmallCNN()
    #print(model)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data_dir = 'dataset'
    n_epochs = 35
    valid_loss_min = np.Inf
    train_loader, valid_loader, test_loader = load_data(data_dir)

    model.load_state_dict(torch.load('model_old.pth'))  # Load the best model
    model.eval()  # Evaluation mode
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():  # No gradient computation during testing
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Convert predictions and targets to NumPy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # test_loss = test_loss/len(test_loader.sampler)
    # test_acc = test_correct / test_total
    # print(f'Test Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.6f}')
    test_acc = accuracy_score(all_targets, all_preds)
    print(f'Test Accuracy: {test_acc:.6f}')

    # Calculate Precision, Recall, and F1-Score
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

    # Print the results
    print(f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1-Score: {f1:.6f}')      

    # # Generate and print confusion matrix
    # cm = confusion_matrix(all_targets, all_preds)
    # # Plot confusion matrix
    # class_names = ['angry', 'bored', 'engaged', 'neutral']
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()

