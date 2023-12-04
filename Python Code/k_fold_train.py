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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

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
        # Linear layer (512 -> num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        # Dropout layer (p=0.5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 4 * 4)
        # Add dropout layer
        x = self.dropout(x)
        # Add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # Add dropout layer
        x = self.dropout(x)
        # Add 2nd hidden layer, with relu activation function
        x = self.fc2(x)

        return x


def load_data(data_dir):
# Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
    valid_dataset = datasets.ImageFolder(root='dataset/validation', transform=transform)
    test_dataset = datasets.ImageFolder(root='dataset/test', transform=transform)

    # Create data loaders for each set
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

    return train_loader, valid_loader, test_loader



def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=35):
    valid_loss_min = np.Inf
    
    for epoch in range(1, n_epochs+1):
        
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_correct = 0
        valid_correct = 0
        train_total = 0
        valid_total = 0
        
        # Train the model #
        
        model.train()
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            
            loss = criterion(output, target)
            
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()*data.size(0)
            
            _, pred = torch.max(output, 1)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
        # Validate the model #
        
        model.eval()
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            
            loss = criterion(output, target)
             
            valid_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)
            valid_correct += pred.eq(target.view_as(pred)).sum().item()
            valid_total += target.size(0)
        
        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        # calculate average accuracy
        train_acc = train_correct / train_total
        valid_acc = valid_correct / valid_total
        # print training/validation statistics 
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tTraining Accuracy: {train_acc:.6f} \tValidation Loss: {valid_loss:.6f} \tValidation Accuracy: {valid_acc:.6f}')
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model_cnn.pth')
            valid_loss_min = valid_loss


def evaluate_model(model, test_loader, criterion):
    model.load_state_dict(torch.load('model_cnn.pth'))  # Load the best model
    model.eval()  # Evaluation mode
    test_loss = 0.0
    test_loss = 0.0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():  # No gradient computation during testing
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)
            all_predictions.extend(pred.cpu().numpy())
            all_true_labels.extend(target.cpu().numpy())

    test_loss = test_loss / len(test_loader.sampler)
    test_acc = accuracy_score(all_true_labels, all_predictions)

    return test_loss, test_acc, all_predictions, all_true_labels

if __name__ == "__main__":
    # Instantiate the CNN
    model = SmallCNN()
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data_dir = 'dataset'
    n_epochs = 10
    train_loader, valid_loader, test_loader = load_data(data_dir)

    n_splits = 10  # 10-fold cross-validation

    # Create the dataset and use StratifiedKFold to maintain class distribution in each fold
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(full_dataset)), full_dataset.targets)):
        print(f"Fold {fold + 1}/{n_splits}")

        # Create data loaders for each fold
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(full_dataset, batch_size=64, sampler=train_sampler)
        valid_loader = DataLoader(full_dataset, batch_size=64, sampler=valid_sampler)
    
        train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=35)


        # Evaluate the model on the test set for the current fold
        test_loader = DataLoader(full_dataset, batch_size=64, sampler=valid_sampler)  # Use validation data for testing
        test_loss, test_acc, predictions, true_labels = evaluate_model(model, test_loader, criterion)

        # Calculate and print metrics
        macro_precision = precision_score(true_labels, predictions, average='macro')
        micro_precision = precision_score(true_labels, predictions, average='micro')

        macro_recall = recall_score(true_labels, predictions, average='macro')
        micro_recall = recall_score(true_labels, predictions, average='micro')

        macro_f1 = f1_score(true_labels, predictions, average='macro')
        micro_f1 = f1_score(true_labels, predictions, average='micro')

        accuracy = accuracy_score(true_labels, predictions)

        print(f'Macro Precision: {macro_precision:.6f}, Micro Precision: {micro_precision:.6f}')
        print(f'Macro Recall: {macro_recall:.6f}, Micro Recall: {micro_recall:.6f}')
        print(f'Macro F1-score: {macro_f1:.6f}, Micro F1-score: {micro_f1:.6f}')
        print(f'Accuracy: {accuracy:.6f}')



