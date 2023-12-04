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
        #
        # print training/validation statistics 
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tTraining Accuracy: {train_acc:.6f} \tValidation Loss: {valid_loss:.6f} \tValidation Accuracy: {valid_acc:.6f}')
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model_old.pth')
            valid_loss_min = valid_loss


def evaluate_model(model, test_loader, criterion):
    model.load_state_dict(torch.load('model_old.pth'))  # Load the best model
    model.eval()  # Evaluation mode
    test_loss = 0.0
    test_loss = 0.0
    all_preds = []
    all_targets = []
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():  # No gradient computation during testing
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())
            all_true_labels.extend(target.cpu().numpy())

    # Convert predictions and targets to NumPy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_loss = test_loss / len(test_loader.sampler)
    test_acc = accuracy_score(all_true_labels, all_predictions)

    return all_preds, all_targets, test_loss, test_acc

if __name__ == "__main__":
    # Instantiate the CNN
    model = SmallCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data_dir = 'dataset'
    n_epochs = 35
    train_loader, valid_loader, test_loader = load_data(data_dir)

    
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=35)

    # Evaluate the model on the test set for the current fold
    all_preds, all_targets, test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    test_acc = accuracy_score(all_targets, all_preds)
    print(f'Test Accuracy: {test_acc:.6f}')

    # Calculate and print metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

    # Print the results
    print(f'Precision: {precision:.6f}, Recall: {recall:.6f}, F1-Score: {f1:.6f}')   

    # Generate and print confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    # Plot confusion matrix
    class_names = ['angry', 'bored', 'engaged', 'neutral']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()   


