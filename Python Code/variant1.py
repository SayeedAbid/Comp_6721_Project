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
class TwoLayerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(TwoLayerCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(32 * 8 * 8, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten image input
        x = x.view(-1, 32 * 8 * 8)
        # Add dropout layer
        x = self.dropout(x)
        
        x = self.fc1(x)
        return x

#dataset = datasets.ImageFolder(root=data_dir, transform=transform)
def load_data(data_dir, batch_size=64, train_split=0.7, valid_split=0.15):
# Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    valid_size = int(0.15 * dataset_size)
    test_size = dataset_size - (train_size + valid_size)

    # Split the dataset into training, validation, and testing sets
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # Create data loaders for each set
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

    return train_loader, valid_loader, test_loader




if __name__ == "__main__":
# Number of epochs to train the model
    # Instantiate the CNN
    model = TwoLayerCNN()
    #print(model)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data_dir = 'Dataset'
    n_epochs = 50
    valid_loss_min = np.Inf
    train_loader, valid_loader, test_loader = load_data('Dataset')
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
            
            optimizer.zero_grad()
            
            output = model(data)
        
            loss = criterion(output, target)
           
            loss.backward()
           
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            # calculate accuracy
            _, pred = torch.max(output, 1)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
        # Validate the model #
        
        model.eval()
        for data, target in valid_loader:
            
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
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
        
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tTraining Accuracy: {train_acc:.6f} \tValidation Loss: {valid_loss:.6f} \tValidation Accuracy: {valid_acc:.6f}')
        
        # save model 
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model_cnn.pth')
            valid_loss_min = valid_loss
    
    model.load_state_dict(torch.load('model_cnn.pth'))  # Load the best model
    model.eval()  # Evaluation mode
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():  # No gradient computation during testing
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += target.size(0)

    test_loss = test_loss/len(test_loader.sampler)
    test_acc = test_correct / test_total
    print(f'Test Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.6f}')

    # Initialize variables for predictions and ground truth
all_predictions = []
all_labels = []

# Make predictions on the test set
with torch.no_grad():
    for inputs, labels in test_loader:
        # inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate micro metrics
micro_accuracy = accuracy_score(all_labels, all_predictions)
micro_precision = precision_score(all_labels, all_predictions, average='micro')
micro_recall = recall_score(all_labels, all_predictions, average='micro')

# Calculate macro metrics
macro_accuracy = accuracy_score(all_labels, all_predictions)
macro_precision = precision_score(all_labels, all_predictions, average='macro')
macro_recall = recall_score(all_labels, all_predictions, average='macro')

# Generate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Plot confusion matrix
class_names = ['angry', 'bored', 'engaged', 'neutral']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print or use the results
print(f'Micro Accuracy: {micro_accuracy}')
print(f'Micro Precision: {micro_precision}')
print(f'Micro Recall: {micro_recall}')
print(f'Macro Accuracy: {macro_accuracy}')
print(f'Macro Precision: {macro_precision}')
print(f'Macro Recall: {macro_recall}')