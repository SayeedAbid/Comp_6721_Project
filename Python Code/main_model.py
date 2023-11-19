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


#dataset = datasets.ImageFolder(root=data_dir, transform=transform)
def load_data(data_dir, batch_size=64, train_split=0.7, valid_split=0.15):
# Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # Assume 'dataset' is your complete dataset loaded using ImageFolder
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
    model = SmallCNN()
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
            # clear the gradients of all optimized variables
            #print(f"Input batch size: {data.size(0)}, Target batch size: {target.size(0)}")
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            #print(f"Output batch size: {output.size(0)}, Target batch size: {target.size(0)}")
            
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
'''
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
'''