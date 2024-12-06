import os
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

dropout_value = 0.01

class AdvanceCNN(nn.Module):
    def __init__(self):
        super(AdvanceCNN, self).__init__()
        self.conv1 = nn.Sequential(
              nn.Conv2d(1, 8, 3, padding=0, bias=False),
              nn.BatchNorm2d(8),
              nn.ReLU(inplace=True),
              nn.Dropout(dropout_value)) # output_size = 26, RF = 3
        self.conv2 = nn.Sequential(
              nn.Conv2d(8, 12, 3, padding=0, bias=False),
              nn.BatchNorm2d(12),
              nn.ReLU(inplace=True),
              nn.Dropout(dropout_value)) # output_size = 24, RF = 5
        self.conv3 = nn.Sequential(
              nn.MaxPool2d(kernel_size=2, stride=2), # output_size = 12, RF = 6
              nn.Conv2d(12, 8, 1, padding=0, bias=False),
              nn.BatchNorm2d(8),
              nn.ReLU(inplace=True),
              nn.Dropout(dropout_value)) # output_size = 12, RF = 6
        self.conv4 = nn.Sequential(
              nn.Conv2d(8, 12, 3, padding=0, bias=False),
              nn.BatchNorm2d(12),
              nn.ReLU(inplace=True),
              nn.Dropout(dropout_value)) # output_size = 10, RF = 10
        self.conv5 = nn.Sequential(
              nn.Conv2d(12, 16, 3, padding=0, bias=False),
              nn.BatchNorm2d(16),
              nn.ReLU(inplace=True),
              nn.Dropout(dropout_value)) # output_size = 8, RF = 14
        self.conv6 = nn.Sequential(
              nn.Conv2d(16, 32, 3, padding=0, bias=False),
              nn.BatchNorm2d(32),
              nn.ReLU(inplace=True),
              nn.Dropout(dropout_value)) # output_size = 6, RF = 18
        self.gap = nn.AvgPool2d(kernel_size=6) # output_size = 1, RF = 28
        self.conv7 = nn.Sequential(
              nn.Conv2d(32, 10, 1, padding=0, bias=False)) # output_size = 1, RF = 28
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


def train_model():
    # Force CPU usage
    device = torch.device("cpu")
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Basic transform for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset with augmentation
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=val_transform)
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    model = AdvanceCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.075, momentum=0.9)
    
    # Train for one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Quick validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')
    
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'models/model_{timestamp}.pth')
    
    return accuracy

if __name__ == "__main__":
    train_model() 