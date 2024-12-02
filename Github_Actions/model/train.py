import os
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)  # 28x28x1 -> 26x26x8
        self.conv2 = nn.Conv2d(8, 10, kernel_size=3)  # 26x26x8 -> 24x24x10
        self.pool = nn.MaxPool2d(2, stride=2)  # 24x24x10 -> 12x12x10
        self.conv3 = nn.Conv2d(10, 16, kernel_size=3)  # 12x12x10 -> 10x10x16
        self.fc1 = nn.Linear(10 * 10 * 16, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 10 * 10 * 16)
        x = self.fc1(x)
        return x

def show_augmented_samples(dataset, num_samples=5):
    # Create a figure
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    # Get some random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Original image (before normalization)
        orig_img = dataset.data[idx].numpy()
        
        # Augmented image
        aug_img, _ = dataset[idx]
        aug_img = aug_img.squeeze().numpy()
        
        # Plot original
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # Plot augmented
        axes[1, i].imshow(aug_img, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Augmented')
    
    plt.tight_layout()
    
    # Create samples directory if it doesn't exist
    if not os.path.exists('samples'):
        os.makedirs('samples')
    
    # Save the figure
    plt.savefig('samples/augmented_samples.png')
    plt.close()

def train_model():
    # Force CPU usage
    device = torch.device("cpu")
    
    # Define augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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
    
    # Show and save augmented samples
    show_augmented_samples(train_dataset)
    print("Augmented samples have been saved to 'samples/augmented_samples.png'")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
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