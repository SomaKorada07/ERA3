import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from cnn import MNISTNet
import random
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Training configuration
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

def train():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    print("-" * 60)
    print(f"{'Epoch':^6}{'Batch':^8}{'Loss':^12}{'Accuracy':^12}")
    print("-" * 60)
    
    training_history = []
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                accuracy = 100. * correct / total
                current_loss = running_loss / (batch_idx + 1)
                
                # Print to console
                print(f"{epoch+1:^6d}{batch_idx:^8d}{current_loss:^12.4f}{accuracy:^12.2f}")
                
                current_stats = {
                    'epoch': epoch + 1,
                    'batch': batch_idx,
                    'loss': current_loss,
                    'accuracy': accuracy
                }
                training_history.append(current_stats)
                
                # Save training history to file
                with open('static/training_history.json', 'w') as f:
                    json.dump(training_history, f)
        
        # Print epoch summary
        print("-" * 60)
        print(f"Epoch {epoch+1} Summary - Loss: {current_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print("-" * 60)
    
    # Save model
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    
    # Generate predictions for 10 random test images
    model.eval()
    test_samples = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            # Get 10 random samples
            for i in range(min(10, len(data))):
                test_samples.append({
                    'image': data[i].cpu().numpy().tolist(),
                    'prediction': int(pred[i].item()),
                    'actual': int(target[i].item())
                })
            
            if len(test_samples) >= 10:
                break
    
    with open('static/test_samples.json', 'w') as f:
        json.dump(test_samples, f)

if __name__ == '__main__':
    train() 