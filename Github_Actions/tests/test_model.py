import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.train import SimpleCNN
import pytest
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

def test_device_is_cpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert str(device) == "cpu", "Device should be CPU"
    
    # Also test if model and data are on CPU
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    assert not test_input.is_cuda, "Input tensor should be on CPU"
    assert next(model.parameters()).device.type == "cpu", "Model should be on CPU"

def test_training_loss_decreasing():
    device = torch.device("cpu")
    
    # Setup data
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model and training components
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epoch_losses = []
    
    # Train for 3 epochs
    for epoch in range(3):
        epoch_loss = 0.0
        batch_count = 0
        
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_epoch_loss = epoch_loss / batch_count
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")
    
    # Check if loss is decreasing
    for i in range(1, len(epoch_losses)):
        assert epoch_losses[i] < epoch_losses[i-1], f"Loss should decrease: previous={epoch_losses[i-1]:.4f}, current={epoch_losses[i]:.4f}"

def test_validation_accuracy_increasing():
    device = torch.device("cpu")
    
    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # Initialize model and training components
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epoch_accuracies = []
    
    # Train and validate for 3 epochs
    for epoch in range(3):
        # Training
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Validation
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
        epoch_accuracies.append(accuracy)
        print(f"Epoch {epoch + 1} Validation Accuracy: {accuracy:.2f}%")
    
    # Check if accuracy is increasing
    for i in range(1, len(epoch_accuracies)):
        assert epoch_accuracies[i] > epoch_accuracies[i-1], \
            f"Accuracy should increase: previous={epoch_accuracies[i-1]:.2f}%, current={epoch_accuracies[i]:.2f}%"

def test_model_architecture():
    model = SimpleCNN()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape should be (1, 10)"
    
    # Test number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_model_accuracy():
    # Force CPU
    device = torch.device("cpu")
    
    # Load test data with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Load the most recent model
    model = SimpleCNN().to(device)
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        pytest.skip("No model file found")
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    model.load_state_dict(torch.load(os.path.join(models_dir, latest_model)))
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Model accuracy is {accuracy}%, should be > 95%"