import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.train import SimpleCNN
import pytest

def test_model_architecture():
    model = SimpleCNN()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape should be (1, 10)"
    
    # Test number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 100000, f"Model has {total_params} parameters, should be less than 10000"

def test_model_accuracy():
    from torchvision import datasets, transforms
    
    # Load test data
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Load the most recent model
    model = SimpleCNN()
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    model.load_state_dict(torch.load(os.path.join(models_dir, latest_model)))
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%" 