import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.train import AdvanceCNN
import pytest
from torchvision import datasets, transforms


def test_model_architecture():
    model = AdvanceCNN()
    
    # Check total parameter count (should be less than 20K)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20_000, f'Model has {total_params} parameters, exceeding 20K limit'
    
    # Check for batch normalization
    has_batchnorm = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_batchnorm, 'Model must include batch normalization layers'
    
    # Check for dropout
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, 'Model must include dropout layers'
    
    # Check for FC layer or GAP
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    has_gap = any(isinstance(m, torch.nn.AvgPool2d) for m in model.modules())
    assert has_fc or has_gap, 'Model must include either fully connected layer or global average pooling'
    
    print('All architecture checks passed!')
