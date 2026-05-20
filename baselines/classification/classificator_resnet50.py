import torch
import torch.nn as nn
import random
import numpy as np
from torchvision import models

# Change according to the data
params = {
    "imagex": 160,
    "imagey": 160,
    "classes_num": 2
}

# Set seeds for reproducibility
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

# ResNet50 model modified for grayscale input
class ResNet50Gray(nn.Module):
    def __init__(self, num_classes=params["classes_num"]):
        super(ResNet50Gray, self).__init__()
        
        # Load pretrained ResNet50
        self.model = models.resnet50(weights=None)  # use weights="IMAGENET1K_V1" if you want pretrained
        
        # Change first conv layer to accept 1 channel instead of 3
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Change the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ResNet18 model modified for grayscale input
class ResNet18Gray(nn.Module):
    def __init__(self, num_classes=params["classes_num"]):
        super(ResNet18Gray, self).__init__()
        
        # Load pretrained ResNet18
        self.model = models.resnet18(weights=None)  # use weights="IMAGENET1K_V1" if you want pretrained
        
        # Change first conv layer to accept 1 channel instead of 3
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Change the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Training function
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X, y in dataloader:
        X = X.to(device)
        y = y.long().to(device)
        optimizer.zero_grad()
        outputs = model(X)
        
        loss_result = criterion(outputs, y)
        loss = loss_result[0] if isinstance(loss_result, tuple) else loss
            
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.long().to(device)
            outputs = model(X)
            
            loss_result = criterion(outputs, y)
            loss = loss_result[0] if isinstance(loss_result, tuple) else loss
                
            running_loss += loss.item() * X.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total
