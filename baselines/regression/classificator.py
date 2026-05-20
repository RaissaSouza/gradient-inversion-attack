import torch
import torch.nn as nn
import random
import numpy as np


# Change according to the data
params = {
    "imagex": 160,
    "imagey": 160,
    "classes_num": 1
}

# Set seeds for reproducibility
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

    
class SFCN(nn.Module):
    def __init__(self):
        super(SFCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.norm5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.conv6 = nn.Conv2d(256, 64, kernel_size=1, padding=0)
        self.norm6 = nn.BatchNorm2d(64)
        
        self.avgpool = nn.AvgPool2d(2)
        self.dropout = nn.Dropout(0.2)
        
        self.flattened_size = self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size, 96)
        self.fc2 = nn.Linear(96, params["classes_num"])
        
    def _get_flattened_size(self):
        x = torch.zeros(1, 1, params['imagex'], params['imagey'])
        x = self.pool1(self.norm1(self.conv1(x)))
        x = self.pool2(self.norm2(self.conv2(x)))
        x = self.pool3(self.norm3(self.conv3(x)))
        x = self.pool4(self.norm4(self.conv4(x)))
        x = self.pool5(self.norm5(self.conv5(x)))
        x = self.norm6(self.conv6(x))
        x = self.avgpool(x)
        x = self.dropout(x)
        return x.view(1, -1).shape[1]
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.norm1(self.conv1(x))))
        x = self.pool2(torch.relu(self.norm2(self.conv2(x))))
        x = self.pool3(torch.relu(self.norm3(self.conv3(x))))
        x = self.pool4(torch.relu(self.norm4(self.conv4(x))))
        x = self.pool5(torch.relu(self.norm5(self.conv5(x))))
        x = torch.relu(self.norm6(self.conv6(x)))
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_abs_error = 0.0
    total = 0

    for X, y in dataloader:
        X = X.to(device)
        y = y.float().to(device)

        optimizer.zero_grad()
        outputs = model(X).squeeze()

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # ---- accumulate ----
        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        running_abs_error += torch.abs(outputs - y).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_mae = running_abs_error / total

    return epoch_loss, epoch_mae



# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_abs_error = 0.0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.float().to(device)

            outputs = model(X).squeeze()
            loss = criterion(outputs, y)

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            running_abs_error += torch.abs(outputs - y).sum().item()
            total += batch_size

    epoch_loss = running_loss / total
    epoch_mae = running_abs_error / total

    return epoch_loss, epoch_mae
