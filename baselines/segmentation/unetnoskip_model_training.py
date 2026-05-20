import os
import pandas as pd
import time

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

import inversefed
from datagenerator_pd import DataGenerator


start = time.time()

# Device
setup = inversefed.utils.system_startup()
device, dtype = setup["device"], setup["dtype"]


epochs = 100
patience = 10
best_val_loss = float('inf')
early_stop_counter = 0


# Parameters
params = {
    "batch_size": 5,
    "imagex": 160,
    "imagey": 192,
    "imagez": 160,
    "column": "Group_bin",
}

csv_dir = 'pd_data_csv'
train_csv = os.path.join(csv_dir, "train_pd_complete_adni.csv")
val_csv = os.path.join(csv_dir, "test_pd_complete_adni.csv")


def _get_meanstd(dataset):
    channel = dataset[0][0].shape[0]
    cc = torch.cat([dataset[i][0].reshape(channel, -1) for i in range(len(dataset))], dim=1)
    data_mean = torch.mean(cc, dim=1)
    data_std = torch.std(cc, dim=1)

    data_mean = torch.as_tensor([data_mean], **setup)
    data_std = torch.as_tensor([data_std], **setup)
    
    return data_mean, data_std




train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(val_csv)

# IDs
train_IDs = train_df['Subject'].to_numpy()
test_IDs = test_df['Subject'].to_numpy()


training_dataset = DataGenerator(train_IDs, (params['imagex'], params['imagez']), 
                                train_csv, params['column'])
train_loader = DataLoader(training_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=1)



test_dataset = DataGenerator(test_IDs, (params['imagex'], params['imagez']), 
                            val_csv, params['column'])
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=1)



print('Model creating...')

# ================== WITHOUT SKIPPED CONNECTIONS ==================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.upconv(x)
        x = self.conv_block(x)
        return x

class UNetNoSkips(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        return self.final_conv(x)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(1)
        dice = (2. * intersection + self.smooth) / (
            probs.sum(1) + targets.sum(1) + self.smooth
        )

        return 1 - dice.mean()

# # Model
model = UNetNoSkips(in_channels=1, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = DiceLoss()
loss_fn = DiceLoss()
print('Model is created')


# ================== TRAINING PROCESS STARTS ==================
print('Training starts... ')
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ========== Validation ==========
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x_val, y_val in test_loader:
            x_val = x_val.to(device, dtype=torch.float32)
            y_val = y_val.to(device, dtype=torch.float32)

            logits = model(x_val)
            loss = criterion(logits, y_val)

            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss / len(train_loader):.4f} "
          f"Val Loss: {val_loss / len(test_loader):.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "unet_noskip_model_seg.pt")
        print(f"Model saved at epoch {epoch+1}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

end = time.time()
print(f"Total time taken: {(end - start)/60:.2f} minutes")