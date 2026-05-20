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
from scipy.spatial.distance import directed_hausdorff
import numpy as np


start = time.time()

# Device
setup = inversefed.utils.system_startup()
device, dtype = setup["device"], setup["dtype"]


image_path = 'segs_noskip'
os.makedirs(image_path, exist_ok=True)


# Parameters
params = {
    "batch_size": 1,
    "imagex": 160,
    "imagey": 192,
    "imagez": 160,
    "column": "Group_bin",
}

csv_dir = 'pd_data_csv'
train_csv = os.path.join(csv_dir, "train_pd_complete_adni.csv")
val_csv = os.path.join(csv_dir, "test_pd_complete_adni.csv")


def compute_dice(pred, target, smooth=1e-6):
    intersection = np.sum(pred * target)
    return (2. * intersection + smooth) / (
        np.sum(pred) + np.sum(target) + smooth
    )


def compute_hausdorff(pred, target):
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)

    if len(pred_points) == 0 or len(target_points) == 0:
        return np.nan  # avoid crash if empty mask

    hd1 = directed_hausdorff(pred_points, target_points)[0]
    hd2 = directed_hausdorff(target_points, pred_points)[0]

    return max(hd1, hd2)



test_df = pd.read_csv(val_csv)
test_IDs = test_df['Subject'].to_numpy()

test_dataset = DataGenerator(test_IDs, (params['imagex'], params['imagez']), 
                            val_csv, params['column'])
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=1)



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


# # Model
model = UNetNoSkips(in_channels=1, out_channels=1).to(device)

# ================== TESTING LOOP WITH CSV ==================
print("Starting testing...")

# Load best model
model.load_state_dict(torch.load("unet_noskip_model_seg.pt", map_location=device))
model.eval()

results = []
dice_total = 0.0
hd_total = 0.0
count = 0

with torch.no_grad():
    for idx, (x_test, y_test) in enumerate(test_loader):

        x_test = x_test.to(device, dtype=torch.float32)
        y_test = y_test.to(device, dtype=torch.float32)

        logits = model(x_test)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        batch_size = preds.size(0)

        for i in range(batch_size):

            pred_np = preds[i].squeeze().cpu().numpy()
            target_np = y_test[i].squeeze().cpu().numpy()

            # ----- Metrics per image -----
            dice_val = compute_dice(pred_np, target_np)
            hd_val = compute_hausdorff(pred_np, target_np)

            img_id = f"img_{idx}_{i}"

            # ----- Save Masks -----
            torchvision.utils.save_image(
                preds[i],
                os.path.join(image_path, f"{img_id}_prediction.png")
            )

            torchvision.utils.save_image(
                y_test[i],
                os.path.join(image_path, f"{img_id}_groundtruth.png")
            )

            results.append({
                "img_id": img_id,
                "dice": dice_val,
                "hausdorff": hd_val
            })

            dice_total += dice_val
            if not np.isnan(hd_val):
                hd_total += hd_val

            count += 1

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("unet_noskip_test_results.csv", index=False)

# Averages
mean_dice = dice_total / count
mean_hd = hd_total / count

print(f"Test Results -> Dice: {mean_dice:.4f} | Hausdorff: {mean_hd:.4f}")
print("Per-image results saved to unet_noskip_test_results.csv")


