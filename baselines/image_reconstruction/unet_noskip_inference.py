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


image_path = 'inference_imgs_noskip_harm'
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


def _get_meanstd(dataset):
    channel = dataset[0][0].shape[0]
    cc = torch.cat([dataset[i][0].reshape(channel, -1) for i in range(len(dataset))], dim=1)
    data_mean = torch.mean(cc, dim=1)
    data_std = torch.std(cc, dim=1)

    data_mean = torch.as_tensor([data_mean], **setup)
    data_std = torch.as_tensor([data_std], **setup)
    
    return data_mean, data_std

def psnr_metric(y_true, y_pred, max_val=1.0):
    mse = F.mse_loss(y_true, y_pred)
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10 * torch.log10(max_val**2 / mse)
    psnr = torch.clamp(psnr, 0.0, 100.0)
    return psnr



test_df = pd.read_csv(val_csv)
test_IDs = test_df['Subject'].to_numpy()

test_dataset = DataGenerator(test_IDs, (params['imagex'], params['imagez']), 
                            val_csv, params['column'])
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=1)

print('Computing mean and std...')
# dm, ds = _get_meanstd(training_dataset)

# Mean and std values were computed earlier
dm = torch.as_tensor([-0.5934], **setup) 
ds = torch.as_tensor([0.3767], **setup)


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
model.load_state_dict(torch.load("unet_no_skip_model_2d_harm.pt", map_location=device))
model.eval()

results = []
mse_total, psnr_total, ssim_total = 0.0, 0.0, 0.0
count = 0

with torch.no_grad():
    for idx, (x_test, y_test) in enumerate(test_loader):
        x_test, y_test = x_test.to(device, dtype=torch.float32), y_test.to(device, dtype=torch.float32)
        outputs = model(x_test)

        # Convert to numpy
        outputs_np = outputs.squeeze().cpu().numpy()
        y_test_np = y_test.squeeze().cpu().numpy()

        batch_size = outputs.size(0)
        for i in range(batch_size):
            # ----- Metrics per image -----
            mse_val = F.mse_loss(outputs[i], y_test[i], reduction="mean").item()
            psnr_val = psnr(y_test_np[i], outputs_np[i], data_range=1.0)
            ssim_val = ssim(y_test_np[i], outputs_np[i], data_range=1.0)

            # Save reconstructed + ground truth images
            img_id = f"img_{idx}_{i}"
            torchvision.utils.save_image(outputs[i], os.path.join(image_path, f"{img_id}_reconstruction.png"))
            torchvision.utils.save_image(y_test[i], os.path.join(image_path, f"{img_id}_groundtruth.png"))

            # Append to results list
            results.append({
                "img_id": img_id,
                "mse": mse_val,
                "psnr": psnr_val,
                "ssim": ssim_val
            })

            mse_total += mse_val
            psnr_total += psnr_val
            ssim_total += ssim_val
            count += 1

# Save all results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("unet_noskip_test_results.csv", index=False)

# Averages
test_mse = mse_total / count
test_psnr = psnr_total / count
test_ssim = ssim_total / count

print(f"Test Results -> MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | SSIM: {test_ssim:2.4f}")
print("Per-image results saved to test_results.csv")


