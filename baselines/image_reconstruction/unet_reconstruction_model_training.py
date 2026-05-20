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

image_path = 'reconstructed_imgs'
os.makedirs(image_path, exist_ok=True)

ground_truth_name = 'input_brain2d.png'
reconstructed_name = 'reconstruction_noskipped_8k.png'

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

def psnr_metric(y_true, y_pred, max_val=1.0):
    mse = F.mse_loss(y_true, y_pred)
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10 * torch.log10(max_val**2 / mse)
    psnr = torch.clamp(psnr, 0.0, 100.0)
    return psnr


train_df = pd.read_csv(train_csv)
#val_df = pd.read_csv(val_csv, nrows=402)
test_df = pd.read_csv(val_csv)

# IDs
train_IDs = train_df['Subject'].to_numpy()
#val_IDs = val_df['Subject'].to_numpy()
test_IDs = test_df['Subject'].to_numpy()


training_dataset = DataGenerator(train_IDs, (params['imagex'], params['imagez']), 
                                train_csv, params['column'])
train_loader = DataLoader(training_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=1)


#val_dataset = DataGenerator(val_IDs, (params['imagex'], params['imagez']), 
#                           val_csv, params['column'])
#val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=1)


test_dataset = DataGenerator(test_IDs, (params['imagex'], params['imagez']), 
                            val_csv, params['column'])
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=1)

print('Computing mean and std...')
# dm, ds = _get_meanstd(training_dataset)

# Mean and std values were computed earlier
dm = torch.as_tensor([-0.5934], **setup) 
ds = torch.as_tensor([0.3767], **setup)

print('Model creating...')

# ================== MODEL WITH SKIPPED CONNECTIONS ==================
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
        conv = self.conv_block(x)
        pooled = self.pool(conv)
        return conv, pooled

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
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
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)

        return self.final_conv(x)


# ================== WITHOUT SKIPPED CONNECTIONS ==================
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         return x

# class EncoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv_block = ConvBlock(in_channels, out_channels)
#         self.pool = nn.MaxPool2d(2)

#     def forward(self, x):
#         x = self.conv_block(x)
#         x = self.pool(x)
#         return x

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv_block = ConvBlock(out_channels, out_channels)

#     def forward(self, x):
#         x = self.upconv(x)
#         x = self.conv_block(x)
#         return x

# class UNetNoSkips(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1):
#         super().__init__()
#         self.enc1 = EncoderBlock(in_channels, 64)
#         self.enc2 = EncoderBlock(64, 128)
#         self.enc3 = EncoderBlock(128, 256)
#         self.enc4 = EncoderBlock(256, 512)

#         self.bottleneck = ConvBlock(512, 1024)

#         self.dec1 = DecoderBlock(1024, 512)
#         self.dec2 = DecoderBlock(512, 256)
#         self.dec3 = DecoderBlock(256, 128)
#         self.dec4 = DecoderBlock(128, 64)

#         self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

#     def forward(self, x):
#         x = self.enc1(x)
#         x = self.enc2(x)
#         x = self.enc3(x)
#         x = self.enc4(x)

#         x = self.bottleneck(x)

#         x = self.dec1(x)
#         x = self.dec2(x)
#         x = self.dec3(x)
#         x = self.dec4(x)

#         return self.final_conv(x)


# # Model
model = UNet(in_channels=1, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
loss_fn = nn.MSELoss()
print('Model is created')


# ================== TRAINING PROCESS STARTS ==================
# # Uncomment if you need to train a new model
print('Training starts... ')
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss = 0
    psnr_val = 0
    model.eval()
    with torch.no_grad():
        for x_val, y_val in test_loader:
            x_val, y_val = x_val.to(device, dtype=torch.float32), y_val.to(device, dtype=torch.float32)
            outputs = model(x_val)
            loss = criterion(outputs, y_val)
            psnr_value = psnr_metric(outputs, y_val)
            psnr_val += psnr_value.item()
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss / len(train_loader):.4f} Val Loss: {val_loss / len(test_loader):.4f} Val PSNR: {psnr_val / len(test_loader):.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "unet_skip_model_2d_harmonized.pt")
        print("Model saved at epoch {}".format(epoch+1))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
# ================== TRAINING PROCESS ENDS ==================


# ground_truth = torchvision.io.read_image(os.path.join(image_path, ground_truth_name)).float() / 255.0
# ground_truth = ground_truth.to(device=device, dtype=dtype)
# ground_truth = ground_truth.unsqueeze(0)
# print('Image is loaded')

# model.eval()
# model.zero_grad()
# ground_truth.requires_grad_(True)

# ================== LOADING MODEL WEIGHTS ==================
#Comment if you retrain model and do not need to load something
# full_state_dict = torch.load("unet_img2img_model_skip_connection.pth")
# model_dict = model.state_dict()
# model_dict.update(full_state_dict)
# model.load_state_dict(model_dict, strict=False)
# print('Model weights are loaded')

# model_output = model(ground_truth)
# target_loss = loss_fn(model_output, ground_truth)

# input_gradient = torch.autograd.grad(
#     outputs=target_loss, 
#     inputs=model.parameters(), 
#     create_graph=False,
#     retain_graph=False,
#     allow_unused=True
# )

# filtered_gradients = []
# for i, (grad, param) in enumerate(zip(input_gradient, model.parameters())):
#     if grad is not None:
#         filtered_gradients.append(grad.detach().clone())

# input_gradient = filtered_gradients

# config = dict(
#     signed=False,
#     boxed=True,
#     cost_fn='sim',
#     indices='def',
#     weights='equal', 
#     lr=0.0005, 
#     optim='adam',
#     restarts=1,
#     max_iterations=1000,
#     total_variation=1e-5,
#     init='randn',                
#     filter='median',
#     lr_decay=True,
#     scoring_choice='loss'
# )
# print('Configs are set up')

# # Perform gradient inversion
# rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
# print(rec_machine)
# output, stats = rec_machine.reconstruct(input_gradient, ground_truth, img_shape=(1, params['imagex'], params['imagez']))
# print('Gradient inversion is performed')


# output = torch.as_tensor(output, device=device, dtype=dtype)
# output = output.mul(ds).add(dm)
# torchvision.utils.save_image(output, os.path.join(image_path, reconstructed_name))
# print(f"Reconstruction saved")


# # Evaluate reconstruction quality
# output_np = output.detach().cpu().numpy().squeeze() 
# ground_truth_np = ground_truth.detach().cpu().numpy().squeeze()

# test_mse = (output.detach() - ground_truth).pow(2).mean()
# feat_mse = (model(output.detach()) - model(ground_truth)).pow(2).mean()  
# test_psnr = psnr(output_np, ground_truth_np, data_range=1.0)
# test_ssim = ssim(output_np, ground_truth_np, data_range=1.0)

# print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
#           f"| PSNR: {test_psnr:4.2f} | SSIM: {test_ssim:2.4f}")



end = time.time()
print(f"Total time taken: {(end - start)/60:.2f} minutes")