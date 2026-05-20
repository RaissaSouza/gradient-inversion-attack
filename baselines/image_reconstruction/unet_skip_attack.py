import os
import pandas as pd
import time
import csv
import os
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


epochs = 1
patience = 10
best_val_loss = float('inf')
early_stop_counter = 0

image_path = 'reconstructed_imgs_skip'
os.makedirs(image_path, exist_ok=True)

# ground_truth_name = 'input_brain2d.png'
# reconstructed_name = 'reconstruction_noskipped_8k.png'

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
    data_min = torch.min(cc, dim=1).values
    data_max = torch.max(cc, dim=1).values


    data_mean = torch.as_tensor([data_mean], **setup)
    data_std = torch.as_tensor([data_std], **setup)
    data_min = torch.as_tensor([data_min], **setup)
    data_max = torch.as_tensor([data_max], **setup)
    
    return data_mean, data_std, data_min, data_max

def psnr_metric(y_true, y_pred, max_val=1.0):
    mse = F.mse_loss(y_true, y_pred)
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10 * torch.log10(max_val**2 / mse)
    psnr = torch.clamp(psnr, 0.0, 100.0)
    return psnr


train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv, nrows=402)
test_df = pd.read_csv(val_csv).tail(5)

# IDs
train_IDs = train_df['Subject'].to_numpy()
val_IDs = val_df['Subject'].to_numpy()
test_IDs = test_df['Subject'].to_numpy()


training_dataset = DataGenerator(train_IDs, (params['imagex'], params['imagez']), 
                                train_csv, params['column'])
train_loader = DataLoader(training_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=1)


test_dataset = DataGenerator(test_IDs, (params['imagex'], params['imagez']), 
                            val_csv, params['column'])
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=1)



print('Computing mean and std...')
dm, ds, di, dx = _get_meanstd(training_dataset)
print(dm)
print(ds)
print(di)
print(dx)

dmt, dst, dit, dxt = _get_meanstd(test_dataset)
print(dmt)
print(dst)
print(dit)
print(dxt)

# Mean and std values were computed earlier
#dm = torch.as_tensor([-0.5934], **setup) 
#ds = torch.as_tensor([0.3767], **setup)

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
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
loss_fn = nn.MSELoss()
print('Model is created')

# Load best model
model.load_state_dict(torch.load("unet_skip_model_2d_harmonized.pt", map_location=device))
model.eval()
print('Model weights are loaded')

# config = dict(
#     signed=False,
#     boxed=True,
#     cost_fn='sim',
#     indices='def',
#     weights='equal', 
#     lr=0.0005, 
#     optim='adam',
#     restarts=5,
#     max_iterations=8_000,
#     total_variation=1e-5,
#     init='randn',                
#     filter='none',
#     lr_decay=True,
#     scoring_choice='loss'
# )

config = dict(
      signed=False,
      boxed=True,
      cost_fn='max',
      indices='def',
      weights='equal',
      lr=0.1,
      optim='adam',
      restarts=1,
      max_iterations=24000,
      total_variation=0.00001,
      init='randn',
      filter='none',
      lr_decay=True,
      scoring_choice='loss'
    )

#False	True	max	0.05	0.0001	none
#False	False	max	0.1	0.00001	none -> second try


print('Configs are set up')

# ================== TRAINING PROCESS STARTS ==================
# Uncomment if you need to train a new model

# Prepare CSV data storage
metrics_data = []

print('Training starts... ')
for epoch in range(epochs):
    #model.train()
    bs = 0

    for x, y in train_loader:
        x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)

        #print("x range:", x.min().item(), x.max().item())
        #print("y range:", y.min().item(), y.max().item())

        #x = (x - dm[None, :, None, None]) / ds[None, :, None, None]
        #y = (y - dm[None, :, None, None]) / ds[None, :, None, None]

        #print("x range:", x.min().item(), x.max().item())
        #print("y range:", y.min().item(), y.max().item())
        torchvision.utils.save_image(y, os.path.join(image_path, "original.png"))

        #optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        # Get gradients for inversion BEFORE optimizer.step()
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=model.parameters(),
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )



        #filtered_gradients = [g.detach().clone() for g in grads if g is not None]
        filtered_gradients = []
        for i, (grad, param) in enumerate(zip(grads, model.parameters())):
            if grad is not None:
                filtered_gradients.append(grad.detach().clone())

        grads = filtered_gradients

        # Perform gradient inversion
        rec_machine = inversefed.GradientReconstructor(model, (dm,ds), config, num_images=1)
        output, stats = rec_machine.reconstruct(grads, x, img_shape=(1, params['imagex'], params['imagez']))
        print('Gradient inversion is performed')

        print("reconstructed")
        print(output.min().item(), output.max().item(), output.mean().item())
        print("y")
        print(y.min().item(), y.max().item(), y.mean().item())


        output = torch.as_tensor(output, device=device, dtype=dtype)
        #output = output.mul(ds).add(dm)
        print("denormalized")
        print(output.min().item(), output.max().item(), output.mean().item())
        #output = output.clamp(-1, 1)
        #output = output.clamp(y.min(), y.max()) 
        img_name = f"img_attacked_epoch{epoch}_bs{bs}.png"
        torchvision.utils.save_image(output, os.path.join(image_path, img_name))
        print(f"Reconstruction saved: {img_name}")

        print("rescaled")
        print(output.min().item(), output.max().item(), output.mean().item())

        # Evaluate reconstruction quality
        output_np = output.detach().cpu().numpy().squeeze()
        ground_truth_np = y.detach().cpu().numpy().squeeze()

        test_mse = (output.detach() - y).pow(2).mean().item()
        test_psnr = psnr(output_np, ground_truth_np, data_range=ground_truth_np.max() - ground_truth_np.min())
        test_ssim = ssim(output_np, ground_truth_np, data_range=ground_truth_np.max() - ground_truth_np.min())


        print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
              f"| PSNR: {test_psnr:4.2f} | SSIM: {test_ssim:2.4f}")
        

        # Accumulate metrics
        metrics_data.append([img_name, stats['opt'], test_mse, test_psnr, test_ssim])


        # Continue training
        #loss.backward()
        #optimizer.step()
        bs += 1

# Save all metrics to a single CSV file
csv_file = os.path.join(image_path, "unet_skip_reconstruction_metrics.csv")
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['img_name', 'rec_loss', 'mse', 'psnr', 'ssim'])
    writer.writerows(metrics_data)

print(f"All metrics saved to {csv_file}")

# ================== TRAINING PROCESS ENDS ==================



end = time.time()
print(f"Total time taken: {(end - start)/60:.2f} minutes")