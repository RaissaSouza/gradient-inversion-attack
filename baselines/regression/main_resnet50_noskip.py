import os
import numpy as np
import pandas as pd
import random
import time

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import inversefed
from datagenerator_pd import DataGenerator
from classificator_resnet50noskip import ResNet50NoSkip,ResNet18NoSkip, validate, train_one_epoch


start = time.time()

setup = inversefed.utils.system_startup()
device, dtype = setup["device"], setup["dtype"]

def _get_meanstd(dataset):
    channel = dataset[0][0].shape[0]
    cc = torch.cat([dataset[i][0].reshape(channel, -1) for i in range(len(dataset))], dim=1)
    data_mean = torch.mean(cc, dim=1)
    data_std = torch.std(cc, dim=1)

    mean_tensor = torch.as_tensor([data_mean], **setup)
    std_tensor = torch.as_tensor([data_std], **setup)
    
    return mean_tensor, std_tensor


# Set seeds for reproducibility
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

trained_model = True
CUDA_LAUNCH_BLOCKING=1


params = {
    "batch_size": 5,
    "imagex": 160,
    "imagey": 192,
    "imagez": 160,
    "column": "Age",
}

csv_dir = 'pd_data_csv'
nifti_dir = '/work/forkert_lab/harmonized_with_masked_include_adni'
num_images = 1

image_path = 'reconstruction_result_brain2d'
ground_truth_name = 'brain2d_ground_truth.png'
reconstruction_name = 'test.png'
os.makedirs(image_path, exist_ok=True)

train_csv = os.path.join(csv_dir, "train_pd_complete_adni_with_age.csv")
val_csv = os.path.join(csv_dir, "test_pd_complete_adni_with_age.csv")

# Create data loaders
train = pd.read_csv(train_csv)
val = pd.read_csv(val_csv)

train_IDs = train['Subject'].to_numpy()
training_dataset = DataGenerator(train_IDs, (params['imagex'], params['imagez']), train_csv, params['column'])
trainloader = DataLoader(training_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=1)

val_IDs = val['Subject'].to_numpy()
val_dataset = DataGenerator(val_IDs, (params['imagex'], params['imagez']), val_csv, params['column'])
validloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=1)




# Create and train the model
print("Creating the model...")
model = ResNet18NoSkip().to(device=device, dtype=dtype)
criterion = nn.MSELoss()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(-0.1))

# Training loop
best_val_loss = float('inf')
patience = 30
patience_counter = 0
num_epochs = 100

print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, validloader, criterion, device)
    scheduler.step()
    print("Epoch {}: Train Loss={:.4f}, Train MAE={:.4f}, Val Loss={:.4f}, Val MAE={:.4f}".format(
        epoch+1, train_loss, train_acc, val_loss, val_acc))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "resnet18_noskip_harm.pt")
        print("Model saved at epoch {}".format(epoch+1))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print("Training completed!")

end = time.time()
print(f"Total time taken: {(end - start)/60:.2f} minutes")