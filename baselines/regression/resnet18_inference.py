import os
import numpy as np
import pandas as pd
import random
import time

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import inversefed
from datagenerator_pd import DataGenerator
from classificator_resnet50 import ResNet50Gray,ResNet18Gray, validate, train_one_epoch


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
    "batch_size": 1,
    "imagex": 160,
    "imagey": 192,
    "imagez": 160,
    "column": "Age",
}

csv_dir = 'pd_data_csv'
nifti_dir = '/work/forkert_lab/harmonized_with_masked_include_adni'
num_images = 1


train_csv = os.path.join(csv_dir, "train_pd_complete_adni_with_age.csv")
val_csv = os.path.join(csv_dir, "test_pd_complete_adni_with_age.csv")

# Create data loaders
train = pd.read_csv(train_csv)
val = pd.read_csv(val_csv)

val_IDs = val['Subject'].to_numpy()
val_dataset = DataGenerator(val_IDs, (params['imagex'], params['imagey'], params['imagez']), val_csv, params['column'])
validloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=1)



# Create and train the model
print("Loading the model...")
model = ResNet18Gray().to(device=device, dtype=dtype)
model.load_state_dict(torch.load("resnet18_skip_harm_age.pt", map_location=device))
model.eval()


results = []

sum_abs_error = 0.0
sum_sq_error = 0.0
total = 0

all_preds = []
all_targets = []

with torch.no_grad():
    for idx, (x_test, y_test) in enumerate(validloader):
        x_test = x_test.to(device, dtype=torch.float32)
        y_test = y_test.to(device, dtype=torch.float32)

        outputs = model(x_test).squeeze(-1)

        batch_size = y_test.size(0)

        abs_error = torch.abs(outputs - y_test)
        sq_error = (outputs - y_test) ** 2

        sum_abs_error += abs_error.sum().item()
        sum_sq_error += sq_error.sum().item()
        total += batch_size

        all_preds.append(outputs.cpu())
        all_targets.append(y_test.cpu())

        for i in range(batch_size):
            img_id = f"img_{idx}_{i}"
            results.append({
                "img_id": img_id,
                "ground_truth": y_test[i].item(),
                "prediction": outputs[i].item(),
                "abs_error": abs_error[i].item(),
                "MAE": None,
                "RMSE": None,
                "R2": None
            })

# ---- concatenate for R² ----
all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

# ---- compute metrics ----
mae = sum_abs_error / total
rmse = np.sqrt(sum_sq_error / total)

ss_res = torch.sum((all_targets - all_preds) ** 2)
ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0)

# ---- add summary row ----
results.append({
    "img_id": "GLOBAL_METRICS",
    "ground_truth": None,
    "prediction": None,
    "abs_error": None,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2.item()
})

# ---- save CSV ----
results_df = pd.DataFrame(results)
results_df.to_csv("inf_resnet18_regression_results.csv", index=False)

print("Inference completed. Results saved to inf_resnet18_regression_results.csv")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2.item():.4f}")
