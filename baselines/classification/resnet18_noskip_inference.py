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
    "batch_size": 1,
    "imagex": 160,
    "imagey": 192,
    "imagez": 160,
    "column": "Group_bin",
}

csv_dir = 'pd_data_csv'
nifti_dir = '/work/forkert_lab/harmonized_with_masked_include_adni'
num_images = 1


train_csv = os.path.join(csv_dir, "train_pd_complete_adni.csv")
val_csv = os.path.join(csv_dir, "test_pd_complete_adni.csv")

# Create data loaders
train = pd.read_csv(train_csv)
val = pd.read_csv(val_csv)

val_IDs = val['Subject'].to_numpy()
val_dataset = DataGenerator(val_IDs, (params['imagex'], params['imagey'], params['imagez']), val_csv, params['column'])
validloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=1)



# Create and train the model
print("Loading the model...")
model = ResNet18NoSkip().to(device=device, dtype=dtype)
model.load_state_dict(torch.load("resnet18_noskip_harm.pt", map_location=device))
model.eval()


results = []

# Counters for metrics
TP = 0
TN = 0
FP = 0
FN = 0

with torch.no_grad():
    for idx, (x_test, y_test) in enumerate(validloader):
        x_test, y_test = x_test.to(device, dtype=torch.float32), y_test.to(device, dtype=torch.float32)
        outputs = model(x_test)

        # Convert logits to probabilities if needed
        #probs = torch.sigmoid(outputs) if outputs.max() > 1 else outputs
        probs = torch.softmax(outputs, dim=1)

        for i in range(x_test.size(0)):
            img_id = f"img_{idx}_{i}"
            
            raw_pred = probs[i, 1].item()
            pred_class = int(raw_pred > 0.5)
            gt_class = int(y_test[i].item())
            
            # Save results for CSV
            results.append({
                "img_id": img_id,
                "ground_truth": gt_class,
                "predicted_class": pred_class,
                "raw_prediction": raw_pred
            })
            
            # Update confusion matrix counters
            if gt_class == 1 and pred_class == 1:
                TP += 1
            elif gt_class == 0 and pred_class == 0:
                TN += 1
            elif gt_class == 0 and pred_class == 1:
                FP += 1
            elif gt_class == 1 and pred_class == 0:
                FN += 1

# Compute metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

# Save CSV
results_df = pd.DataFrame(results)
results_df.to_csv("inf_resnet18_noskip_results.csv", index=False)

print(f"Inference completed. Results saved to inference_results.csv")
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
