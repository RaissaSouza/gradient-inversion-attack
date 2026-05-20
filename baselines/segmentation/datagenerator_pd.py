from PIL import Image
from datetime import datetime
import os
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


MAIN_DIRECTORY = '/work/forkert_lab/harmonized_with_masked_include_adni'
MAIN_DIRECTORY_SEG = '/work/forkert_lab/adni-fastsurfer-seg/nifti'

# Set seeds for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

class DataGenerator(Dataset):
    """Generates data for PyTorch"""

    def __init__(self, list_IDs, dim, filename, column, transform=None):
        """
        Args:
            list_IDs (list): List of subject IDs.
            dim (tuple): Dimensions of the input images (D, H, W).
            filename (str): Path to the CSV file.
            column (str): Name of the label column.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dim = dim
        self.list_IDs = list_IDs
        self.filename = filename
        self.column = column
        self.transform = transform
        self.dataset = pd.read_csv(self.filename)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        ID = self.list_IDs[idx]
        # Filter dataset for the current subject
        subject_data = self.dataset[self.dataset['Subject'] == ID]
        subject_sbj = subject_data['Subject'].values[0]
        subject_str = subject_sbj
        subject_seg = subject_sbj+"-seg"
        extension_str = subject_data['Extension'].values[0]
        path_x = os.path.join(MAIN_DIRECTORY, f"{subject_str}{extension_str}")
        path_y = os.path.join(MAIN_DIRECTORY_SEG, f"{subject_seg}{extension_str}")

        itk_img_x = sitk.ReadImage(path_x)
        np_img_x = sitk.GetArrayFromImage(itk_img_x)

        itk_img_y = sitk.ReadImage(path_y)
        np_img_y = sitk.GetArrayFromImage(itk_img_y)

        # TO GET A SLICE
        y_index   = 86
        slice_arr_x = np_img_x[:, y_index, :]
        slice_arr_x = slice_arr_x.astype(np.float32)
        slice_arr_x = slice_arr_x.reshape(self.dim[0], self.dim[1], 1)
        X = torch.from_numpy(slice_arr_x).permute(2, 0, 1) # (C, D, H, W)


        if self.transform:
            X = self.transform(X)
        
         # TO GET A SLICE
        y_index   = 86
        slice_arr_y = np_img_y[:, y_index, :]
        slice_arr_y = (slice_arr_y == 16).astype(np.float32)   # ensure binary
        slice_arr_y = slice_arr_y.reshape(self.dim[0], self.dim[1], 1)

        Y = torch.from_numpy(slice_arr_y).permute(2, 0, 1)   # (1, H, W)



        if self.transform:
            Y = self.transform(Y)

        return X, Y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    
    
