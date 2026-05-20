from PIL import Image
from datetime import datetime
import os
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

# Set seeds for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

class MITACS_ADNI_Dataset(Dataset):
    """Generates data for PyTorch"""

    def __init__(self, image_paths, transform=None, slice_idx=None):
        """
        Args:
            image_paths (list): paths to load images from.
            transform (callable, optional): Optional transform to be applied on a sample.
            slice (int): reduce the image to 2D by extracting a coronal slice at the given index
        """
        self.image_paths = image_paths
        self.transform = transform
        self.slice_idx = slice_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        imdat = sitk.GetArrayFromImage(sitk.ReadImage(self.image_paths[idx]))

        if self.slice_idx is not None:
            imdat = imdat[np.newaxis, :, self.slice_idx, ...]

        X = torch.from_numpy(imdat)

        if self.transform:
            X = self.transform(X)

        return X
    

