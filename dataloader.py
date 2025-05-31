
import os
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class SentinelDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".tif")
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        src = rasterio.open(img_path)
        image = src.read()

        image = image / 10000.  # scale data

        # Adding additional 2 channels to cope with the trained model
        first_channel = image[0:1, :, :]  # shape: (1, H, W)
        new_channels = np.repeat(first_channel, 2, axis=0)  # shape: (2, H, W)
        image = np.concatenate([image, new_channels], axis=0)  # shape: (4 + 2, H, W)

        image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]).astype('float32')  # CxHxW to HxWxC

        # Apply transform if given
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return {'image': image, 'filename': os.path.basename(img_path)}