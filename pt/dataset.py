import os
import shutil
import pandas as pd
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split

TRAIN_DIR = "../dataset"

class DetectCenterDataset(Dataset):
    def __init__(self, path:str, transform=None):
        self.path = path
        self.transform = transform

        train_dir = os.path.join(TRAIN_DIR, "train")
        self.files = [f for f in os.listdir(train_dir) if not f.startswith('.')]

        filename = os.path.join(path, "coords.json")
        with open(filename, "r") as f:
            self.dict_coords = json.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = os.path.join(self.path, "train", filename)

        # sample = np.array(Image.open(filepath))
        sample = Image.open(filepath)
        # coord = np.array(self.dict_coords[filename])
        coord = torch.tensor(self.dict_coords[filename], dtype=torch.float32)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, coord

if __name__ == "__main__":
    BATCH_SIZE = 128
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    training_data = DetectCenterDataset(TRAIN_DIR, transform=transform)
    train_data, val_data = random_split(training_data, [0.8, 0.2])
    print(f"train - {len(train_data)}")
    print(f"val - {len(val_data)}")

    sample, coord = training_data[11]
    plt.scatter(coord[1], coord[0], marker='o', color='r')
    img_np = sample.squeeze().numpy()  # (1, 64, 64) → (64, 64)
    plt.imshow(img_np, cmap='gray')
    plt.show()

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # train_loop = tqdm(train_loader, desc=f"Training - Epoch {1}/{1}", leave=True)
    # for samples, targets in train_loop:
    #     print("samples - ",samples)
    #     print("targets - ", targets)

