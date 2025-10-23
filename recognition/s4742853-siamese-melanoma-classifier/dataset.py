import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from torchvision import transforms


class SiameseMelanomaClassifierDataset(Dataset):
    def __init__(self, pairs_csv_path, img_dir, mode):
        self.data = pd.read_csv(pairs_csv_path)
        self.img_dir = img_dir
        self.mode = mode
        self.transform = self._build_transforms(mode)

    @staticmethod
    def _build_transforms(mode):
        # TODO: Choose an image size.
        # TODO: Choose image transforms.
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        elif mode in ['val', 'test']:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_paths = [os.path.join(self.img_dir, f"{row[f'image_{i}']}.jpg") for i in (1, 2)]
        images = [self.transform(Image.open(p).convert('RGB')) for p in image_paths]
        return tuple(images), row['pair_label']

