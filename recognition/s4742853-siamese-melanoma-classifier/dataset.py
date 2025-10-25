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
        mean = [0.8057231307029724, 0.6201786994934082, 0.5902535915374756]
        std = [0.0848047286272049, 0.09797607362270355, 0.1101665124297142]

        if mode == 'train':
            return transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif mode in ['val', 'test']:
            return transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_paths = [os.path.join(self.img_dir, f"{row[f'image_{i}']}.jpg") for i in (1, 2)]
        images = [self.transform(Image.open(p).convert('RGB')) for p in image_paths]
        label = torch.tensor(row['pair_label'], dtype=torch.float32)
        return tuple(images), label
