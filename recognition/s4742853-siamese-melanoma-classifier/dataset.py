import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class SiameseMelanomaClassifierDataset(Dataset):
    def __init__(self, pairs_csv_path, img_dir, mode):
        self.data = pd.read_csv(pairs_csv_path)
        self.img_dir = img_dir
        self.mode = mode
        self.transform = self._build_transforms(mode)

        logger.info(f"Loading dataset: {pairs_csv_path}")
        logger.info(f"Image directory: {img_dir}")
        logger.info(f"Mode: {mode}")

    @staticmethod
    def _build_transforms(mode):
        mean = [0.8057231307029724, 0.6201786994934082, 0.5902535915374756]
        std = [0.0848047286272049, 0.09797607362270355, 0.1101665124297142]

        if mode == 'train':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif mode in ['val', 'test']:
            return transforms.Compose([
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

        images_raw = [Image.open(p).convert('RGB') for p in image_paths]

        images = [self.transform(img) for img in images_raw]

        label = torch.tensor(row['pair_label'], dtype=torch.float32)
        return tuple(images), label


class TestDataset(Dataset):
    def __init__(self, test_csv, test_img_dir, ref_csv, ref_img_dir, k):
        self.test_data = pd.read_csv(test_csv)
        self.ref_data = pd.read_csv(ref_csv)
        self.test_img_dir = test_img_dir
        self.ref_img_dir = ref_img_dir
        self.k = k

        mean = [0.8057231307029724, 0.6201786994934082, 0.5902535915374756]
        std = [0.0848047286272049, 0.09797607362270355, 0.1101665124297142]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.ref_by_class = {}
        for _, row in self.ref_data.iterrows():
            label = row['target']
            if label not in self.ref_by_class:
                self.ref_by_class[label] = []
            self.ref_by_class[label].append(row['image_name'])

        for label in self.ref_by_class:
            if len(self.ref_by_class[label]) > k:
                self.ref_by_class[label] = self.ref_by_class[label][:k]

        logger.info(f"Test images: {len(self.test_data)}")
        logger.info(f"Classes: {len(self.ref_by_class)}")
        logger.info(f"K per class: {k}")

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        test_row = self.test_data.iloc[idx]
        test_img_name = test_row['image_name']
        test_label = test_row['target']

        test_path = os.path.join(self.test_img_dir, f"{test_img_name}.jpg")
        test_img = self.transform(Image.open(test_path).convert('RGB'))

        ref_imgs = {}
        for label, img_names in self.ref_by_class.items():
            imgs = []
            for img_name in img_names:
                ref_path = os.path.join(self.ref_img_dir, f"{img_name}.jpg")
                imgs.append(self.transform(Image.open(ref_path).convert('RGB')))
            ref_imgs[label] = torch.stack(imgs)

        return test_img, ref_imgs, test_label, test_img_name

