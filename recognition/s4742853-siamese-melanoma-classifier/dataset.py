import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class SiameseMelanomaClassifierDataset(Dataset):
    """
    Dataset for loading image pairs (anchor/positive or anchor/negative)
    for Siamese network training and validation.
    """
    def __init__(self, pairs_csv_path, img_dir, mode):
        """
        Initialises the dataset.

        Args:
            pairs_csv_path (str): Path to the CSV file containing image pair information.
            img_dir (str): Root directory containing the image files.
            mode (str): 'train' or 'val', to select appropriate data augmentation.
        """
        # Load the CSV file containing pairs and labels
        self.data = pd.read_csv(pairs_csv_path)
        self.img_dir = img_dir
        self.mode = mode
        # Build image transformation pipeline
        self.transform = self._build_transforms(mode)

        logger.info(f"Loading dataset: {pairs_csv_path}")
        logger.info(f"Image directory: {img_dir}")
        logger.info(f"Mode: {mode}")

    @staticmethod
    def _build_transforms(mode):
        """Helper function to define transformation based on mode."""
        # Pre-calculated mean and std of train set for normalisation
        mean = [0.8057231307029724, 0.6201786994934082, 0.5902535915374756]
        std = [0.0848047286272049, 0.09797607362270355, 0.1101665124297142]

        if mode == 'train':
            # Apply data augmentation for training
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif mode == 'val':
            # Only normalisation for validation
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __len__(self):
        """Returns the total number of pairs in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Loads and returns an image pair and its label at the specified index."""
        row = self.data.iloc[idx]
        # Construct full paths for the two images in the pair
        image_paths = [os.path.join(self.img_dir, f"{row[f'image_{i}']}.jpg") for i in (1, 2)]

        # Open, convert to RGB, and load the raw images
        images_raw = [Image.open(p).convert('RGB') for p in image_paths]

        # Apply the defined transformations to the images
        images = [self.transform(img) for img in images_raw]

        # Get the label (0 or 1) and convert to a float tensor for BCELoss
        label = torch.tensor(row['pair_label'], dtype=torch.float32)
        return tuple(images), label


class TestDataset(Dataset):
    """
    Dataset for testing. It loads a test image and its corresponding
    reference images from a pre-selected 'k' images per class for comparison.
    """
    def __init__(self, test_csv, test_img_dir, ref_csv, ref_img_dir, k):
        """
        Initialises the test dataset.

        Args:
            test_csv (str): Path to the CSV with test image names and true labels.
            test_img_dir (str): Directory for test images.
            ref_csv (str): Path to the CSV with all reference image names and labels.
            ref_img_dir (str): Directory for reference images.
            k (int): Number of reference images to sample per class.
        """
        self.test_data = pd.read_csv(test_csv)
        self.ref_data = pd.read_csv(ref_csv)
        self.test_img_dir = test_img_dir
        self.ref_img_dir = ref_img_dir
        self.k = k

        # Define standard normalisation-only transform
        mean = [0.8057231307029724, 0.6201786994934082, 0.5902535915374756]
        std = [0.0848047286272049, 0.09797607362270355, 0.1101665124297142]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # Pre-select and group k reference images per class
        self.ref_by_class = {}
        for _, row in self.ref_data.iterrows():
            label = row['target']
            if label not in self.ref_by_class:
                self.ref_by_class[label] = []
            self.ref_by_class[label].append(row['image_name'])

        # Limit to k reference images per class
        for label in self.ref_by_class:
            if len(self.ref_by_class[label]) > k:
                self.ref_by_class[label] = self.ref_by_class[label][:k]

        logger.info(f"Test images: {len(self.test_data)}")
        logger.info(f"Classes: {len(self.ref_by_class)}")
        logger.info(f"K per class: {k}")

    def __len__(self):
        """Returns the total number of test images."""
        return len(self.test_data)

    def __getitem__(self, idx):
        """
        Loads a single test image and all pre-selected reference images.

        Returns:
            tuple: (test_img, ref_imgs, test_label, test_img_name)
                - test_img (Tensor): The test image.
                - ref_imgs (dict): Dictionary mapping class label to a stacked Tensor of k reference images.
                - test_label (int): The true label of the test image.
                - test_img_name (str): The name of the test image file.
        """
        test_row = self.test_data.iloc[idx]
        test_img_name = test_row['image_name']
        test_label = test_row['target']

        # Load and transform the test image
        test_path = os.path.join(self.test_img_dir, f"{test_img_name}.jpg")
        test_img = self.transform(Image.open(test_path).convert('RGB'))

        # Load and stack all reference images for comparison
        ref_imgs = {}
        for label, img_names in self.ref_by_class.items():
            imgs = []
            for img_name in img_names:
                ref_path = os.path.join(self.ref_img_dir, f"{img_name}.jpg")
                imgs.append(self.transform(Image.open(ref_path).convert('RGB')))
            # Stack all k reference images for the current class into a single tensor
            ref_imgs[label] = torch.stack(imgs)

        return test_img, ref_imgs, test_label, test_img_name
