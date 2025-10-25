import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import SiameseMelanomaClassifierDataset
from modules import SiameseNetwork, PretrainedSiameseNetwork
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

matplotlib.use('Agg')
import os
import argparse


class Plotter:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def add(self, train_loss, val_loss):
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))

    def plot(self, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()


class Trainer:
    def __init__(self,
                 network: SiameseNetwork,
                 train_loader: SiameseMelanomaClassifierDataset,
                 val_loader: SiameseMelanomaClassifierDataset,
                 device: torch.device,
                 lr: float = 0.0001
                 ):
        self.network = network.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimiser = optim.Adam(network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimiser, gamma=0.99)
        self.plotter = Plotter()

    def train_epoch(self, pbar):
        self.network.train()
        total_loss = 0

        for (img1, img2), labels in tqdm(self.train_loader, desc="Training", leave=False):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.float().to(self.device)

            self.optimiser.zero_grad()
            outputs = self.network(img1, img2)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimiser.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        pbar.set_postfix({'train_loss': f'{avg_loss:.4f}'})
        return avg_loss

    def validate(self, pbar):
        self.network.eval()
        total_loss = 0

        with torch.no_grad():
            for (img1, img2), labels in tqdm(self.val_loader, desc="Validation", leave=False):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.float().to(self.device)

                outputs = self.network(img1, img2)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        pbar.set_postfix({'train_loss': pbar.postfix['train_loss'], 'val_loss': f'{avg_loss:.4f}'})
        return avg_loss

    def train(self, epochs, save_dir):
        pbar = tqdm(range(epochs), desc="Epochs")
        for epoch in pbar:
            train_loss = self.train_epoch(pbar)
            val_loss = self.validate(pbar)
            self.plotter.add(train_loss, val_loss)
            self.scheduler.step()

        timestamp = int(time.time())
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'siamese_melanoma_classifier_{timestamp}.pt')
        torch.save(self.network.state_dict(), model_path)
        self.plotter.plot(os.path.join(save_dir, f'siamese_melanoma_classifier_{timestamp}_loss.png'))
        print(f"Model saved to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=str, default='data/cleaned/train_pairs.csv')
    parser.add_argument('--train-img-dir', type=str, default='data/cleaned/train_images_224')
    parser.add_argument('--val-csv', type=str, default='data/cleaned/validation_pairs.csv')
    parser.add_argument('--val-img-dir', type=str, default='data/cleaned/validation_images_224')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--model', type=str, default='custom', choices=['pretrained', 'custom'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(
        "mps" if torch.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_dataset = SiameseMelanomaClassifierDataset(args.train_csv, args.train_img_dir, mode='train')
    val_dataset = SiameseMelanomaClassifierDataset(args.val_csv, args.val_img_dir, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    network = PretrainedSiameseNetwork() if args.model == 'pretrained' else SiameseNetwork()
    trainer = Trainer(network, train_loader, val_loader, device, lr=args.lr)
    trainer.train(epochs=args.epochs, save_dir=args.save_dir)
