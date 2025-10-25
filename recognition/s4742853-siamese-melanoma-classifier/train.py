import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import SiameseMelanomaClassifierDataset
from modules import SiameseNetwork
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os


class Plotter:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def add(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

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

    def train(self, epochs):
        pbar = tqdm(range(epochs), desc="Epochs")
        for epoch in pbar:
            train_loss = self.train_epoch(pbar)
            val_loss = self.validate(pbar)
            self.plotter.add(train_loss, val_loss)
            self.scheduler.step()

        timestamp = int(time.time())
        os.makedirs('models', exist_ok=True)
        model_path = f'models/siamese_melanoma_classifier_{timestamp}.pt'
        torch.save(self.network.state_dict(), model_path)
        self.plotter.plot(f'models/siamese_melanoma_classifier_{timestamp}_loss.png')


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    train_dataset = SiameseMelanomaClassifierDataset('data/cleaned/train_pairs.csv',
                                                     'data/cleaned/train_images',
                                                     mode='train')
    val_dataset = SiameseMelanomaClassifierDataset('data/cleaned/validation_pairs.csv',
                                                   'data/cleaned/validation_images',
                                                   mode='val')

    batch_size = 1
    epochs = 50
    lr = 1e-4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    network = SiameseNetwork()
    trainer = Trainer(network, train_loader, val_loader, device, lr=lr)
    trainer.train(epochs=epochs)