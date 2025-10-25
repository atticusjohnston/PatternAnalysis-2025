import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import SiameseMelanomaClassifierDataset
from modules import SiameseNetwork, PretrainedSiameseNetwork
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import logging
import sys
import os
import argparse

matplotlib.use('Agg')  # Only pngs


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


logger = logging.getLogger(__name__)


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
        logger.info(f"Loss plot saved to {save_path}")


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

        logger.info(f"Trainer initialized with device: {device}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")

    def train_epoch(self, epoch):
        self.network.train()
        total_loss = 0
        batch_count = len(self.train_loader)
        epoch_start = time.time()

        for batch_idx, ((img1, img2), labels) in enumerate(self.train_loader):
            batch_start = time.time()

            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.float().to(self.device)
            data_time = time.time() - batch_start

            forward_start = time.time()
            self.optimiser.zero_grad()
            outputs = self.network(img1, img2)
            loss = self.criterion(outputs, labels)
            forward_time = time.time() - forward_start

            backward_start = time.time()
            loss.backward()
            backward_time = time.time() - backward_start

            optim_start = time.time()
            self.optimiser.step()
            optim_time = time.time() - optim_start

            total_loss += loss.item()
            batch_time = time.time() - batch_start

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - epoch_start
                avg_batch_time = elapsed / (batch_idx + 1)
                eta = avg_batch_time * (batch_count - batch_idx - 1)
                logger.info(
                    f"Epoch {epoch + 1} - Batch {batch_idx + 1}/{batch_count} - Loss: {loss.item():.4f} - Total: {batch_time:.3f}s (data: {data_time:.3f}s, fwd: {forward_time:.3f}s, bwd: {backward_time:.3f}s, opt: {optim_time:.3f}s) - Avg: {avg_batch_time:.3f}s - ETA: {eta:.1f}s")

        avg_loss = total_loss / batch_count
        logger.info(f"Epoch {epoch + 1} - Training complete - Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, epoch):
        self.network.eval()
        total_loss = 0
        batch_count = len(self.val_loader)

        with torch.no_grad():
            for batch_idx, ((img1, img2), labels) in enumerate(self.val_loader):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.float().to(self.device)

                outputs = self.network(img1, img2)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        avg_loss = total_loss / batch_count
        logger.info(f"Epoch {epoch + 1} - Validation complete - Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, epochs, save_dir):
        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()
            logger.info(f"{'=' * 50}")
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.plotter.add(train_loss, val_loss)

            lr_before = self.optimiser.param_groups[0]['lr']
            self.scheduler.step()
            lr_after = self.optimiser.param_groups[0]['lr']

            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            logger.info(f"Learning rate: {lr_before:.6f} -> {lr_after:.6f}")

        total_time = time.time() - start_time
        logger.info(f"{'=' * 50}")
        logger.info(f"Training completed in {total_time:.2f}s ({total_time / 60:.2f}m)")

        timestamp = int(time.time())
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'siamese_melanoma_classifier_{timestamp}.pt')
        torch.save(self.network.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        plot_path = os.path.join(save_dir, f'siamese_melanoma_classifier_{timestamp}_loss.png')
        self.plotter.plot(plot_path)


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
    parser.add_argument('--log-file', type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.log_file is None:
        args.log_file = os.path.join('logs', f'train_{int(time.time())}.log')

    setup_logging(args.log_file)

    logger.info("=" * 50)
    logger.info("Starting melanoma classification training")
    logger.info("=" * 50)
    logger.info(f"Arguments: {vars(args)}")

    device = torch.device(
        "mps" if torch.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    train_dataset = SiameseMelanomaClassifierDataset(args.train_csv, args.train_img_dir, mode='train')
    val_dataset = SiameseMelanomaClassifierDataset(args.val_csv, args.val_img_dir, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    network = PretrainedSiameseNetwork() if args.model == 'pretrained' else SiameseNetwork()
    trainer = Trainer(network, train_loader, val_loader, device, lr=args.lr)
    trainer.train(epochs=args.epochs, save_dir=args.save_dir)

    logger.info("Training script finished")
