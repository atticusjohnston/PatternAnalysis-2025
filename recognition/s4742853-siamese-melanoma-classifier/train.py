# train.py
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import SiameseMelanomaClassifierDataset, TestDataset
from modules import SiameseNetwork, PretrainedSiameseNetwork
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import logging
import sys
import os
import argparse
import pandas as pd

matplotlib.use('Agg')


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
                 network: nn.Module,
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
        iter_start = time.time()

        for batch_idx, ((img1, img2), labels) in enumerate(self.train_loader):
            data_time = time.time() - iter_start

            transfer_start = time.time()
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.float().to(self.device)
            if batch_idx == 0 and epoch == 0:
                logger.debug(f"BATCH 0: img1==img2: {torch.allclose(img1, img2)}")
                logger.debug(f"BATCH 0: labels: {labels[:10]}")
            transfer_time = time.time() - transfer_start

            forward_start = time.time()
            self.optimiser.zero_grad()
            outputs = self.network(img1, img2)
            loss = self.criterion(outputs, labels)
            forward_time = time.time() - forward_start

            backward_start = time.time()
            loss.backward()
            backward_time = time.time() - backward_start

            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

            grad_norms = {}
            for name, param in self.network.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.norm().item()

            optim_start = time.time()
            self.optimiser.step()
            optim_time = time.time() - optim_start

            total_loss += loss.item()
            batch_time = time.time() - iter_start

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - epoch_start
                avg_batch_time = elapsed / (batch_idx + 1)
                eta = avg_batch_time * (batch_count - batch_idx - 1)

                output_stats = f"outputs[min:{outputs.min():.3f}, max:{outputs.max():.3f}, mean:{outputs.mean():.3f}, std:{outputs.std():.3f}]"

                if 'fc1.weight' in grad_norms:
                    grad_fc = grad_norms.get('fc1.weight', 0)
                    grad_conv = grad_norms.get('conv1.weight', 0)
                    grad_alpha = grad_norms.get('alpha', 0)
                    grad_stats = f"grads[conv1:{grad_conv:.6f}, fc1:{grad_fc:.6f}, alpha:{grad_alpha:.6f}]"
                else:
                    grad_fc0 = grad_norms.get('fc.0.weight', 0)
                    grad_fc2 = grad_norms.get('fc.2.weight', 0)
                    grad_feat = grad_norms.get('feature_extractor.7.1.conv2.weight', 0)
                    grad_alpha = grad_norms.get('alpha', 0)
                    grad_stats = f"grads[feat:{grad_feat:.6f}, fc0:{grad_fc0:.6f}, fc2:{grad_fc2:.6f}, alpha:{grad_alpha:.6f}]"

                logger.info(
                    f"Epoch {epoch + 1} - Batch {batch_idx + 1}/{batch_count} - Loss: {loss.item():.4f} - ETA: {eta:.1f}s")
                logger.debug(
                    f"{output_stats} - {grad_stats} - Total: {batch_time:.3f}s (data: {data_time:.3f}s, xfer: {transfer_time:.3f}s, fwd: {forward_time:.3f}s, bwd: {backward_time:.3f}s, opt: {optim_time:.3f}s) - Avg: {avg_batch_time:.3f}s")

            iter_start = time.time()

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

        return timestamp


class Tester:
    def __init__(self,
                 network: nn.Module,
                 test_loader: TestDataset,
                 device: torch.device,
                 output_path: str
                 ):
        self.network = network.to(device)
        self.test_loader = test_loader
        self.device = device
        self.output_path = output_path

        logger.info(f"Tester initialized with device: {device}")
        logger.info(f"Test batches: {len(test_loader)}")

    def test(self):
        self.network.eval()
        predictions = []

        logger.info("Starting testing...")
        start_time = time.time()

        with torch.no_grad():
            for idx, (test_img, ref_imgs, true_label, img_name) in enumerate(self.test_loader):
                test_img = test_img.to(self.device)

                class_probs = {}
                for label, ref_batch in ref_imgs.items():
                    ref_batch = ref_batch.squeeze(0).to(self.device)
                    test_batch = test_img.repeat(ref_batch.size(0), 1, 1, 1)

                    probs = self.network(test_batch, ref_batch)
                    class_probs[label] = probs.mean().item()

                pred_label = max(class_probs, key=class_probs.get)
                predictions.append({
                    'image_name': img_name[0],
                    'true_label': true_label.item(),
                    'pred_label': pred_label
                })

                if (idx + 1) % 100 == 0:
                    logger.info(f"Tested {idx + 1}/{len(self.test_loader)} images")

        total_time = time.time() - start_time
        correct = sum(1 for p in predictions if p['true_label'] == p['pred_label'])
        accuracy = correct / len(predictions)

        logger.info(f"{'=' * 50}")
        logger.info(f"Testing completed in {total_time:.2f}s ({total_time / 60:.2f}m)")
        logger.info(f"Test Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df = pd.DataFrame(predictions)
        df.to_csv(self.output_path, index=False)
        logger.info(f"Predictions saved to {self.output_path}")

        return accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'])
    parser.add_argument('--train-csv', type=str, default='data/cleaned/train_pairs.csv')
    parser.add_argument('--train-img-dir', type=str, default='data/cleaned/train_images_224')
    parser.add_argument('--val-csv', type=str, default='data/cleaned/validation_pairs.csv')
    parser.add_argument('--val-img-dir', type=str, default='data/cleaned/validation_images_224')
    parser.add_argument('--test-csv', type=str, default='data/cleaned/test.csv')
    parser.add_argument('--test-img-dir', type=str, default='data/cleaned/test_images_224')
    parser.add_argument('--ref-csv', type=str, default='data/cleaned/train.csv')
    parser.add_argument('--ref-img-dir', type=str, default='data/cleaned/train_images_224')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--model', type=str, default='custom', choices=['pretrained', 'custom'])
    parser.add_argument('--model-timestamp', type=str, default=None)
    parser.add_argument('--log-file', type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    args = parse_args()

    if args.log_file is None:
        args.log_file = os.path.join('logs', f'{args.mode}_{int(time.time())}.log')

    setup_logging(args.log_file)

    logger.info("=" * 50)
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 50)
    logger.info(f"Arguments: {vars(args)}")

    device = torch.device(
        "mps" if torch.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    model_timestamp = args.model_timestamp

    if args.mode in ['train', 'both']:
        train_dataset = SiameseMelanomaClassifierDataset(args.train_csv, args.train_img_dir, mode='train')
        val_dataset = SiameseMelanomaClassifierDataset(args.val_csv, args.val_img_dir, mode='val')

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, persistent_workers=True,
                                  shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, persistent_workers=True,
                                shuffle=False, pin_memory=True)

        network = PretrainedSiameseNetwork(pretrained=True) if args.model == 'pretrained' else SiameseNetwork()
        trainer = Trainer(network, train_loader, val_loader, device, lr=args.lr)
        model_timestamp = trainer.train(epochs=args.epochs, save_dir=args.save_dir)

    if args.mode in ['test', 'both']:
        if model_timestamp is None:
            raise ValueError("--model-timestamp required for test mode")

        model_path = os.path.join(args.save_dir, f'siamese_melanoma_classifier_{model_timestamp}.pt')
        logger.info(f"Loading model from {model_path}")

        network = PretrainedSiameseNetwork(pretrained=False) if args.model == 'pretrained' else SiameseNetwork()
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.to(device)

        test_dataset = TestDataset(args.test_csv, args.test_img_dir, args.ref_csv, args.ref_img_dir, args.k)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        output_path = os.path.join(args.results_dir, f'predictions_{model_timestamp}.csv')
        tester = Tester(network, test_loader, device, output_path)
        tester.test()

    logger.info("Script finished")
