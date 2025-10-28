import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import SiameseMelanomaClassifierDataset, TestDataset
from modules import SiameseNetwork, PretrainedSiameseNetwork
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import time
import matplotlib.pyplot as plt
import logging
import sys
import os
import argparse
import pandas as pd

# non-GUI environments require 'Agg' matplotlib backend
matplotlib.use('Agg')


def setup_logging(log_file=None):
    """
    Sets up the global logging configuration, supporting both console output and file logging.
    """
    handlers = [logging.StreamHandler(sys.stdout)]  # Log to console
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))  # Log to file

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the entire training lifecycle, including optimisation, validation,
    early stopping, and model saving.
    """

    def __init__(self,
                 network: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 lr: float = 0.0001,
                 patience: int = 5
                 ):
        self.network = network.to(device)  # Move model to specified device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for probability output
        self.optimiser = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-5)
        # Learning rate scheduler to gradually decay the LR
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimiser, gamma=0.99)

        self.patience = patience
        self.best_val_acc = 0.0
        self.best_balanced_acc = 0.0
        self.patience_counter = 0

        logger.info(f"Trainer initialised with device: {device}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Early stopping patience: {patience}")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation images: {len(val_loader.dataset)}")

    def train_epoch(self, epoch):
        """Performs a single training epoch."""
        self.network.train()  # Set network to training mode

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
            transfer_time = time.time() - transfer_start

            forward_start = time.time()
            self.optimiser.zero_grad()  # Reset gradients
            outputs = self.network(img1, img2)  # Forward pass
            loss = self.criterion(outputs, labels)  # Calculate loss
            forward_time = time.time() - forward_start

            backward_start = time.time()
            loss.backward()  # Backpropagation
            backward_time = time.time() - backward_start

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

            # Record gradient norms for debugging
            grad_norms = {}
            for name, param in self.network.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.norm().item()

            optim_start = time.time()
            self.optimiser.step()  # Update weights
            optim_time = time.time() - optim_start

            total_loss += loss.item()
            batch_time = time.time() - iter_start

            # Logging updates every 10 batches
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - epoch_start
                avg_batch_time = elapsed / (batch_idx + 1)
                eta = avg_batch_time * (batch_count - batch_idx - 1)

                output_stats = f"outputs[min:{outputs.min():.3f}, max:{outputs.max():.3f}, mean:{outputs.mean():.3f}, std:{outputs.std():.3f}]"

                # Logging of gradient norms based on model type
                if 'fc1.weight' in grad_norms:
                    # Custom model gradient names
                    grad_fc = grad_norms.get('fc1.weight', 0)
                    grad_conv = grad_norms.get('conv1.weight', 0)
                    grad_alpha = grad_norms.get('alpha', 0)
                    grad_stats = f"grads[conv1:{grad_conv:.6f}, fc1:{grad_fc:.6f}, alpha:{grad_alpha:.6f}]"
                else:
                    # Pretrained model gradient names
                    grad_fc0 = grad_norms.get('fc.0.weight', 0)
                    grad_fc2 = grad_norms.get('fc.2.weight', 0)
                    # Example feature extractor layer
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
        """
        Performs validation using soft top-k voting accuracy approach.

        Returns:
            tuple: (accuracy, balanced_accuracy)
        """
        self.network.eval()
        correct = 0
        total = len(self.val_loader.dataset)

        # Track per-class predictions for balanced accuracy
        predictions = []
        true_labels = []

        logger.info(f"Running validation...")
        start_time = time.time()

        with torch.no_grad():
            for idx, (test_img, ref_imgs, true_label, img_name) in enumerate(self.val_loader):
                # test_img shape is (1, C, H, W)
                test_img = test_img.to(self.device)

                class_probs = {}
                for label, ref_batch in ref_imgs.items():
                    # ref_batch shape is (1, k, C, H, W). Squeeze the batch dim
                    ref_batch = ref_batch.squeeze(0).to(self.device)
                    # Repeat the single test image k times to match the ref_batch size
                    test_batch = test_img.repeat(ref_batch.size(0), 1, 1, 1)

                    # Get the probabilities that the test image matches each reference image
                    probs = self.network(test_batch, ref_batch)

                    # Get the mean probability of matching the top-5 references
                    k = min(5, len(probs))
                    top_k_probs = probs.topk(k=k).values
                    class_probs[label] = top_k_probs.mean().item()

                # The predicted class is the one with the highest mean probability
                pred_label = max(class_probs, key=class_probs.get)
                predictions.append(pred_label)
                true_labels.append(true_label.item())

                if pred_label == true_label.item():
                    correct += 1

                if (idx + 1) % 200 == 0:
                    logger.info(f"Validated {idx + 1}/{total} images")

        accuracy = correct / total

        # Compute per-class recalls for balanced accuracy
        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
        recall_0 = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        recall_1 = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        balanced_accuracy = (recall_0 + recall_1) / 2

        val_time = time.time() - start_time

        logger.info(
            f"Epoch {epoch + 1} - Validation complete - Accuracy: {accuracy:.4f} ({correct}/{total}) - Balanced Acc: {balanced_accuracy:.4f} - Time: {val_time:.2f}s")
        logger.info(f"Per-class Recall - Class 0: {recall_0:.4f}, Class 1: {recall_1:.4f}")

        return accuracy, balanced_accuracy

    def train(self, epochs, save_dir):
        """
        Runs the full training loop across all epochs with early stopping.

        Returns:
            int: Timestamp of the training run, used to identify the saved model.
        """
        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()

        os.makedirs(save_dir, exist_ok=True)
        timestamp = int(time.time())
        model_path = os.path.join(save_dir, f'siamese_melanoma_classifier_{timestamp}.pt')

        for epoch in range(epochs):
            epoch_start = time.time()
            logger.info(f"{'=' * 50}")
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch(epoch)
            val_acc, balanced_acc = self.validate(epoch)

            # Early stopping based on balanced accuracy
            if balanced_acc > self.best_balanced_acc:
                self.best_balanced_acc = balanced_acc
                self.best_val_acc = val_acc
                self.patience_counter = 0
                torch.save(self.network.state_dict(), model_path)
                logger.info(f"New best balanced accuracy: {balanced_acc:.4f} (acc: {val_acc:.4f}) - Model saved")
            else:
                self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Step the learning rate scheduler
            lr_before = self.optimiser.param_groups[0]['lr']
            self.scheduler.step()
            lr_after = self.optimiser.param_groups[0]['lr']

            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            logger.info(f"Learning rate: {lr_before:.6f} -> {lr_after:.6f}")

        total_time = time.time() - start_time
        logger.info(f"{'=' * 50}")
        logger.info(f"Training completed in {total_time:.2f}s ({total_time / 60:.2f}m)")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Best balanced accuracy: {self.best_balanced_acc:.4f}")
        logger.info(f"Best model saved to {model_path}")

        return timestamp


class Tester:
    """Handles the testing phase using the soft top-k voting approach."""

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

        logger.info(f"Tester initialised with device: {device}")
        logger.info(f"Test batches: {len(test_loader)}")

    def test(self):
        """
        Runs the k-NN testing protocol: compares each test image against k
        reference images for each class and makes a prediction.

        Returns:
            float: The calculated test accuracy.
        """
        self.network.eval()
        predictions = []

        logger.info("Starting testing...")
        start_time = time.time()

        with torch.no_grad():
            for idx, (test_img, ref_imgs, true_label, img_name) in enumerate(self.test_loader):
                # test_img shape is (1, C, H, W)
                test_img = test_img.to(self.device)

                class_probs = {}
                for label, ref_batch in ref_imgs.items():
                    # ref_batch shape is (1, k, C, H, W). Squeeze the batch dim
                    ref_batch = ref_batch.squeeze(0).to(self.device)
                    # Repeat the single test image k times to match the ref_batch size
                    test_batch = test_img.repeat(ref_batch.size(0), 1, 1, 1)

                    # Get the probabilities that the test image matches each reference image
                    probs = self.network(test_batch, ref_batch)

                    # Get the mean probability of matching the top-3 references
                    k = min(3, len(probs))
                    top_k_probs = probs.topk(k=k).values
                    class_probs[label] = top_k_probs.mean().item()

                # The predicted class is the one with the highest mean probability
                pred_label = max(class_probs, key=class_probs.get)
                predictions.append({
                    'image_name': img_name[0],
                    'true_label': true_label.item(),
                    'pred_label': pred_label
                })

                if (idx + 1) % 100 == 0:
                    logger.info(f"Tested {idx + 1}/{len(self.test_loader)} images")

        total_time = time.time() - start_time

        # Calculate metrics
        correct = sum(1 for p in predictions if p['true_label'] == p['pred_label'])
        accuracy = correct / len(predictions)

        true_labels = [p['true_label'] for p in predictions]
        pred_labels = [p['pred_label'] for p in predictions]

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

        recall_0 = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        recall_1 = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        balanced_accuracy = (recall_0 + recall_1) / 2

        logger.info(f"{'=' * 50}")
        logger.info(f"Testing completed in {total_time:.2f}s ({total_time / 60:.2f}m)")
        logger.info(f"Test Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
        logger.info(f"Test Balanced Accuracy: {balanced_accuracy:.4f}")
        logger.info("Confusion Matrix:")
        logger.info(f"True Negative: {cm[0, 0]} | False Positive: {cm[0, 1]}")
        logger.info(f"False Negative: {cm[1, 0]} | True Positive: {cm[1, 1]}")
        logger.info(f"Per-class Recall - Class 0: {recall_0:.4f}, Class 1: {recall_1:.4f}")

        # Save predictions to CSV
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df = pd.DataFrame(predictions)
        df.to_csv(self.output_path, index=False)
        logger.info(f"Predictions saved to {self.output_path}")

        return accuracy


def parse_args():
    """Parses command-line arguments for running the training or testing script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'],
                        help="Operation mode: train, test, or both.")
    parser.add_argument('--train-csv', type=str, default='data/cleaned/train_pairs.csv',
                        help="Path to the training pairs CSV.")
    parser.add_argument('--train-img-dir', type=str, default='data/cleaned/train_images_224',
                        help="Directory containing training images.")
    parser.add_argument('--val-csv', type=str, default='data/cleaned/validation.csv',
                        help="Path to the validation CSV (images to validate).")
    parser.add_argument('--val-img-dir', type=str, default='data/cleaned/validation_images_224',
                        help="Directory containing validation images.")
    parser.add_argument('--test-csv', type=str, default='data/cleaned/test.csv',
                        help="Path to the test image CSV.")
    parser.add_argument('--test-img-dir', type=str, default='data/cleaned/test_images_224',
                        help="Directory containing test images.")
    parser.add_argument('--batch-size', type=int, default=512,
                        help="Training batch size.")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epochs to train for.")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Initial learning rate.")
    parser.add_argument('--k', type=int, default=10,
                        help="Number of reference images per class for validation and testing.")
    parser.add_argument('--save-dir', type=str, default='models',
                        help="Directory to save model checkpoints.")
    parser.add_argument('--results-dir', type=str, default='results',
                        help="Directory to save test results.")
    parser.add_argument('--model', type=str, default='custom', choices=['pretrained', 'custom'],
                        help="Type of Siamese network to use ('pretrained' or 'custom').")
    parser.add_argument('--model-timestamp', type=str, default=None,
                        help="Timestamp of a saved model to load for testing.")
    parser.add_argument('--log-file', type=str, default=None,
                        help="Path for saving the log file.")
    return parser.parse_args()


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(2025)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2025)

    args = parse_args()

    # Determine log file path if not explicitly provided
    if args.log_file is None:
        args.log_file = os.path.join('logs', f'{args.mode}_{int(time.time())}.log')

    setup_logging(args.log_file)

    logger.info("=" * 50)
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 50)
    logger.info(f"Arguments: {vars(args)}")

    # Determine which device to use
    device = torch.device(
        "mps" if torch.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    model_timestamp = args.model_timestamp

    if args.mode in ['train', 'both']:
        # Setup training dataset and loader
        train_dataset = SiameseMelanomaClassifierDataset(args.train_csv, args.train_img_dir, mode='train')

        # Setup validation dataset using TestDataset with test images as references
        val_dataset = TestDataset(args.val_csv, args.val_img_dir, args.test_csv, args.test_img_dir, args.k)

        # Configure DataLoaders with multiprocessing and memory pinning
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, persistent_workers=True,
                                  shuffle=True, pin_memory=True)
        # Validation loader uses batch_size=1 like test loader
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Initialise the network
        network = PretrainedSiameseNetwork(pretrained=True) if args.model == 'pretrained' else SiameseNetwork()
        trainer = Trainer(network, train_loader, val_loader, device, lr=args.lr)
        # Start training and get the unique timestamp for the best model
        model_timestamp = trainer.train(epochs=args.epochs, save_dir=args.save_dir)

    if args.mode in ['test', 'both']:
        if model_timestamp is None:
            raise ValueError("--model-timestamp required for test mode")

        # Load the model's weights
        model_path = os.path.join(args.save_dir, f'siamese_melanoma_classifier_{model_timestamp}.pt')
        logger.info(f"Loading model from {model_path}")

        # Initialise the network
        network = PretrainedSiameseNetwork(pretrained=False) if args.model == 'pretrained' else SiameseNetwork()
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.to(device)

        # Setup test dataset and loader
        test_dataset = TestDataset(args.test_csv, args.test_img_dir, args.val_csv, args.val_img_dir, args.k)
        # Batch size of 1 for TestDataset
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Initialise and run the tester
        output_path = os.path.join(args.results_dir, f'predictions_{model_timestamp}.csv')
        tester = Tester(network, test_loader, device, output_path)
        tester.test()

    logger.info("Script finished")