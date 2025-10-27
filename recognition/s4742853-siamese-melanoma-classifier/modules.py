import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights
import logging

logger = logging.getLogger(__name__)


class SiameseNetwork(nn.Module):
    """
    Custom convolutional Siamese Network for comparing image pairs.

    The network takes two input images (x1, x2) and computes the probability
    (p) that they belong to the same class (a 'positive' pair).
    It uses a shared weight architecture to create embeddings.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=10, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1)

        # Pooling and Dropout layers
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Fully connected layer for embedding
        # Input size is calculated based on input size 224x224 and pooling steps
        self.fc1 = nn.Linear(256 * 20 * 20, 512)

        # Learnable weight vector (alpha) for weighted distance calculation
        self.alpha = nn.Parameter(torch.ones(512) * 0.01)

        logger.info("Initialised SiameseNetwork with custom architecture")

    def forward_one(self, x):
        """
        Forward pass for a single image (shared weights).

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The 512-dimensional feature embedding.
        """
        # Block 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        # Block 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        # Block 3 & 4
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))

        # Flatten the feature map for the fully connected layer
        x = x.view(x.size(0), -1)
        # Final embedding layer with sigmoid activation
        x = torch.sigmoid(self.fc1(x))

        return x

    def forward(self, x1, x2):
        """
        Forward pass for a pair of images.

        Args:
            x1 (torch.Tensor): The first image in the pair.
            x2 (torch.Tensor): The second image in the pair.

        Returns:
            torch.Tensor: Probability tensor that the pair is a match (p).
        """
        # Get feature embeddings for both inputs
        h1 = self.forward_one(x1)
        h2 = self.forward_one(x2)

        # Calculate the absolute difference (L1 distance) between the embeddings
        distance = torch.abs(h1 - h2)
        # Compute the weighted sum of the distance (learnable alpha)
        weighted_distance = torch.sum(self.alpha * distance, dim=1)

        # Apply sigmoid to map the weighted distance to a probability [0, 1]
        p = torch.sigmoid(weighted_distance)

        return p


class PretrainedSiameseNetwork(nn.Module):
    """
    Siamese Network using a pre-trained ResNet-18 as the feature extractor.

    This network leverages transfer learning for feature extraction, followed by
    custom linear layers to generate the final embedding.
    """
    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool, optional): Whether to use pre-trained ImageNet weights. Defaults to True.
        """
        super(PretrainedSiameseNetwork, self).__init__()

        # Load ResNet-18 and use its weights
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        # Use all layers except the final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Custom fully connected head
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # ResNet18 features are 512 after AvgPool
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),  # Final embedding dimension
            nn.Dropout(0.7)
        )

        # Learnable weight vector (alpha) for weighted distance calculation
        self.alpha = nn.Parameter(torch.ones(128) * 0.01)

        # Ensure all feature extractor parameters are trainable
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        logger.info(f"Initialised PretrainedSiameseNetwork (pretrained={pretrained})")

    def forward_one(self, x):
        """
        Forward pass for a single image (shared weights).

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The 128-dimensional feature embedding.
        """
        # Pass through the ResNet feature extractor
        x = self.feature_extractor(x)
        # Flatten the (B, 512, 1, 1) output to (B, 512)
        x = x.view(x.size(0), -1)

        # Forward pass through the custom fully connected layers
        # The logging statements below are for debugging training stability
        if torch.rand(1) < 0.01:
            has_nan = torch.isnan(x).any().item()
            has_inf = torch.isinf(x).any().item()
            logger.debug(f"Before fc: has_nan={has_nan}, has_inf={has_inf}")

        x = self.fc[0](x)  # Linear 512 -> 256

        if torch.rand(1) < 0.01:
            logger.debug(
                f"After fc[0] (Linear 512->256): min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}, std={x.std():.3f}")
            zero_activations = (x == 0).float().mean().item()
            logger.debug(f"Zero activations after fc[0]: {zero_activations:.3%}")

        x = self.fc[1](x)  # Leaky ReLU activation

        if torch.rand(1) < 0.01:
            logger.debug(
                f"After fc[1] (ReLU): min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}, std={x.std():.3f}")
            zero_activations = (x == 0).float().mean().item()
            logger.debug(f"Dead neurons after ReLU: {zero_activations:.3%}")

        x = self.fc[2](x)  # Linear 256 -> 128 (Embedding output)

        if torch.rand(1) < 0.01:
            logger.debug(
                f"After fc[2] (Linear 256->128): min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}, std={x.std():.3f}")
            has_nan = torch.isnan(x).any().item()
            has_inf = torch.isinf(x).any().item()
            logger.debug(f"After fc: has_nan={has_nan}, has_inf={has_inf}")

        return x

    def forward(self, x1, x2):
        """
        Forward pass for a pair of images.

        Args:
            x1 (torch.Tensor): The first image in the pair.
            x2 (torch.Tensor): The second image in the pair.

        Returns:
            torch.Tensor: Probability tensor that the pair is a match (p).
        """
        # Get feature embeddings for both inputs
        h1 = self.forward_one(x1)
        h2 = self.forward_one(x2)

        # Debugging: Check for embedding collapse (low variance)
        if torch.rand(1) < 0.01:
            h1_var_across_batch = h1.std(dim=0).mean().item()
            h2_var_across_batch = h2.std(dim=0).mean().item()
            logger.debug(f"Embedding variance across batch: h1={h1_var_across_batch:.6f}, h2={h2_var_across_batch:.6f}")

            # Check pairwise distances within batch
            pairwise_dist = torch.cdist(h1, h2, p=2).mean().item()
            logger.debug(f"Mean pairwise distance h1 to h2: {pairwise_dist:.6f}")

        # Calculate the absolute difference (L1 distance) between the embeddings
        distance = torch.abs(h1 - h2)
        # Compute the weighted sum of the distance
        weighted_distance = torch.sum(self.alpha * distance, dim=1)

        # Debugging: Log weighted distance distribution
        if torch.rand(1) < 0.01:
            logger.debug(
                f"weighted_distance: min={weighted_distance.min():.6f}, max={weighted_distance.max():.6f}, mean={weighted_distance.mean():.6f}")

        # Apply sigmoid to map the weighted distance to a probability [0, 1]
        p = torch.sigmoid(weighted_distance)
        return p
