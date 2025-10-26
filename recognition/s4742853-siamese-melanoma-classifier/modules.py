import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights
import logging

logger = logging.getLogger(__name__)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=10, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256 * 20 * 20, 512)

        self.alpha = nn.Parameter(torch.ones(512) * 0.01)

        logger.info("Initialized SiameseNetwork with custom architecture")

    def forward_one(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))

        return x

    def forward(self, x1, x2):
        h1 = self.forward_one(x1)
        h2 = self.forward_one(x2)

        distance = torch.abs(h1 - h2)
        weighted_distance = torch.sum(self.alpha * distance, dim=1)

        p = torch.sigmoid(weighted_distance)

        return p


class PretrainedSiameseNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super(PretrainedSiameseNetwork, self).__init__()

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.alpha = nn.Parameter(torch.ones(128))
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        logger.info(f"Initialized PretrainedSiameseNetwork (pretrained={pretrained})")

    def forward_one(self, x):
        # Track through network
        input_stats = (x.min().item(), x.max().item(), x.mean().item(), x.std().item())

        x = self.feature_extractor(x)
        feat_stats = (x.min().item(), x.max().item(), x.mean().item(), x.std().item())

        x = x.view(x.size(0), -1)
        flat_stats = (x.min().item(), x.max().item(), x.mean().item(), x.std().item())

        x = self.fc(x)
        output_stats = (x.min().item(), x.max().item(), x.mean().item(), x.std().item())

        if torch.rand(1) < 0.01:
            logger.debug(
                f"forward_one: input{input_stats} -> feat{feat_stats} -> flat{flat_stats} -> out{output_stats}")

        return x

    def forward(self, x1, x2):
        h1 = self.forward_one(x1)
        h2 = self.forward_one(x2)

        # Check if embeddings are collapsing across batch
        if torch.rand(1) < 0.01:
            h1_var_across_batch = h1.std(dim=0).mean().item()
            h2_var_across_batch = h2.std(dim=0).mean().item()
            logger.debug(f"Embedding variance across batch: h1={h1_var_across_batch:.6f}, h2={h2_var_across_batch:.6f}")

            # Check pairwise distances within batch
            pairwise_dist = torch.cdist(h1, h2, p=2).mean().item()
            logger.debug(f"Mean pairwise distance h1 to h2: {pairwise_dist:.6f}")

        distance = torch.abs(h1 - h2)
        weighted_distance = torch.sum(self.alpha * distance, dim=1)

        # Log weighted distance before sigmoid
        if torch.rand(1) < 0.01:
            logger.debug(
                f"weighted_distance: min={weighted_distance.min():.6f}, max={weighted_distance.max():.6f}, mean={weighted_distance.mean():.6f}")

        p = torch.sigmoid(weighted_distance)
        return p
