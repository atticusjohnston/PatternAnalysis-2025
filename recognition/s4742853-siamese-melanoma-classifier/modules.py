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
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        # Debug: Check inputs
        if torch.rand(1) < 0.01:
            x_identical = torch.allclose(x1, x2, atol=1e-6)
            logger.debug(f"INPUT: x1==x2: {x_identical}")
            logger.debug(f"INPUT: x1[min:{x1.min():.3f}, max:{x1.max():.3f}, mean:{x1.mean():.3f}]")
            logger.debug(f"INPUT: x2[min:{x2.min():.3f}, max:{x2.max():.3f}, mean:{x2.mean():.3f}]")

        h1 = self.forward_one(x1)
        h2 = self.forward_one(x2)

        # Debug: Check outputs
        if torch.rand(1) < 0.01:
            h_identical = torch.allclose(h1, h2, atol=1e-6)
            logger.debug(f"OUTPUT: h1==h2: {h_identical}")
            logger.debug(
                f"OUTPUT: h1[min:{h1.min():.3f}, max:{h1.max():.3f}, mean:{h1.mean():.3f}, std:{h1.std():.3f}]")
            logger.debug(
                f"OUTPUT: h2[min:{h2.min():.3f}, max:{h2.max():.3f}, mean:{h2.mean():.3f}, std:{h2.std():.3f}]")
            logger.debug(f"OUTPUT: alpha[min:{self.alpha.min():.3f}, max:{self.alpha.max():.3f}]")

        distance = torch.abs(h1 - h2)
        weighted_distance = torch.sum(self.alpha * distance, dim=1)
        p = torch.sigmoid(weighted_distance)
        return p
