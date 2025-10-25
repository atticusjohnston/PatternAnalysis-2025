import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=10, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 20 * 20, 4096)

        self.alpha = nn.Parameter(torch.ones(4096))

    def forward_one(self, x):
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv3(x)))
        print(x.shape)
        x = F.relu(self.conv4(x))
        print(x.shape)

        x = x.view(x.size(0), -1)
        print(x.shape)
        x = torch.sigmoid(self.fc1(x))
        print(x.shape)

        return x

    def forward(self, x1, x2):
        h1 = self.forward_one(x1)
        h2 = self.forward_one(x2)

        distance = torch.abs(h1 - h2)
        weighted_distance = torch.sum(self.alpha * distance, dim=1)

        p = torch.sigmoid(weighted_distance)

        return p
