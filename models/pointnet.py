import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, k * k),
        )
        nn.init.zeros_(self.fc[-1].weight)
        self.fc[-1].bias.data.copy_(torch.eye(k).flatten())

    def forward(self, x):
        x = self.conv(x).max(dim=2).values
        return self.fc(x).view(-1, self.k, self.k)


class PointNet(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.input_transform = TNet(3)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.feature_transform = TNet(64)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, points):
        transform = self.input_transform(points)
        points = torch.bmm(transform, points)
        features = self.mlp1(points)
        feature_transform = self.feature_transform(features)
        features = torch.bmm(feature_transform, features)
        features = self.mlp2(features).max(dim=2).values
        return self.classifier(features)


if __name__ == "__main__":
    model = PointNet(num_classes=40)
    points = torch.randn(4, 3, 1024)
    logits = model(points)
    print("output:", tuple(logits.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
