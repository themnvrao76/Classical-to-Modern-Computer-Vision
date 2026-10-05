import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        return F.normalize(self.fc(self.features(x).flatten(1)), dim=-1)


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = EmbeddingCNN(embedding_dim)

    def forward(self, left, right):
        return self.encoder(left), self.encoder(right)


def contrastive_loss(z1, z2, label, margin=1.0):
    distance = F.pairwise_distance(z1, z2)
    positive = label * distance.pow(2)
    negative = (1 - label) * F.relu(margin - distance).pow(2)
    return 0.5 * (positive + negative).mean()


if __name__ == "__main__":
    model = SiameseNetwork()
    a = torch.randn(4, 3, 96, 96)
    b = torch.randn(4, 3, 96, 96)
    z1, z2 = model(a, b)
    labels = torch.tensor([1., 0., 1., 0.])
    print("embedding:", tuple(z1.shape))
    print("loss:", float(contrastive_loss(z1, z2, labels)))
