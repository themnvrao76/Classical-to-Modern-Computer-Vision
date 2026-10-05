import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, x):
        return F.normalize(self.projection(self.features(x).flatten(1)), dim=-1)


class TripletNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = TripletEncoder(embedding_dim)

    def forward(self, anchor, positive, negative):
        return self.encoder(anchor), self.encoder(positive), self.encoder(negative)


def triplet_loss(anchor, positive, negative, margin=0.2):
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin, p=2)


if __name__ == "__main__":
    model = TripletNetwork()
    a = torch.randn(4, 3, 96, 96)
    p = torch.randn(4, 3, 96, 96)
    n = torch.randn(4, 3, 96, 96)
    za, zp, zn = model(a, p, n)
    print("embedding:", tuple(za.shape))
    print("loss:", float(triplet_loss(za, zp, zn)))
