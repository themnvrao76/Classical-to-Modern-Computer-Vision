import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, out_dim, 3, 2, 1, bias=False), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class SimCLR(nn.Module):
    def __init__(self, feature_dim=512, projection_dim=128):
        super().__init__()
        self.encoder = ConvEncoder(feature_dim)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = F.normalize(self.projector(h), dim=-1)
        return h, z


def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    logits = z @ z.t() / temperature
    n = z1.shape[0]
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(mask, float("-inf"))
    targets = torch.arange(2 * n, device=z.device)
    targets = (targets + n) % (2 * n)
    return F.cross_entropy(logits, targets)


if __name__ == "__main__":
    model = SimCLR()
    x1 = torch.randn(4, 3, 96, 96)
    x2 = torch.randn(4, 3, 96, 96)
    _, z1 = model(x1)
    _, z2 = model(x2)
    print("projection:", tuple(z1.shape))
    print("loss:", float(nt_xent_loss(z1, z2)))
    print("parameters:", sum(p.numel() for p in model.parameters()))
