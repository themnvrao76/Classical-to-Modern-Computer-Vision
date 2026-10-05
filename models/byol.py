import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
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


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    def __init__(self, feature_dim=512, projection_dim=256, momentum=0.996):
        super().__init__()
        self.online_encoder = Encoder(feature_dim)
        self.online_projector = MLP(feature_dim, out_dim=projection_dim)
        self.predictor = MLP(projection_dim, hidden_dim=512, out_dim=projection_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        self.momentum = momentum
        for module in (self.target_encoder, self.target_projector):
            for p in module.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        online = list(self.online_encoder.parameters()) + list(self.online_projector.parameters())
        target = list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
        for o, t in zip(online, target):
            t.data.mul_(self.momentum).add_(o.data, alpha=1.0 - self.momentum)

    def encode_online(self, x):
        z = self.online_projector(self.online_encoder(x))
        return self.predictor(z)

    @torch.no_grad()
    def encode_target(self, x):
        return self.target_projector(self.target_encoder(x))

    def forward(self, view1, view2):
        p1 = self.encode_online(view1)
        p2 = self.encode_online(view2)
        with torch.no_grad():
            z1 = self.encode_target(view1)
            z2 = self.encode_target(view2)
        loss = 2 - 2 * (F.cosine_similarity(p1, z2.detach(), dim=-1).mean() +
                        F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) / 2
        return loss


if __name__ == "__main__":
    model = BYOL()
    a = torch.randn(4, 3, 96, 96)
    b = torch.randn(4, 3, 96, 96)
    print("loss:", float(model(a, b)))
    print("parameters:", sum(p.numel() for p in model.parameters()))
