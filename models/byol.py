import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=4096, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class BYOLEncoder(nn.Module):
    def __init__(self, projection_dim=256, hidden_dim=4096):
        super().__init__()
        self.backbone = resnet50(weights=None)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projector = MLP(feature_dim, hidden_dim, projection_dim)

    def forward(self, x):
        return self.projector(self.backbone(x))


class BYOL(nn.Module):
    def __init__(self, projection_dim=256, hidden_dim=4096, momentum=0.996):
        super().__init__()
        self.online_encoder = BYOLEncoder(projection_dim, hidden_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.predictor = MLP(projection_dim, hidden_dim, projection_dim)
        self.momentum = momentum

        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        for online, target in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(self.momentum).add_(online.data, alpha=1.0 - self.momentum)

    @staticmethod
    def regression_loss(prediction, target):
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target.detach(), dim=-1)
        return 2 - 2 * (prediction * target).sum(dim=-1)

    def forward(self, view_one, view_two):
        online_one = self.predictor(self.online_encoder(view_one))
        online_two = self.predictor(self.online_encoder(view_two))

        with torch.no_grad():
            target_one = self.target_encoder(view_one)
            target_two = self.target_encoder(view_two)

        loss_one = self.regression_loss(online_one, target_two)
        loss_two = self.regression_loss(online_two, target_one)
        return (loss_one + loss_two).mean()

    def train_step(self, view_one, view_two):
        loss = self(view_one, view_two)
        self.update_target_encoder()
        return loss


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    model = BYOL()
    model.train()
    view_one = torch.randn(2, 3, 224, 224)
    view_two = torch.randn(2, 3, 224, 224)
    loss = model(view_one, view_two)
    print("Loss:", f"{loss.item():.4f}")
    print("Parameters:", f"{parameter_count(model):,}")
