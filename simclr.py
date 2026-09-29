import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class SimCLR(nn.Module):
    def __init__(self, projection_dim=128, temperature=0.5):
        super().__init__()
        backbone = resnet50(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projector = ProjectionHead(2048, 2048, projection_dim)
        self.temperature = temperature

    def encode(self, x):
        return torch.flatten(self.encoder(x), 1)

    def forward(self, x):
        h = self.encode(x)
        z = F.normalize(self.projector(h), dim=1)
        return h, z

    def nt_xent_loss(self, z1, z2):
        if z1.shape != z2.shape:
            raise ValueError("Augmented views must produce matching embeddings")
        n = z1.size(0)
        z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)
        logits = z @ z.T / self.temperature
        mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(mask, float("-inf"))
        targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
        return F.cross_entropy(logits, targets)

def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    model = SimCLR()
    a = torch.randn(4, 3, 224, 224)
    b = torch.randn(4, 3, 224, 224)
    _, z1 = model(a)
    _, z2 = model(b)
    print("Projection shape:", tuple(z1.shape))
    print("NT-Xent loss:", float(model.nt_xent_loss(z1, z2)))
    print("Parameters:", f"{parameter_count(model):,}")
