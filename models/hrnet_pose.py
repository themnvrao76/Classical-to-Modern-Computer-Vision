import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class HighResolutionModule(nn.Module):
    def __init__(self, channels=(32, 64, 128)):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(BasicBlock(c), BasicBlock(c)) for c in channels
        )
        self.channels = channels
        self.projections = nn.ModuleDict()
        for i, ci in enumerate(channels):
            for j, cj in enumerate(channels):
                if i != j:
                    self.projections[f"{i}_{j}"] = nn.Sequential(
                        nn.Conv2d(ci, cj, 1, bias=False),
                        nn.BatchNorm2d(cj),
                    )

    def forward(self, xs):
        xs = [branch(x) for branch, x in zip(self.branches, xs)]
        outputs = []
        for j, target in enumerate(xs):
            fused = target
            target_size = target.shape[-2:]
            for i, source in enumerate(xs):
                if i == j:
                    continue
                z = self.projections[f"{i}_{j}"](source)
                z = F.interpolate(z, size=target_size, mode="bilinear", align_corners=False)
                fused = fused + z
            outputs.append(F.relu(fused, inplace=True))
        return outputs


class HRNetPose(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.branch1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.branch2 = nn.Conv2d(64, 64, 3, 2, 1)
        self.branch3 = nn.Conv2d(64, 128, 3, 4, 1)
        self.stage = nn.Sequential(
            HighResolutionModule(),
            HighResolutionModule(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(32 + 64 + 128, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_joints, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        xs = [self.branch1(x), self.branch2(x), self.branch3(x)]
        for module in self.stage:
            xs = module(xs)
        size = xs[0].shape[-2:]
        fused = [xs[0]]
        fused += [F.interpolate(t, size=size, mode="bilinear", align_corners=False) for t in xs[1:]]
        return self.head(torch.cat(fused, dim=1))


if __name__ == "__main__":
    model = HRNetPose()
    x = torch.randn(1, 3, 256, 192)
    y = model(x)
    print("heatmaps:", tuple(y.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
