import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))
        return F.relu(x + identity, inplace=True)


class HighResolutionModule(nn.Module):
    def __init__(self, channels, num_blocks=4):
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList([
            nn.Sequential(*[BasicBlock(c, c) for _ in range(num_blocks)])
            for c in channels
        ])
        self.fuse_layers = nn.ModuleList()
        for i, out_c in enumerate(channels):
            row = nn.ModuleList()
            for j, in_c in enumerate(channels):
                if i == j:
                    row.append(nn.Identity())
                elif j > i:
                    row.append(nn.Sequential(
                        nn.Conv2d(in_c, out_c, 1, bias=False),
                        nn.BatchNorm2d(out_c),
                    ))
                else:
                    ops = []
                    c = in_c
                    for k in range(i - j):
                        next_c = out_c if k == i - j - 1 else c
                        ops.extend([
                            nn.Conv2d(c, next_c, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(next_c),
                        ])
                        if k != i - j - 1:
                            ops.append(nn.ReLU(inplace=True))
                        c = next_c
                    row.append(nn.Sequential(*ops))
            self.fuse_layers.append(row)

    def forward(self, xs):
        xs = [branch(x) for branch, x in zip(self.branches, xs)]
        outputs = []
        for i, row in enumerate(self.fuse_layers):
            y = None
            target_size = xs[i].shape[-2:]
            for j, transform in enumerate(row):
                z = transform(xs[j])
                if j > i:
                    z = F.interpolate(z, size=target_size, mode="bilinear", align_corners=False)
                y = z if y is None else y + z
            outputs.append(F.relu(y, inplace=True))
        return outputs


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = []
        for i, c in enumerate(out_channels):
            if i < len(in_channels):
                if in_channels[i] == c:
                    layers.append(nn.Identity())
                else:
                    layers.append(nn.Sequential(
                        nn.Conv2d(in_channels[i], c, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(c),
                        nn.ReLU(inplace=True),
                    ))
            else:
                ops = []
                current = in_channels[-1]
                for k in range(i + 1 - len(in_channels)):
                    next_c = c if k == i - len(in_channels) else current
                    ops.extend([
                        nn.Conv2d(current, next_c, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(next_c),
                        nn.ReLU(inplace=True),
                    ])
                    current = next_c
                layers.append(nn.Sequential(*ops))
        self.layers = nn.ModuleList(layers)

    def forward(self, xs):
        out = []
        for i, layer in enumerate(self.layers):
            source = xs[i] if i < len(xs) else xs[-1]
            out.append(layer(source))
        return out


class HRNetW32Pose(nn.Module):
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

        stage1 = [Bottleneck(64, 64)]
        stage1.extend(Bottleneck(256, 64) for _ in range(3))
        self.stage1 = nn.Sequential(*stage1)

        self.transition1 = Transition([256], [32, 64])
        self.stage2 = nn.Sequential(*[HighResolutionModule([32, 64]) for _ in range(1)])

        self.transition2 = Transition([32, 64], [32, 64, 128])
        self.stage3 = nn.Sequential(*[HighResolutionModule([32, 64, 128]) for _ in range(4)])

        self.transition3 = Transition([32, 64, 128], [32, 64, 128, 256])
        self.stage4 = nn.Sequential(*[HighResolutionModule([32, 64, 128, 256]) for _ in range(3)])

        self.final_layer = nn.Conv2d(32, num_joints, 1)

    def forward(self, x):
        x = self.stage1(self.stem(x))
        xs = self.stage2(self.transition1([x]))
        xs = self.stage3(self.transition2(xs))
        xs = self.stage4(self.transition3(xs))
        return self.final_layer(xs[0])


if __name__ == "__main__":
    model = HRNetW32Pose(num_joints=17)
    x = torch.randn(1, 3, 256, 192)
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print("Output:", tuple(y.shape))
    print(f"Parameters: {params:,}")
