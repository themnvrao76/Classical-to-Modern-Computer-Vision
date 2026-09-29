import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.in_channels = 64
        self.layer1 = self._make(64, 2, 1)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)

    def _make(self, channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers += [BasicBlock(channels, channels) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        f0 = x
        x = self.pool(x)
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        return [f0, f1, f2, f3, x]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return F.elu(self.conv(x), inplace=True)


class DepthDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_channels = [64, 64, 128, 256, 512]
        decoder_channels = [16, 32, 64, 128, 256]
        self.upconv_0 = nn.ModuleList()
        self.upconv_1 = nn.ModuleList()
        for i in range(4, -1, -1):
            in_channels = encoder_channels[-1] if i == 4 else decoder_channels[i + 1]
            self.upconv_0.append(ConvBlock(in_channels, decoder_channels[i]))
            skip_channels = encoder_channels[i - 1] if i > 0 else 0
            self.upconv_1.append(ConvBlock(decoder_channels[i] + skip_channels, decoder_channels[i]))
        self.disparity_heads = nn.ModuleList([
            nn.Conv2d(decoder_channels[i], 1, 3, 1, 1) for i in range(4)
        ])

    def forward(self, features):
        outputs = {}
        x = features[-1]
        for stage, i in enumerate(range(4, -1, -1)):
            x = self.upconv_0[stage](x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if i > 0:
                x = torch.cat([x, features[i - 1]], dim=1)
            x = self.upconv_1[stage](x)
            if i < 4:
                outputs[i] = torch.sigmoid(self.disparity_heads[i](x))
        return outputs


class Monodepth2(nn.Module):
    def __init__(self, min_depth=0.1, max_depth=100.0):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.decoder = DepthDecoder()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def disparity_to_depth(self, disparity):
        min_disparity = 1.0 / self.max_depth
        max_disparity = 1.0 / self.min_depth
        scaled_disparity = min_disparity + (max_disparity - min_disparity) * disparity
        return scaled_disparity, 1.0 / scaled_disparity


if __name__ == "__main__":
    model = Monodepth2()
    image = torch.randn(1, 3, 192, 640)
    outputs = model(image)
    print({scale: tuple(value.shape) for scale, value in outputs.items()})
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
