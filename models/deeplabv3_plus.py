import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dilation=1):
        padding = dilation if kernel == 3 else 0
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = nn.Sequential(
            ConvBNReLU(3, 64, 3, 2),
            ConvBNReLU(64, 64),
        )
        self.mid = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2),
            ConvBNReLU(128, 128),
            ConvBNReLU(128, 256, 3, 2),
        )
        self.high = nn.Sequential(
            ConvBNReLU(256, 512, 3, 2),
            ConvBNReLU(512, 512, dilation=2),
        )

    def forward(self, x):
        low = self.low(x)
        x = self.mid(low)
        high = self.high(x)
        return low, high


class ASPP(nn.Module):
    def __init__(self, in_ch=512, out_ch=256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([ConvBNReLU(in_ch, out_ch, kernel=1)])
        self.branches.extend(ConvBNReLU(in_ch, out_ch, dilation=r) for r in rates)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBNReLU(in_ch, out_ch, kernel=1))
        self.project = ConvBNReLU(out_ch * (len(rates) + 2), out_ch, kernel=1)

    def forward(self, x):
        size = x.shape[-2:]
        features = [b(x) for b in self.branches]
        pooled = F.interpolate(self.pool(x), size=size, mode="bilinear", align_corners=False)
        features.append(pooled)
        return self.project(torch.cat(features, dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.backbone = TinyBackbone()
        self.aspp = ASPP()
        self.low_proj = ConvBNReLU(64, 48, kernel=1)
        self.decoder = nn.Sequential(
            ConvBNReLU(304, 256),
            ConvBNReLU(256, 256),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        low, high = self.backbone(x)
        high = self.aspp(high)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([self.low_proj(low), high], dim=1))
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    model = DeepLabV3Plus(num_classes=21)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("output:", tuple(y.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
