import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):
    def __init__(self, channels=512, scale=20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((channels,), scale))

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-10)
        return self.weight[None, :, None, None] * x / norm


def vgg_block(in_channels, out_channels, n):
    layers = []
    for _ in range(n):
        layers += [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True)]
        in_channels = out_channels
    return nn.Sequential(*layers)


class SSD300(nn.Module):
    """SSD300 with the VGG-16 backbone and six multi-scale prediction maps."""

    def __init__(self, num_classes=21):
        super().__init__()
        self.num_classes = num_classes
        self.block1 = vgg_block(3, 64, 2)
        self.block2 = vgg_block(64, 128, 2)
        self.block3 = vgg_block(128, 256, 3)
        self.block4 = vgg_block(256, 512, 3)
        self.block5 = vgg_block(512, 512, 3)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.pool5 = nn.MaxPool2d(3, 1, padding=1)

        self.conv6 = nn.Conv2d(512, 1024, 3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, 1)
        self.conv8 = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.ReLU(inplace=True), nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(nn.Conv2d(512, 128, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv10 = nn.Sequential(nn.Conv2d(256, 128, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 256, 3), nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(nn.Conv2d(256, 128, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 256, 3), nn.ReLU(inplace=True))
        self.norm4 = L2Norm(512)

        channels = [512, 1024, 512, 256, 256, 256]
        boxes = [4, 6, 6, 6, 4, 4]
        self.loc = nn.ModuleList(nn.Conv2d(c, b * 4, 3, padding=1) for c, b in zip(channels, boxes))
        self.cls = nn.ModuleList(nn.Conv2d(c, b * num_classes, 3, padding=1) for c, b in zip(channels, boxes))
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        conv4_3 = self.norm4(self.block4(x))
        x = self.pool(conv4_3)
        x = self.block5(x)
        x = F.relu(self.conv6(self.pool5(x)), inplace=True)
        conv7 = F.relu(self.conv7(x), inplace=True)
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(conv8)
        conv10 = self.conv10(conv9)
        conv11 = self.conv11(conv10)
        features = [conv4_3, conv7, conv8, conv9, conv10, conv11]

        locations, logits = [], []
        for feature, loc, cls in zip(features, self.loc, self.cls):
            locations.append(loc(feature).permute(0, 2, 3, 1).contiguous().view(feature.size(0), -1, 4))
            logits.append(cls(feature).permute(0, 2, 3, 1).contiguous().view(feature.size(0), -1, self.num_classes))
        return torch.cat(locations, 1), torch.cat(logits, 1)


def ssd300(num_classes=21):
    return SSD300(num_classes=num_classes)


if __name__ == "__main__":
    model = ssd300()
    model.eval()
    with torch.no_grad():
        boxes, scores = model(torch.randn(1, 3, 300, 300))
    print("Locations:", boxes.shape)
    print("Class logits:", scores.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
