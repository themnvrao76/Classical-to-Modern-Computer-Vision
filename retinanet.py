import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + identity)


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)

    def _make_layer(self, channels, blocks, stride):
        layers = [Bottleneck(self.in_channels, channels, stride)]
        self.in_channels = channels * Bottleneck.expansion
        layers.extend(Bottleneck(self.in_channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(self.stem(x))
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.lateral3 = nn.Conv2d(512, channels, 1)
        self.lateral4 = nn.Conv2d(1024, channels, 1)
        self.lateral5 = nn.Conv2d(2048, channels, 1)
        self.output3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.output4 = nn.Conv2d(channels, channels, 3, padding=1)
        self.output5 = nn.Conv2d(channels, channels, 3, padding=1)
        self.p6 = nn.Conv2d(2048, channels, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, features):
        c3, c4, c5 = features
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.output3(p3)
        p4 = self.output4(p4)
        p5 = self.output5(p5)
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p3, p4, p5, p6, p7]


class RetinaHead(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        layers = []
        for _ in range(4):
            layers.extend([nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)])
        self.tower = nn.Sequential(*layers)
        self.output = nn.Conv2d(256, output_channels, 3, padding=1)

    def forward(self, feature):
        return self.output(self.tower(feature))


class AnchorGenerator(nn.Module):
    def __init__(self, sizes=(32, 64, 128, 256, 512), ratios=(0.5, 1.0, 2.0), scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3))):
        super().__init__()
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales

    def forward(self, features, image_size):
        image_h, image_w = image_size
        anchors = []
        for feature, size in zip(features, self.sizes):
            h, w = feature.shape[-2:]
            stride_y, stride_x = image_h / h, image_w / w
            dtype, device = feature.dtype, feature.device
            shifts_x = (torch.arange(w, dtype=dtype, device=device) + 0.5) * stride_x
            shifts_y = (torch.arange(h, dtype=dtype, device=device) + 0.5) * stride_y
            yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            centers = torch.stack((xx, yy, xx, yy), dim=-1).reshape(-1, 4)
            base = []
            for scale in self.scales:
                for ratio in self.ratios:
                    area = (size * scale) ** 2
                    aw = math.sqrt(area / ratio)
                    ah = aw * ratio
                    base.append((-aw / 2, -ah / 2, aw / 2, ah / 2))
            base = torch.tensor(base, dtype=dtype, device=device)
            anchors.append((centers[:, None, :] + base[None, :, :]).reshape(-1, 4))
        return torch.cat(anchors)


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    probabilities = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    loss = ce * (1 - p_t).pow(gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


class RetinaNet(nn.Module):
    def __init__(self, num_classes=80, num_anchors=9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.backbone = ResNet50Backbone()
        self.fpn = FeaturePyramidNetwork()
        self.classification_head = RetinaHead(num_anchors * num_classes)
        self.regression_head = RetinaHead(num_anchors * 4)
        self.anchor_generator = AnchorGenerator()
        self._init_heads()

    def _init_heads(self):
        for head in (self.classification_head, self.regression_head):
            for module in head.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, std=0.01)
                    nn.init.zeros_(module.bias)
        prior = 0.01
        nn.init.constant_(self.classification_head.output.bias, -math.log((1 - prior) / prior))

    def forward(self, images):
        features = self.fpn(self.backbone(images))
        class_outputs = []
        box_outputs = []
        for feature in features:
            cls = self.classification_head(feature)
            box = self.regression_head(feature)
            batch, _, height, width = cls.shape
            cls = cls.view(batch, self.num_anchors, self.num_classes, height, width)
            cls = cls.permute(0, 3, 4, 1, 2).reshape(batch, -1, self.num_classes)
            box = box.view(batch, self.num_anchors, 4, height, width)
            box = box.permute(0, 3, 4, 1, 2).reshape(batch, -1, 4)
            class_outputs.append(cls)
            box_outputs.append(box)
        anchors = self.anchor_generator(features, images.shape[-2:])
        return {
            "class_logits": torch.cat(class_outputs, dim=1),
            "box_deltas": torch.cat(box_outputs, dim=1),
            "anchors": anchors,
        }


def retinanet_resnet50(num_classes=80):
    return RetinaNet(num_classes=num_classes)


if __name__ == "__main__":
    model = retinanet_resnet50()
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model(x)
    print(out["class_logits"].shape, out["box_deltas"].shape, out["anchors"].shape)
    print(sum(p.numel() for p in model.parameters()))
