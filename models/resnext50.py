import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, groups=32, width_per_group=4):
        super().__init__()
        width = int(channels * (width_per_group / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += identity
        return self.relu(x)


class ResNeXt(nn.Module):
    def __init__(self, layers, num_classes=1000, groups=32, width_per_group=4):
        super().__init__()
        self.in_channels = 64
        self.groups = groups
        self.width_per_group = width_per_group

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, channels, blocks, stride):
        layers = [
            Bottleneck(
                self.in_channels,
                channels,
                stride,
                self.groups,
                self.width_per_group,
            )
        ]
        self.in_channels = channels * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.in_channels,
                    channels,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnext50_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 6, 3], num_classes=num_classes, groups=32, width_per_group=4)


if __name__ == "__main__":
    model = resnext50_32x4d()
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)
    print(sum(parameter.numel() for parameter in model.parameters()))
