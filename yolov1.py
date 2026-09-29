import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


class YOLOv1(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.features = nn.Sequential(
            ConvBlock(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 192, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(192, 128, 1),
            ConvBlock(128, 256, 3, padding=1),
            ConvBlock(256, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            ConvBlock(1024, 1024, 3, padding=1),
            ConvBlock(1024, 1024, 3, stride=2, padding=1),
            ConvBlock(1024, 1024, 3, padding=1),
            ConvBlock(1024, 1024, 3, padding=1),
        )

        output_size = split_size * split_size * (num_classes + num_boxes * 5)
        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * split_size * split_size, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_size),
        )

    def forward(self, x):
        x = self.features(x)
        if x.shape[-2:] != (self.split_size, self.split_size):
            raise ValueError(
                f"YOLOv1 expects features of size {self.split_size}x{self.split_size}; "
                "use 448x448 inputs with the default configuration."
            )
        x = self.detector(x)
        return x.view(
            x.size(0),
            self.split_size,
            self.split_size,
            self.num_classes + self.num_boxes * 5,
        )


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    model = YOLOv1()
    print(f"Parameters: {count_parameters(model):,}")
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 448, 448))
    print(f"Output shape: {tuple(output.shape)}")
