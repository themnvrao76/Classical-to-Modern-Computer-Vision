import torch
import torch.nn as nn


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class StochasticDepth(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6, drop_path=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=1000, drop_path_rate=0.1):
        super().__init__()
        depths = (3, 3, 9, 3)
        dims = (96, 192, 384, 768)

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )

        total_blocks = sum(depths)
        rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        stages = []
        downsample_layers = []
        block_index = 0

        for stage_index, (depth, dim) in enumerate(zip(depths, dims)):
            blocks = []
            for _ in range(depth):
                blocks.append(ConvNeXtBlock(dim, drop_path=rates[block_index]))
                block_index += 1
            stages.append(nn.Sequential(*blocks))

            if stage_index < len(dims) - 1:
                downsample_layers.append(
                    nn.Sequential(
                        LayerNorm2d(dim, eps=1e-6),
                        nn.Conv2d(dim, dims[stage_index + 1], kernel_size=2, stride=2),
                    )
                )

        self.stages = nn.ModuleList(stages)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < len(self.downsample_layers):
                x = self.downsample_layers[index](x)
        x = self.avgpool(x).flatten(1)
        x = self.norm(x)
        return self.head(x)


def convnext_tiny(num_classes=1000, drop_path_rate=0.1):
    return ConvNeXtTiny(num_classes=num_classes, drop_path_rate=drop_path_rate)


if __name__ == "__main__":
    model = convnext_tiny()
    model.eval()
    sample = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(sample)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Parameters: {parameters:,}")
