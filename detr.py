import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        out_channels = channels * 4
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)) if stride != 1 or in_channels != out_channels else nn.Identity()
    def forward(self, x):
        identity = self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))
        return F.relu(x + identity, inplace=True)

class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.in_channels = 64
        self.layer1 = self._layer(64, 3)
        self.layer2 = self._layer(128, 4, 2)
        self.layer3 = self._layer(256, 6, 2)
        self.layer4 = self._layer(512, 3, 2)
    def _layer(self, channels, blocks, stride=1):
        layers = [Bottleneck(self.in_channels, channels, stride)]
        self.in_channels = channels * 4
        layers.extend(Bottleneck(self.in_channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats, self.temperature = num_pos_feats, temperature
    def forward(self, features):
        b, _, h, w = features.shape
        y = (torch.arange(h, device=features.device, dtype=features.dtype) + 1)[:, None].expand(h, w)
        x = (torch.arange(w, device=features.device, dtype=features.dtype) + 1)[None, :].expand(h, w)
        y, x = y / h * 2 * math.pi, x / w * 2 * math.pi
        dim = torch.arange(self.num_pos_feats, device=features.device, dtype=features.dtype)
        dim = self.temperature ** (2 * torch.div(dim, 2, rounding_mode="floor") / self.num_pos_feats)
        px, py = x[..., None] / dim, y[..., None] / dim
        px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), 3).flatten(2)
        py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), 3).flatten(2)
        pos = torch.cat((py, px), 2).permute(2, 0, 1)
        return pos.unsqueeze(0).expand(b, -1, -1, -1)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)])
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

class DETR(nn.Module):
    def __init__(self, num_classes=91, num_queries=100, hidden_dim=256):
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.input_proj = nn.Conv2d(2048, hidden_dim, 1)
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        encoder = nn.TransformerEncoderLayer(hidden_dim, 8, 2048, dropout=0.1)
        decoder = nn.TransformerDecoderLayer(hidden_dim, 8, 2048, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder, 6)
        self.decoder = nn.TransformerDecoder(decoder, 6)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4)
    def forward(self, images):
        features = self.backbone(images)
        src = self.input_proj(features)
        pos = self.position_embedding(src)
        batch = src.shape[0]
        src = src.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src + pos)
        queries = self.query_embed.weight.unsqueeze(1).expand(-1, batch, -1)
        hidden = self.decoder(torch.zeros_like(queries) + queries, memory + pos).transpose(0, 1)
        return {"pred_logits": self.class_embed(hidden), "pred_boxes": self.bbox_embed(hidden).sigmoid()}

if __name__ == "__main__":
    model = DETR()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))
    print(out["pred_logits"].shape, out["pred_boxes"].shape)
