import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x):
        h, w = x.shape[-2:]
        y = torch.arange(h, device=x.device, dtype=torch.float32)
        xx = torch.arange(w, device=x.device, dtype=torch.float32)
        dim_t = torch.arange(self.num_pos_feats, device=x.device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_y = y[:, None] / dim_t[None, :]
        pos_x = xx[:, None] / dim_t[None, :]
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=-1).flatten(1)
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=-1).flatten(1)
        pos = torch.cat([
            pos_y[:, None, :].expand(h, w, -1),
            pos_x[None, :, :].expand(h, w, -1),
        ], dim=-1)
        return pos.permute(2, 0, 1).unsqueeze(0)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class DETR(nn.Module):
    def __init__(self, num_classes=91, num_queries=100, hidden_dim=256, nheads=8,
                 encoder_layers=6, decoder_layers=6):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 3, 2, 1), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
        )
        self.position = PositionEmbeddingSine(hidden_dim // 2)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=False,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, x):
        features = self.backbone(x)
        pos = self.position(features).expand(features.shape[0], -1, -1, -1)
        src = features.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)
        queries = self.query_embed.weight[:, None, :].expand(-1, x.shape[0], -1)
        target = torch.zeros_like(queries)
        memory = self.transformer.encoder(src + pos)
        hs = self.transformer.decoder(target + queries, memory)
        hs = hs.transpose(0, 1)
        return {
            "pred_logits": self.class_embed(hs),
            "pred_boxes": self.bbox_embed(hs).sigmoid(),
        }


if __name__ == "__main__":
    model = DETR(num_classes=20, num_queries=100)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print("logits:", tuple(out["pred_logits"].shape))
    print("boxes:", tuple(out["pred_boxes"].shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
