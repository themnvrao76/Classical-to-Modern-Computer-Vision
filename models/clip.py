import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        return self.proj(self.net(x).flatten(1))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=30000, width=256, layers=4, heads=8,
                 context_length=32, embed_dim=256):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.pos = nn.Parameter(torch.randn(1, context_length, width) * 0.01)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, embed_dim)

    def forward(self, tokens):
        x = self.token(tokens) + self.pos[:, :tokens.shape[1]]
        x = self.transformer(x)
        x = self.norm(x)
        indices = tokens.ne(0).sum(dim=1).sub(1).clamp_min(0)
        pooled = x[torch.arange(x.shape[0], device=x.device), indices]
        return self.proj(pooled)


class CLIP(nn.Module):
    def __init__(self, embed_dim=256, vocab_size=30000, context_length=32):
        super().__init__()
        self.image_encoder = VisionEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim, context_length=context_length, embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def encode_image(self, images):
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, tokens):
        return F.normalize(self.text_encoder(tokens), dim=-1)

    def forward(self, images, tokens):
        image_features = self.encode_image(images)
        text_features = self.encode_text(tokens)
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * image_features @ text_features.t()


if __name__ == "__main__":
    model = CLIP(vocab_size=10000)
    images = torch.randn(4, 3, 128, 128)
    tokens = torch.randint(1, 10000, (4, 16))
    logits = model(images, tokens)
    print("similarity:", tuple(logits.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
