import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x):
        return self.projection(x).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim=768, heads=12, mlp_dim=3072, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x):
        q = self.norm1(x)
        attention, _ = self.attention(q, q, q, need_weights=False)
        x = x + attention
        return x + self.mlp(self.norm2(x))


class DistilledDeiTBase(nn.Module):
    def __init__(self, num_classes=1000, image_size=224, patch_size=16, dropout=0.0):
        super().__init__()
        dim = 768
        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embedding.num_patches + 2, dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(dim, 12, 3072, dropout) for _ in range(12)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.distillation_head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.distillation_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.distillation_head.weight)
        nn.init.zeros_(self.distillation_head.bias)

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size = x.shape[0]
        cls = self.class_token.expand(batch_size, -1, -1)
        dist = self.distillation_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, dist, x), dim=1)
        x = self.dropout(x + self.position_embedding)
        x = self.norm(self.blocks(x))
        cls_logits = self.head(x[:, 0])
        dist_logits = self.distillation_head(x[:, 1])
        if self.training:
            return cls_logits, dist_logits
        return (cls_logits + dist_logits) / 2


def deit_base_distilled(num_classes=1000):
    return DistilledDeiTBase(num_classes=num_classes)


if __name__ == "__main__":
    model = deit_base_distilled()
    sample = torch.randn(1, 3, 224, 224)
    model.eval()
    output = model(sample)
    parameters = sum(p.numel() for p in model.parameters())
    print(output.shape)
    print(f"Parameters: {parameters:,}")
