import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=3072, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        x = x + attended
        return x + self.mlp(self.norm2(x))


class ViTB16(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size, patch_size, 3, embed_dim
        )
        num_patches = self.patch_embedding.num_patches

        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(
            *[
                EncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.head.weight)

    def forward(self, x):
        x = self.patch_embedding(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.dropout(x + self.position_embedding)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def vit_b16(num_classes=1000):
    return ViTB16(num_classes=num_classes)


if __name__ == "__main__":
    model = vit_b16()
    sample = torch.randn(1, 3, 224, 224)
    output = model(sample)
    parameters = sum(p.numel() for p in model.parameters())
    print(output.shape)
    print(f"Parameters: {parameters:,}")
