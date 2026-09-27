import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x, attention_mask=None):
        normalized = self.ln1(x)
        attended, _ = self.attention(normalized, normalized, normalized, attn_mask=attention_mask, need_weights=False)
        x = x + attended
        return x + self.mlp(self.ln2(x))

class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=32, width=768, layers=12, heads=12, embed_dim=512):
        super().__init__()
        grid_size = image_size // patch_size
        self.patch_embedding = nn.Conv2d(3, width, patch_size, patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.empty(1, 1, width))
        self.position_embedding = nn.Parameter(torch.empty(1, grid_size * grid_size + 1, width))
        self.pre_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])
        self.post_norm = nn.LayerNorm(width)
        self.projection = nn.Parameter(torch.empty(width, embed_dim))
        nn.init.normal_(self.class_embedding, std=width ** -0.5)
        nn.init.normal_(self.position_embedding, std=width ** -0.5)
        nn.init.normal_(self.projection, std=width ** -0.5)

    def forward(self, images):
        x = self.patch_embedding(images).flatten(2).transpose(1, 2)
        x = torch.cat((self.class_embedding.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pre_norm(x + self.position_embedding)
        for block in self.blocks:
            x = block(x)
        return self.post_norm(x[:, 0]) @ self.projection

class TextTransformer(nn.Module):
    def __init__(self, vocab_size=49408, context_length=77, width=512, layers=12, heads=8, embed_dim=512):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Parameter(torch.empty(context_length, width))
        self.blocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        self.projection = nn.Parameter(torch.empty(width, embed_dim))
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)
        nn.init.normal_(self.projection, std=width ** -0.5)

    def forward(self, tokens):
        if tokens.shape[1] > self.context_length:
            raise ValueError("Token sequence exceeds context length.")
        x = self.token_embedding(tokens) + self.position_embedding[:tokens.shape[1]]
        mask = torch.full((tokens.shape[1], tokens.shape[1]), float("-inf"), device=tokens.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        end_positions = tokens.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), end_positions]
        return x @ self.projection

class CLIP(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.visual = VisionTransformer(embed_dim=embed_dim)
        self.text = TextTransformer(embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images):
        return F.normalize(self.visual(images), dim=-1)

    def encode_text(self, tokens):
        return F.normalize(self.text(tokens), dim=-1)

    def forward(self, images, tokens):
        image_features = self.encode_image(images)
        text_features = self.encode_text(tokens)
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.t()
        return logits_per_image, logits_per_image.t()

    def contrastive_loss(self, images, tokens):
        image_logits, text_logits = self(images, tokens)
        targets = torch.arange(images.shape[0], device=images.device)
        return (F.cross_entropy(image_logits, targets) + F.cross_entropy(text_logits, targets)) / 2

def clip_vit_b32():
    return CLIP()

if __name__ == "__main__":
    model = clip_vit_b32()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    images = torch.randn(2, 3, 224, 224)
    tokens = torch.randint(0, 49407, (2, 77))
    tokens[:, -1] = 49407
    print(model(images, tokens)[0].shape)
