import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


def build_encoder(projection_dim=128):
    encoder = resnet50(weights=None)
    encoder.fc = nn.Linear(encoder.fc.in_features, projection_dim)
    return encoder


class MoCo(nn.Module):
    def __init__(self, projection_dim=128, queue_size=65536, momentum=0.999, temperature=0.07):
        super().__init__()
        self.encoder_q = build_encoder(projection_dim)
        self.encoder_k = build_encoder(projection_dim)
        self.momentum = momentum
        self.temperature = temperature

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(projection_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        queue_size = self.queue.shape[1]
        if batch_size > queue_size:
            keys = keys[-queue_size:]
            batch_size = queue_size

        ptr = int(self.queue_ptr.item())
        end = ptr + batch_size
        if end <= queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            first = queue_size - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :end - queue_size] = keys[first:].T
        self.queue_ptr[0] = end % queue_size

    def forward(self, im_q, im_k):
        q = F.normalize(self.encoder_q(im_q), dim=1)

        with torch.no_grad():
            self.momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(im_k), dim=1)

        positive = torch.einsum("nc,nc->n", q, k).unsqueeze(1)
        negative = torch.einsum("nc,ck->nk", q, self.queue.detach())
        logits = torch.cat([positive, negative], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        self.dequeue_and_enqueue(k)
        return logits, labels


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = MoCo(queue_size=4096)
    im_q = torch.randn(2, 3, 224, 224)
    im_k = torch.randn(2, 3, 224, 224)
    logits, labels = model(im_q, im_k)
    print("Logits shape:", tuple(logits.shape))
    print("Labels shape:", tuple(labels.shape))
    print("Parameters:", f"{parameter_count(model):,}")
