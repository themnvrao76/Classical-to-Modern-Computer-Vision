import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, dim)

    def forward(self, x):
        return F.normalize(self.fc(self.net(x).flatten(1)), dim=-1)


class MoCo(nn.Module):
    def __init__(self, dim=128, queue_size=4096, momentum=0.999, temperature=0.07):
        super().__init__()
        self.encoder_q = Encoder(dim)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        self.momentum = momentum
        self.temperature = temperature
        self.register_buffer("queue", F.normalize(torch.randn(dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data.mul_(self.momentum).add_(q.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch <= self.queue.shape[1]:
            self.queue[:, ptr:ptr + batch] = keys.T
        else:
            first = self.queue.shape[1] - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :batch - first] = keys[first:].T
        self.queue_ptr[0] = (ptr + batch) % self.queue.shape[1]

    def forward(self, query, key):
        q = self.encoder_q(query)
        with torch.no_grad():
            self.momentum_update()
            k = self.encoder_k(key)
        positive = torch.sum(q * k, dim=1, keepdim=True)
        negative = q @ self.queue.detach()
        logits = torch.cat([positive, negative], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self.dequeue_and_enqueue(k)
        return logits, labels


if __name__ == "__main__":
    model = MoCo(queue_size=256)
    q = torch.randn(4, 3, 96, 96)
    k = torch.randn(4, 3, 96, 96)
    logits, labels = model(q, k)
    print("logits:", tuple(logits.shape), "labels:", tuple(labels.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
