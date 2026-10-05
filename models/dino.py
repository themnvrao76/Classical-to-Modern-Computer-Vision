import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTransformer(nn.Module):
    def __init__(self, image_size=96, patch_size=8, dim=256, depth=4, heads=8, out_dim=1024):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch_size, patch_size)
        n = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, :x.shape[1]]
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


class DINO(nn.Module):
    def __init__(self, out_dim=1024, teacher_momentum=0.996,
                 student_temp=0.1, teacher_temp=0.04):
        super().__init__()
        self.student = PatchTransformer(out_dim=out_dim)
        self.teacher = copy.deepcopy(self.student)
        self.teacher_momentum = teacher_momentum
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.register_buffer("center", torch.zeros(1, out_dim))
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data.mul_(self.teacher_momentum).add_(s.data, alpha=1.0 - self.teacher_momentum)

    def forward(self, student_views, teacher_views):
        student_logits = [self.student(v) / self.student_temp for v in student_views]
        with torch.no_grad():
            teacher_logits = [
                F.softmax((self.teacher(v) - self.center) / self.teacher_temp, dim=-1)
                for v in teacher_views
            ]
        losses = []
        for s in student_logits:
            logp = F.log_softmax(s, dim=-1)
            for t in teacher_logits:
                losses.append(-(t * logp).sum(dim=-1).mean())
        return torch.stack(losses).mean()


if __name__ == "__main__":
    model = DINO(out_dim=256)
    crops = [torch.randn(2, 3, 96, 96), torch.randn(2, 3, 96, 96)]
    print("loss:", float(model(crops, crops)))
    print("parameters:", sum(p.numel() for p in model.parameters()))
