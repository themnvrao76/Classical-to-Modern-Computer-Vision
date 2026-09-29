import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, image_size=224, patch_size=16, dim=384):
        super().__init__()
        self.n = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, dim, patch_size, patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim=384, heads=6):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b,n,3,self.heads,c//self.heads).permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)
        attn = ((q @ k.transpose(-2,-1)) * self.scale).softmax(-1)
        return self.proj((attn @ v).transpose(1,2).reshape(b,n,c))

class Block(nn.Module):
    def __init__(self, dim=384, heads=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class ViTSmall(nn.Module):
    def __init__(self, image_size=224, patch_size=16, dim=384, depth=12, heads=6):
        super().__init__()
        self.patch = PatchEmbed(image_size, patch_size, dim)
        self.cls = nn.Parameter(torch.zeros(1,1,dim))
        self.pos = nn.Parameter(torch.zeros(1,self.patch.n+1,dim))
        self.blocks = nn.ModuleList([Block(dim,heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.patch(x)
        x = torch.cat((self.cls.expand(x.size(0),-1,-1),x),1) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:,0]

class DINOHead(nn.Module):
    def __init__(self, dim=384, out_dim=65536):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim,2048),nn.GELU(),nn.Linear(2048,256))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(256,out_dim,bias=False))
    def forward(self, x):
        return self.last(F.normalize(self.mlp(x),dim=-1))

class DINO(nn.Module):
    def __init__(self, out_dim=65536, momentum=0.996, student_temp=0.1, teacher_temp=0.04):
        super().__init__()
        self.student_backbone = ViTSmall()
        self.student_head = DINOHead(out_dim=out_dim)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = copy.deepcopy(self.student_head)
        self.momentum = momentum
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.register_buffer("center", torch.zeros(1,out_dim))
        for module in (self.teacher_backbone,self.teacher_head):
            for parameter in module.parameters():
                parameter.requires_grad = False
    def student(self, x):
        return self.student_head(self.student_backbone(x))
    @torch.no_grad()
    def teacher(self, x):
        return self.teacher_head(self.teacher_backbone(x))
    @torch.no_grad()
    def update_teacher(self):
        student = list(self.student_backbone.parameters()) + list(self.student_head.parameters())
        teacher = list(self.teacher_backbone.parameters()) + list(self.teacher_head.parameters())
        for source,target in zip(student,teacher):
            target.data.mul_(self.momentum).add_(source.data,alpha=1-self.momentum)
    def loss(self, student_views, teacher_views):
        student = [self.student(view)/self.student_temp for view in student_views]
        with torch.no_grad():
            teacher_logits = [self.teacher(view) for view in teacher_views]
            teacher = [F.softmax((z-self.center)/self.teacher_temp,dim=-1) for z in teacher_logits]
        total = student[0].new_tensor(0.0)
        terms = 0
        for i,target in enumerate(teacher):
            for j,prediction in enumerate(student):
                if i == j:
                    continue
                total -= (target * F.log_softmax(prediction,dim=-1)).sum(-1).mean()
                terms += 1
        with torch.no_grad():
            batch_center = torch.cat(teacher_logits).mean(0,keepdim=True)
            self.center.mul_(0.9).add_(batch_center,alpha=0.1)
        return total / terms

if __name__ == "__main__":
    model = DINO()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    views = [torch.randn(2,3,224,224) for _ in range(2)]
    print(f"Loss: {model.loss(views,views).item():.4f}")
