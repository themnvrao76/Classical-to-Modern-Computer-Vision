import torch
import torch.nn as nn


def square_distance(src, dst):
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    batch = torch.arange(points.shape[0], device=points.device)
    view = [points.shape[0]] + [1] * (idx.dim() - 1)
    return points[batch.view(*view), idx]


def farthest_point_sample(xyz, npoint):
    b, n, _ = xyz.shape
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((b, n), 1e10, device=xyz.device)
    farthest = torch.randint(0, n, (b,), device=xyz.device)
    batch = torch.arange(b, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=-1).indices
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    idx = dist.argsort(dim=-1)[:, :, :nsample]
    grouped_dist = torch.gather(dist, -1, idx)
    first = idx[:, :, :1].expand_as(idx)
    return torch.where(grouped_dist <= radius * radius, idx, first)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, mlp_channels, group_all=False):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.group_all = npoint, radius, nsample, group_all
        layers = []
        channels = in_channels
        for out_channels in mlp_channels:
            layers += [nn.Conv2d(channels, out_channels, 1, bias=False),
                       nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            channels = out_channels
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points=None):
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            grouped_xyz = xyz.unsqueeze(1) - new_xyz.unsqueeze(2)
            features = torch.cat([grouped_xyz, points.unsqueeze(1)], dim=-1) if points is not None else grouped_xyz
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)
            features = torch.cat([grouped_xyz, index_points(points, group_idx)], dim=-1) if points is not None else grouped_xyz
        features = self.mlp(features.permute(0, 3, 2, 1))
        return new_xyz, torch.max(features, dim=2).values.transpose(1, 2)


class PointNet2SSG(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 6, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 131, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None, 259, [256, 512, 1024], group_all=True)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, xyz):
        l1_xyz, l1_points = self.sa1(xyz, xyz)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, global_features = self.sa3(l2_xyz, l2_points)
        return self.classifier(global_features.squeeze(1))


if __name__ == "__main__":
    model = PointNet2SSG()
    model.eval()
    points = torch.randn(2, 1024, 3)
    with torch.no_grad():
        output = model(points)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
