import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.downsample = None
        if stride != 1 or in_channels != channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(channels * 4),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))
        return F.relu(x + identity, inplace=True)


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.in_channels = 64
        self.layer1 = self._layer(64, 3, 1)
        self.layer2 = self._layer(128, 4, 2)
        self.layer3 = self._layer(256, 6, 2)
        self.layer4 = self._layer(512, 3, 2)

    def _layer(self, channels, blocks, stride):
        layers = [Bottleneck(self.in_channels, channels, stride)]
        self.in_channels = channels * 4
        layers.extend(Bottleneck(self.in_channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class FPN(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        channels = [256, 512, 1024, 2048]
        self.lateral = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in channels)
        self.output = nn.ModuleList(nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in channels)

    def forward(self, features):
        laterals = [conv(x) for conv, x in zip(self.lateral, features)]
        for i in range(2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:], mode="nearest")
        pyramids = [conv(x) for conv, x in zip(self.output, laterals)]
        pyramids.append(F.max_pool2d(pyramids[-1], 1, 2))
        return pyramids


class AnchorGenerator:
    def __init__(self, sizes=(32, 64, 128, 256, 512), ratios=(0.5, 1.0, 2.0)):
        self.sizes = sizes
        self.ratios = ratios

    def __call__(self, features, image_size):
        anchors = []
        ih, iw = image_size
        for feature, size in zip(features, self.sizes):
            h, w = feature.shape[-2:]
            sy, sx = ih / h, iw / w
            cy = (torch.arange(h, device=feature.device, dtype=feature.dtype) + 0.5) * sy
            cx = (torch.arange(w, device=feature.device, dtype=feature.dtype) + 0.5) * sx
            yy, xx = torch.meshgrid(cy, cx, indexing="ij")
            centers = torch.stack((xx, yy, xx, yy), -1).reshape(-1, 1, 4)
            base = []
            for ratio in self.ratios:
                aw = size / math.sqrt(ratio)
                ah = size * math.sqrt(ratio)
                base.append(feature.new_tensor([-aw / 2, -ah / 2, aw / 2, ah / 2]))
            anchors.append((centers + torch.stack(base)).reshape(-1, 4))
        return anchors


class RPNHead(nn.Module):
    def __init__(self, channels=256, anchors_per_location=3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.objectness = nn.Conv2d(channels, anchors_per_location, 1)
        self.box_deltas = nn.Conv2d(channels, anchors_per_location * 4, 1)
        for layer in (self.conv, self.objectness, self.box_deltas):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        logits, deltas = [], []
        for feature in features:
            x = F.relu(self.conv(feature), inplace=True)
            logits.append(self.objectness(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1))
            deltas.append(self.box_deltas(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1, 4))
        return logits, deltas


def decode_boxes(deltas, anchors):
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    cx = anchors[:, 0] + 0.5 * widths
    cy = anchors[:, 1] + 0.5 * heights
    dx, dy, dw, dh = deltas.unbind(-1)
    dw, dh = dw.clamp(max=math.log(1000.0 / 16)), dh.clamp(max=math.log(1000.0 / 16))
    pcx, pcy = dx * widths + cx, dy * heights + cy
    pw, ph = widths * dw.exp(), heights * dh.exp()
    return torch.stack((pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2), -1)


def box_iou(box, boxes):
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


def nms(boxes, scores, threshold=0.7):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou(boxes[i], boxes[remaining]) <= threshold]
    return torch.stack(keep) if keep else boxes.new_empty((0,), dtype=torch.long)


class RegionProposalNetwork(nn.Module):
    def __init__(self, pre_nms=1000, post_nms=300):
        super().__init__()
        self.head = RPNHead()
        self.anchor_generator = AnchorGenerator()
        self.pre_nms = pre_nms
        self.post_nms = post_nms

    def forward(self, features, image_size):
        logits, deltas = self.head(features)
        anchors = self.anchor_generator(features, image_size)
        proposals = []
        for batch_idx in range(features[0].shape[0]):
            boxes_all, scores_all = [], []
            for level_logits, level_deltas, level_anchors in zip(logits, deltas, anchors):
                scores = level_logits[batch_idx].sigmoid()
                count = min(self.pre_nms, scores.numel())
                scores, indices = scores.topk(count)
                boxes = decode_boxes(level_deltas[batch_idx, indices], level_anchors[indices])
                boxes[:, 0::2].clamp_(0, image_size[1])
                boxes[:, 1::2].clamp_(0, image_size[0])
                valid = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
                boxes_all.append(boxes[valid])
                scores_all.append(scores[valid])
            boxes, scores = torch.cat(boxes_all), torch.cat(scores_all)
            keep = nms(boxes, scores, 0.7)[:self.post_nms]
            proposals.append(boxes[keep])
        return proposals, (logits, deltas)


def roi_align(feature, boxes, output_size, spatial_scale):
    if boxes.numel() == 0:
        return feature.new_empty((0, feature.shape[1], output_size, output_size))
    outputs = []
    h, w = feature.shape[-2:]
    for box in boxes:
        x1, y1, x2, y2 = box * spatial_scale
        ys = torch.linspace(y1, y2, output_size, device=feature.device, dtype=feature.dtype)
        xs = torch.linspace(x1, x2, output_size, device=feature.device, dtype=feature.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((2 * xx / max(w - 1, 1) - 1, 2 * yy / max(h - 1, 1) - 1), -1).unsqueeze(0)
        outputs.append(F.grid_sample(feature, grid, mode="bilinear", align_corners=True)[0])
    return torch.stack(outputs)


class MultiScaleRoIAlign(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, features, proposals, image_size):
        pooled = []
        for batch_idx, boxes in enumerate(proposals):
            if boxes.numel() == 0:
                continue
            area = torch.sqrt(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=1))
            levels = (4 + torch.log2(area / 224 + 1e-6)).floor().clamp(2, 5).long()
            sample = features[0].new_empty((boxes.shape[0], 256, self.output_size, self.output_size))
            for level in range(2, 6):
                idx = torch.where(levels == level)[0]
                if idx.numel():
                    scale = features[level - 2].shape[-1] / image_size[1]
                    sample[idx] = roi_align(features[level - 2][batch_idx:batch_idx + 1], boxes[idx], self.output_size, scale)
            pooled.append(sample)
        return torch.cat(pooled) if pooled else features[0].new_empty((0, 256, self.output_size, self.output_size))


class BoxHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, num_classes)
        self.box_regressor = nn.Linear(1024, num_classes * 4)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return self.classifier(x), self.box_regressor(x)


class MaskHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)]
        self.convs = nn.Sequential(*layers)
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.predictor = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x = self.convs(x)
        x = F.relu(self.deconv(x), inplace=True)
        return self.predictor(x)


class MaskRCNN(nn.Module):
    def __init__(self, num_classes=81):
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.fpn = FPN()
        self.rpn = RegionProposalNetwork()
        self.box_pool = MultiScaleRoIAlign(7)
        self.mask_pool = MultiScaleRoIAlign(14)
        self.box_head = BoxHead(num_classes)
        self.mask_head = MaskHead(num_classes)

    def forward(self, images):
        image_size = images.shape[-2:]
        features = self.fpn(self.backbone(images))
        proposals, rpn_outputs = self.rpn(features, image_size)
        box_features = self.box_pool(features, proposals, image_size)
        class_logits, box_deltas = self.box_head(box_features)
        mask_features = self.mask_pool(features, proposals, image_size)
        mask_logits = self.mask_head(mask_features)
        return {
            "proposals": proposals,
            "class_logits": class_logits,
            "box_deltas": box_deltas,
            "mask_logits": mask_logits,
            "rpn_outputs": rpn_outputs,
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MaskRCNN(num_classes=81)
    print(f"Parameters: {count_parameters(model):,}")
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))
    print("Proposals:", output["proposals"][0].shape)
    print("Class logits:", output["class_logits"].shape)
    print("Mask logits:", output["mask_logits"].shape)
