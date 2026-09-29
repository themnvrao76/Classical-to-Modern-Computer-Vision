import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, roi_align


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + identity)


class ResNetC4(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)

    def _make_layer(self, channels, blocks, stride):
        layers = [Bottleneck(self.in_channels, channels, stride)]
        self.in_channels = channels * Bottleneck.expansion
        layers.extend(Bottleneck(self.in_channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(self.stem(x))))


class AnchorGenerator(nn.Module):
    def __init__(self, sizes=(128, 256, 512), ratios=(0.5, 1.0, 2.0), stride=16):
        super().__init__()
        self.sizes = sizes
        self.ratios = ratios
        self.stride = stride

    def forward(self, feature):
        h, w = feature.shape[-2:]
        device, dtype = feature.device, feature.dtype
        shifts_x = (torch.arange(w, device=device, dtype=dtype) + 0.5) * self.stride
        shifts_y = (torch.arange(h, device=device, dtype=dtype) + 0.5) * self.stride
        yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        centers = torch.stack((xx, yy, xx, yy), dim=-1).reshape(-1, 4)
        base = []
        for size in self.sizes:
            for ratio in self.ratios:
                aw = size / math.sqrt(ratio)
                ah = size * math.sqrt(ratio)
                base.append((-aw / 2, -ah / 2, aw / 2, ah / 2))
        base = torch.tensor(base, device=device, dtype=dtype)
        return (centers[:, None, :] + base[None, :, :]).reshape(-1, 4)


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=1024, num_anchors=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.objectness = nn.Conv2d(512, num_anchors, 1)
        self.box_deltas = nn.Conv2d(512, num_anchors * 4, 1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.objectness(x)
        deltas = self.box_deltas(x)
        logits = logits.permute(0, 2, 3, 1).reshape(x.size(0), -1)
        deltas = deltas.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        return logits, deltas


def decode_boxes(anchors, deltas):
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights
    dx, dy, dw, dh = deltas.unbind(-1)
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw.clamp(max=math.log(1000.0 / 16))) * widths
    pred_h = torch.exp(dh.clamp(max=math.log(1000.0 / 16))) * heights
    return torch.stack((pred_ctr_x - pred_w / 2, pred_ctr_y - pred_h / 2,
                        pred_ctr_x + pred_w / 2, pred_ctr_y + pred_h / 2), dim=-1)


def clip_boxes(boxes, height, width):
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    return boxes


class RoIHead(nn.Module):
    def __init__(self, in_channels=1024, num_classes=21, pool_size=7):
        super().__init__()
        hidden = 1024
        self.pool_size = pool_size
        self.fc1 = nn.Linear(in_channels * pool_size * pool_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_classes)
        self.box_regressor = nn.Linear(hidden, num_classes * 4)

    def forward(self, feature, proposals):
        pooled = roi_align(feature, proposals, output_size=self.pool_size, spatial_scale=1 / 16, aligned=True)
        x = pooled.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.classifier(x), self.box_regressor(x)


class FasterRCNN(nn.Module):
    def __init__(self, num_classes=21, pre_nms_topk=6000, post_nms_topk=300):
        super().__init__()
        self.backbone = ResNetC4()
        self.anchor_generator = AnchorGenerator()
        self.rpn = RegionProposalNetwork()
        self.roi_head = RoIHead(num_classes=num_classes)
        self.num_classes = num_classes
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

    def proposals(self, anchors, objectness, deltas, image_size):
        boxes = decode_boxes(anchors, deltas)
        boxes = clip_boxes(boxes, *image_size)
        scores = objectness.sigmoid()
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
        boxes, scores = boxes[keep], scores[keep]
        topk = min(self.pre_nms_topk, scores.numel())
        idx = scores.topk(topk).indices
        boxes, scores = boxes[idx], scores[idx]
        keep = nms(boxes, scores, 0.7)[: self.post_nms_topk]
        return boxes[keep]

    def forward(self, images):
        feature = self.backbone(images)
        anchors = self.anchor_generator(feature)
        objectness, rpn_deltas = self.rpn(feature)
        image_size = images.shape[-2:]
        batch_proposals = [
            self.proposals(anchors, objectness[i], rpn_deltas[i], image_size)
            for i in range(images.size(0))
        ]
        rois = [torch.cat((p.new_full((p.size(0), 1), i), p), dim=1) for i, p in enumerate(batch_proposals)]
        rois = torch.cat(rois, dim=0)
        class_logits, box_deltas = self.roi_head(feature, rois)
        return {
            "proposals": batch_proposals,
            "class_logits": class_logits,
            "box_deltas": box_deltas,
            "rpn_objectness": objectness,
            "rpn_box_deltas": rpn_deltas,
        }


def faster_rcnn(num_classes=21):
    return FasterRCNN(num_classes=num_classes)


if __name__ == "__main__":
    model = faster_rcnn()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(out["class_logits"].shape, out["box_deltas"].shape)
    print(sum(p.numel() for p in model.parameters()))
