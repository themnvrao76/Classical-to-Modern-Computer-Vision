# Classical to Modern Computer Vision

<p align="center">
  <strong>PyTorch implementations of landmark computer vision architectures — from LeNet and AlexNet to ResNet, EfficientNet, Vision Transformers, detection, segmentation, self-supervised learning, multimodal vision, and 3D vision.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-Computer%20Vision-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Models-23-blue" alt="Models">
  <img src="https://img.shields.io/badge/Papers-Original%20Sources-success" alt="Original Papers">
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen" alt="Active Development">
</p>

---

## About

This repository is a growing collection of **computer vision architectures implemented in PyTorch**, organized as a journey from classical convolutional neural networks to modern transformer-based and multimodal vision systems.

The goal is to keep the implementations readable and close to the architectural ideas introduced in the original papers while building a practical reference for studying the evolution of computer vision.

**Current coverage:** CNNs · residual networks · efficient/mobile vision · vision transformers · modern ConvNet backbones · semantic segmentation · object detection · instance segmentation  
**Planned coverage:** pose estimation · self-supervised learning · vision-language models · generative vision · optical flow · tracking · 3D vision

### Why this repository?

- Landmark computer vision architectures in one place
- Direct links to the original research papers
- Parameter counts for each implementation
- Runnable PyTorch implementations
- Historical progression from classical CNNs to modern vision models
- Expanding into detection, segmentation, multimodal learning, video, and 3D vision

---

## Architecture Evolution

<p align="center">
  <strong>LeNet-5 → AlexNet → VGG / Inception → FCN / U-Net → ResNet / ResNeXt / DenseNet → Faster R-CNN / SSD / YOLO / RetinaNet / Mask R-CNN → DeepLabV3 → MobileNet / EfficientNet → ViT / DeiT / Swin → ConvNeXt → Self-Supervised & Multimodal → 3D & Human-Centric Vision</strong>
</p>

| Era | Representative Models | Main Idea |
|---|---|---|
| Classical CNNs | LeNet-5, AlexNet | Learned hierarchical visual features |
| Deep CNNs | VGG, GoogLeNet | Depth and multi-scale feature extraction |
| Dense Prediction | FCN, U-Net, DeepLabV3 | Segmentation, encoder-decoder paths, skip fusion, atrous convolution, and multi-scale context |
| Residual Networks | ResNet, ResNeXt, DenseNet | Skip connections and feature reuse |
| Efficient Vision | MobileNet, EfficientNet | Lightweight and scalable architectures |
| Vision Transformers | ViT, DeiT, Swin Transformer | Patch-based attention and hierarchical shifted windows |
| Modern ConvNets | ConvNeXt | Transformer-era design principles applied to convolutional networks |
| Object & Instance Detection | Faster R-CNN, SSD, YOLOv1, RetinaNet, Mask R-CNN, DETR | Region proposals, direct grid prediction, feature pyramids, focal loss, and parallel instance-mask prediction |
| Representation Learning | SimCLR, MoCo, BYOL, DINO | Self-supervised visual representations |
| Multimodal Vision | CLIP-style models | Joint image-text representations |
| 3D & Human Vision | PointNet, HRNet, pose/mesh models | Geometry and human-centric understanding |

---

## Model Index

| Model | Year | Parameters | Original Paper | PyTorch Implementation |
|---|---:|---:|---|---|
| LeNet-5 | 1998 | 44,426 | [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791) | [`models/lenet.py`](models/lenet.py) |
| AlexNet | 2012 | 61,100,840 | [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | [`models/alexnet.py`](models/alexnet.py) |
| VGG-16 | 2014 | 138,357,544 | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | [`models/vgg16.py`](models/vgg16.py) |
| GoogLeNet / Inception v1 | 2014 | 6,991,272 | [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) | [`models/googlenet.py`](models/googlenet.py) |
| FCN-8s | 2015 | 134,362,751 | [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) | [`models/fcn8s.py`](models/fcn8s.py) |
| U-Net | 2015 | 31,031,810 | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | [`models/unet.py`](models/unet.py) |
| ResNet-18 | 2015 | 11,689,512 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | [`models/resnet18.py`](models/resnet18.py) |
| ResNet-50 | 2015 | 25,557,032 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | [`models/resnet50.py`](models/resnet50.py) |
| Faster R-CNN (ResNet-C4) | 2015 | 65,823,958 | [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) | [`models/faster_rcnn.py`](models/faster_rcnn.py) |
| SSD300 (VGG-16) | 2016 | 26,285,486 | [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) | [`models/ssd300.py`](models/ssd300.py) |
| YOLOv1 | 2016 | 271,703,550 | [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) | [`models/yolov1.py`](models/yolov1.py) |
| MobileNet v1 | 2017 | 4,231,976 | [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) | [`models/mobilenet_v1.py`](models/mobilenet_v1.py) |
| ResNeXt-50 32x4d | 2017 | 25,028,904 | [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) | [`models/resnext50.py`](models/resnext50.py) |
| DenseNet-121 | 2017 | 7,978,856 | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) | [`models/densenet121.py`](models/densenet121.py) |
| DeepLabV3 (ResNet-50) | 2017 | 42,004,074 | [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) | [`models/deeplabv3.py`](models/deeplabv3.py) |
| RetinaNet (ResNet-50 FPN) | 2017 | 37,968,692 | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) | [`models/retinanet.py`](models/retinanet.py) |
| Mask R-CNN (ResNet-50 FPN) | 2017 | 44,400,693 | [Mask R-CNN](https://arxiv.org/abs/1703.06870) | [`models/mask_rcnn.py`](models/mask_rcnn.py) |
| MobileNetV2 | 2018 | 3,504,872 | [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) | [`models/mobilenet_v2.py`](models/mobilenet_v2.py) |
| EfficientNet-B0 | 2019 | 5,288,548 | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | [`models/efficientnet_b0.py`](models/efficientnet_b0.py) |
| ViT-B/16 | 2020 | 86,567,656 | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | [`models/vit_b16.py`](models/vit_b16.py) |
| DeiT-Base Distilled | 2021 | 87,338,192 | [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) | [`models/deit_base_distilled.py`](models/deit_base_distilled.py) |
| Swin Transformer-T | 2021 | 28,288,354 | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | [`models/swin_t.py`](models/swin_t.py) |
| ConvNeXt-Tiny | 2022 | 28,589,128 | [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) | [`models/convnext_tiny.py`](models/convnext_tiny.py) |

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/themnvrao76/Classical-to-Modern-Computer-Vision.git
cd Classical-to-Modern-Computer-Vision
```

Install PyTorch:

```bash
pip install torch torchvision
```

Run an implementation directly:

```bash
python models/mask_rcnn.py
```

Each model file includes a small runnable check so the architecture can be instantiated and its output shape or parameter count verified.

---

## What Is Coming Next?

The repository is expanding beyond image classification into the major branches of modern computer vision:

- **Modern backbones:** MobileNetV3, SENet, Xception, ShuffleNet, RegNet
- **Semantic segmentation:** SegNet, PSPNet, DeepLabV3+
- **Object detection:** DETR, later YOLO generations
- **Human pose:** SimpleBaseline, HRNet, OpenPose-style methods, 3D human understanding
- **Self-supervised learning:** SimCLR, MoCo, BYOL, DINO
- **Vision-language learning:** CLIP-style image-text representation learning and multimodal vision
- **Generative vision:** Autoencoders, VAE, DCGAN, Pix2Pix, CycleGAN
- **Video and motion:** optical flow, tracking, temporal vision models
- **3D vision:** PointNet-family methods, depth, geometry, and human-centric 3D vision

---

## Repository Topics

`computer-vision` · `deep-learning` · `pytorch` · `cnn` · `vision-transformer` · `image-classification` · `object-detection` · `semantic-segmentation` · `pose-estimation` · `self-supervised-learning` · `vision-language-models` · `3d-vision` · `research-papers`

---

## References

Every architecture in the model index links to its original paper or primary publication. Parameter counts correspond to the implementations in this repository and may differ from other variants or training configurations.

---

<p align="center">
  <strong>From the foundations of convolutional networks to modern multimodal and 3D vision.</strong>
</p>
