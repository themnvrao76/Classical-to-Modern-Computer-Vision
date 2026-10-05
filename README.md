# Classical to Modern Computer Vision

<p align="center">
  <strong>PyTorch implementations of landmark computer vision architectures — from LeNet and AlexNet to ResNet, EfficientNet, Vision Transformers, detection, segmentation, self-supervised learning, multimodal vision, and 3D vision.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-Computer%20Vision-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Models-36-blue" alt="Models">
  <img src="https://img.shields.io/badge/Papers-Original%20Sources-success" alt="Original Papers">
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen" alt="Active Development">
</p>

---

## About

This repository is a growing collection of **computer vision architectures implemented in PyTorch**, organized as a journey from classical convolutional neural networks to modern transformer-based and multimodal vision systems.

The goal is to keep the implementations readable and close to the architectural ideas introduced in the original papers while building a practical reference for studying the evolution of computer vision.

**Current coverage:** CNNs · residual networks · efficient/mobile vision · vision transformers · modern ConvNet backbones · semantic segmentation · object/instance detection · pose estimation · metric learning · self-supervised learning · vision-language learning · 3D vision  
**Planned coverage:** generative vision · optical flow · tracking · deeper human-mesh and 3D reconstruction methods

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
| Siamese Network | 1993 | Config-dependent | [Signature Verification using a Siamese Time Delay Neural Network](https://papers.nips.cc/paper/1993/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html) | [`models/siamese_network.py`](models/siamese_network.py) |
| LeNet-5 | 1998 | 44,426 | [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791) | [`models/lenet.py`](models/lenet.py) |
| AlexNet | 2012 | 61,100,840 | [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | [`models/alexnet.py`](models/alexnet.py) |
| VGG-16 | 2014 | 138,357,544 | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | [`models/vgg16.py`](models/vgg16.py) |
| GoogLeNet / Inception v1 | 2014 | 6,991,272 | [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) | [`models/googlenet.py`](models/googlenet.py) |
| FCN-8s | 2015 | 134,362,751 | [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) | [`models/fcn8s.py`](models/fcn8s.py) |
| U-Net | 2015 | 31,031,810 | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | [`models/unet.py`](models/unet.py) |
| ResNet-18 | 2015 | 11,689,512 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | [`models/resnet18.py`](models/resnet18.py) |
| ResNet-50 | 2015 | 25,557,032 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | [`models/resnet50.py`](models/resnet50.py) |
| Faster R-CNN (ResNet-C4) | 2015 | 65,823,958 | [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) | [`models/faster_rcnn.py`](models/faster_rcnn.py) |
| Triplet Metric Learning | 2015 | Config-dependent | [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) | [`models/triplet_network.py`](models/triplet_network.py) |
| SSD300 (VGG-16) | 2016 | 26,285,486 | [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) | [`models/ssd300.py`](models/ssd300.py) |
| YOLOv1 | 2016 | 271,703,550 | [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) | [`models/yolov1.py`](models/yolov1.py) |
| MobileNet v1 | 2017 | 4,231,976 | [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) | [`models/mobilenet_v1.py`](models/mobilenet_v1.py) |
| ResNeXt-50 32x4d | 2017 | 25,028,904 | [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) | [`models/resnext50.py`](models/resnext50.py) |
| DenseNet-121 | 2017 | 7,978,856 | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) | [`models/densenet121.py`](models/densenet121.py) |
| DeepLabV3 (ResNet-50) | 2017 | 42,004,074 | [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) | [`models/deeplabv3.py`](models/deeplabv3.py) |
| RetinaNet (ResNet-50 FPN) | 2017 | 37,968,692 | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) | [`models/retinanet.py`](models/retinanet.py) |
| Mask R-CNN (ResNet-50 FPN) | 2017 | 44,400,693 | [Mask R-CNN](https://arxiv.org/abs/1703.06870) | [`models/mask_rcnn.py`](models/mask_rcnn.py) |
| PointNet | 2017 | Config-dependent | [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) | [`models/pointnet.py`](models/pointnet.py) |
| MobileNetV2 | 2018 | 3,504,872 | [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) | [`models/mobilenet_v2.py`](models/mobilenet_v2.py) |
| DeepLabV3+ | 2018 | Config-dependent | [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) | [`models/deeplabv3_plus.py`](models/deeplabv3_plus.py) |
| SimpleBaseline Pose | 2018 | Config-dependent | [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/abs/1804.06208) | [`models/simple_baseline_pose.py`](models/simple_baseline_pose.py) |
| EfficientNet-B0 | 2019 | 5,288,548 | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | [`models/efficientnet_b0.py`](models/efficientnet_b0.py) |
| MobileNetV3-Small | 2019 | Config-dependent | [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) | [`models/mobilenet_v3_small.py`](models/mobilenet_v3_small.py) |
| HRNet Pose | 2019 | Config-dependent | [Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212) | [`models/hrnet_pose.py`](models/hrnet_pose.py) |
| ViT-B/16 | 2020 | 86,567,656 | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | [`models/vit_b16.py`](models/vit_b16.py) |
| MoCo | 2020 | Encoder-dependent | [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) | [`models/moco.py`](models/moco.py) |
| SimCLR | 2020 | Encoder-dependent | [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) | [`models/simclr.py`](models/simclr.py) |
| DETR | 2020 | Config-dependent | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | [`models/detr.py`](models/detr.py) |
| BYOL | 2020 | Encoder-dependent | [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733) | [`models/byol.py`](models/byol.py) |
| DeiT-Base Distilled | 2021 | 87,338,192 | [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) | [`models/deit_base_distilled.py`](models/deit_base_distilled.py) |
| Swin Transformer-T | 2021 | 28,288,354 | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | [`models/swin_t.py`](models/swin_t.py) |
| CLIP-style Image-Text Model | 2021 | Config-dependent | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | [`models/clip.py`](models/clip.py) |
| DINO | 2021 | Config-dependent | [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) | [`models/dino.py`](models/dino.py) |
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

- **Modern backbones:** MobileNetV3 ✓; next SENet, Xception, ShuffleNet, RegNet
- **Semantic segmentation:** DeepLabV3+ ✓; next SegNet and PSPNet
- **Object detection:** DETR ✓; next later YOLO generations and deformable attention detectors
- **Human pose:** SimpleBaseline ✓, HRNet ✓; next OpenPose-style methods and 3D human understanding
- **Metric learning:** Siamese and triplet-learning references ✓; next stronger retrieval/embedding objectives
- **Self-supervised learning:** SimCLR ✓, MoCo ✓, BYOL ✓, DINO ✓
- **Vision-language learning:** CLIP-style image-text representation learning ✓; next multimodal fusion and grounded VLM components
- **Generative vision:** Autoencoders, VAE, DCGAN, Pix2Pix, CycleGAN
- **Video and motion:** optical flow, tracking, temporal vision models
- **3D vision:** PointNet ✓; next PointNet-family extensions, depth, geometry, and human-centric 3D vision

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
