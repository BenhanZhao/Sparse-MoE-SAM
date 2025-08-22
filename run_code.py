# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Data processing
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

# Model visualization
import cv2


SMOOTH = 1e-5

import re
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class BaseProcessor:
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


class ImageTrainProcessor(BaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def preprocess(self, item, return_tensors):
        return {'pixel_values': [self.transform(item)]}


class ImageEvalProcessor(BaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def preprocess(self, item, return_tensors):
        return {'pixel_values': [self.transform(item)]}


class QWenImageProcessor(BaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.transform = transforms.Compose([
            transforms.Resize(
                (448, 448),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def preprocess(self, item, return_tensors):
        return {'pixel_values': [self.transform(item)]}

def _gather_channels(x, indexes, **kwargs):
    """Slice tensor along channels axis by given indexes"""
    backend = kwargs['backend']
    if backend.image_data_format() == 'channels_last':
        x = backend.permute_dimensions(x, (3, 0, 1, 2))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 2, 3, 0))
    else:
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
    return x


def get_reduce_axes(per_image, **kwargs):
    backend = kwargs['backend']
    axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes


def gather_channels(*xs, indexes=None, **kwargs):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes, **kwargs) for x in xs]
    return xs


def round_if_needed(x, threshold, **kwargs):
    backend = kwargs['backend']
    if threshold is not None:
        x = backend.greater(x, threshold)
        x = backend.cast(x, backend.floatx())
    return x


def average(x, per_image=False, class_weights=None, **kwargs):
    backend = kwargs['backend']
    if per_image:
        x = backend.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return backend.mean(x)


# ----------------------------------------------------------------
#   Metric Functions
# ----------------------------------------------------------------

def iou_score(gt, pr, class_weights=1., class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None, **kwargs):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:

    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
        IoU/Jaccard score in range [0, 1]

    .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index

    """

    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    # score calculation
    intersection = backend.sum(gt * pr, axis=axes)
    union = backend.sum(gt + pr, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


def f_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None,
            **kwargs):
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:

    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    The formula in terms of *Type I* and *Type II* errors:

    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}


    where:
        TP - true positive;
        FP - false positive;
        FN - false negative;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        beta: f-score coefficient
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
        F-score in range [0, 1]

    """

    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    # calculate score
    tp = backend.sum(gt * pr, axis=axes)
    fp = backend.sum(pr, axis=axes) - tp
    fn = backend.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


def precision(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None, **kwargs):
    r"""Calculate precision between the ground truth (gt) and the prediction (pr).

    .. math:: F_\beta(tp, fp) = \frac{tp} {(tp + fp)}

    where:
         - tp - true positives;
         - fp - false positives;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``precision`` name is used.

    Returns:
        float: precision score
    """
    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    # score calculation
    tp = backend.sum(gt * pr, axis=axes)
    fp = backend.sum(pr, axis=axes) - tp
    
    score = (tp + smooth) / (tp + fp + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


def recall(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None, **kwargs):
    r"""Calculate recall between the ground truth (gt) and the prediction (pr).

    .. math:: F_\beta(tp, fn) = \frac{tp} {(tp + fn)}

    where:
         - tp - true positives;
         - fp - false positives;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``precision`` name is used.

    Returns:
        float: recall score
    """
    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    tp = backend.sum(gt * pr, axis=axes)
    fn = backend.sum(gt, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


# ----------------------------------------------------------------
#   Loss Functions
# ----------------------------------------------------------------

def categorical_crossentropy(gt, pr, class_weights=1., class_indexes=None, **kwargs):
    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)

    # scale predictions so that the class probas of each sample sum to 1
    axis = 3 if backend.image_data_format() == 'channels_last' else 1
    pr /= backend.sum(pr, axis=axis, keepdims=True)

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1 - backend.epsilon())

    # calculate loss
    output = gt * backend.log(pr) * class_weights
    return - backend.mean(output)


def binary_crossentropy(gt, pr, **kwargs):
    backend = kwargs['backend']
    return backend.mean(backend.binary_crossentropy(gt, pr))


def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, class_indexes=None, **kwargs):
    r"""Implementation of Focal Loss from the paper in multiclass classification

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

    """

    backend = kwargs['backend']
    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

    # Calculate focal loss
    loss = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))

    return backend.mean(loss)


def binary_focal_loss(gt, pr, gamma=2.0, alpha=0.25, **kwargs):
    r"""Implementation of Focal Loss from the paper in binary classification

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr) \
               - (1 - gt) * alpha * (pr^gamma) * log(1 - pr)

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0

    """
    backend = kwargs['backend']

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

    loss_1 = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))
    loss_0 = - (1 - gt) * ((1 - alpha) * backend.pow((pr), gamma) * backend.log(1 - pr))
    loss = backend.mean(loss_0 + loss_1)
    return loss

class PlantDiseaseDataset(Dataset):
    """Custom dataset for plant disease segmentation"""
    
    def __init__(self, root_path, image_size=256, train=True, split_ratio=0.8):
        self.image_size = image_size
        self.image_paths = sorted(glob(os.path.join(root_path, "images/*.jpg")))
        self.mask_paths = [path.replace("images", "masks").replace("jpg", "png") 
                          for path in self.image_paths]
        
        # Split data
        split_idx = int(len(self.image_paths) * split_ratio)
        if train:
            self.image_paths = self.image_paths[:split_idx]
            self.mask_paths = self.mask_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
            self.mask_paths = self.mask_paths[split_idx:]
            
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        
        # Load mask
        mask = Image.open(self.mask_paths[idx]).convert('L')
        mask = mask.resize((self.image_size, self.image_size))
        mask = np.array(mask) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return image, mask

class ConvBlock(nn.Module):
    """Convolution block: Conv2D -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ConvBlock, self).__init__()
        padding = dilation if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SparseAttention(nn.Module):
    """Sparse Self-Attention Mechanism for efficient long-range dependencies"""
    
    def __init__(self, dim, num_heads=8, sparsity_ratio=0.1):
        super(SparseAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # Reshape for attention computation
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # B, N, C
        
        # Compute Q, K, V
        Q = self.query(x_flat).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x_flat).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x_flat).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sparsity - keep only top-k attention weights
        topk = max(1, int(N * self.sparsity_ratio))
        sparse_scores = torch.zeros_like(scores)
        
        # Get top-k indices and values
        values, indices = torch.topk(scores, topk, dim=-1)
        sparse_scores.scatter_(-1, indices, values)
        
        # Apply softmax and attention
        attn_weights = F.softmax(sparse_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        
        # Output projection and residual connection
        out = self.output(out)
        out = self.layer_norm(out + x_flat)
        
        # Reshape back to original format
        out = out.permute(0, 2, 1).view(B, C, H, W)
        
        return out

class ExpertLayer(nn.Module):
    """Individual expert in the MoE system"""
    
    def __init__(self, in_channels, out_channels, expert_type='conv'):
        super(ExpertLayer, self).__init__()
        self.expert_type = expert_type
        
        if expert_type == 'conv':
            self.expert = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif expert_type == 'dilated':
            self.expert = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif expert_type == 'depthwise':
            self.expert = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        return self.expert(x)

class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer for adaptive feature processing"""
    
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=2):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create different types of experts
        expert_types = ['conv', 'dilated', 'depthwise', 'conv']
        self.experts = nn.ModuleList([
            ExpertLayer(in_channels, out_channels, expert_types[i % len(expert_types)])
            for i in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_experts, 1),
            nn.Softmax(dim=1)
        )
        
        # Load balancing
        self.noise_scale = 0.1
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Compute gating weights
        gate_weights = self.gate(x)  # B, num_experts, 1, 1
        
        # Add noise for load balancing during training
        if self.training:
            noise = torch.randn_like(gate_weights) * self.noise_scale
            gate_weights = gate_weights + noise
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(gate_weights.squeeze(), self.top_k, dim=1)
        top_k_weights = F.softmax(top_k_weights, dim=1)
        
        # Compute expert outputs
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.experts[i](x))
        
        # Combine expert outputs using gating weights
        output = torch.zeros_like(expert_outputs[0])
        for b in range(B):
            for k in range(self.top_k):
                expert_idx = top_k_indices[b, k]
                weight = top_k_weights[b, k]
                output[b] += weight * expert_outputs[expert_idx][b]
        
        return output

class EnhancedASPP(nn.Module):
    """Enhanced Atrous Spatial Pyramid Pooling with Sparse Attention"""
    
    def __init__(self, in_channels, out_channels=256):
        super(EnhancedASPP, self).__init__()
        
        # Image pooling
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(in_channels, out_channels, 1),
        )
        
        # Atrous convolutions
        self.conv_1x1 = ConvBlock(in_channels, out_channels, 1)
        self.conv_3x3_6 = ConvBlock(in_channels, out_channels, 3, dilation=6)
        self.conv_3x3_12 = ConvBlock(in_channels, out_channels, 3, dilation=12)
        self.conv_3x3_18 = ConvBlock(in_channels, out_channels, 3, dilation=18)
        
        # Sparse attention module
        self.sparse_attention = SparseAttention(out_channels * 5, num_heads=8, sparsity_ratio=0.1)
        
        # Output projection
        self.project = ConvBlock(out_channels * 5, out_channels, 1)
        
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        
        # Image pooling
        image_pool = self.image_pool(x)
        image_pool = F.interpolate(image_pool, size=(h, w), mode='bilinear', align_corners=False)
        
        # Atrous convolutions
        conv_1x1 = self.conv_1x1(x)
        conv_3x3_6 = self.conv_3x3_6(x)
        conv_3x3_12 = self.conv_3x3_12(x)
        conv_3x3_18 = self.conv_3x3_18(x)
        
        # Concatenate features
        features = torch.cat([image_pool, conv_1x1, conv_3x3_6, conv_3x3_12, conv_3x3_18], dim=1)
        
        # Apply sparse attention
        features = self.sparse_attention(features)
        
        # Project to output channels
        return self.project(features)

class KerasObject:
    _backend = None
    _models = None
    _layers = None
    _utils = None

    def __init__(self, name=None):
        if (self.backend is None or
                self.utils is None or
                self.models is None or
                self.layers is None):
            raise RuntimeError('You cannot use `KerasObjects` with None submodules.')

        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name

    @property
    def name(self):
        return self.__name__

    @name.setter
    def name(self, name):
        self._name = name

    @classmethod
    def set_submodules(cls, backend, layers, models, utils):
        cls._backend = backend
        cls._layers = layers
        cls._models = models
        cls._utils = utils

    @property
    def submodules(self):
        return {
            'backend': self.backend,
            'layers': self.layers,
            'models': self.models,
            'utils': self.utils,
        }

    @property
    def backend(self):
        return self._backend

    @property
    def layers(self):
        return self._layers

    @property
    def models(self):
        return self._models

    @property
    def utils(self):
        return self._utils


class Metric(KerasObject):
    pass


class Loss(KerasObject):

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `Loss` class')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __rmul__(self, other):
        return self.__mul__(other)


class MultipliedLoss(Loss):

    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split('+')) > 1:
            name = '{}({})'.format(multiplier, loss.__name__)
        else:
            name = '{}{}'.format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, gt, pr):
        return self.multiplier * self.loss(gt, pr)


class SumOfLosses(Loss):

    def __init__(self, l1, l2):
        name = '{}_plus_{}'.format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, gt, pr):
        return self.l1(gt, pr) + self.l2(gt, pr)

class EnhancedDeepLabV3Plus(nn.Module):
    """Enhanced DeepLabV3+ with Sparse Attention and Mixture of Experts"""
    
    def __init__(self, num_classes=1, backbone='resnet50'):
        super(EnhancedDeepLabV3Plus, self).__init__()
        
        # Backbone
        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            # Remove the last two layers (avgpool and fc)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            low_level_channels = 256
            high_level_channels = 2048
        
        # Enhanced ASPP with sparse attention
        self.aspp = EnhancedASPP(high_level_channels, 256)
        
        # Low-level feature processing
        self.low_level_conv = ConvBlock(low_level_channels, 48, 1)
        
        # Enhanced decoder with MoE
        self.moe_layer1 = MixtureOfExperts(256 + 48, 256, num_experts=4, top_k=2)
        self.conv_block = ConvBlock(256, 256, 3)
        self.moe_layer2 = MixtureOfExperts(256, 256, num_experts=3, top_k=2)
        
        # Output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(256, num_classes, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features from backbone
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 4:  # After conv2_x (low-level features)
                low_level_features = x
            features.append(x)
        
        # High-level features through enhanced ASPP
        high_level_features = self.aspp(x)
        
        # Upsample high-level features
        high_level_features = F.interpolate(
            high_level_features, 
            size=low_level_features.size()[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Process low-level features
        low_level_features = self.low_level_conv(low_level_features)
        
        # Concatenate features
        combined_features = torch.cat([high_level_features, low_level_features], dim=1)
        
        # Enhanced decoder with MoE
        x = self.moe_layer1(combined_features)
        x = self.conv_block(x)
        x = self.moe_layer2(x)
        
        # Final output
        output = self.output_conv(x)
        
        # Upsample to input size
        output = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
        
        return output

def visualize_predictions(model, dataloader, device, num_samples=5):
    """Visualize model predictions"""
    model.eval()
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            predictions = model(images)
            
            # Convert to numpy for visualization
            image_np = images[0].cpu().permute(1, 2, 0).numpy()
            mask_np = masks[0].cpu().squeeze().numpy()
            pred_np = predictions[0].cpu().squeeze().numpy()
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image_np)
            plt.title("Input Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(mask_np, cmap='gray')
            plt.title("Ground Truth")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred_np, cmap='gray')
            plt.title("Prediction")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()

def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=0.001):
    """Enhanced training function with adaptive learning rate"""
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Show predictions every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_predictions(model, val_loader, device, num_samples=2)
    
    return train_losses, val_losses

def main():
    """Main training pipeline with enhanced model"""
    # Configuration
    ROOT_PATH = './data/'  # Update this path
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(ROOT_PATH, IMAGE_SIZE, train=True)
    val_dataset = PlantDiseaseDataset(ROOT_PATH, IMAGE_SIZE, train=False)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize enhanced model
    model = EnhancedDeepLabV3Plus(num_classes=1).to(device)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, device, NUM_EPOCHS, LEARNING_RATE
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save model
    torch.save(model.state_dict(), 'model_checkpoint.pth')
    print('Model saved as model_checkpoint.pth')
    
    # Final predictions visualization
    print('Final enhanced model predictions:')
    visualize_predictions(model, val_loader, device, num_samples=10)

from typing import Type

import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

if __name__ == "__main__":
    main()