from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn.tasks import DetectionModel


def _finite_real(name: str, value: Real) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f'{name} must be a real scalar')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, coefficient: float) -> torch.Tensor:
        coefficient = _finite_real('coefficient', coefficient)
        if not 0.0 <= coefficient <= 1.0:
            raise ValueError('coefficient must be within [0, 1]')
        ctx.coefficient = coefficient
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * gradient, None


def gradient_reverse(inputs: torch.Tensor, coefficient: float) -> torch.Tensor:
    if not isinstance(inputs, torch.Tensor):
        raise TypeError('inputs must be a torch.Tensor')
    return GradientReversalFunction.apply(inputs, coefficient)


class GradientReversalLayer(nn.Module):
    def forward(self, inputs: torch.Tensor, coefficient: float) -> torch.Tensor:
        return gradient_reverse(inputs, coefficient)


def dann_lambda(progress: float, lambda_max: float = 1.0, gamma: float = 10.0) -> float:
    progress = _finite_real('progress', progress)
    lambda_max = _finite_real('lambda_max', lambda_max)
    gamma = _finite_real('gamma', gamma)
    if not 0.0 <= progress <= 1.0:
        raise ValueError('progress must be within [0, 1]')
    if not 0.0 <= lambda_max <= 1.0:
        raise ValueError('lambda_max must be within [0, 1]')
    if gamma <= 0.0:
        raise ValueError('gamma must be greater than zero')
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


class DomainClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        if not isinstance(in_channels, int) or isinstance(in_channels, bool) or in_channels <= 0:
            raise ValueError('in_channels must be a positive integer')
        if not isinstance(hidden_channels, int) or isinstance(hidden_channels, bool) or hidden_channels <= 0:
            raise ValueError('hidden_channels must be a positive integer')
        dropout = _finite_real('dropout', dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be within [0, 1)')
        self.in_channels = in_channels
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor) or features.ndim != 4:
            raise ValueError('features must be a 4D tensor [batch, channels, height, width]')
        if features.shape[1] != self.in_channels:
            raise ValueError(f'expected {self.in_channels} P5 channels, received {features.shape[1]}')
        return self.classifier(features)


class DANNDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg='yolov8n.yaml',
        ch=3,
        nc=None,
        verbose=True,
        *,
        p5_layer=31,
        domain_hidden=256,
        domain_dropout=0.1,
        domain_loss_gain=1.0,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if p5_layer != 31:
            raise ValueError('the first DANN version requires p5_layer=31')
        domain_loss_gain = _finite_real('domain_loss_gain', domain_loss_gain)
        if domain_loss_gain < 0.0:
            raise ValueError('domain_loss_gain must be non-negative')
        self.p5_layer = p5_layer
        self.domain_loss_gain = domain_loss_gain
        p5_channels = self.model[-1].cv2[-1][0].conv.in_channels
        self.grl = GradientReversalLayer()
        self.domain_classifier = DomainClassifier(p5_channels, domain_hidden, domain_dropout)

    def forward_with_p5(self, images: torch.Tensor):
        y = []
        p5 = None
        output = images
        for module in self.model:
            if module.f != -1:
                output = y[module.f] if isinstance(module.f, int) else [output if j == -1 else y[j] for j in module.f]
            output = module(output)
            if module.i == self.p5_layer:
                p5 = output
            y.append(output if module.i in self.save else None)
        if p5 is None:
            raise RuntimeError(f'P5 layer {self.p5_layer} was not executed')
        return output, p5

    @staticmethod
    def _validate_domain_images(source: torch.Tensor, target: torch.Tensor) -> None:
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError('source and target images must be tensors')
        if source.ndim != 4 or target.ndim != 4 or source.shape[0] == 0 or target.shape[0] == 0:
            raise ValueError('source and target images must be non-empty 4D tensors')
        if source.shape[1:] != target.shape[1:]:
            raise ValueError('source and target images must have matching non-batch shapes')
        if source.device != target.device or source.dtype != target.dtype:
            raise ValueError('source and target images must share device and dtype')

    @staticmethod
    def _source_predictions(predictions, source_batch_size: int):
        if not isinstance(predictions, (list, tuple)) or not predictions:
            raise TypeError('training Detect predictions must be a non-empty list or tuple')
        if not all(isinstance(feature, torch.Tensor) for feature in predictions):
            raise TypeError('every Detect prediction must be a tensor')
        return [feature[:source_batch_size] for feature in predictions]

    @staticmethod
    def _source_batch(batch: dict) -> dict:
        return {
            key: value
            for key, value in batch.items()
            if not key.startswith('target_') and key not in {'dann_enabled', 'dann_lambda'}
        }

    def loss(self, batch, preds=None):
        if 'target_img' not in batch:
            detection_loss, detection_items = super().loss(batch, preds=preds)
            return detection_loss, torch.cat((detection_items, detection_items.new_zeros(3)))
        if preds is not None:
            raise ValueError('precomputed predictions are not supported for mixed-domain loss')
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()

        source_images = batch['img']
        target_images = batch['target_img']
        self._validate_domain_images(source_images, target_images)
        source_batch_size = source_images.shape[0]
        predictions, p5 = self.forward_with_p5(torch.cat((source_images, target_images), dim=0))
        source_predictions = self._source_predictions(predictions, source_batch_size)
        detection_loss, detection_items = self.criterion(source_predictions, self._source_batch(batch))

        enabled = batch.get('dann_enabled', True)
        coefficient = _finite_real('dann_lambda', batch.get('dann_lambda', 1.0))
        if not 0.0 <= coefficient <= 1.0:
            raise ValueError('dann_lambda must be within [0, 1]')
        if not isinstance(enabled, bool):
            raise TypeError('dann_enabled must be a bool')
        if not enabled:
            return detection_loss, torch.cat((detection_items, detection_items.new_zeros(3)))

        target_batch_size = target_images.shape[0]
        domain_labels = torch.cat(
            (
                torch.zeros(source_batch_size, dtype=torch.long, device=p5.device),
                torch.ones(target_batch_size, dtype=torch.long, device=p5.device),
            )
        )
        domain_logits = self.domain_classifier(self.grl(p5, coefficient))
        domain_loss = F.cross_entropy(domain_logits, domain_labels)
        domain_accuracy = (domain_logits.argmax(dim=1) == domain_labels).float().mean()
        total_loss = detection_loss + self.domain_loss_gain * domain_loss * source_batch_size
        metrics = detection_items.new_tensor((domain_loss.detach(), domain_accuracy.detach(), coefficient))
        return total_loss, torch.cat((detection_items, metrics))
