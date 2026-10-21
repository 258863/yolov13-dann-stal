from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import scale_img


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
    DOMAIN_FEATURE_BRANCHES = {23: 0, 27: 1, 31: 2}
    BACKBONE_DOMAIN_FEATURE_LAYERS = {1}
    VERIFIED_DOMAIN_FEATURE_LAYERS = (1, 23, 27, 31)

    def __init__(
        self,
        cfg='yolov8n.yaml',
        ch=3,
        nc=None,
        verbose=True,
        *,
        p5_layer=31,
        domain_feature_layers=None,
        domain_hidden=256,
        domain_dropout=0.1,
        domain_loss_gain=1.0,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if domain_feature_layers is None:
            domain_feature_layers = (p5_layer,)
        elif not isinstance(domain_feature_layers, (list, tuple)):
            raise TypeError('domain_feature_layers must be a list or tuple')
        else:
            domain_feature_layers = tuple(domain_feature_layers)
            if p5_layer != 31 and p5_layer != domain_feature_layers[0]:
                raise ValueError('p5_layer and domain_feature_layers conflict')
        if not domain_feature_layers:
            raise ValueError('domain_feature_layers must not be empty')
        if any(not isinstance(layer, int) or isinstance(layer, bool) for layer in domain_feature_layers):
            raise TypeError('domain_feature_layers must contain integers')
        if len(set(domain_feature_layers)) != len(domain_feature_layers):
            raise ValueError('domain_feature_layers must contain unique layers')
        if any(layer not in self.VERIFIED_DOMAIN_FEATURE_LAYERS for layer in domain_feature_layers):
            raise ValueError('domain_feature_layers contains an unverified YOLOv13 feature layer')
        domain_loss_gain = _finite_real('domain_loss_gain', domain_loss_gain)
        if domain_loss_gain < 0.0:
            raise ValueError('domain_loss_gain must be non-negative')
        self.domain_feature_layers = domain_feature_layers
        self.p5_layer = domain_feature_layers[0]
        self.domain_loss_gain = domain_loss_gain
        self.grl = GradientReversalLayer()
        self.extra_grls = nn.ModuleList(GradientReversalLayer() for _ in domain_feature_layers[1:])
        channels = [self._domain_feature_channels(layer) for layer in domain_feature_layers]
        self.domain_classifier = DomainClassifier(channels[0], domain_hidden, domain_dropout)
        self.extra_domain_classifiers = nn.ModuleList(
            DomainClassifier(channel, domain_hidden, domain_dropout) for channel in channels[1:]
        )

    @property
    def grls(self):
        return (self.grl, *self.extra_grls)

    @property
    def domain_classifiers(self):
        return (self.domain_classifier, *self.extra_domain_classifiers)

    def _domain_feature_channels(self, layer: int) -> int:
        if layer in self.BACKBONE_DOMAIN_FEATURE_LAYERS:
            return self.model[layer].conv.out_channels
        detect = self.model[-1]
        if detect.f != list(self.DOMAIN_FEATURE_BRANCHES):
            raise RuntimeError(f'expected YOLOv13 Detect inputs {list(self.DOMAIN_FEATURE_BRANCHES)}, received {detect.f}')
        return detect.cv2[self.DOMAIN_FEATURE_BRANCHES[layer]][0].conv.in_channels

    def _predict_augment(self, images: torch.Tensor):
        if getattr(self, 'end2end', False):
            return super()._predict_augment(images)
        image_size = images.shape[-2:]
        outputs = []
        for scale, flip in zip((1, 0.83, 0.67), (None, 3, None)):
            augmented = scale_img(images.flip(flip) if flip else images, scale, gs=int(self.stride.max()))
            prediction = super().predict(augmented)[0]
            outputs.append(self._descale_pred(prediction, flip, scale, image_size))
        return torch.cat(self._clip_augmented(outputs), -1), None

    def forward_with_domain_features(self, images: torch.Tensor):
        y = []
        selected_layers = set(self.domain_feature_layers)
        domain_features = {}
        output = images
        for module in self.model:
            if module.f != -1:
                output = y[module.f] if isinstance(module.f, int) else [output if j == -1 else y[j] for j in module.f]
            output = module(output)
            if module.i in selected_layers:
                domain_features[module.i] = output
            y.append(output if module.i in self.save else None)
        missing = selected_layers.difference(domain_features)
        if missing:
            raise RuntimeError(f'domain feature layers were not executed: {sorted(missing)}')
        return output, tuple(domain_features[layer] for layer in self.domain_feature_layers)

    def forward_with_p5(self, images: torch.Tensor):
        if len(self.domain_feature_layers) != 1:
            raise RuntimeError('forward_with_p5 requires exactly one configured domain feature layer')
        output, domain_features = self.forward_with_domain_features(images)
        return output, domain_features[0]

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
        predictions, domain_features = self.forward_with_domain_features(torch.cat((source_images, target_images), dim=0))
        target_supervised = batch.get('target_supervised', False)
        if not isinstance(target_supervised, bool):
            raise TypeError('target_supervised must be a bool')
        detection_predictions = predictions if target_supervised else self._source_predictions(predictions, source_batch_size)
        detection_loss, detection_items = self.criterion(detection_predictions, self._source_batch(batch))

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
                torch.zeros(source_batch_size, dtype=torch.long, device=domain_features[0].device),
                torch.ones(target_batch_size, dtype=torch.long, device=domain_features[0].device),
            )
        )
        domain_losses = []
        domain_accuracies = []
        for features, grl, classifier in zip(domain_features, self.grls, self.domain_classifiers):
            domain_logits = classifier(grl(features, coefficient))
            domain_losses.append(F.cross_entropy(domain_logits, domain_labels))
            domain_accuracies.append((domain_logits.argmax(dim=1) == domain_labels).float().mean())
        domain_loss = torch.stack(domain_losses).mean()
        domain_accuracy = torch.stack(domain_accuracies).mean()
        total_loss = detection_loss + self.domain_loss_gain * domain_loss * source_batch_size
        metrics = detection_items.new_tensor((domain_loss.detach(), domain_accuracy.detach(), coefficient))
        return total_loss, torch.cat((detection_items, metrics))
