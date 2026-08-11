from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch
from torch import nn


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
