from __future__ import annotations

import math

import pytest
import torch

from ultralytics.models.yolo.detect.dann import DomainClassifier, dann_lambda, gradient_reverse


def test_gradient_reversal_is_identity_in_forward() -> None:
    inputs = torch.tensor([[1.0, -2.0]], requires_grad=True)

    outputs = gradient_reverse(inputs, 0.75)

    assert torch.equal(outputs, inputs)
    assert outputs is not inputs


def test_gradient_reversal_scales_negative_gradient() -> None:
    inputs = torch.tensor([1.0, 2.0], requires_grad=True)

    gradient_reverse(inputs, 0.25).backward(torch.tensor([4.0, -8.0]))

    assert torch.equal(inputs.grad, torch.tensor([-1.0, 2.0]))


@pytest.mark.parametrize('coefficient', [-0.01, 1.01, math.inf, math.nan, True, '1.0'])
def test_gradient_reversal_rejects_invalid_coefficient(coefficient) -> None:
    error = TypeError if isinstance(coefficient, (bool, str)) else ValueError
    with pytest.raises(error, match='coefficient'):
        gradient_reverse(torch.ones(1), coefficient)


def test_dann_lambda_matches_schedule_endpoints() -> None:
    assert dann_lambda(0.0, lambda_max=0.8, gamma=10.0) == pytest.approx(0.0)
    assert dann_lambda(1.0, lambda_max=0.8, gamma=10.0) == pytest.approx(
        0.8 * (2.0 / (1.0 + math.exp(-10.0)) - 1.0)
    )


@pytest.mark.parametrize(
    ('progress', 'lambda_max', 'gamma', 'error'),
    [
        (-0.01, 1.0, 10.0, ValueError),
        (1.01, 1.0, 10.0, ValueError),
        (0.5, -0.1, 10.0, ValueError),
        (0.5, 1.1, 10.0, ValueError),
        (0.5, 1.0, 0.0, ValueError),
        (True, 1.0, 10.0, TypeError),
    ],
)
def test_dann_lambda_validates_arguments(progress, lambda_max, gamma, error) -> None:
    with pytest.raises(error):
        dann_lambda(progress, lambda_max=lambda_max, gamma=gamma)


def test_domain_classifier_returns_two_logits() -> None:
    classifier = DomainClassifier(32, hidden_channels=8, dropout=0.0)

    logits = classifier(torch.randn(3, 32, 5, 5))

    assert logits.shape == (3, 2)


@pytest.mark.parametrize(
    ('kwargs', 'error'),
    [
        ({'in_channels': 0}, ValueError),
        ({'in_channels': 16, 'hidden_channels': 0}, ValueError),
        ({'in_channels': 16, 'dropout': -0.1}, ValueError),
        ({'in_channels': 16, 'dropout': 1.0}, ValueError),
    ],
)
def test_domain_classifier_validates_dimensions(kwargs, error) -> None:
    with pytest.raises(error):
        DomainClassifier(**kwargs)


def test_domain_classifier_requires_p5_tensor() -> None:
    classifier = DomainClassifier(16, hidden_channels=8)

    with pytest.raises(ValueError, match='4D'):
        classifier(torch.randn(2, 16))
