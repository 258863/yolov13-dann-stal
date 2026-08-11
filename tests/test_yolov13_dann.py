from __future__ import annotations

import math

import pytest
import torch

from ultralytics.models.yolo.detect.dann import DANNDetectionModel, DomainClassifier, dann_lambda, gradient_reverse


class RecordingCriterion:
    def __init__(self) -> None:
        self.batch_sizes = None
        self.batch_keys = None

    def __call__(self, predictions, batch):
        self.batch_sizes = [prediction.shape[0] for prediction in predictions]
        self.batch_keys = set(batch)
        zero = sum((prediction.sum() * 0.0 for prediction in predictions), torch.zeros((), device=predictions[0].device))
        return zero, torch.zeros(3, device=predictions[0].device)


def make_detection_model(**kwargs) -> DANNDetectionModel:
    return DANNDetectionModel('cfg/models/v13/yolov13.yaml', nc=2, verbose=False, domain_hidden=8, **kwargs)


def assert_nested_close(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        return
    assert type(left) is type(right)
    assert len(left) == len(right)
    for left_item, right_item in zip(left, right):
        assert_nested_close(left_item, right_item)


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


def test_dann_model_uses_verified_yolov13_layers() -> None:
    model = make_detection_model()

    assert model.model[31].i == 31
    assert model.model[32].i == 32
    assert model.model[32].f == [23, 27, 31]
    assert model.p5_layer == 31
    assert model.domain_classifier.in_channels == model.model[32].cv2[-1][0].conv.in_channels


def test_forward_with_p5_preserves_prediction_contract() -> None:
    torch.manual_seed(2)
    model = make_detection_model().eval()
    images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        expected = model(images)
        actual, p5 = model.forward_with_p5(images)

    assert p5.shape[0] == 2
    assert p5.shape[1] == model.domain_classifier.in_channels
    assert_nested_close(actual, expected)


def test_dann_loss_sends_only_source_predictions_to_detection_criterion() -> None:
    model = make_detection_model().train()
    criterion = RecordingCriterion()
    model.criterion = criterion
    source = torch.randn(1, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)

    loss, items = model.loss(
        {'img': source, 'target_img': target, 'dann_enabled': False, 'dann_lambda': 0.5, 'target_cls': 'unused'}
    )

    assert criterion.batch_sizes == [1, 1, 1]
    assert 'target_img' not in criterion.batch_keys
    assert 'target_cls' not in criterion.batch_keys
    assert loss.shape == ()
    assert items.shape == (6,)
    assert torch.equal(items[3:], torch.zeros(3))


def test_dann_loss_validates_mixed_image_shapes() -> None:
    model = make_detection_model().train()
    model.criterion = RecordingCriterion()

    with pytest.raises(ValueError, match='matching non-batch shapes'):
        model.loss({'img': torch.randn(1, 3, 64, 64), 'target_img': torch.randn(1, 3, 32, 64)})


def test_dann_loss_traverses_backbone_once_for_both_domains() -> None:
    model = make_detection_model().train()
    model.criterion = RecordingCriterion()
    calls = []
    handle = model.model[0].register_forward_hook(lambda *args: calls.append(1))
    try:
        model.loss(
            {
                'img': torch.randn(1, 3, 64, 64),
                'target_img': torch.randn(1, 3, 64, 64),
                'dann_enabled': False,
            }
        )
    finally:
        handle.remove()

    assert len(calls) == 1


def test_dann_loss_backpropagates_domain_and_backbone_gradients() -> None:
    torch.manual_seed(3)
    model = make_detection_model(domain_loss_gain=0.5).train()
    model.criterion = RecordingCriterion()

    loss, items = model.loss(
        {
            'img': torch.randn(1, 3, 64, 64),
            'target_img': torch.randn(1, 3, 64, 64) + 2.0,
            'dann_enabled': True,
            'dann_lambda': 0.75,
        }
    )
    loss.backward()

    assert items.shape == (6,)
    assert items[3] > 0
    assert 0 <= items[4] <= 1
    assert items[5] == pytest.approx(0.75)
    assert model.domain_classifier.classifier[-1].weight.grad is not None
    assert model.model[0].conv.weight.grad is not None
    assert torch.isfinite(model.model[0].conv.weight.grad).all()
