from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ultralytics.models.yolo.detect.dann import DANNDetectionModel, DomainClassifier, dann_lambda, gradient_reverse
from ultralytics.models.yolo.detect.train_dann import DANNDetectionTrainer, validate_dann_config
from ultralytics.nn.tasks import DetectionModel
from ultralytics.scripts import train_yolov13_dann
from ultralytics.scripts.train_yolov13_dann import build_trainer, load_experiment_config, select_trainer


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
    if left is None:
        assert right is None
        return
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


def test_source_only_validation_loss_keeps_six_metric_slots() -> None:
    model = make_detection_model().train()
    model.criterion = RecordingCriterion()
    predictions = [torch.randn(1, 4, size, size) for size in (8, 4, 2)]

    loss, items = model.loss({'img': torch.randn(1, 3, 64, 64)}, preds=predictions)

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


def test_validate_dann_config_applies_defaults() -> None:
    config = validate_dann_config({'target_data': 'target.yaml', 'experiment': 'e1'})

    assert config['target_batch'] is None
    assert config['p5_layer'] == 31
    assert config['domain_hidden'] == 256
    assert config['domain_loss_gain'] == pytest.approx(1.0)
    assert config['lambda_max'] == pytest.approx(1.0)
    assert config['gamma'] == pytest.approx(10.0)


def test_validate_e0bn_config_requires_supervised_batch_without_target_data() -> None:
    config = validate_dann_config({'experiment': 'e0bn', 'supervised_batch': 8})

    assert config['target_data'] is None
    assert config['supervised_batch'] == 8
    with pytest.raises(ValueError, match='supervised_batch'):
        validate_dann_config({'experiment': 'e0bn'})
    with pytest.raises(ValueError, match='supervised_batch'):
        validate_dann_config({'experiment': 'e0bn', 'supervised_batch': 0})


@pytest.mark.parametrize(
    'updates',
    [
        {},
        {'target_data': 'target.yaml', 'experiment': 'e0'},
        {'target_data': 'target.yaml', 'experiment': 'e1', 'p5_layer': 27},
        {'target_data': 'target.yaml', 'experiment': 'e1', 'target_batch': 0},
        {'target_data': 'target.yaml', 'experiment': 'e1', 'lambda_max': 1.1},
    ],
)
def test_validate_dann_config_rejects_invalid_values(updates) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_dann_config(updates)


def make_trainer_without_init(experiment='e1') -> DANNDetectionTrainer:
    trainer = object.__new__(DANNDetectionTrainer)
    trainer.device = torch.device('cpu')
    trainer.args = SimpleNamespace(multi_scale=False, imgsz=64)
    trainer.dann_config = validate_dann_config({'target_data': 'target.yaml', 'experiment': experiment})
    trainer.target_loader = [{'img': torch.full((1, 3, 32, 32), 255, dtype=torch.uint8)}]
    trainer._target_iterator = iter(trainer.target_loader)
    trainer._target_epoch = None
    trainer.train_loader = [None, None]
    trainer.epochs = 2
    trainer.epoch = 0
    trainer.start_epoch = 0
    trainer._dann_step = 0
    return trainer


def test_trainer_cycles_target_batches_and_matches_source_shape() -> None:
    trainer = make_trainer_without_init()
    source = {'img': torch.zeros(1, 3, 64, 64, dtype=torch.uint8), 'cls': torch.zeros(0, 1)}

    first = trainer.preprocess_batch(source.copy())
    second = trainer.preprocess_batch(source.copy())

    assert first['img'].shape == first['target_img'].shape
    assert second['target_img'].shape == second['img'].shape
    assert torch.all(first['target_img'] == 1.0)
    assert first['dann_enabled'] is True
    assert first['dann_lambda'] == pytest.approx(0.0)
    assert second['dann_lambda'] > first['dann_lambda']
    assert 'target_cls' not in first


def test_e05_trainer_disables_domain_objective_but_keeps_target_images() -> None:
    trainer = make_trainer_without_init(experiment='e05')

    batch = trainer.preprocess_batch({'img': torch.zeros(1, 3, 64, 64, dtype=torch.uint8)})

    assert batch['target_img'].shape == batch['img'].shape
    assert batch['dann_enabled'] is False


def test_e0bn_trainer_splits_source_batch_and_filters_detection_targets() -> None:
    trainer = object.__new__(DANNDetectionTrainer)
    trainer.device = torch.device('cpu')
    trainer.args = SimpleNamespace(multi_scale=False, imgsz=64)
    trainer.dann_config = validate_dann_config({'experiment': 'e0bn', 'supervised_batch': 8})
    trainer._e0bn_supervised_batch = 8
    batch = {
        'img': torch.arange(16 * 3 * 8 * 8, dtype=torch.uint8).reshape(16, 3, 8, 8),
        'batch_idx': torch.arange(16),
        'cls': torch.ones(16, 1),
        'bboxes': torch.full((16, 4), 0.5),
    }

    result = trainer.preprocess_batch(batch)

    assert result['img'].shape == (8, 3, 8, 8)
    assert result['target_img'].shape == (8, 3, 8, 8)
    assert torch.equal(result['batch_idx'], torch.arange(8))
    assert result['cls'].shape == (8, 1)
    assert result['bboxes'].shape == (8, 4)
    assert result['dann_enabled'] is False
    assert result['dann_lambda'] == pytest.approx(0.0)


def test_e0bn_doubles_only_the_training_loader_batch(monkeypatch) -> None:
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    class Loader:
        def __init__(self) -> None:
            self.batch_sampler = SimpleNamespace(sampler=SimpleNamespace(drop_last=False))
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    observed = []
    loaders = []

    def make_loader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        observed.append((batch_size, mode))
        loader = Loader()
        loaders.append(loader)
        return loader

    monkeypatch.setattr(
        DetectionTrainer,
        'get_dataloader',
        make_loader,
    )
    trainer = object.__new__(DANNDetectionTrainer)
    trainer.dann_config = validate_dann_config({'experiment': 'e0bn', 'supervised_batch': 8})

    train_loader = trainer.get_dataloader('source', batch_size=8, mode='train')
    val_loader = trainer.get_dataloader('source', batch_size=16, mode='val')
    assert observed == [(16, 'train'), (16, 'val')]
    assert train_loader.batch_sampler.sampler.drop_last is True
    assert train_loader.reset_calls == 1
    assert val_loader.batch_sampler.sampler.drop_last is False
    assert val_loader.reset_calls == 0


def test_target_loader_uses_target_dataset_metadata_and_restores_source() -> None:
    trainer = object.__new__(DANNDetectionTrainer)
    source_data = {'nc': 2, 'names': {0: 'source'}}
    target_data = {'nc': 1, 'names': {0: 'target'}, 'train': 'target/images'}
    trainer.data = source_data
    observed = []
    trainer.get_dataloader = lambda *args, **kwargs: observed.append(trainer.data) or 'target-loader'

    loader = trainer._build_target_loader(target_data, target_batch=2)

    assert loader == 'target-loader'
    assert observed == [target_data]
    assert trainer.data is source_data


def test_target_batch_requires_exact_ddp_division() -> None:
    assert DANNDetectionTrainer._target_batch_per_rank(8, world_size=2) == 4
    with pytest.raises(ValueError, match='divisible'):
        DANNDetectionTrainer._target_batch_per_rank(7, world_size=2)
    with pytest.raises(ValueError, match='at least'):
        DANNDetectionTrainer._target_batch_per_rank(2, world_size=4)


def test_target_iterator_resets_infinite_loader_at_epoch_boundary() -> None:
    class ResettableLoader:
        sampler = None

        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

        def __iter__(self):
            return iter([{'img': torch.zeros(1, 3, 8, 8)}])

    trainer = object.__new__(DANNDetectionTrainer)
    trainer.target_loader = ResettableLoader()
    trainer.epoch = 3

    trainer._reset_target_iterator()

    assert trainer.target_loader.reset_calls == 1
    assert trainer._target_epoch == 3


def test_select_trainer_keeps_e0_on_original_trainer() -> None:
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    assert select_trainer('e0') is DetectionTrainer
    assert select_trainer('e0bn') is DANNDetectionTrainer
    assert select_trainer('e05') is DANNDetectionTrainer
    assert select_trainer('e1') is DANNDetectionTrainer


def test_load_experiment_config_separates_upstream_and_dann_keys(tmp_path: Path) -> None:
    config_path = tmp_path / 'experiment.yaml'
    config_path.write_text(
        'experiment: e1\n'
        'source_data: source.yaml\n'
        'target_data: target.yaml\n'
        'model: cfg/models/v13/yolov13.yaml\n'
        'weights: weights.pt\n'
        'train:\n  epochs: 3\n  batch: 2\n'
        'dann:\n  domain_hidden: 16\n',
        encoding='utf-8',
    )

    overrides, dann_config = load_experiment_config(config_path)

    assert overrides['data'] == 'source.yaml'
    assert overrides['model'] == 'cfg/models/v13/yolov13.yaml'
    assert overrides['pretrained'] == 'weights.pt'
    assert 'dann' not in overrides
    assert dann_config['target_data'] == 'target.yaml'
    assert dann_config['domain_hidden'] == 16


def test_load_experiment_config_rejects_missing_source(tmp_path: Path) -> None:
    config_path = tmp_path / 'experiment.yaml'
    config_path.write_text('experiment: e0\nmodel: cfg/models/v13/yolov13.yaml\n', encoding='utf-8')

    with pytest.raises(ValueError, match='source_data'):
        load_experiment_config(config_path)


def test_build_trainer_routes_e0_without_dann_arguments(tmp_path: Path, monkeypatch) -> None:
    calls = []

    class FakeDetectionTrainer:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(train_yolov13_dann, 'DetectionTrainer', FakeDetectionTrainer)
    config_path = tmp_path / 'e0.yaml'
    config_path.write_text(
        'experiment: e0\nsource_data: source.yaml\nmodel: cfg/models/v13/yolov13.yaml\n', encoding='utf-8'
    )

    trainer = build_trainer(config_path)

    assert isinstance(trainer, FakeDetectionTrainer)
    assert set(calls[0]) == {'overrides'}


def test_trainer_recovers_dann_config_in_generated_ddp_child(monkeypatch) -> None:
    payload = {'target_data': 'target.yaml', 'experiment': 'e1', 'domain_hidden': 32}
    monkeypatch.setenv('ULTRALYTICS_DANN_CONFIG', json.dumps(payload))
    monkeypatch.setattr(
        'ultralytics.models.yolo.detect.train.DetectionTrainer.__init__', lambda self, *args, **kwargs: None
    )

    trainer = DANNDetectionTrainer(dann_config=None)

    assert trainer.dann_config['target_data'] == 'target.yaml'
    assert trainer.dann_config['domain_hidden'] == 32


def test_trainer_exposes_dann_config_only_while_parent_train_runs(monkeypatch) -> None:
    trainer = object.__new__(DANNDetectionTrainer)
    trainer.dann_config = validate_dann_config({'target_data': 'target.yaml', 'experiment': 'e1'})
    monkeypatch.delenv('ULTRALYTICS_DANN_CONFIG', raising=False)
    observed = []
    monkeypatch.setattr(
        'ultralytics.models.yolo.detect.train.DetectionTrainer.train',
        lambda self: observed.append(json.loads(os.environ['ULTRALYTICS_DANN_CONFIG'])),
    )

    trainer.train()

    assert observed[0]['target_data'] == 'target.yaml'
    assert 'ULTRALYTICS_DANN_CONFIG' not in os.environ


@pytest.mark.parametrize(('enabled', 'expect_domain_gradient'), [(False, False), (True, True)])
def test_real_detection_loss_backward_smoke(enabled, expect_domain_gradient) -> None:
    torch.manual_seed(4)
    model = make_detection_model(domain_dropout=0.0).train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    batch = {
        'img': torch.randn(1, 3, 64, 64),
        'target_img': torch.randn(1, 3, 64, 64) + 1.0,
        'batch_idx': torch.tensor([0]),
        'cls': torch.tensor([[1.0]]),
        'bboxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        'dann_enabled': enabled,
        'dann_lambda': 0.5,
    }

    loss, items = model.loss(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(items).all()
    assert model.model[0].conv.weight.grad is not None
    assert torch.isfinite(model.model[0].conv.weight.grad).all()
    classifier_gradient = model.domain_classifier.classifier[-1].weight.grad
    assert (classifier_gradient is not None) is expect_domain_gradient


def test_dann_training_path_does_not_change_inference_contract() -> None:
    torch.manual_seed(5)
    model = make_detection_model(domain_dropout=0.0).eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        before = model(image)

    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    model.criterion = RecordingCriterion()
    model.loss(
        {
            'img': torch.randn(1, 3, 64, 64),
            'target_img': torch.randn(1, 3, 64, 64),
            'dann_enabled': True,
            'dann_lambda': 0.5,
        }
    )
    model.eval()
    with torch.no_grad():
        after = model(image)

    assert_nested_close(after, before)


def test_dann_model_preserves_augmented_inference() -> None:
    torch.manual_seed(6)
    baseline = DetectionModel('cfg/models/v13/yolov13.yaml', nc=2, verbose=False).eval()
    dann = make_detection_model(domain_dropout=0.0).eval()
    dann.model.load_state_dict(baseline.model.state_dict())
    image = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        expected = baseline(image, augment=True)
        actual = dann(image, augment=True)

    assert_nested_close(actual, expected)


def test_onnx_export_excludes_training_only_domain_branch(tmp_path: Path) -> None:
    onnx = pytest.importorskip('onnx', reason='onnx is required to inspect the exported graph contract')
    baseline = DetectionModel('cfg/models/v13/yolov13.yaml', nc=2, verbose=False).eval()
    dann = make_detection_model(domain_dropout=0.0).eval()
    graphs = []

    for name, model in (('baseline', baseline), ('dann', dann)):
        head = model.model[-1]
        head.export = True
        head.format = 'onnx'
        head.dynamic = False
        path = tmp_path / f'{name}.onnx'
        torch.onnx.export(
            model,
            torch.randn(1, 3, 64, 64),
            path,
            opset_version=17,
            input_names=['images'],
            output_names=['output0'],
        )
        graphs.append(onnx.load(path))

    baseline_graph, dann_graph = (model.graph for model in graphs)
    assert len(dann_graph.input) == 1
    assert len(dann_graph.output) == len(baseline_graph.output)
    assert [dim.dim_value for dim in dann_graph.output[0].type.tensor_type.shape.dim] == [
        dim.dim_value for dim in baseline_graph.output[0].type.tensor_type.shape.dim
    ]
    initializer_names = [initializer.name.lower() for initializer in dann_graph.initializer]
    node_names = [node.name.lower() for node in dann_graph.node]
    assert not any('domain_classifier' in name for name in initializer_names)
    assert not any('domain_classifier' in name or 'gradientreversal' in name for name in node_names)
