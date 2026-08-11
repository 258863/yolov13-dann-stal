from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from ultralytics.models.yolo.detect.dann import DANNDetectionModel


class RecordingCriterion:
    def __init__(self, criterion) -> None:
        self.criterion = criterion
        self.batch_sizes = None

    def __call__(self, predictions, batch):
        self.batch_sizes = [feature.shape[0] for feature in predictions]
        return self.criterion(predictions, batch)


def make_batch(dann_enabled: bool) -> dict:
    return {
        'img': torch.randn(1, 3, 64, 64),
        'target_img': torch.randn(1, 3, 64, 64) + 1.0,
        'batch_idx': torch.tensor([0]),
        'cls': torch.tensor([[1.0]]),
        'bboxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        'dann_enabled': dann_enabled,
        'dann_lambda': 0.5,
    }


def output_shapes(output):
    if isinstance(output, torch.Tensor):
        return tuple(output.shape)
    return [output_shapes(item) for item in output]


def main() -> None:
    torch.manual_seed(0)
    model = DANNDetectionModel(
        'cfg/models/v13/yolov13.yaml', nc=2, verbose=False, domain_hidden=8, domain_dropout=0.0
    )
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model.criterion = RecordingCriterion(model.init_criterion())

    model.eval()
    with torch.no_grad():
        inference, p5 = model.forward_with_p5(torch.randn(2, 3, 64, 64))

    print(f'P5 layer: {model.model[31].i}; Detect layer: {model.model[32].i}; Detect inputs: {model.model[32].f}')
    print(f'P5 shape: {tuple(p5.shape)}')
    print(f'Inference output structure: {output_shapes(inference)}')

    for experiment, enabled in (('E0.5', False), ('E1', True)):
        model.train()
        model.zero_grad(set_to_none=True)
        loss, items = model.loss(make_batch(enabled))
        loss.backward()
        classifier_grad = model.domain_classifier.classifier[-1].weight.grad
        backbone_grad = model.model[0].conv.weight.grad
        print(
            f'{experiment}: source prediction batches={model.criterion.batch_sizes}, total={loss.item():.6f}, '
            f'domain={items[3].item():.6f}, domain_acc={items[4].item():.3f}, lambda={items[5].item():.3f}, '
            f'backbone_grad={backbone_grad is not None}, classifier_grad={classifier_grad is not None}'
        )


if __name__ == '__main__':
    main()
