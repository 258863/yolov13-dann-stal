from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


ANCHORS = torch.tensor([[4.0, 4.0], [12.0, 4.0], [4.0, 12.0], [12.0, 12.0]])


def legacy_candidates_in_gts(xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
    deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2)
    return deltas.view(bs, n_boxes, n_anchors, -1).amin(3).gt_(eps)


def make_assigner(stal: bool) -> TaskAlignedAssigner:
    return TaskAlignedAssigner(topk=4, num_classes=1, stride=[8, 16, 32], stal=stal)


class RecordingTaskAlignedAssigner(TaskAlignedAssigner):
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        self.metric_gt_bboxes = gt_bboxes.clone()
        return super().get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)


def test_stal_adds_four_candidates_for_four_pixel_gt() -> None:
    gt_bboxes = torch.tensor([[[6.0, 6.0, 10.0, 10.0]]])
    mask_gt = torch.ones(1, 1, 1)

    tal_mask = make_assigner(False).select_candidates_in_gts(ANCHORS, gt_bboxes, mask_gt)
    stal_mask = make_assigner(True).select_candidates_in_gts(ANCHORS, gt_bboxes, mask_gt)

    assert tal_mask.sum().item() == 0
    assert stal_mask.sum().item() == 4


def test_stal_expands_only_dimensions_below_minimum_stride() -> None:
    gt_bboxes = torch.tensor([[[6.0, 2.0, 10.0, 14.0]]])  # width 4, height 12
    anchors = torch.tensor([[4.0, 0.0], [4.0, 4.0], [12.0, 12.0], [4.0, 16.0], [20.0, 4.0]])

    candidates = make_assigner(True).select_candidates_in_gts(anchors, gt_bboxes, torch.ones(1, 1, 1))

    assert candidates.tolist() == [[[False, True, True, False, False]]]


def test_stal_does_not_expand_dimension_equal_to_minimum_stride() -> None:
    gt_bboxes = torch.tensor([[[4.0, 6.0, 12.0, 10.0]]])  # width 8, height 4

    candidates = make_assigner(True).select_candidates_in_gts(ANCHORS, gt_bboxes, torch.ones(1, 1, 1))

    assert not candidates.any()


def test_stal_does_not_expand_padded_gt() -> None:
    gt_bboxes = torch.tensor([[[6.0, 6.0, 10.0, 10.0]]])

    candidates = make_assigner(True).select_candidates_in_gts(ANCHORS, gt_bboxes, torch.zeros(1, 1, 1))

    assert not candidates.any()


def test_stal_disabled_matches_legacy_candidate_selection() -> None:
    torch.manual_seed(7)
    anchors = torch.rand(20, 2) * 32
    xy1 = torch.rand(2, 3, 2) * 16
    gt_bboxes = torch.cat((xy1, xy1 + torch.rand(2, 3, 2) * 16 + 0.1), dim=-1)

    actual = make_assigner(False).select_candidates_in_gts(anchors, gt_bboxes, torch.ones(2, 3, 1))

    assert torch.equal(actual, legacy_candidates_in_gts(anchors, gt_bboxes))


def test_stal_does_not_modify_original_gt_and_regresses_original_box() -> None:
    gt_bboxes = torch.tensor([[[6.0, 6.0, 10.0, 10.0]]])
    original = gt_bboxes.clone()
    assigner = RecordingTaskAlignedAssigner(topk=4, num_classes=1, stride=[8, 16, 32], stal=True)
    pd_bboxes = gt_bboxes.expand(1, len(ANCHORS), 4).clone()

    _, target_bboxes, _, fg_mask, _ = assigner(
        pd_scores=torch.full((1, len(ANCHORS), 1), 0.9),
        pd_bboxes=pd_bboxes,
        anc_points=ANCHORS,
        gt_labels=torch.zeros(1, 1, 1),
        gt_bboxes=gt_bboxes,
        mask_gt=torch.ones(1, 1, 1),
    )

    assert fg_mask.all()
    assert torch.equal(gt_bboxes, original)
    assert torch.equal(assigner.metric_gt_bboxes, original)
    assert torch.equal(target_bboxes[fg_mask], original[0, 0].expand(len(ANCHORS), 4))


class FakeDetectHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nc = 2
        self.reg_max = 16
        self.stride = torch.tensor([8.0, 16.0, 32.0])


class FakeDetectionModel(nn.Module):
    def __init__(self, stal=None) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.model = nn.ModuleList([FakeDetectHead()])
        args = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
        if stal is not None:
            args["stal"] = stal
        self.args = SimpleNamespace(**args)


def test_stal_is_disabled_by_default_and_accepts_config_override() -> None:
    assert DEFAULT_CFG_DICT["stal"] is False
    assert get_cfg(overrides={"stal": True}).stal is True


@pytest.mark.parametrize(("configured", "expected"), [(None, False), (False, False), (True, True)])
def test_detection_loss_passes_stal_and_strides_to_assigner(configured, expected) -> None:
    criterion = v8DetectionLoss(FakeDetectionModel(configured))

    assert criterion.assigner.stal is expected
    assert criterion.assigner.stride == [8.0, 16.0, 32.0]


@pytest.mark.parametrize("stal", [False, True])
def test_yolov13_detection_loss_backward_smoke(stal: bool) -> None:
    torch.manual_seed(11)
    model = DetectionModel("cfg/models/v13/yolov13.yaml", nc=2, verbose=False).train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, stal=stal)
    images = torch.randn(1, 3, 64, 64)
    predictions = model(images)
    criterion = v8DetectionLoss(model)
    batch = {
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.125, 0.125]]),
    }

    loss, loss_items = criterion(predictions, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(loss_items).all()
    assert model.model[0].conv.weight.grad is not None
