# Target-Style Source Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, lightweight Source-to-Target style augmentation to the existing P2 DANN trainer and provide a reproducible seed42 A1 configuration.

**Architecture:** Keep the transform inside `DANNDetectionTrainer`, after Source and Target tensors have independently passed their existing loaders and preprocessing. A small pure tensor helper transfers per-image RGB moments from the current Target batch to Source, then applies probability and blend controls; the model, loss, validation, and inference paths remain unchanged.

**Tech Stack:** Python 3, PyTorch, Ultralytics YOLOv13 trainer, pytest, YAML

## Global Constraints

- Preserve P2 layer 1, FP32, `lambda_max=0.02`, Source batch 8, Target batch 8, `target_supervised=true`, and STAL disabled.
- Apply style augmentation only to Source training tensors; Target, validation, inference, boxes, and classes remain unchanged.
- Use `source_style_prob=0.50` and `source_style_alpha=0.30` for A1.
- Keep the feature disabled by default and add no dependency.
- Preserve user changes already present in `models/yolo/detect/train_dann.py` and `tests/test_yolov13_dann.py`.

---

### Task 1: Add the Source-to-Target Moment Transform

**Files:**
- Modify: `models/yolo/detect/train_dann.py:17-116,225-286`
- Test: `tests/test_yolov13_dann.py:365-575`

**Interfaces:**
- Consumes: preprocessed `source_images: torch.Tensor[Bs,C,H,W]`, `target_images: torch.Tensor[Bt,C,H,W]`, and validated probability/alpha scalars.
- Produces: `DANNDetectionTrainer._target_style_source(source_images, target_images, probability, alpha) -> torch.Tensor` and config keys `source_style_aug`, `source_style_prob`, `source_style_alpha`.

- [x] **Step 1: Write failing configuration tests**

Add assertions that defaults are disabled with `0.50/0.30`, valid boundary values are accepted, and booleans, values below zero, and values above one are rejected:

```python
config = validate_dann_config({'target_data': 'target.yaml', 'experiment': 'e1'})
assert config['source_style_aug'] is False
assert config['source_style_prob'] == pytest.approx(0.50)
assert config['source_style_alpha'] == pytest.approx(0.30)

for key in ('source_style_prob', 'source_style_alpha'):
    for value in (-0.01, 1.01, True):
        with pytest.raises((TypeError, ValueError), match=key):
            validate_dann_config({'target_data': 'target.yaml', 'experiment': 'e1', key: value})
```

- [x] **Step 2: Write failing transform tests**

Cover disabled/no-op behavior, deterministic full-probability transfer, batch cycling, range preservation, and Target immutability:

```python
source = torch.tensor([[[[0.0, 0.2], [0.4, 0.6]]]]).repeat(2, 3, 1, 1)
target = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]]).repeat(1, 3, 1, 1)
original_target = target.clone()
actual = DANNDetectionTrainer._target_style_source(source, target, probability=1.0, alpha=1.0)
assert actual.shape == source.shape
assert torch.allclose(actual.mean((-2, -1)), target.mean((-2, -1)).expand(2, -1), atol=1e-6)
assert torch.allclose(actual.std((-2, -1), unbiased=False), target.std((-2, -1), unbiased=False).expand(2, -1), atol=1e-6)
assert actual.min() >= 0 and actual.max() <= 1
assert torch.equal(target, original_target)
```

Add a trainer integration assertion showing disabled augmentation preserves `batch['img']` and enabled augmentation changes Source while `batch['target_img']` remains equal to the preprocessed Target tensor.

- [x] **Step 3: Run the focused tests and verify failure**

Run:

```powershell
python -m pytest tests/test_yolov13_dann.py -k "style or validate_dann_config" -q
```

Expected: new tests fail because the config keys and `_target_style_source` do not exist.

- [x] **Step 4: Implement validated opt-in configuration**

Extend `DEFAULT_DANN_CONFIG`:

```python
'source_style_aug': False,
'source_style_prob': 0.50,
'source_style_alpha': 0.30,
```

Require `source_style_aug` to be a real boolean. Parse the other two values through `_finite_number`, validate the closed interval `[0, 1]`, and write the normalized floats back to the merged config.

- [x] **Step 5: Implement the minimum tensor helper**

Add a static method to `DANNDetectionTrainer`:

```python
@staticmethod
def _target_style_source(source, target, probability, alpha):
    if probability == 0.0 or alpha == 0.0:
        return source
    dims = (-2, -1)
    source_mean = source.mean(dims, keepdim=True)
    source_std = source.std(dims, keepdim=True, unbiased=False).clamp_min(torch.finfo(source.dtype).eps)
    target = target.detach()
    target_mean = target.mean(dims, keepdim=True)
    target_std = target.std(dims, keepdim=True, unbiased=False)
    indices = torch.arange(source.shape[0], device=source.device) % target.shape[0]
    transferred = (source - source_mean) / source_std
    transferred = transferred * target_std[indices] + target_mean[indices]
    blended = source.lerp(transferred, alpha).clamp_(0.0, 1.0)
    selected = torch.rand((source.shape[0], 1, 1, 1), device=source.device) < probability
    return torch.where(selected, blended, source)
```

After Target resize and before assigning `batch['target_img']`, call the helper only when `source_style_aug` is true and replace `batch['img']` with the returned Source tensor. Do not modify the separate `target_images` tensor.

- [x] **Step 6: Run focused and complete DANN tests**

Run:

```powershell
python -m pytest tests/test_yolov13_dann.py -q
```

Expected: all DANN tests pass, including the new style tests.

---

### Task 2: Add and Verify the A1 Seed42 Experiment Configuration

**Files:**
- Create: `configs/dann/ic_p2_target_style_a1_seed42.yaml`
- Modify: `tests/test_yolov13_dann.py`

**Interfaces:**
- Consumes: `scripts/train_yolov13_dann.py::load_experiment_config()` and the new DANN config keys from Task 1.
- Produces: one reproducible A1 config that differs from the P2 baseline only by the Source style controls and output naming.

- [x] **Step 1: Write the A1 config contract test**

Load the repository config through `load_experiment_config` and assert:

```python
overrides, dann = load_experiment_config('configs/dann/ic_p2_target_style_a1_seed42.yaml')
assert overrides['batch'] == 8
assert overrides['amp'] is False
assert overrides['seed'] == 42
assert overrides['deterministic'] is True
assert overrides['stal'] is False
assert dann['target_batch'] == 8
assert dann['target_supervised'] is True
assert dann['domain_feature_layers'] == (1,)
assert dann['lambda_max'] == pytest.approx(0.02)
assert dann['source_style_aug'] is True
assert dann['source_style_prob'] == pytest.approx(0.50)
assert dann['source_style_alpha'] == pytest.approx(0.30)
```

- [x] **Step 2: Add the A1 YAML**

Create the config by copying the historical P2 seed42 paths and training fields, then changing only the run name and Source style controls:

```yaml
experiment: e1
source_data: F:/dann_compare_20260811/experiment/source.yaml
target_data: F:/dann_compare_20260811/experiment/target.yaml
model: F:/dann_compare_20260811/ultralytics/cfg/models/v13/yolov13.yaml
weights: F:/dann_compare_20260811/yolov13n.pt

train:
  epochs: 10
  imgsz: 640
  batch: 8
  device: 0
  workers: 4
  amp: false
  seed: 42
  deterministic: true
  stal: false
  project: F:/dann_compare_20260811/runs
  exist_ok: true
  plots: false
  verbose: true
  save: true
  val: false
  name: p2_dann_target_style_a1_seed42

dann:
  target_batch: 8
  target_supervised: true
  p5_layer: 1
  domain_hidden: 256
  domain_dropout: 0.1
  domain_loss_gain: 1.0
  lambda_max: 0.02
  gamma: 10.0
  warmup_fraction: 0.0
  source_style_aug: true
  source_style_prob: 0.50
  source_style_alpha: 0.30
```

- [x] **Step 3: Run config and regression verification**

Run:

```powershell
python -m pytest tests/test_yolov13_dann.py tests/test_stal.py -q
python scripts/check_yolov13_dann.py
git diff --check
```

Expected: both test modules and backward smoke script pass; `git diff --check` reports no whitespace errors. Remote seed42 training is not part of this code verification.

- [x] **Step 4: Review the final diff**

Run:

```powershell
git diff -- models/yolo/detect/train_dann.py tests/test_yolov13_dann.py configs/dann/ic_p2_target_style_a1_seed42.yaml
```

Expected: no Detect/criterion/export changes, no Target tensor rewrite, and no unrelated user changes removed.

## Plan Self-Review

- Spec coverage: Source-only batch-stat transfer, `0.50/0.30` controls, disabled defaults, numerical boundaries, Target immutability, P2 seed42 config, and regression checks are covered.
- Placeholder scan: no deferred implementation markers or unspecified error handling remain.
- Type consistency: configuration names and `_target_style_source(source, target, probability, alpha)` match across tasks and tests.
