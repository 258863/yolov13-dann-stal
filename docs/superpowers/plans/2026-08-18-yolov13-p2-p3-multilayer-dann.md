# YOLOv13 P2+P3 Multi-Layer DANN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible two-layer P2+P3 DANN training path and run a fair three-seed comparison against the existing single-layer experiments.

**Architecture:** Generalize the training-only domain feature selection from one layer to an ordered tuple. Capture both requested features during the existing single detector forward, run one independent GRL/classifier per feature, and average the two cross-entropy losses before applying the existing gain and Source-batch scaling.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics YOLOv13, pytest, PowerShell, RTX 3090 Ti.

## Global Constraints

- Multi-layer experiment uses exactly layers 1 and 23.
- Nano channels must resolve to P2=32 and P3=64.
- Keep lambda_max=0.02, gamma=10.0, no warmup, 10 epochs, seeds 42/43/44, Source batch 8, Target batch 8, AMP disabled, and deterministic mode enabled.
- Average the two domain losses; do not sum them.
- Preserve legacy p5_layer configs and all single-layer behavior.
- Preserve inference, Detect, model YAML, exporter, and ONNX contracts.
- Preserve unrelated dirty-worktree changes.
- Save experiment artifacts under D:\work\dz\dann_compare_20260818\p2_p3_dann_lambda002.

---

### Task 1: Lock multi-layer behavior with failing tests

**Files:**
- Modify: `tests/test_yolov13_dann.py`

**Interfaces:**
- Consumes: `DANNDetectionModel(..., domain_feature_layers=(1, 23))`.
- Produces: regression coverage for channel inference, feature capture, averaged loss, gradients, configuration validation, and ONNX exclusion.

- [ ] **Step 1: Add construction and channel tests**

```python
def test_dann_model_supports_p2_p3_feature_layers() -> None:
    model = make_detection_model(domain_feature_layers=(1, 23))

    assert model.domain_feature_layers == (1, 23)
    assert [classifier.in_channels for classifier in model.domain_classifiers] == [32, 64]
```

- [ ] **Step 2: Add mixed-domain backward coverage**

```python
def test_p2_p3_dann_loss_backward_reaches_both_classifiers() -> None:
    model = make_detection_model(domain_feature_layers=(1, 23), domain_dropout=0.0).train()
    model.criterion = RecordingCriterion()
    loss, items = model.loss({
        'img': torch.randn(1, 3, 64, 64),
        'target_img': torch.randn(1, 3, 64, 64) + 1.0,
        'dann_enabled': True,
        'dann_lambda': 0.02,
    })
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(items).all()
    assert all(c.classifier[-1].weight.grad is not None for c in model.domain_classifiers)
    assert model.model[0].conv.weight.grad is not None
```

- [ ] **Step 3: Add an exact averaged-loss test**

Patch the classifiers with deterministic logits, compute each branch cross-entropy separately, and assert the reported domain loss equals `torch.stack(branch_losses).mean()`, not their sum.

- [ ] **Step 4: Add config validation tests**

```python
def test_validate_dann_config_accepts_p2_p3_layers() -> None:
    config = validate_dann_config({
        'target_data': 'target.yaml',
        'experiment': 'e1',
        'domain_feature_layers': [1, 23],
    })
    assert config['domain_feature_layers'] == (1, 23)
```

Reject empty lists, duplicates, booleans, unverified layers, and simultaneous conflicting `p5_layer`/`domain_feature_layers` inputs.

- [ ] **Step 5: Run the focused tests and confirm failure**

Run:

```powershell
$env:PYTHONPATH='D:\dip'
python -m pytest tests/test_yolov13_dann.py -q -k 'multilayer or p2_p3'
```

Expected: failures because `domain_feature_layers` and `domain_classifiers` do not exist.

### Task 2: Implement multi-layer model support

**Files:**
- Modify: `models/yolo/detect/dann.py`
- Test: `tests/test_yolov13_dann.py`

**Interfaces:**
- Consumes: `domain_feature_layers: tuple[int, ...] | None` and legacy `p5_layer: int`.
- Produces: `domain_feature_layers`, `domain_classifiers`, `forward_with_domain_features(images)`, and an averaged domain objective.

- [ ] **Step 1: Normalize selected layers**

```python
layers = (p5_layer,) if domain_feature_layers is None else tuple(domain_feature_layers)
if not layers or len(set(layers)) != len(layers):
    raise ValueError('domain_feature_layers must contain unique verified layers')
if any(layer not in self.VERIFIED_DOMAIN_FEATURE_LAYERS for layer in layers):
    raise ValueError('domain_feature_layers contains an unverified YOLOv13 feature layer')
self.domain_feature_layers = layers
self.p5_layer = layers[0] if len(layers) == 1 else None
```

Only reject a legacy/new-field conflict when the caller explicitly supplies a non-default `p5_layer` that is not the first selected layer.

- [ ] **Step 2: Reuse one channel inference helper**

```python
def _domain_feature_channels(self, layer: int) -> int:
    if layer in self.BACKBONE_DOMAIN_FEATURE_LAYERS:
        return self.model[layer].conv.out_channels
    detect = self.model[-1]
    return detect.cv2[self.DOMAIN_FEATURE_BRANCHES[layer]][0].conv.in_channels
```

Create `nn.ModuleList` instances for GRLs and classifiers in layer order. Keep `grl` and `domain_classifier` aliases only for single-layer backward compatibility.

- [ ] **Step 3: Capture every selected feature in one forward**

```python
def forward_with_domain_features(self, images: torch.Tensor):
    selected = set(self.domain_feature_layers)
    features = {}
    # existing model loop
    if module.i in selected:
        features[module.i] = output
    missing = selected.difference(features)
    if missing:
        raise RuntimeError(f'domain feature layers were not executed: {sorted(missing)}')
    return output, tuple(features[layer] for layer in self.domain_feature_layers)
```

Keep `forward_with_p5` as a legacy wrapper that returns the only feature and raises when more than one layer is configured.

- [ ] **Step 4: Average branch objectives**

For every selected feature, construct logits using its matching GRL/classifier, compute cross-entropy and accuracy, then use:

```python
domain_loss = torch.stack(domain_losses).mean()
domain_accuracy = torch.stack(domain_accuracies).mean()
total_loss = detection_loss + self.domain_loss_gain * domain_loss * source_batch_size
```

Keep the existing three appended loss items so Ultralytics logging remains compatible; the reported domain loss and accuracy are branch means.

- [ ] **Step 5: Run focused and complete tests**

Run:

```powershell
$env:PYTHONPATH='D:\dip'
python -m pytest tests/test_yolov13_dann.py -q -k 'multilayer or p2_p3'
python -m pytest tests/test_yolov13_dann.py -q
```

Expected: all tests pass, including ONNX exclusion.

### Task 3: Add trainer configuration compatibility

**Files:**
- Modify: `models/yolo/detect/train_dann.py`
- Modify: `tests/test_yolov13_dann.py`

**Interfaces:**
- Consumes: optional YAML key `domain_feature_layers`.
- Produces: normalized tuple passed to `DANNDetectionModel` while preserving legacy `p5_layer` behavior.

- [ ] **Step 1: Add the optional default**

```python
'domain_feature_layers': None,
```

- [ ] **Step 2: Validate and normalize the field**

When absent, set it to `(p5_layer,)`. When present, require a non-empty list/tuple of unique non-boolean integers from `VERIFIED_DOMAIN_FEATURE_LAYERS`, convert it to a tuple, and reject a conflicting explicitly supplied `p5_layer`.

- [ ] **Step 3: Pass the normalized tuple from trainer to model**

```python
domain_feature_layers=self.dann_config['domain_feature_layers'],
```

- [ ] **Step 4: Run configuration and complete regression tests**

Run:

```powershell
$env:PYTHONPATH='D:\dip'
python -m pytest tests/test_yolov13_dann.py -q -k 'config and (layer or feature)'
python -m pytest tests/test_yolov13_dann.py -q
```

Expected: all tests pass and legacy configs still resolve to one layer.

### Task 4: Execute the controlled RTX 3090 Ti experiment

**Files:**
- Create: `D:\work\dz\dann_compare_20260818\p2_p3_dann_lambda002\p2_p3_dann_seed42.yaml`
- Create: matching seed43/44 configs, evaluation scripts, class-aware analysis, resumable `run_all.ps1`, and final summary.

**Interfaces:**
- Consumes: the verified training code, existing Source/Target YAML files, Target holdout, and P2 evaluation workflow.
- Produces: three checkpoints, validation JSON files, fixed-threshold CSV files, class-aware misses, and `three_seed_summary.json`.

- [ ] **Step 1: Create configs with only the intended architectural change**

```yaml
dann:
  target_batch: 8
  target_supervised: true
  domain_feature_layers: [1, 23]
  domain_hidden: 256
  domain_dropout: 0.1
  domain_loss_gain: 1.0
  lambda_max: 0.02
  gamma: 10.0
  warmup_fraction: 0.0
```

- [ ] **Step 2: Run local tests and remote smoke test**

The remote smoke test must assert:

```python
assert model.domain_feature_layers == (1, 23)
assert [c.in_channels for c in model.domain_classifiers] == [32, 64]
```

- [ ] **Step 3: Run seeds sequentially using a removable scheduled task**

Skip a seed only when both its checkpoint and zero exit-code file exist. Run training then source/target validation and threshold analysis for each seed. Remove the scheduled task immediately after completion.

- [ ] **Step 4: Compare all five groups**

Generate a summary containing no-DANN, P5, P3, P2, and P2+P3 at confidence 0.40/0.55, Source/Target mAP50-95, HD/LX misses, image30 status, empty-label overkill, and final mean domain metrics.

- [ ] **Step 5: Verify completion**

Require every training/evaluation/analysis exit code to equal zero, all local artifacts to exist under `D:\work\dz`, and the remote scheduled task to be deleted.
