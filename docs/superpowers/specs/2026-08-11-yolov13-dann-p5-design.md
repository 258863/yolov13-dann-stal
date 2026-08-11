# YOLOv13 P5 DANN Design

Date: 2026-08-11

## Objective

Add unsupervised domain-adversarial training to the bundled Ultralytics 8.3.63 YOLOv13 implementation without changing the inference graph, Detect head, model YAML, global detection loss, class count, or exporter.

The first implementation attaches a training-only domain branch to layer 31 (P5). Source and target images are augmented independently, concatenated, and passed through the YOLO backbone and neck once. Detection loss is computed only from the source slice. Domain loss is computed from all P5 features through a gradient reversal layer (GRL).

## Repository Boundary

The imported source did not include Git metadata or the normal upstream repository wrapper. `D:/dip/ultralytics` is therefore the repository root and also the Python package directory. Project-only artifacts such as `scripts/`, `configs/`, `tests/`, and `docs/` live under this repository root. Tests and scripts must add the repository parent (`D:/dip`) to `sys.path` when run directly so `import ultralytics` resolves to this checkout.

## Existing Contract

The checked-in `cfg/models/v13/yolov13.yaml` defines:

- P3: layer 23
- P4: layer 27
- P5: layer 31
- Detect: layer 32, consuming `[23, 27, 31]`

The following remain unchanged in the first version:

- `DSC3k2`
- `HyperACE`
- `FullPAD_Tunnel`
- `Detect`
- P3/P4/P5 main path
- `v8DetectionLoss`
- `cfg/models/v13/yolov13.yaml`
- prediction API
- exporter implementation
- ONNX input and output contract

## Architecture

### Training Graph

```text
source images ----+
                  +--> concatenate --> YOLOv13 layers 0..31 --> P3/P4/P5
target images ----+                                      |          |
                                                         |          +--> GRL --> GAP --> MLP --> 2 domains
                                                         +--> Detect --> source prediction slice only
```

The Detect head still receives the original `[P3, P4, P5]` features. The domain classifier is a registered training attribute of `DANNDetectionModel`; it is not added to the YOLO YAML and is not called by normal `forward()`.

### New Model Components

`models/yolo/detect/dann.py` contains:

- `GradientReversalFunction`: identity forward, `-lambda * gradient` backward.
- `GradientReversalLayer`: module wrapper that validates and applies the current coefficient.
- `DomainClassifier`: adaptive global average pooling, flatten, one hidden linear block, and two output logits.
- `DANNDetectionModel`: `DetectionModel` subclass that captures layer 31 during a single mixed-domain forward and owns the training-only domain branch.

The P5 input channel count is derived from the last Detect scale input (`Detect.cv2[-1][0].conv.in_channels`) after the base model has been constructed. This avoids hard-coding a channel count and supports YOLOv13 compound scales.

### P5 Forward Capture

`DANNDetectionModel.forward_with_p5(images)` performs one layer traversal equivalent to `BaseModel._predict_once`, retaining layer 31 and returning:

```python
predictions_all, p5_all
```

Normal `forward()` is inherited and remains unchanged. No persistent forward hook is used, avoiding mutable captured state during DDP, export, validation, or concurrent inference.

The custom traversal is covered by a parity test: in evaluation mode its prediction output must match the inherited forward output for the same tensor.

## Data Flow

### Source Loader

The source loader is the normal Ultralytics detection loader and supplies labels. Its existing augmentations remain available.

### Target Loader

The target loader is a separate detection-style loader configured by `target_data`. Target labels are not consumed. A target image folder without labels is valid; any empty-label warnings are informational because target annotations never enter detection loss.

The trainer maintains a target iterator and recreates it after `StopIteration`, so target data cycles until the source epoch finishes.

### Independent Augmentation

Source and target batches are produced by separate datasets/loaders. Mosaic or MixUp can occur within a domain according to that loader's settings, but no transform receives samples from both domains. The two preprocessed image tensors must have identical non-batch shapes before concatenation.

### Batch Rules

- Source and target batch sizes default to equal values.
- A smaller final target batch is allowed; domain labels use actual `Bs` and `Bt`.
- Source and target image height/width, dtype, and device must match.
- The effective forward batch is `Bs + Bt`, so memory checks and smoke tests report both sizes.
- DDP target sampling uses the same rank/world-size rules as the source loader.

## Trainer Integration

`models/yolo/detect/train_dann.py` contains `DANNDetectionTrainer`, derived from `DetectionTrainer`.

The trainer does not copy or replace `BaseTrainer._do_train`. It overrides narrow extension points:

- `get_model()`: construct `DANNDetectionModel`.
- `_setup_train()`: call the base implementation, then construct the target loader and iterator.
- `preprocess_batch()`: preprocess source normally, fetch and preprocess one target batch independently, and attach only `target_img` plus DANN progress metadata to the source batch.
- `label_loss_items()` and `progress_string()`: add domain loss, domain accuracy, and GRL lambda while preserving box/cls/dfl ordering.

The standard training loop still calls `self.model(batch)`. `DANNDetectionModel.loss()` recognizes the additional target image and executes the DANN path. Validation batches do not contain `target_img`, so they use the inherited source-only detection behavior.

Custom DANN configuration is loaded by `scripts/train_yolov13_dann.py` and passed separately from Ultralytics overrides. Unknown DANN keys are not inserted into the upstream configuration validator.

## Loss Definition

For source batch size `Bs` and target batch size `Bt`:

```python
images_all = torch.cat((source_images, target_images), dim=0)
predictions_all, p5_all = model.forward_with_p5(images_all)
source_predictions = [prediction[:Bs] for prediction in predictions_all]
detection_loss, detection_items = original_criterion(source_predictions, source_batch)
```

Target predictions are discarded before `v8DetectionLoss`. The target batch is never converted into an empty detection batch.

For E1:

```python
domain_labels = torch.cat(
    (
        torch.zeros(Bs, dtype=torch.long, device=p5_all.device),
        torch.ones(Bt, dtype=torch.long, device=p5_all.device),
    )
)
domain_logits = domain_classifier(grl(p5_all, current_lambda))
domain_loss_mean = cross_entropy(domain_logits, domain_labels)
domain_term = domain_loss_gain * domain_loss_mean * Bs
total_loss = detection_loss + domain_term
```

The multiplication by `Bs` matches Ultralytics detection-loss scaling. `domain_loss_gain` controls the scalar contribution of domain classification. GRL lambda independently controls only the reversed gradient reaching P5 and earlier layers.

Metrics recorded per batch/epoch:

- box loss
- cls loss
- dfl loss
- domain loss before `Bs` scaling
- domain accuracy
- GRL lambda

## Lambda Schedule

Training progress is clamped to `[0, 1]` and includes batch-level progress within the epoch:

```text
p = (epoch * steps_per_epoch + batch_index) / max(total_steps - 1, 1)
lambda(p) = lambda_max * (2 / (1 + exp(-gamma * p)) - 1)
```

Defaults:

- `lambda_max: 1.0`
- `gamma: 10.0`

The original YOLO optimizer and learning-rate schedule remain unchanged.

## Experiment Modes

The same entry point supports:

| Mode | Mixed-domain forward | Domain classifier/GRL | Detection loss |
|---|---:|---:|---|
| `e0` | No | No | Source only, normal trainer |
| `e05` | Yes | No | Source prediction slice only |
| `e1` | Yes | Yes | Source prediction slice plus domain term |

`e05` must still concatenate source and target images and execute one backbone/neck forward. It skips domain-classifier computation and returns zero-valued domain metrics. This isolates the effect of mixed BatchNorm statistics.

## Configuration

`configs/dann/ic_p5.yaml` contains a DANN section separate from normal YOLO arguments:

```yaml
experiment: e1
source_data: D:/datasets/ic_source/data.yaml
target_data: D:/datasets/ic_target/data.yaml
model: cfg/models/v13/yolov13.yaml
weights: null
train:
  epochs: 100
  imgsz: 640
  batch: 8
  device: 0
dann:
  target_batch: 8
  p5_layer: 31
  domain_hidden: 256
  domain_dropout: 0.1
  domain_loss_gain: 1.0
  lambda_max: 1.0
  gamma: 10.0
```

The script validates required paths, experiment name, positive batch sizes, finite non-negative gain, P5 layer identity, lambda range, and gamma.

## Checkpoint, Inference, and Export

Training checkpoints may contain domain-classifier parameters because they are part of `DANNDetectionModel`. Normal prediction calls inherited `forward()` and never invoke GRL or the domain classifier.

Export validation must be empirical rather than inferred. A small-model export test checks:

- one ONNX input
- the same number of outputs as the baseline export
- matching output shapes
- no initializer name belonging to `domain_classifier`
- no graph node associated with GRL/domain classification

If checkpoint loading into a plain `DetectionModel` reports domain-branch keys, the training entry point will provide an explicit inference-weight extraction helper that filters only `domain_classifier.*`; it will not modify the global checkpoint loader in the first version.

## Tests

New focused tests cover:

1. GRL identity forward and exact negative scaled gradient.
2. GRL coefficient type/range validation and schedule endpoints.
3. Domain classifier input/output shapes for multiple P5 channel counts.
4. `forward_with_p5()` prediction parity with normal forward.
5. P5 is captured from layer 31 and has the combined batch dimension.
6. Detection criterion receives only `Bs` predictions.
7. Target labels cannot enter detection loss.
8. Domain labels are `[0] * Bs + [1] * Bt`.
9. E0.5 performs one mixed forward with no domain gradient/loss.
10. E1 backward gives normal gradients to the classifier and reversed gradients to the P5 path.
11. Target iterator restarts after exhaustion.
12. Source/target shape mismatch fails before model forward.
13. Random-tensor AMP-compatible backward smoke test.
14. Normal prediction output parity before and after adding DANN attributes.
15. ONNX contract and initializer inspection when ONNX dependencies are available.

Tests use the smallest YOLOv13 scale and small tensors where possible. Export tests are skipped with an explicit reason when optional ONNX packages are absent.

## Files

New files:

- `models/yolo/detect/dann.py`
- `models/yolo/detect/train_dann.py`
- `scripts/train_yolov13_dann.py`
- `configs/dann/ic_p5.yaml`
- `tests/test_yolov13_dann.py`
- `docs/superpowers/specs/2026-08-11-yolov13-dann-p5-design.md`

Minimally updated files, only if import exposure is needed:

- `models/yolo/detect/__init__.py`

No dependency is added. PyTorch, PyYAML, pytest, and existing Ultralytics utilities are reused.

## Acceptance Criteria

- The repository remains clean after tests except intentional source changes.
- E0 uses the existing YOLO training behavior.
- E0.5 and E1 perform exactly one model feature traversal per paired source/target step.
- Detection loss sees only source predictions and source labels.
- E1 domain loss sees all P5 samples and produces two-class logits.
- GRL lambda follows the configured progressive schedule.
- Original YOLOv13 YAML and Detect head are byte-for-byte unchanged.
- Standard prediction requires only an image and matches baseline output structure.
- ONNX has no extra input/output and contains no domain-classifier initializer.
- Tests and random backward smoke test pass before dataset experiments begin.

## Deferred Work

- P3/P4 multi-level domain branches
- AdaIN or other style-transfer adaptation
- domain-specific BatchNorm
- DANN paper learning-rate schedule
- target pseudo-labeling
- cross-domain Mosaic/MixUp
- exporter or predictor customization
