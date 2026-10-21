# Target-Style Source Augmentation Design

## Goal

Add one controlled A1 experiment to the existing YOLOv13 P2 DANN training path. Source images receive a light target-style transfer during training so the experiment can test whether reducing the visible domain gap improves recall without increasing false positives.

## Fixed Baseline

The experiment preserves the current best settings:

- P2 domain feature layer: layer 1
- FP32 training
- `lambda_max: 0.02`
- Source batch: 8
- Target batch: 8
- `target_supervised: true`
- STAL disabled

Only Source style augmentation changes between A0 and A1.

## Data Flow

After the normal Source and Target augmentations and conversion to `[0, 1]` tensors:

1. Calculate per-image, per-channel spatial mean and standard deviation for Source and Target.
2. Pair Source samples with Target statistics within the current batch. Cycle Target statistics when batch sizes differ.
3. Normalize Source with its own statistics and restore it with the paired Target statistics.
4. Blend the transferred image with the original Source using `alpha = 0.30`.
5. Apply the blend independently to each Source sample with probability `0.50`.
6. Clamp the result to `[0, 1]`.
7. Continue the existing single combined Source and Target forward pass.

Target images, annotations, validation images, and inference images are not modified.

## Configuration

Add disabled-by-default DANN options:

```yaml
source_style_aug: false
source_style_prob: 0.50
source_style_alpha: 0.30
```

Create a dedicated A1 experiment configuration rather than changing the existing baseline configuration. The A1 configuration enables the augmentation and records the fixed P2 DANN settings.

## Numerical Behavior

- Use a small epsilon when dividing by Source standard deviation.
- Perform the operation in the same floating dtype and device as the preprocessed Source batch.
- `prob = 0.00` and `alpha = 0.00` must preserve Source exactly.
- Validate probability and alpha in the closed interval `[0, 1]`.
- Do not derive gradients through Target images; this is an input transform, not a trainable module.

## Verification

Automated tests cover:

- configuration defaults and boundaries;
- exact no-op behavior when disabled;
- deterministic full-probability moment transfer on synthetic tensors;
- Target tensors remain unchanged;
- output range and shape;
- the existing DANN test suite remains green.

A quick trainer/config smoke check must prove that the dedicated A1 configuration selects layer 1, `lambda_max = 0.02`, supervised Target training, and FP32 execution. Full training is a separate remote experiment.

## Success Criterion

Run seed42 first and evaluate at `conf = 0.40`. Compare against the current A0 result `TP=58, FP=2, FN=1`. Continue to more seeds only if A1 does not worsen FN and does not materially increase FP.
