# YOLOv13 P3 Weak-DANN Experiment Design

## Objective

Measure whether weak adversarial alignment on the shallow P3 feature reduces threshold-2 misses without increasing false positives or damaging threshold-1 retention.

## Approved experiment

- Baseline: existing full dual-domain supervised model without DANN.
- Treatment: full dual-domain supervised training with one training-only domain classifier attached to layer 23 (P3).
- Source and Target are concatenated for one forward pass.
- Detection loss uses both domains because `target_supervised: true`.
- Domain loss uses all Source and Target P3 features.
- `lambda_max: 0.02`, `gamma: 10.0`, no DANN warmup.
- Seeds: 42, 43, 44.
- Training: 10 epochs, image size 640, 8 Source + 8 Target, AMP disabled, deterministic enabled.
- Test data remains the independent 74-image threshold-2 test set; training labels must not be mixed into it.

## Implementation boundary

Keep the existing `p5_layer` configuration key for backward compatibility, but allow only the verified YOLOv13 pyramid output layers 23, 27, and 31. Select the matching Detect input branch to infer the domain classifier input channels. Do not alter the YOLOv13 YAML, Detect head, inference path, exporter, or existing P5 experiment behavior.

## Evaluation

Report conf=0.40 and conf=0.55 TP/FP/FN, hd/lx misses, false positives on all 15 empty-label images, Source/Target mAP50-95, and three-seed mean. Compare primarily against the existing full dual-domain supervised no-DANN baseline and secondarily against P5 DANN lambda=0.10.

## Stop condition

Do not proceed to multi-layer DANN unless P3-only alignment shows a repeatable benefit across seeds without unacceptable old-domain degradation.
