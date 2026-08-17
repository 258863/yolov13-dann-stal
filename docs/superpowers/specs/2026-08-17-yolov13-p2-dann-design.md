# YOLOv13 P2 Weak-DANN Experiment Design

## Objective

Measure whether aligning the paper-corresponding P2/4 downsampling output improves cross-site detection compared with the no-DANN, P5-DANN, and P3-DANN experiments.

## Feature position

- Selected feature: YOLOv13 layer 1, the P2/4 downsampling output.
- Base configuration channels: 128.
- Nano model channels: 32.
- Layer 2 is intentionally excluded because the requested position is the direct P2/4 downsampling output used by the paper mapping.

## Controlled experiment

- One training-only domain classifier attached only to layer 1.
- `lambda_max: 0.02`, `gamma: 10.0`, no warmup.
- Full dual-domain supervision with 8 Source + 8 Target.
- 10 epochs; seeds 42, 43, and 44; AMP disabled; deterministic enabled.
- Same Source, Target, independent 74-image Target test set, weights, augmentations, and evaluation scripts as P3.
- Evaluate conf=0.40 and conf=0.55, per-class misses, 15 empty-label images, and Source/Target mAP50-95.

## Compatibility

Keep the legacy `p5_layer` key and existing layers 23, 27, and 31 unchanged. Layer 1 channels are inferred from its Conv output. Inference, Detect, YAML, exporter, and ONNX outputs remain unchanged.
