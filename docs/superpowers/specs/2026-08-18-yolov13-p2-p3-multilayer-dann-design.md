# YOLOv13 P2+P3 Multi-Layer DANN Experiment Design

## Objective

Test whether the complementary behavior observed in the single-layer experiments can be combined: P2 produced the lowest fixed-threshold miss rate, while P3 produced the highest Target mAP50-95.

## Alternatives considered

1. Tune the P2 GRL coefficient before changing architecture. This is lower risk, but it does not test the planned multi-layer hypothesis.
2. Attach domain branches to P2 and P3 only. This is selected because it is the smallest multi-layer experiment and preserves interpretability.
3. Attach domain branches to P2, P3, P4, and P5. This is rejected for the first run because it changes too many gradient paths and makes a negative result difficult to diagnose.

## Architecture

- P2 branch: layer 1, nano output channels 32.
- P3 branch: layer 23, nano output channels 64.
- Each feature has its own GRL and DomainClassifier.
- Source and Target images are concatenated and pass through the detector once.
- Detection loss uses both domains because the current controlled experiment uses labeled Target training data.
- Domain labels use every Source and Target feature at both selected layers.
- The two cross-entropy losses are averaged before applying domain_loss_gain and Source batch scaling:

```python
domain_loss = (p2_domain_loss + p3_domain_loss) / 2
```

This prevents the total domain-loss scale from doubling relative to the single-layer experiments.

## Configuration and compatibility

- Add a multi-layer configuration field while retaining the existing single p5_layer field for backward compatibility.
- The accepted multi-layer value for this experiment is [1, 23].
- Single-layer P2, P3, P4, and P5 behavior must remain unchanged.
- The additional domain modules remain training-only attributes.
- Detect, model YAML, prediction output, exporter, and ONNX input/output contracts remain unchanged.

## Controlled experiment

- 10 epochs.
- Source batch 8 and Target batch 8.
- Target supervision enabled.
- Seeds 42, 43, and 44.
- lambda_max=0.02, gamma=10.0, no DANN warmup.
- AMP disabled and deterministic mode enabled.
- Use the same weights, datasets, augmentations, Target holdout, and evaluation scripts as the P2 and P3 experiments.

## Evaluation and acceptance

Compare no-DANN, P5, P3, P2, and P2+P3 using:

- TP, FP, and FN at confidence 0.40 and 0.55 with IoU matching 0.30.
- Overkill on the 15 empty-label Target images.
- HD and LX misses.
- Source and Target mAP50-95.
- Stability of image30.png detection across three seeds.
- Per-layer domain loss and domain accuracy plus their averaged training loss.

The experiment is promising only if it improves or matches P2's miss rate without materially increasing false positives or empty-label overkill. A Target mAP gain alone is insufficient if fixed-threshold FN regresses.

## Verification

- Unit tests cover feature capture, channel inference, averaged loss, gradients to both classifiers and both feature paths, legacy single-layer behavior, and invalid configurations.
- A random-tensor backward smoke test must pass.
- The complete DANN test suite and ONNX exclusion test must pass before remote training.
- Remote smoke testing must confirm P2=32 channels and P3=64 channels.
