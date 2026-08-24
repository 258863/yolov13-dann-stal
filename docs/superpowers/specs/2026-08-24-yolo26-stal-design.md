# YOLO26 STAL Reproduction Design

## Goal

Reproduce YOLO26 small-target-aware label assignment (STAL) in the local
YOLOv13 training code so that controlled TAL-versus-STAL experiments can be
run without changing inference behavior or box regression targets.

The implementation follows Ultralytics commit
`ca4ca9e4c4c2fccbae45413f738078958f79f1e1`.

## Behavior

STAL is disabled by default through `stal: false`. When enabled, the
axis-aligned `TaskAlignedAssigner` builds a temporary proxy box only for the
anchor-center inclusion mask:

- A GT width or height smaller than the smallest detection stride is replaced
  independently by the second-smallest stride.
- The proxy box center is unchanged.
- Dimensions that are not smaller than the smallest stride are unchanged.
- Invalid or padded GT rows remain masked out.

For the normal `[8, 16, 32]` strides, a centered `4 x 4` GT box therefore uses
a `16 x 16` proxy box for candidate selection.

The original GT boxes continue to feed alignment metric IoU, conflict
resolution, target generation, and box/DFL regression. STAL does not enlarge
the learned target.

## Integration

`cfg/default.yaml` exposes the boolean `stal` training option. Detection loss
reads the option from model arguments and passes the detection-head stride list
and the enabled flag to `TaskAlignedAssigner`.

`TaskAlignedAssigner` owns proxy-box construction because candidate selection
is its responsibility. With STAL disabled, it follows the current code path
unchanged. This keeps baseline behavior stable and avoids a separate model,
trainer, or inference graph.

The first implementation is scoped to axis-aligned detection assignment. It
does not modify rotated-box assignment because the requested experiment uses
YOLOv13 detection and the rotated geometry requires separate validation.
Detection-derived losses that instantiate `v8DetectionLoss` inherit the option,
but no prediction or export code changes.

## Experiment Contract

The comparison changes only `stal`:

```text
E-TAL:  stal=false
E-STAL: stal=true
```

Dataset split, seed, initialization, batch size, optimizer, augmentations,
epochs or training steps, and confidence threshold remain fixed. Evaluation
prioritizes missed detections and false positives on the same untouched
validation set.

## Verification

Tests must prove:

1. A `[6, 6, 10, 10]` GT with stride list `[8, 16, 32]` has no candidates at
   centers `(4,4)`, `(12,4)`, `(4,12)`, `(12,12)` under TAL and four under
   STAL.
2. Only dimensions smaller than the minimum stride are expanded.
3. Padded GT entries cannot create candidates.
4. STAL-disabled assignment matches the existing implementation.
5. IoU metrics and generated target boxes use the original GT.
6. YOLOv13 loss can run forward and backward with both settings.

## Compatibility

The feature changes training-time assignment only. Checkpoints, prediction,
validation inference, ONNX export, model YAML, Detect head output, and DANN
domain branches remain unchanged.
