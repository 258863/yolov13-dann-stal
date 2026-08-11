# E0-BN BatchNorm Control Design

## Goal

Add a control experiment that separates the effect of a 16-image forward/BatchNorm batch from the effect of mixing Source and Target domains.

## Experiment Matrix

| Experiment | Forward batch | Detection supervision | Target images | Domain objective |
|---|---|---|---:|---:|
| E0 | Source 8 | Source 8 | No | No |
| E0-BN | Source 8 + Source 8 | First Source 8 only | No | No |
| E0.5 | Source 8 + Target 8 | Source 8 only | Yes | No |
| E1 | Source 8 + Target 8 | Source 8 only | Yes | Yes |

## Data Path

E0-BN uses one Source training loader with a physical batch size of 16. After normal YOLO preprocessing and augmentation, the trainer splits the batch at index 8:

- img[:8] becomes the supervised Source batch.
- img[8:16] becomes the BN-only companion batch through target_img.
- Detection annotations are filtered to batch_idx < 8.
- The model concatenates both halves and performs one backbone/neck forward pass.
- Detect predictions are sliced back to the first 8 images before the original detection criterion.
- dann_enabled=False and dann_lambda=0.0 keep domain loss, domain accuracy, and GRL disabled.

Using one 16-image loader avoids the identical-order risk of two independently constructed Source loaders with the same seed.

## Configuration

Add experiment: e0bn and dann.supervised_batch: 8. The E0-BN run keeps the existing comparison settings:

- epochs: 10
- imgsz: 640
- physical train batch: 16
- supervised batch: 8
- device: 0
- workers: 4
- seed: 42
- deterministic: true
- weights: the same yolov13n.pt

target_data is not required for E0-BN because all 16 images come from Source.

## Validation

Validate best.pt on the same Source val and original independent Target val used for E0 and E0.5. Report Source/Target precision, recall, mAP50, mAP50-95, training time, and peak GPU memory. Confirm all domain metrics remain zero.

After the cleaned Target labels are synchronized, revalidate E0, E0-BN, and E0.5 together; do not mix cleaned Target metrics with the existing pre-cleaning table.

## Acceptance Criteria

- Each E0-BN training step forwards 16 Source images once.
- Only annotations for images 0 through 7 reach detection loss.
- Images 8 through 15 influence shared BatchNorm statistics but receive no detection supervision.
- Domain classifier parameters receive no gradient.
- Domain loss, domain accuracy, and GRL lambda log as zero.
- Existing E0, E0.5, E1, inference, and export behavior remain unchanged.
