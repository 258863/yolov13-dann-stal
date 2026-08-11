# E0-BN BatchNorm Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add and run an E0-BN control where 16 distinct Source images share one forward pass but only the first 8 receive detection supervision.

**Architecture:** Extend the existing DANN trainer with an e0bn preprocessing branch. The configured optimizer batch remains 8 while only the training DataLoader fetch is doubled to 16; that physical batch is split into supervised img[:8] and BN-only target_img=img[8:], while annotations are filtered to batch_idx < 8.

**Tech Stack:** Python, PyTorch, Ultralytics YOLOv13, pytest, YAML, Windows PowerShell, RTX 3090 Ti.

## Global Constraints

- Do not change Backbone, Neck, Detect Head, inference, export, class count, or ONNX contract.
- E0-BN uses configured/optimizer batch 8 and one Source loader with physical fetch batch 16; do not substitute two independently seeded Source loaders.
- Drop only the final incomplete E0-BN training batch so every forward remains exactly 8+8.
- Only images 0-7 and their annotations reach detection loss.
- Images 8-15 affect BatchNorm but receive no detection or domain supervision.
- Keep the prior comparison settings: epochs 10, imgsz 640, seed 42, deterministic true, workers 4, same yolov13n.pt weights.

---

### Task 1: Lock E0-BN Configuration and Batch Splitting

**Files:**
- Modify: tests/test_yolov13_dann.py
- Modify: models/yolo/detect/train_dann.py
- Modify: scripts/train_yolov13_dann.py
- Create: configs/dann/ic_p5_e0bn.yaml

**Interfaces:**
- Consumes: DANNDetectionTrainer.preprocess_batch(batch), validate_dann_config(config), select_trainer(experiment).
- Produces: experiment name e0bn and config key supervised_batch: int.

- [ ] **Step 1: Add failing configuration tests**

Add assertions that select_trainer('e0bn') returns DANNDetectionTrainer, validate_dann_config accepts e0bn without target_data when supervised_batch=8, and rejects missing/non-positive supervised_batch.

- [ ] **Step 2: Add failing preprocessing test**

Construct a 16-image uint8 batch with batch_idx values spanning 0-15. Assert preprocess_batch returns img shape (8, C, H, W), target_img shape (8, C, H, W), keeps only annotations with batch_idx < 8, sets dann_enabled False, and sets dann_lambda 0.0.

- [ ] **Step 3: Run focused tests and confirm failure**

Run: python -m pytest tests/test_yolov13_dann.py -q

Expected: failures because e0bn and supervised_batch are not implemented.

- [ ] **Step 4: Implement minimal trainer support**

In validate_dann_config, allow e0bn, require target_data only for e05/e1, and require a positive supervised_batch for e0bn. In _setup_train, skip target-loader construction for e0bn. In preprocess_batch, split the already-preprocessed physical Source batch and filter batch_idx, cls, and bboxes with the same mask before setting target_img, dann_enabled=False, and dann_lambda=0.0.

- [ ] **Step 5: Extend the training entry point**

Map e0bn to DANNDetectionTrainer and update validation/help text to list e0, e0bn, e05, and e1.

- [ ] **Step 6: Add the reusable E0-BN YAML**

Create configs/dann/ic_p5_e0bn.yaml with experiment e0bn, train.batch 8, and dann.supervised_batch 8. The trainer alone doubles the train-loader fetch to 16.

- [ ] **Step 7: Run focused verification**

Run: python -m pytest tests/test_yolov13_dann.py -q

Expected: all focused DANN tests pass.

### Task 2: Regression and Runtime Smoke Verification

**Files:**
- Test: tests/test_yolov13_dann.py
- Run: scripts/check_yolov13_dann.py

**Interfaces:**
- Consumes: completed e0bn trainer/config behavior.
- Produces: evidence that E0/E0.5/E1 and inference contracts remain unchanged.

- [ ] **Step 1: Run the DANN smoke checker**

Run: python scripts/check_yolov13_dann.py

Expected: forward/backward checks complete without error.

- [ ] **Step 2: Run syntax and diff checks**

Run: python -m compileall models/yolo/detect/dann.py models/yolo/detect/train_dann.py scripts/train_yolov13_dann.py

Run: git diff --check

Expected: exit code 0 for both.

- [ ] **Step 3: Commit the implementation**

Stage only the trainer, entry point, config, tests, and plan. Use a Lore commit describing the supervised/BN-only split and verification.

### Task 3: Remote 3090 E0-BN Experiment

**Files:**
- Create remotely: F:/dann_compare_20260811/experiment/e0bn.yaml
- Produce remotely: F:/dann_compare_20260811/runs/e0bn_10ep
- Copy locally: D:/work/dz/dann_compare_20260811/e0bn_results.csv
- Copy locally: D:/work/dz/dann_compare_20260811/e0bn_best.pt

**Interfaces:**
- Consumes: source.yaml, target.yaml, yolov13n.pt, implemented e0bn trainer.
- Produces: E0-BN best checkpoint and Source/Target metrics comparable with E0 and E0.5.

- [ ] **Step 1: Sync the committed branch files to the remote experiment root**

Copy the modified code/config/tests needed by the remote checkout while preserving the existing E0/E0.5 outputs.

- [ ] **Step 2: Write the remote experiment YAML**

Use experiment e0bn, source_data F:/dann_compare_20260811/experiment/source.yaml, model F:/dann_compare_20260811/cfg/models/v13/yolov13.yaml, weights F:/dann_compare_20260811/yolov13n.pt, train.batch 8, and dann.supervised_batch 8.

- [ ] **Step 3: Run a one-batch backward smoke test**

Confirm physical batch 16, supervised img batch 8, BN-only target_img batch 8, finite detection loss, and zero domain metrics.

- [ ] **Step 4: Train 10 epochs on the RTX 3090 Ti**

Run through a hidden scheduled task, monitor results.csv, GPU memory, temperature, and exit code until completion.

- [ ] **Step 5: Uniformly validate best.pt**

Run Source val and the same original Target val with imgsz 640, batch 8, device 0, workers 4, plots false.

- [ ] **Step 6: Save comparison artifacts**

Copy best.pt, results.csv, validation JSON, logs, and a concise E0/E0-BN/E0.5 table to D:/work/dz/dann_compare_20260811.

## Self-Review

- Spec coverage: configuration, unique 16-Source sampling, supervision slicing, zero domain objective, regression checks, remote training, and validation are covered.
- Placeholder scan: no TBD/TODO or deferred implementation steps.
- Type consistency: supervised_batch is an integer config value; annotation filtering uses the boolean mask derived from batch_idx.
