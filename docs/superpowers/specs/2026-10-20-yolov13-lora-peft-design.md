# YOLOv13 LoRA / PEFT Design (structure-aware adapter planning)

## Goal

Add an opt-in, structure-aware LoRA path to the existing YOLOv13 DANN trainers so that a pretrained YOLOv13n can be adapted to the STAL Source/Target data by training a small set of adapter parameters instead of the whole network. The path must (a) decide *where* adapters may go with an explicit, auditable planner, (b) never silently disable the DANN adversarial signal, and (c) end with a plain, fusable, exportable checkpoint.

The approach follows the planner idea of YOLO-PEFT (arXiv 2608.07051): a pure function `P(graph, request, budget) -> (decision, rank map, ordered rejection log)` that applies operator, semantic, architecture-policy, rank and reliability filters in a fixed order, and refuses with named reason codes instead of degrading silently. This repository implements its own planner and adapter modules; no code is copied from the public reference implementation (Tencent/YOLO-Master, AGPL-3.0), which was read for behaviour only.

## Non-Goals

- No multi-GPU/DDP support (the trainer raises if `world_size > 1`).
- No Linear/attention-projection adapters, DoRA, HRA, alpha warm-up, orthogonality regularisation, or leave-one-out calibration in v1.
- No dependency on Hugging Face PEFT.
- No change to the default behaviour of any existing experiment; the feature is enabled only by a `peft:` section in an experiment YAML.
- The adapter does not replace the existing E0 / E0-BN / E0.5 controls; it is an additional arm.

## Evidence And Regime Caution

- YOLO-PEFT reports defaults of rank 16, alpha 32, dropout 0.05, rank-stabilised scaling, and a catastrophe criterion of mAP50:95 dropping more than 0.05 below the full fine-tune anchor. YOLOv13 was not among its evaluated families, and the paper lists domain-shift and few-shot evaluation as limitations. Every YOLOv13 plan is therefore marked `UNVERIFIED` and requires `allow_unverified: true`.
- The reference implementation's own guidance warns that fewer than about 500 images often make LoRA underperform full fine-tuning. This project trains on a small Source/Target set, so LoRA may well lose to full fine-tuning here. The experiment matrix below is designed so that a negative result is decisive rather than ambiguous.
- Independent evidence on cross-domain few-shot detection (arXiv 2504.06330) shows small gains for LoRA applied after an initial fine-tune at 1/5 shots and results below full fine-tuning at 10 or more shots.
- With scale `n` the adapters are not small relative to the model at higher ranks, so rank is capped by a budget ratio rather than fixed.

## Codebase Findings

| # | Finding | Consequence |
|---|---------|-------------|
| F1 | All experiments use `yolov13n`: depth 0.5, width 0.25, widths 16/32/64/64/128/128/128/256/256 for layers 0-8. Layer 1 is a grouped Conv (g=2) and layer 3 a grouped Conv (g=4). | Grouped and depthwise convs cannot take a rank-r update (`E-GROUP-RANK`, `E-DEPTHWISE-MIX`). |
| F2 | The P2 DANN hook is layer 1. Its upstream is only layers {0, 1}: 432+32+2,304+64 = 2,832 parameters. Hooks 23/27/31 have 23/28/31 upstream layers. | The adversarial gradient reaches only parameters upstream of the hook. Freezing the stem would make P2 DANN a no-op for the detector, so layers 0 and 1 are trained in full (`full_train_layers: [0, 1]`) and the planner checks reach. |
| F3 | YOLOv13 attention modules use `qk`, `v`, `proj`, `pe` names, not YOLO12's `qkv`. | Name-substring exclusions do not transfer; rules use class ancestry (`AAttn`, `HyperACE`, ...). |
| F4 | The stock trainer's freeze block runs in `_setup_train` and resets `requires_grad`. | The trainable set must be applied inside `build_optimizer`, after the freeze block. |
| F5 | `optimizer=auto` selects AdamW with a fitted learning rate and forces `warmup_bias_lr=0.0`. Stock groups put biases first and warm-up treats group index 0 specially. | Adapter parameters go in one extra group appended last; optimizer, `lr0`, momentum and `warmup_bias_lr` are pinned identically in every arm. |
| F6 | `save_model` pickles `deepcopy(ema.ema).half()`; `BaseModel.load` silently drops keys whose names or shapes differ; resume loads the EMA strictly but the live model only through `get_model(weights)`. | Wrappers are stripped from a resumed checkpoint before the stock load and adapters are restored strictly. |
| F7 | `final_eval` builds an AutoBackend, which calls `fuse()` and `fuse_conv_and_bn`; these ignore adapters. | Adapters are merged and verified before final evaluation. The unmerged wrapper exposes no `weight`, so an accidental fuse fails loudly. |
| F8 | The new class-prediction conv `Detect.cv3.*.2` is re-initialised when `nc != 80`. | It is trained directly (`X-REINIT`), never given an adapter. |
| F9 | BatchNorm running statistics update in train mode on mixed Source+Target batches. | An AdaBN-like effect exists in every arm; the existing E0-BN control isolates it, and `bn_mode` is an explicit ablation. |

## Planner

Pure functions in `yolo_peft/planner.py`; no torch import. Input: a list of `Leaf` records (one per parameter-owning module), the layer parent graph, DANN hook layers, a family label, a `pretrained` flag and a `PeftConfig`. Output: a `Plan`.

Evaluation order for each Conv2d leaf: operator, semantic, architecture policy, rank allocation, then user scope (`layers`). User scope is applied only after the mandatory filters.

| Code | Meaning in this repository |
|------|----------------------------|
| `E-OP-UNSUPPORTED` | Linear or bare `nn.Parameter` leaf (v1 scope) |
| `E-DEPTHWISE-MIX` | Depthwise convolution (groups equal to input channels) |
| `E-GROUP-RANK` | Other grouped convolution |
| `E-DFL-FIXED` | Distribution Focal Loss conv or any already-frozen leaf |
| `E-HEAD-GEOMETRY` | Any conv in the `Detect` box-regression branch (`cv2`) |
| `E-BUDGET` | Trainable parameters exceed `budget_ratio` even at `min_rank` |
| `E-EMPTY-PLAN` | No leaf received an adapter |
| `E-FAMILY-UNKNOWN` | Family outside the rule set, or `UNVERIFIED` without `allow_unverified` |
| `R-OUT-OF-SCOPE` | Warning attached to an `UNVERIFIED` plan (YOLOv13 is outside the YOLO-PEFT families) |
| `X-NO-PRETRAINED` | Refusal: no pretrained weights, adapters on a random backbone are meaningless |
| `X-PROFILE-ATTN` | Attention / hypergraph leaf excluded by profile `core` |
| `X-USER-LAYERS` | Leaf outside the requested `layers` interval (applied after mandatory filters) |
| `X-RANK-CAPACITY` | Leaf capacity `min(c_out, c_in*kh*kw)//2` is below `min_rank` |
| `X-NO-SAVING` | Adapter parameters reach `saving_ratio` of the layer's own weights |
| `X-FULL-LAYER` | Leaf is in `full_train_layers` and is trained directly |
| `X-REINIT` | Leaf was freshly initialised and is trained directly |
| `X-DANN-REACH-PROMOTE` | Leaf promoted to full training so a DANN hook has trainable parameters |
| `X-DANN-UNREACHABLE` | Refusal: a DANN hook cannot be reached within `reach_param_cap` |

The `Plan` records status (`ACCEPT`, `UNVERIFIED`, `REFUSE`), family, rank, the `adapt` map (leaf to rank), the `full` map, the ordered refusal log, per-hook reach, counts and warnings. `to_json()` is deterministic and a short fingerprint of the leaf list is stored so a checkpoint can be matched to the architecture it was planned for.

Rank allocation tries `rank`, then every value of `rank_set` below it in descending order, until the trainable ratio fits `budget_ratio`; `min_rank` is a per-leaf floor (`X-RANK-CAPACITY`). DANN reach: if a hook has no trainable parameter, the planner promotes the single cheapest upstream Conv2d (`X-DANN-REACH-PROMOTE`) when it fits `reach_param_cap`; otherwise it refuses with `X-DANN-UNREACHABLE`. With the default `full_train_layers: [0, 1]` the P2 hook is reached without promotion. `dann_reach: off` disables the check for ablations.

## Trainable-Set Policy

- Adapted: planned Conv2d leaves (ordinary `Conv.conv` and pointwise convs, rank per plan).
- Full: `full_train_layers` (default layers 0 and 1), re-initialised leaves, promoted DANN-reach leaves, DANN domain classifiers.
- Head: classification output conv is full (re-initialised); box output conv is frozen unless `head_box: full`.
- BatchNorm: affine parameters frozen by default (`bn_mode: train` only keeps running statistics updating); `bn_mode: affine` trains them.
- FullPAD gates frozen unless `train_gates: true`.
- DFL always frozen.

## Adapter Module

`LoRAConv2d` replaces a `nn.Conv2d` with groups=1 and zero padding: `y = base(x) + scale * B(A(dropout(x)))`. `A` is a conv with the base kernel, stride, padding and dilation producing `r` channels; `B` is 1x1. `A` uses Kaiming-uniform (a = sqrt(5)); `B` starts at zero so a fresh adapter is an identity edit. `scale = alpha / sqrt(r)` with rank-stabilised scaling, else `alpha / r`; `alpha = alpha_ratio * r`.

Merge: `W' = W + scale * einsum('or,rihw->oihw', B[:, :, 0, 0], A)`, computed in float32 and cast back; bias unchanged. The algebra was checked in NumPy over 72 configurations (kernels, strides, ranks, dilations).

Adapter parameter count for a leaf: `r * (c_in*kh*kw + c_out)`.

## Trainer Integration

`PEFTTrainerMixin` is placed first in the MRO of `PEFTDANNDetectionTrainer(PEFTTrainerMixin, DANNDetectionTrainer)`:

1. `get_model`: strip wrappers from a resumed checkpoint, call the stock `get_model`, plan, write `peft_plan.json` into the run directory, refuse or attach adapters, then restore adapter tensors strictly on resume.
2. `build_optimizer`: apply the trainable policy, call the stock builder, then drop frozen/adapter parameters from the stock groups and append one adapter group (`lr = lr0 * lr_mult`, `weight_decay` from config).
3. `final_eval`: on rank 0 verify and write merged checkpoints for `last.pt` and `best.pt` (keeping `<stem>_peft.pt` and `<stem>_adapter.pt`), then run the stock final evaluation.

Experiments `e0bn`, `e05` and `e1` may use PEFT; `e0` (plain `DetectionTrainer`) keeps its existing branch unchanged. Routing lives in `scripts/train_yolov13_dann.py` (`select_trainer(experiment, peft=False)` and `build_trainer`).

## Configuration

Optional top-level `peft:` section in an experiment YAML; unknown keys and wrong types are rejected.

| Key | Default | Notes |
|-----|---------|-------|
| `enabled` | true | false disables the section |
| `method` | lora | only value in v1 |
| `profile` | core | `core` or `core_attn` |
| `rank` / `rank_set` / `min_rank` | 8 / [4, 8, 16] / 4 | budget fallback uses rank_set values <= rank, descending; min_rank is a per-leaf floor |
| `alpha_ratio` | 2.0 | alpha = ratio * rank |
| `rslora` | true | scale by 1/sqrt(r) |
| `dropout` | 0.05 | on the adapter input |
| `saving_ratio` | 0.5 | adapter must be smaller than this fraction of its layer |
| `budget_ratio` | 0.20 | max trainable params / base params |
| `layers` | null | optional `[lo, hi]` scope, applied last |
| `full_train_layers` | [0, 1] | trained directly; needed for P2 reach |
| `head_box` | frozen | or `full` |
| `bn_mode` | train | or `affine` |
| `train_gates` | false | FullPAD gates |
| `dann_reach` | auto | or `off` |
| `reach_param_cap` | 50000 | max promoted parameters per hook |
| `lr_mult` / `weight_decay` | 1.0 / 0.0 | adapter group |
| `allow_unverified` | false | must be true for YOLOv13 |

## Experiment Matrix

All arms: seeds 42/43/44, 10 epochs, batch 8, imgsz 640, amp false, deterministic, pinned `optimizer`, `lr0`, `momentum` and `warmup_bias_lr`. Path fields are copied from `configs/dann/ic_p2_a0_closemosaic_seed42.yaml`.

| Arm | Trainer | Change | Question |
|-----|---------|--------|----------|
| R0 | e1 | full fine-tune, current P2 TAL candidate | anchor |
| R1 | e1 | LoRA r=8, core, layers 0-1 full | does LoRA match full fine-tuning? |
| R2 | e05 | R1 with adversarial term off | is the adversarial term still useful under LoRA? |
| R3 | e1 | R1 with rank 4 | rank sensitivity |
| R4 | e1 | R1 with `head_box: full` | is a frozen box head limiting? |
| R5 | e1 | R1 with `bn_mode: affine` | BN affine contribution |
| R6 | e1 | R1 with `profile: core_attn` | attention adapters |

Decision rules (to be confirmed before running):

- Catastrophe gate: any arm with mAP50:95 more than 0.05 below R0 on the mean of three seeds is closed as a failure.
- Non-inferiority: an arm is competitive if its three-seed mean TP is within 1 of R0, FP not more than 1 above, FN not more than 1 above, at conf=0.40 over the 59 ground-truth boxes, with the 15 empty-label Target images counted for FP.
- R1 vs R2 and R1 vs the E0-BN control are the only comparisons used to attribute effects to the adversary or to BatchNorm statistics.

## Verification

- Planner and configuration logic: torch-free unit tests (17 cases, mutation-checked on rule order).
- Merge algebra: NumPy check over 72 configurations.
- Graph, adapter, state and optimizer logic: exercised in the authoring sandbox on a NumPy-backed torch stand-in with YOLOv13-shaped fake modules. These files were **not** executed against real PyTorch in the authoring environment; the plan's Task 8 runs them with real torch.
- Required on the user's machine: the full existing test suite, `scripts/peft_plan.py` on the real YAML, a one-epoch GPU smoke run, and a merge-equivalence check on a trained checkpoint.

## Open Inputs

- Number of Source training images and of labelled Target images (decides whether the unfavourable regime applies).
- Class count `nc` and the effective optimizer line from an existing run log (`optimizer: AdamW(lr=...)`), to pin `lr0`.
- Whether separately deployable per-domain adapters matter (this would favour keeping adapter-only checkpoints).
