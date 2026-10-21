# YOLOv13 LoRA / PEFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, structure-aware LoRA path to the YOLOv13 DANN trainers (experiments `e0bn`, `e05`, `e1`), with a deterministic placement planner, a DANN-reach guarantee for the P2 hook, resume-safe adapter state, and merged exportable checkpoints.

**Architecture:** A torch-free planner (`yolo_peft/planner.py`, `yolo_peft/config.py`) turns parameter-owning leaves into an auditable plan. `yolo_peft/graph.py` extracts leaves from a built model, `yolo_peft/lora.py` holds the adapter module and checkpoint helpers, and `yolo_peft/trainer.py` adds a mixin placed first in the MRO of the existing `DANNDetectionTrainer`. `scripts/train_yolov13_dann.py` routes to it only when the YAML has a `peft:` section. Design: `docs/superpowers/specs/2026-10-20-yolov13-lora-peft-design.md`.

**Tech Stack:** Python 3, PyTorch, Ultralytics YOLOv13 trainer, pytest, YAML

## Global Constraints

- Keep the feature disabled by default; an experiment YAML without `peft:` must behave exactly as before.
- Do not change `feature/yolov13-dann-p5` or any existing experiment config.
- Add no dependency (no Hugging Face PEFT).
- Single GPU only; the mixin raises for `world_size > 1`.
- Pin `optimizer`, `lr0`, `momentum` and `warmup_bias_lr` identically in every compared arm (`optimizer=auto` changes the learning rate with the iteration count and forces `warmup_bias_lr=0.0`).
- Config files must not contain machine-specific paths beyond those already used by the existing configs; copy path fields from `configs/dann/ic_p2_a0_closemosaic_seed42.yaml`.
- The code below was written against the repository layout (package root = repository root, imported as `ultralytics`). `config.py`, `planner.py` and their tests were executed in an authoring sandbox without PyTorch. `graph.py`, `lora.py`, `trainer.py` and `scripts/peft_plan.py` were lint-checked and exercised only on a NumPy-backed torch stand-in; **they have not been run against real PyTorch.** Task 8 does that.

---

### Task 1: Planner And Configuration (torch-free)

**Files:**
- Create: `yolo_peft/__init__.py` (empty)
- Create: `yolo_peft/config.py`
- Create: `yolo_peft/planner.py`
- Create: `tests/test_yolo_peft_planner.py`

**Interfaces:**
- Produces: `PeftConfig`, `validate_peft_config(raw)`, `load_peft_config(path)`, `Leaf`, `Plan`, `plan_placement(leaves, parents, hooks, family, pretrained, cfg)`, `detect_family`, `upstream_layers`, `model_fingerprint`.

- [ ] **Step 1: Create the package and `config.py`**

```python
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

PROFILES = ('core', 'core_attn')
HEAD_BOX_POLICIES = ('frozen', 'full')
BN_MODES = ('train', 'affine')
DANN_REACH_MODES = ('auto', 'off')


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _real(name: str, value: Any, low: float, high: float, *, low_open: bool = False) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f'{name} must be a real scalar')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    if result < low or result > high or (low_open and result == low):
        raise ValueError(f'{name} must be within {"(" if low_open else "["}{low}, {high}]')
    return result


def _positive_int(name: str, value: Any) -> int:
    if not _is_int(value) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _choice(name: str, value: Any, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(f'{name} must be one of {allowed}')
    return value


def _bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f'{name} must be a bool')
    return value


@dataclass(frozen=True)
class PeftConfig:
    """Validated `peft:` section. Defaults follow YOLO-PEFT where the paper states one."""

    method: str = 'lora'
    profile: str = 'core'
    rank: int = 8
    rank_set: tuple[int, ...] = (4, 8, 16)
    min_rank: int = 4
    alpha_ratio: float = 2.0
    rslora: bool = True
    dropout: float = 0.05
    saving_ratio: float = 0.5
    budget_ratio: float = 0.20
    layers: tuple[int, int] | None = None
    full_train_layers: tuple[int, ...] = (0, 1)
    head_box: str = 'frozen'
    bn_mode: str = 'train'
    train_gates: bool = False
    dann_reach: str = 'auto'
    reach_param_cap: int = 50000
    lr_mult: float = 1.0
    weight_decay: float = 0.0
    allow_unverified: bool = False

    def alpha(self, rank: int) -> float:
        return self.alpha_ratio * rank

    def scale(self, rank: int) -> float:
        return self.alpha(rank) / math.sqrt(rank) if self.rslora else self.alpha(rank) / rank


def validate_peft_config(raw: Mapping[str, Any]) -> PeftConfig:
    """Validate a `peft:` mapping. `enabled` is consumed by `load_peft_config`; unknown keys are errors."""
    if not isinstance(raw, Mapping):
        raise TypeError('peft section must be a mapping')
    known = {f.name for f in fields(PeftConfig)}
    unknown = sorted(set(raw) - known - {'enabled'})
    if unknown:
        raise ValueError(f'unknown peft keys: {unknown}')
    merged = {**{f.name: getattr(PeftConfig(), f.name) for f in fields(PeftConfig)}, **raw}
    merged.pop('enabled', None)

    if merged['method'] != 'lora':
        raise ValueError("method must be 'lora'")
    profile = _choice('profile', merged['profile'], PROFILES)
    rank = _positive_int('rank', merged['rank'])
    rank_set = merged['rank_set']
    if not isinstance(rank_set, (list, tuple)) or not rank_set:
        raise TypeError('rank_set must be a non-empty list')
    rank_set = tuple(_positive_int('rank_set item', r) for r in rank_set)
    min_rank = _positive_int('min_rank', merged['min_rank'])
    if min_rank > min(rank_set):
        raise ValueError('min_rank must not exceed the smallest rank_set entry')
    alpha_ratio = _real('alpha_ratio', merged['alpha_ratio'], 0.0, 64.0, low_open=True)
    rslora = _bool('rslora', merged['rslora'])
    dropout = _real('dropout', merged['dropout'], 0.0, 0.5)
    saving_ratio = _real('saving_ratio', merged['saving_ratio'], 0.0, 1.0, low_open=True)
    budget_ratio = _real('budget_ratio', merged['budget_ratio'], 0.0, 1.0, low_open=True)

    layers = merged['layers']
    if layers is not None:
        if not isinstance(layers, (list, tuple)) or len(layers) != 2 or not all(_is_int(v) for v in layers):
            raise TypeError('layers must be null or [low, high] integers')
        if layers[0] < 0 or layers[0] > layers[1]:
            raise ValueError('layers must satisfy 0 <= low <= high')
        layers = (layers[0], layers[1])

    full_train_layers = merged['full_train_layers']
    if not isinstance(full_train_layers, (list, tuple)) or not all(_is_int(v) and v >= 0 for v in full_train_layers):
        raise TypeError('full_train_layers must be a list of non-negative integers')
    full_train_layers = tuple(sorted(set(full_train_layers)))

    head_box = _choice('head_box', merged['head_box'], HEAD_BOX_POLICIES)
    bn_mode = _choice('bn_mode', merged['bn_mode'], BN_MODES)
    train_gates = _bool('train_gates', merged['train_gates'])
    dann_reach = _choice('dann_reach', merged['dann_reach'], DANN_REACH_MODES)
    reach_param_cap = _positive_int('reach_param_cap', merged['reach_param_cap'])
    lr_mult = _real('lr_mult', merged['lr_mult'], 0.0, 100.0, low_open=True)
    weight_decay = _real('weight_decay', merged['weight_decay'], 0.0, 1.0)
    allow_unverified = _bool('allow_unverified', merged['allow_unverified'])

    return PeftConfig(
        method='lora',
        profile=profile,
        rank=rank,
        rank_set=rank_set,
        min_rank=min_rank,
        alpha_ratio=alpha_ratio,
        rslora=rslora,
        dropout=dropout,
        saving_ratio=saving_ratio,
        budget_ratio=budget_ratio,
        layers=layers,
        full_train_layers=full_train_layers,
        head_box=head_box,
        bn_mode=bn_mode,
        train_gates=train_gates,
        dann_reach=dann_reach,
        reach_param_cap=reach_param_cap,
        lr_mult=lr_mult,
        weight_decay=weight_decay,
        allow_unverified=allow_unverified,
    )


def load_peft_config(path: str | Path) -> PeftConfig | None:
    """Read the optional top-level `peft:` section of an experiment YAML; None when absent or disabled."""
    import yaml

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(raw, dict):
        raise TypeError('experiment config must contain a YAML mapping')
    section = raw.get('peft')
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise TypeError('peft section must be a mapping')
    enabled = section.get('enabled', True)
    if not isinstance(enabled, bool):
        raise TypeError('peft.enabled must be a bool')
    return validate_peft_config(section) if enabled else None
```

- [ ] **Step 2: Create `planner.py`**

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ultralytics.yolo_peft.config import PeftConfig

SCHEMA_VERSION = 1

# Reason codes. E-* and R-* names follow YOLO-PEFT (arXiv 2608.07051, Table 2); X-* are repository extensions.
E_OP_UNSUPPORTED = 'E-OP-UNSUPPORTED'
E_GROUP_RANK = 'E-GROUP-RANK'
E_DEPTHWISE_MIX = 'E-DEPTHWISE-MIX'
E_DFL_FIXED = 'E-DFL-FIXED'
E_HEAD_GEOMETRY = 'E-HEAD-GEOMETRY'
E_FAMILY_UNKNOWN = 'E-FAMILY-UNKNOWN'
E_BUDGET = 'E-BUDGET'
E_EMPTY_PLAN = 'E-EMPTY-PLAN'
R_OUT_OF_SCOPE = 'R-OUT-OF-SCOPE'
X_NO_PRETRAINED = 'X-NO-PRETRAINED'
X_FULL_LAYER = 'X-FULL-LAYER'
X_REINIT = 'X-REINIT'
X_PROFILE_ATTN = 'X-PROFILE-ATTN'
X_USER_LAYERS = 'X-USER-LAYERS'
X_RANK_CAPACITY = 'X-RANK-CAPACITY'
X_NO_SAVING = 'X-NO-SAVING'
X_DANN_UNREACHABLE = 'X-DANN-UNREACHABLE'
X_DANN_REACH_PROMOTE = 'X-DANN-REACH-PROMOTE'

# YOLOv13 is outside the architectures YOLO-PEFT calibrated, so it is accepted only as UNVERIFIED.
FAMILY_STATUS = {'yolov13': 'UNVERIFIED'}
# Modules that exist only in the area-attention / hypergraph parts of YOLOv13.
ATTENTION_CLASSES = frozenset(
    {'AAttn', 'ABlock', 'AdaHyperedgeGen', 'AdaHGConv', 'AdaHGComputation', 'C3AH', 'FuseModule', 'HyperACE'}
)


@dataclass(frozen=True)
class Leaf:
    """One module that directly owns parameters, as seen under `model.model`."""

    name: str  # path below model.model, e.g. '2.cv1.conv'
    layer: int  # top-level layer index
    kind: str  # 'conv2d' | 'linear' | 'norm' | 'param'
    owner: str  # class name of the module that owns the parameters
    ancestors: tuple[str, ...] = ()  # class names from the top-level layer module down to the owner
    c_in: int = 0
    c_out: int = 0
    kernel: tuple[int, int] = (1, 1)
    groups: int = 1
    n_params: int = 0  # parameters owned directly (weight + bias, or norm affine)
    reinit: bool = False  # not restored from the checkpoint (shape mismatch or missing)
    fixed: bool = False  # not trainable by construction (DFL)

    @property
    def weights(self) -> int:
        return self.c_out * (self.c_in // max(self.groups, 1)) * self.kernel[0] * self.kernel[1]

    def adapter_params(self, rank: int) -> int:
        return rank * (self.c_in * self.kernel[0] * self.kernel[1] + self.c_out)

    def rank_cap(self) -> int:
        return min(self.c_out, self.c_in * self.kernel[0] * self.kernel[1]) // 2


@dataclass
class Plan:
    status: str
    family: str
    fingerprint: str
    rank: int
    adapt: dict[str, int] = field(default_factory=dict)
    full_layers: list[int] = field(default_factory=list)
    full: dict[str, str] = field(default_factory=dict)
    refused: list[dict] = field(default_factory=list)
    hooks: dict[int, dict] = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        body = {
            'schema_version': SCHEMA_VERSION,
            'status': self.status,
            'family': self.family,
            'fingerprint': self.fingerprint,
            'rank': self.rank,
            'codes': self.codes,
            'adapt': [{'name': n, 'rank': r} for n, r in sorted(self.adapt.items())],
            'full_layers': self.full_layers,
            'full': [{'name': n, 'reason': r} for n, r in sorted(self.full.items())],
            'refused': self.refused,
            'hooks': {str(k): v for k, v in sorted(self.hooks.items())},
            'counts': self.counts,
            'warnings': self.warnings,
        }
        return json.dumps(body, indent=2, sort_keys=True)


def detect_family(layer_classes: Sequence[str]) -> str:
    names = set(layer_classes)
    if {'HyperACE', 'FullPAD_Tunnel'} <= names:
        return 'yolov13'
    if 'A2C2f' in names:
        return 'yolo12'
    if 'C3k2' in names:
        return 'yolo11'
    return 'unknown'


def upstream_layers(parents: Mapping[int, Sequence[int]], layer: int) -> list[int]:
    seen: set[int] = set()
    stack = [layer]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(parents.get(current, ()))
    return sorted(seen)


def model_fingerprint(leaves: Sequence[Leaf]) -> str:
    rows = [
        [leaf.name, leaf.kind, leaf.owner, leaf.c_in, leaf.c_out, list(leaf.kernel), leaf.groups, leaf.n_params]
        for leaf in leaves
    ]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]


def _refusal(leaf: Leaf, code: str, rule: str) -> dict:
    return {'leaf': leaf.name, 'code': code, 'rule': rule}


def _first_refusal(leaf: Leaf, cfg: PeftConfig, rank: int) -> tuple[str, str] | None:
    """Evaluate the adapter rules in YOLO-PEFT order: operator, semantic, policy, user scope, rank allocation."""
    parts = leaf.name.split('.')
    # 1. operator validity
    if leaf.groups > 1 and leaf.groups == leaf.c_in:
        return E_DEPTHWISE_MIX, 'depthwise convolution cannot take a channel-mixing low-rank update'
    if leaf.groups > 1:
        return E_GROUP_RANK, f'grouped convolution (groups={leaf.groups}) has no single low-rank factorisation'
    # 2. semantic safety
    if leaf.owner == 'DFL' or leaf.fixed:
        return E_DFL_FIXED, 'fixed projection; must keep its constant weights'
    if 'Detect' in leaf.ancestors and len(parts) > 1 and parts[1] == 'cv2':
        return E_HEAD_GEOMETRY, 'box-regression branch is geometry sensitive'
    # 4. architecture policy (profile gate)
    if cfg.profile == 'core' and ATTENTION_CLASSES.intersection(leaf.ancestors):
        return X_PROFILE_ATTN, 'attention / hypergraph convolution is gated behind profile core_attn'
    # 5. user scope, intersected only after the mandatory filters
    if cfg.layers is not None and not cfg.layers[0] <= leaf.layer <= cfg.layers[1]:
        return X_USER_LAYERS, f'layer {leaf.layer} is outside the requested interval {list(cfg.layers)}'
    # 6. rank allocation
    leaf_rank = min(rank, leaf.rank_cap())
    if leaf_rank < cfg.min_rank:
        return X_RANK_CAPACITY, f'rank capacity {leaf.rank_cap()} is below min_rank {cfg.min_rank}'
    if leaf.adapter_params(leaf_rank) >= cfg.saving_ratio * leaf.weights:
        return X_NO_SAVING, (
            f'adapter {leaf.adapter_params(leaf_rank)} params >= {cfg.saving_ratio} x layer {leaf.weights} params'
        )
    return None


def _counted_full(leaf: Leaf, plan_full: Mapping[str, str], full_layers: set[int], cfg: PeftConfig) -> bool:
    if leaf.layer in full_layers or leaf.name in plan_full:
        return True
    if leaf.kind == 'norm' and cfg.bn_mode == 'affine':
        return True
    return leaf.kind == 'param' and leaf.owner == 'FullPAD_Tunnel' and leaf.name.endswith('.gate') and cfg.train_gates


def _trainable_in(
    leaves: Sequence[Leaf], layers: set[int], adapt: Mapping[str, int], full: Mapping[str, str],
    full_layers: set[int], cfg: PeftConfig,
) -> int:
    total = 0
    for leaf in leaves:
        if leaf.layer not in layers:
            continue
        if leaf.name in adapt:
            total += leaf.adapter_params(adapt[leaf.name])
        elif not leaf.fixed and _counted_full(leaf, full, full_layers, cfg):
            total += leaf.n_params
    return total


def _build(
    leaves: Sequence[Leaf], parents: Mapping[int, Sequence[int]], hooks: Sequence[int], cfg: PeftConfig, rank: int,
) -> tuple[Plan, float]:
    full_layers = set(cfg.full_train_layers)
    plan = Plan(status='ACCEPT', family='', fingerprint='', rank=rank, full_layers=sorted(full_layers))
    for leaf in leaves:
        if leaf.kind == 'norm':
            continue  # governed by bn_mode / full_train_layers, never adapted
        if leaf.kind == 'param':
            if not (leaf.owner == 'FullPAD_Tunnel' and leaf.name.endswith('.gate')):
                plan.refused.append(_refusal(leaf, E_OP_UNSUPPORTED, 'direct nn.Parameter is not a Conv2d weight'))
            continue  # FullPAD gates are governed by train_gates
        if leaf.kind == 'linear':
            plan.refused.append(_refusal(leaf, E_OP_UNSUPPORTED, 'Linear adapters are outside the v1 scope'))
            continue
        if leaf.layer in full_layers:
            plan.full[leaf.name] = 'FULL-LAYER'
            plan.refused.append(_refusal(leaf, X_FULL_LAYER, 'layer is in full_train_layers and is trained directly'))
            continue
        if leaf.reinit:
            plan.full[leaf.name] = 'REINIT'
            plan.refused.append(_refusal(leaf, X_REINIT, 'weights were freshly initialised, so they are trained directly'))
            continue
        refusal = _first_refusal(leaf, cfg, rank)
        if refusal is None:
            plan.adapt[leaf.name] = min(rank, leaf.rank_cap())
            continue
        code, rule = refusal
        plan.refused.append(_refusal(leaf, code, rule))
        if code == E_HEAD_GEOMETRY and cfg.head_box == 'full':
            plan.full[leaf.name] = 'HEAD-BOX'

    # DANN reach: every hooked layer needs at least one trainable parameter upstream of (and including) it.
    by_name = {leaf.name: leaf for leaf in leaves}
    for hook in hooks:
        upstream = set(upstream_layers(parents, hook))
        reach = _trainable_in(leaves, upstream, plan.adapt, plan.full, set(plan.full_layers), cfg)
        record = {'upstream_layers': sorted(upstream), 'trainable_params': reach, 'promoted': []}
        if reach == 0:
            if cfg.dann_reach == 'off':
                plan.warnings.append(f'{X_DANN_UNREACHABLE}: hook layer {hook} has no trainable parameter upstream')
            else:
                candidates = sorted(
                    (
                        leaf for leaf in leaves
                        if leaf.layer in upstream and leaf.kind == 'conv2d' and leaf.name not in plan.adapt
                        and not leaf.fixed
                    ),
                    key=lambda leaf: (leaf.n_params, leaf.name),
                )
                if candidates and candidates[0].n_params <= cfg.reach_param_cap:
                    chosen = candidates[0]
                    plan.full[chosen.name] = 'DANN-REACH'
                    record['promoted'].append(chosen.name)
                    record['trainable_params'] = chosen.n_params
                    plan.warnings.append(
                        f'{X_DANN_REACH_PROMOTE}: {chosen.name} ({chosen.n_params} params) is trained directly '
                        f'so the DANN hook at layer {hook} can act'
                    )
                else:
                    plan.codes.append(X_DANN_UNREACHABLE)
                    plan.warnings.append(
                        f'{X_DANN_UNREACHABLE}: no upstream leaf of hook layer {hook} fits reach_param_cap '
                        f'{cfg.reach_param_cap}'
                    )
        plan.hooks[hook] = record

    adapter_total = sum(by_name[n].adapter_params(r) for n, r in plan.adapt.items())
    full_total = sum(
        leaf.n_params for leaf in leaves
        if leaf.name not in plan.adapt and not leaf.fixed and _counted_full(leaf, plan.full, set(plan.full_layers), cfg)
    )
    base_total = sum(leaf.n_params for leaf in leaves)
    ratio = (adapter_total + full_total) / base_total if base_total else 0.0
    plan.counts = {
        'leaves': len(leaves),
        'adapted': len(plan.adapt),
        'full': len(plan.full),
        'refused': len(plan.refused),
        'adapter_params': adapter_total,
        'full_params': full_total,
        'base_params': base_total,
        'trainable_ratio': round(ratio, 6),
        'refused_by_code': {},
    }
    for item in plan.refused:
        by_code = plan.counts['refused_by_code']
        by_code[item['code']] = by_code.get(item['code'], 0) + 1
    return plan, ratio


def plan_placement(
    leaves: Sequence[Leaf],
    parents: Mapping[int, Sequence[int]],
    hooks: Sequence[int],
    family: str,
    pretrained: bool,
    cfg: PeftConfig,
) -> Plan:
    """Deterministic placement plan. A REFUSE plan carries its reason codes and must stop the run."""
    names = [leaf.name for leaf in leaves]
    if len(set(names)) != len(names):
        raise ValueError('leaf names must be unique')
    fingerprint = model_fingerprint(leaves)

    def reject(code: str, message: str, rank: int | None = None) -> Plan:
        plan = Plan(status='REFUSE', family=family, fingerprint=fingerprint, rank=rank or cfg.rank, codes=[code])
        plan.warnings.append(f'{code}: {message}')
        return plan

    if not pretrained:
        return reject(X_NO_PRETRAINED, 'adapters on a randomly initialised backbone are meaningless; supply weights')
    if family not in FAMILY_STATUS:
        return reject(E_FAMILY_UNKNOWN, f"family '{family}' is outside this repository's rule set")
    if FAMILY_STATUS[family] == 'UNVERIFIED' and not cfg.allow_unverified:
        return reject(E_FAMILY_UNKNOWN, f"family '{family}' is unverified by YOLO-PEFT; set allow_unverified: true")

    ranks = sorted({r for r in cfg.rank_set if r <= cfg.rank} | {cfg.rank}, reverse=True)
    plan = None
    for rank in ranks:
        plan, ratio = _build(leaves, parents, hooks, cfg, rank)
        if ratio <= cfg.budget_ratio:
            break
    else:
        refusal = reject(
            E_BUDGET, f'trainable ratio {plan.counts["trainable_ratio"]} exceeds budget_ratio {cfg.budget_ratio} '
            f'at every rank in {ranks}', rank=ranks[-1],
        )
        refusal.counts = plan.counts
        refusal.refused = plan.refused
        return refusal
    plan.family = family
    plan.fingerprint = fingerprint
    if plan.rank != cfg.rank:
        plan.warnings.append(f'budget: rank reduced from {cfg.rank} to {plan.rank}')
    if X_DANN_UNREACHABLE in plan.codes:
        plan.status = 'REFUSE'
        return plan
    if not plan.adapt:
        plan.status = 'REFUSE'
        plan.codes.append(E_EMPTY_PLAN)
        plan.warnings.append(f'{E_EMPTY_PLAN}: no convolution survived the rules; this would be a partial freeze')
        return plan
    plan.status = FAMILY_STATUS[family]
    if plan.status == 'UNVERIFIED':
        plan.warnings.append(f'{R_OUT_OF_SCOPE}: {family} is outside the architectures YOLO-PEFT validated')
    return plan
```

- [ ] **Step 3: Create the planner tests**

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ultralytics.yolo_peft.config import PeftConfig, load_peft_config, validate_peft_config
from ultralytics.yolo_peft.planner import (
    Leaf,
    detect_family,
    model_fingerprint,
    plan_placement,
    upstream_layers,
)

V13_YAML = Path(__file__).resolve().parents[1] / 'cfg' / 'models' / 'v13' / 'yolov13.yaml'


def yaml_parents():
    rows = yaml.safe_load(V13_YAML.read_text(encoding='utf-8'))
    rows = rows['backbone'] + rows['head']
    parents = {}
    for index, (source, _, _, _) in enumerate(rows):
        sources = [source] if isinstance(source, int) else list(source)
        parents[index] = tuple(sorted({index - 1 if s == -1 else s for s in sources if not (s == -1 and index == 0)}))
    return parents


def conv(name, c_in, c_out, k=1, groups=1, ancestors=('Conv',), reinit=False, bias=False, fixed=False, owner='Conv'):
    weights = c_out * (c_in // groups) * k * k
    return Leaf(
        name=name, layer=int(name.split('.')[0]), kind='conv2d', owner=owner, ancestors=ancestors, c_in=c_in,
        c_out=c_out, kernel=(k, k), groups=groups, n_params=weights + (c_out if bias else 0), reinit=reinit,
        fixed=fixed,
    )


def norm(name, channels, ancestors=('Conv',)):
    return Leaf(
        name=name, layer=int(name.split('.')[0]), kind='norm', owner='Conv', ancestors=ancestors, c_in=channels,
        c_out=channels, n_params=2 * channels,
    )


def synthetic_v13_leaves(nc=2):
    """Layers 0-3 use the exact yolov13n shapes; deeper layers are representative of each module pattern."""
    ds = ('DSC3k2', 'ModuleList', 'DSBottleneck', 'DSConv')
    attn = ('A2C2f', 'ModuleList', 'Sequential', 'ABlock', 'AAttn', 'Conv')
    return [
        conv('0.conv', 3, 16, 3), norm('0.bn', 16),
        conv('1.conv', 16, 32, 3, groups=2), norm('1.bn', 32),
        conv('2.cv1.conv', 32, 32, ancestors=('DSC3k2', 'Conv')), norm('2.cv1.bn', 32, ('DSC3k2', 'Conv')),
        conv('2.cv2.conv', 48, 64, ancestors=('DSC3k2', 'Conv')), norm('2.cv2.bn', 64, ('DSC3k2', 'Conv')),
        conv('2.m.0.cv1.dw', 16, 16, 3, groups=16, ancestors=ds, owner='DSConv'),
        conv('2.m.0.cv1.pw', 16, 16, ancestors=ds, owner='DSConv'),
        conv('3.conv', 64, 64, 3, groups=4), norm('3.bn', 64),
        conv('5.dw', 128, 128, 3, groups=128, ancestors=('DSConv',), owner='DSConv'),
        conv('5.pw', 128, 128, ancestors=('DSConv',), owner='DSConv'),
        conv('6.cv1.conv', 128, 64, ancestors=('A2C2f', 'Conv')),
        conv('6.m.0.0.attn.qk.conv', 64, 128, ancestors=attn),
        conv('6.m.0.0.attn.pe.conv', 64, 64, 5, groups=64, ancestors=attn),
        conv('6.m.0.0.mlp.0.conv', 64, 128, ancestors=('A2C2f', 'ModuleList', 'Sequential', 'ABlock', 'Sequential', 'Conv')),
        conv('7.pw', 128, 256, ancestors=('DSConv',), owner='DSConv'),
        conv('9.cv1.conv', 256, 384, ancestors=('HyperACE', 'Conv')),
        Leaf(
            name='9.branch1.m.hgnn.edge_proj.0', layer=9, kind='linear', owner='Sequential',
            ancestors=('HyperACE', 'C3AH', 'AdaHGComputation', 'AdaHGConv', 'Sequential'), c_in=64, c_out=64,
            n_params=64 * 64 + 64,
        ),
        Leaf(
            name='9.branch1.m.hgnn.edge_generator.prototype_base', layer=9, kind='param', owner='AdaHyperedgeGen',
            ancestors=('HyperACE', 'C3AH', 'AdaHGComputation', 'AdaHGConv', 'AdaHyperedgeGen'), n_params=4 * 64,
        ),
        conv('11.channel_adjust.conv', 128, 256, ancestors=('DownsampleConv', 'Conv')),
        Leaf(name='12.gate', layer=12, kind='param', owner='FullPAD_Tunnel', ancestors=('FullPAD_Tunnel',), n_params=1),
        conv('17.cv1.conv', 384, 128, ancestors=('DSC3k2', 'Conv')),
        conv('22.conv', 256, 64, ancestors=('Conv',)),
        conv('24.conv', 64, 64, 3, ancestors=('Conv',)),
        conv('32.cv2.0.0.conv', 64, 64, 3, ancestors=('Detect', 'ModuleList', 'Sequential', 'Conv')),
        conv('32.cv2.0.2', 64, 64, ancestors=('Detect', 'ModuleList', 'Sequential'), bias=True, owner='Sequential'),
        conv('32.cv3.0.0.0.conv', 64, 64, 3, groups=64, ancestors=('Detect', 'ModuleList', 'Sequential', 'Sequential', 'DWConv')),
        conv('32.cv3.0.0.1.conv', 64, 64, ancestors=('Detect', 'ModuleList', 'Sequential', 'Sequential', 'Conv')),
        conv('32.cv3.0.2', 64, nc, ancestors=('Detect', 'ModuleList', 'Sequential'), bias=True, reinit=True, owner='Sequential'),
        conv('32.dfl.conv', 16, 1, ancestors=('Detect', 'DFL'), owner='DFL', fixed=True),
    ]


def make_plan(cfg=None, hooks=(1,), pretrained=True, family='yolov13', leaves=None, parents=None):
    cfg = cfg or PeftConfig(allow_unverified=True)
    return plan_placement(
        leaves or synthetic_v13_leaves(), parents or yaml_parents(), hooks, family, pretrained, cfg
    )


def decisions(plan):
    return {item['leaf']: item['code'] for item in plan.refused}


def test_config_defaults_and_strictness():
    cfg = validate_peft_config({})
    assert cfg.rank == 8 and cfg.rank_set == (4, 8, 16) and cfg.full_train_layers == (0, 1)
    assert cfg.allow_unverified is False and cfg.profile == 'core' and cfg.bn_mode == 'train'
    assert cfg.scale(8) == pytest.approx(16.0 / 8 ** 0.5)
    assert validate_peft_config({'rslora': False}).scale(8) == pytest.approx(2.0)
    for bad in ({'unknown': 1}, {'rank': True}, {'rank': 0}, {'profile': 'x'}, {'layers': [5, 1]}, {'bn_mode': 'eval'},
                {'dropout': 0.9}, {'rslora': 1}, {'rank_set': []}, {'min_rank': 99}, {'method': 'dora'}):
        with pytest.raises((TypeError, ValueError)):
            validate_peft_config(bad)


def test_load_peft_config_reads_optional_section(tmp_path):
    plain = tmp_path / 'plain.yaml'
    plain.write_text('experiment: e1\n', encoding='utf-8')
    assert load_peft_config(plain) is None
    disabled = tmp_path / 'disabled.yaml'
    disabled.write_text('peft:\n  enabled: false\n', encoding='utf-8')
    assert load_peft_config(disabled) is None
    enabled = tmp_path / 'enabled.yaml'
    enabled.write_text('peft:\n  rank: 4\n  allow_unverified: true\n', encoding='utf-8')
    assert load_peft_config(enabled).rank == 4


def test_hook_reach_follows_the_real_layer_graph():
    parents = yaml_parents()
    assert upstream_layers(parents, 1) == [0, 1]
    assert len(upstream_layers(parents, 23)) == 23
    assert len(upstream_layers(parents, 27)) == 28
    assert len(upstream_layers(parents, 31)) == 31
    assert set(upstream_layers(parents, 1)) < set(upstream_layers(parents, 23))


def test_worked_examples_match_the_spec():
    plan = make_plan()
    codes = decisions(plan)
    assert plan.status == 'UNVERIFIED' and plan.rank == 8
    assert plan.adapt['2.cv2.conv'] == 8  # 896 adapter params vs 3072
    assert plan.adapt['5.pw'] == 8 and plan.adapt['7.pw'] == 8
    assert codes['2.cv1.conv'] == 'X-NO-SAVING'  # 512 adapter params == 50% of 1024
    assert codes['2.m.0.cv1.pw'] == 'X-NO-SAVING'  # 256 == 256
    assert codes['2.m.0.cv1.dw'] == 'E-DEPTHWISE-MIX' and codes['5.dw'] == 'E-DEPTHWISE-MIX'
    assert codes['3.conv'] == 'E-GROUP-RANK'
    assert codes['0.conv'] == 'X-FULL-LAYER' and codes['1.conv'] == 'X-FULL-LAYER'
    assert plan.full['0.conv'] == 'FULL-LAYER' and plan.full['1.conv'] == 'FULL-LAYER'
    assert codes['32.dfl.conv'] == 'E-DFL-FIXED'
    assert codes['9.branch1.m.hgnn.edge_proj.0'] == 'E-OP-UNSUPPORTED'
    assert codes['9.branch1.m.hgnn.edge_generator.prototype_base'] == 'E-OP-UNSUPPORTED'
    assert '12.gate' not in codes  # governed by train_gates, not by the adapter rules


def test_rank_16_is_refused_where_it_saves_nothing():
    plan = make_plan(PeftConfig(rank=16, allow_unverified=True, budget_ratio=0.9))
    assert plan.rank == 16
    assert decisions(plan)['2.cv2.conv'] == 'X-NO-SAVING'  # 1792 adapter params >= 50% of 3072
    assert plan.adapt['5.pw'] == 16


def test_rule_order_operator_then_semantic_then_policy_then_scope():
    leaves = synthetic_v13_leaves() + [
        conv('32.cv2.1.0.conv', 64, 64, 3, groups=64, ancestors=('Detect', 'ModuleList', 'Sequential', 'DWConv')),
    ]
    plan = make_plan(leaves=leaves)
    codes = decisions(plan)
    assert codes['32.cv2.1.0.conv'] == 'E-DEPTHWISE-MIX'  # operator filter fires before the semantic filter
    assert codes['32.cv2.0.0.conv'] == 'E-HEAD-GEOMETRY'
    assert codes['32.cv2.0.2'] == 'E-HEAD-GEOMETRY'
    scoped = make_plan(PeftConfig(allow_unverified=True, layers=(6, 8)))
    scoped_codes = decisions(scoped)
    assert scoped_codes['5.dw'] == 'E-DEPTHWISE-MIX'  # mandatory filters before the user interval
    assert scoped_codes['9.cv1.conv'] == 'X-PROFILE-ATTN'  # policy filter also fires before the user interval
    assert scoped_codes['17.cv1.conv'] == 'X-USER-LAYERS'
    assert '5.pw' not in scoped.adapt and '7.pw' in scoped.adapt


def test_attention_profile_gate():
    core = make_plan()
    assert decisions(core)['6.m.0.0.attn.qk.conv'] == 'X-PROFILE-ATTN'
    assert decisions(core)['9.cv1.conv'] == 'X-PROFILE-ATTN'
    assert '6.cv1.conv' in core.adapt  # A2C2f.cv1 is an ordinary convolution
    assert '11.channel_adjust.conv' in core.adapt
    wide = make_plan(PeftConfig(allow_unverified=True, profile='core_attn', budget_ratio=0.9))
    assert '6.m.0.0.attn.qk.conv' in wide.adapt and '9.cv1.conv' in wide.adapt
    assert decisions(wide)['6.m.0.0.attn.pe.conv'] == 'E-DEPTHWISE-MIX'


def test_new_head_is_trained_in_full_and_box_branch_policy():
    plan = make_plan()
    assert plan.full['32.cv3.0.2'] == 'REINIT' and '32.cv3.0.2' not in plan.adapt
    assert '32.cv2.0.0.conv' not in plan.full
    boxed = make_plan(PeftConfig(allow_unverified=True, head_box='full'))
    assert boxed.full['32.cv2.0.0.conv'] == 'HEAD-BOX' and boxed.full['32.cv2.0.2'] == 'HEAD-BOX'


def test_p2_hook_reaches_the_stem_by_default():
    plan = make_plan(hooks=(1,))
    assert plan.hooks[1]['trainable_params'] == 432 + 32 + 2304 + 64
    assert plan.hooks[1]['promoted'] == []


def test_dann_reach_promotes_the_cheapest_upstream_conv():
    plan = make_plan(PeftConfig(allow_unverified=True, full_train_layers=()), hooks=(1,))
    assert plan.status == 'UNVERIFIED'
    assert plan.full['0.conv'] == 'DANN-REACH'
    assert plan.hooks[1]['promoted'] == ['0.conv'] and plan.hooks[1]['trainable_params'] == 432
    assert any(w.startswith('X-DANN-REACH-PROMOTE') for w in plan.warnings)


def test_dann_reach_refusal_and_off_switch():
    refused = make_plan(PeftConfig(allow_unverified=True, full_train_layers=(), reach_param_cap=100), hooks=(1,))
    assert refused.status == 'REFUSE' and 'X-DANN-UNREACHABLE' in refused.codes
    off = make_plan(PeftConfig(allow_unverified=True, full_train_layers=(), dann_reach='off'), hooks=(1,))
    assert off.status == 'UNVERIFIED' and off.hooks[1]['trainable_params'] == 0
    assert any(w.startswith('X-DANN-UNREACHABLE') for w in off.warnings)
    deep = make_plan(PeftConfig(allow_unverified=True, full_train_layers=()), hooks=(23, 27, 31))
    assert all(deep.hooks[h]['trainable_params'] > 0 and deep.hooks[h]['promoted'] == [] for h in (23, 27, 31))


def test_budget_lowers_rank_then_refuses():
    base = make_plan(PeftConfig(allow_unverified=True, budget_ratio=0.9))
    tight = base.counts['trainable_ratio']
    lowered = make_plan(PeftConfig(allow_unverified=True, budget_ratio=tight * 0.8))
    assert lowered.status == 'UNVERIFIED' and lowered.rank == 4
    assert any(w.startswith('budget: rank reduced from 8 to 4') for w in lowered.warnings)
    impossible = make_plan(PeftConfig(allow_unverified=True, budget_ratio=0.0001))
    assert impossible.status == 'REFUSE' and impossible.codes == ['E-BUDGET']


def test_empty_plan_is_refused():
    plan = make_plan(PeftConfig(allow_unverified=True, layers=(40, 41)))
    assert plan.status == 'REFUSE' and 'E-EMPTY-PLAN' in plan.codes


def test_family_and_pretrained_gates():
    assert make_plan(PeftConfig()).codes == ['E-FAMILY-UNKNOWN']  # allow_unverified defaults to False
    assert make_plan(family='yolo12').codes == ['E-FAMILY-UNKNOWN']
    assert make_plan(pretrained=False).codes == ['X-NO-PRETRAINED']
    ok = make_plan()
    assert ok.status == 'UNVERIFIED' and any(w.startswith('R-OUT-OF-SCOPE') for w in ok.warnings)
    assert detect_family(['Conv', 'HyperACE', 'FullPAD_Tunnel']) == 'yolov13'
    assert detect_family(['Conv', 'A2C2f']) == 'yolo12' and detect_family(['C3k2']) == 'yolo11'
    assert detect_family(['Conv']) == 'unknown'


def test_bn_affine_and_gate_accounting():
    base = make_plan(PeftConfig(allow_unverified=True, budget_ratio=0.9))
    leaves = synthetic_v13_leaves()
    norm_params = sum(leaf.n_params for leaf in leaves if leaf.kind == 'norm' and leaf.layer not in (0, 1))
    affine = make_plan(PeftConfig(allow_unverified=True, bn_mode='affine', budget_ratio=0.9))
    assert affine.counts['full_params'] - base.counts['full_params'] == norm_params
    gates = make_plan(PeftConfig(allow_unverified=True, train_gates=True, budget_ratio=0.9))
    assert gates.counts['full_params'] - base.counts['full_params'] == 1


def test_plan_json_is_deterministic_and_keeps_log_order():
    first, second = make_plan(), make_plan()
    assert first.to_json() == second.to_json()
    body = json.loads(first.to_json())
    assert body['schema_version'] == 1 and body['status'] == 'UNVERIFIED'
    logged = [item['leaf'] for item in body['refused']]
    order = [leaf.name for leaf in synthetic_v13_leaves() if leaf.name in set(logged)]
    assert logged == order


def test_fingerprint_tracks_structure():
    leaves = synthetic_v13_leaves()
    changed = list(leaves)
    changed[0] = conv('0.conv', 3, 32, 3)
    assert model_fingerprint(leaves) == model_fingerprint(synthetic_v13_leaves())
    assert model_fingerprint(leaves) != model_fingerprint(changed)
    with pytest.raises(ValueError):
        plan_placement(leaves + [leaves[0]], yaml_parents(), (1,), 'yolov13', True, PeftConfig(allow_unverified=True))
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_yolo_peft_planner.py -q`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add yolo_peft/__init__.py yolo_peft/config.py yolo_peft/planner.py tests/test_yolo_peft_planner.py
git commit -m "Add YOLO-PEFT style adapter placement planner"
```

---

### Task 2: Leaf Extraction And Plan Report CLI

**Files:**
- Create: `yolo_peft/graph.py`
- Create: `scripts/peft_plan.py`

**Interfaces:**
- Consumes: Task 1 planner types; a built `DetectionModel` / `DANNDetectionModel` with `model.model` layers carrying `.f`.
- Produces: `graph.layer_parents`, `graph.layer_classes`, `graph.loaded_state_keys`, `graph.collect_leaves`; CLI `python scripts/peft_plan.py --config <yaml> [--weights W] [--nc N] [--json out.json] [--verbose]` (exit code 1 on REFUSE).

- [ ] **Step 1: Create `graph.py`**

```python
from __future__ import annotations

from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics.yolo_peft.planner import Leaf

_NORMS = (_BatchNorm, nn.GroupNorm, nn.LayerNorm)


def layer_parents(model) -> dict[int, tuple[int, ...]]:
    """Layer graph from the `f` attribute that parse_model stores on every top-level layer of `model.model`."""
    parents = {}
    for index, layer in enumerate(model.model):
        source = layer.f
        sources = [source] if isinstance(source, int) else list(source)
        parents[index] = tuple(sorted({index - 1 if s == -1 else s for s in sources if not (s == -1 and index == 0)}))
    return parents


def layer_classes(model) -> list[str]:
    return [type(layer).__name__ for layer in model.model]


def loaded_state_keys(model, weights) -> set[str]:
    """State-dict keys of `model` that a checkpoint restores (same name and same shape), as BaseModel.load does."""
    if weights is None:
        return set()
    source = weights['model'] if isinstance(weights, dict) else weights
    target = model.state_dict()
    return {key for key, value in source.state_dict().items() if key in target and target[key].shape == value.shape}


def collect_leaves(model, loaded_keys: set[str] | None = None) -> list[Leaf]:
    """Parameter-owning modules under `model.model` in deterministic depth-first order.

    `loaded_keys=None` means every parameter was restored; otherwise a leaf with any parameter outside the set is
    marked `reinit`. Domain classifiers and other modules outside `model.model` are never listed.
    """
    leaves: list[Leaf] = []
    for index, layer in enumerate(model.model):
        _walk(layer, str(index), (), index, loaded_keys, leaves)
    return leaves


def _walk(module, path, parents, layer, loaded_keys, out) -> None:
    own = list(module.named_parameters(recurse=False))
    if own:
        out.extend(_leaves(module, path, parents, layer, own, loaded_keys))
    chain = parents + (type(module).__name__,)
    for name, child in module.named_children():
        _walk(child, f'{path}.{name}', chain, layer, loaded_keys, out)


def _leaves(module, path, parents, layer, own, loaded_keys) -> list[Leaf]:
    def restored(parameter_name: str) -> bool:
        return loaded_keys is None or f'model.{path}.{parameter_name}' in loaded_keys

    if isinstance(module, nn.Conv2d):
        kind, c_in, c_out = 'conv2d', module.in_channels, module.out_channels
        kernel, groups = tuple(module.kernel_size), module.groups
    elif isinstance(module, nn.Linear):
        kind, c_in, c_out, kernel, groups = 'linear', module.in_features, module.out_features, (1, 1), 1
    elif isinstance(module, _NORMS):
        features = own[0][1].shape[0]
        kind, c_in, c_out, kernel, groups = 'norm', features, features, (1, 1), 1
    else:  # the module itself owns bare parameters, e.g. FullPAD_Tunnel.gate: one leaf per parameter
        owner = type(module).__name__
        return [
            Leaf(
                name=f'{path}.{name}', layer=layer, kind='param', owner=owner, ancestors=parents + (owner,),
                n_params=parameter.numel(), reinit=not restored(name), fixed=not parameter.requires_grad,
            )
            for name, parameter in own
        ]
    return [
        Leaf(
            name=path,
            layer=layer,
            kind=kind,
            owner=parents[-1] if parents else type(module).__name__,
            ancestors=parents,
            c_in=c_in,
            c_out=c_out,
            kernel=kernel,
            groups=groups,
            n_params=sum(parameter.numel() for _, parameter in own),
            reinit=not all(restored(name) for name, _ in own),
            fixed=all(not parameter.requires_grad for _, parameter in own),
        )
    ]
```

- [ ] **Step 2: Create `scripts/peft_plan.py`**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.dann import DANNDetectionModel
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.scripts.train_yolov13_dann import load_experiment_config
from ultralytics.yolo_peft import graph
from ultralytics.yolo_peft.config import PeftConfig, load_peft_config
from ultralytics.yolo_peft.planner import detect_family, plan_placement


def build_plan(config_path, weights=None, nc=None):
    """Build the model on CPU exactly as the trainer would and return (plan, leaves)."""
    overrides, dann_config = load_experiment_config(config_path)
    peft_config = load_peft_config(config_path) or PeftConfig(allow_unverified=True)
    if nc is None:
        nc = check_det_dataset(overrides['data'])['nc']
    if dann_config['experiment'] == 'e0':
        model = DetectionModel(overrides['model'], nc=nc, verbose=False)
    else:
        model = DANNDetectionModel(
            overrides['model'],
            nc=nc,
            verbose=False,
            p5_layer=dann_config['p5_layer'],
            domain_feature_layers=dann_config['domain_feature_layers'],
            domain_hidden=dann_config['domain_hidden'],
            domain_dropout=dann_config['domain_dropout'],
            domain_loss_gain=dann_config['domain_loss_gain'],
        )
    weights = weights or overrides.get('pretrained')
    pretrained = None
    if weights:
        pretrained, _ = attempt_load_one_weight(weights)
        model.load(pretrained, verbose=False)
    leaves = graph.collect_leaves(model, graph.loaded_state_keys(model, pretrained))
    plan = plan_placement(
        leaves,
        graph.layer_parents(model),
        tuple(getattr(model, 'domain_feature_layers', ())),
        detect_family(graph.layer_classes(model)),
        pretrained is not None,
        peft_config,
    )
    return plan, leaves


def format_report(plan, leaves, verbose=False):
    counts = plan.counts or {}
    lines = [
        f'status: {plan.status}   family: {plan.family}   rank: {plan.rank}   fingerprint: {plan.fingerprint}',
        f'codes: {plan.codes or "-"}',
    ]
    if counts:
        lines += [
            f'leaves: {counts["leaves"]}   adapted: {counts["adapted"]}   full: {counts["full"]}   '
            f'refused: {counts["refused"]}',
            f'parameters: adapter {counts["adapter_params"]:,} + full {counts["full_params"]:,} '
            f'of {counts["base_params"]:,} = {counts["trainable_ratio"]:.2%}',
            'refusals by code:',
        ]
        lines += [f'  {code:<22} {n}' for code, n in sorted(counts['refused_by_code'].items())]
    for hook, record in sorted(plan.hooks.items()):
        lines.append(
            f'hook layer {hook}: {len(record["upstream_layers"])} upstream layers, '
            f'{record["trainable_params"]:,} trainable params, promoted {record["promoted"] or "-"}'
        )
    lines += [f'warning: {warning}' for warning in plan.warnings]
    if verbose:
        by_name = {leaf.name: leaf for leaf in leaves}
        lines.append('adapted:')
        lines += [
            f'  {name:<44} r={rank:<3} {by_name[name].adapter_params(rank):>8,} of {by_name[name].weights:>9,}'
            for name, rank in sorted(plan.adapt.items())
        ]
        lines.append('trained in full:')
        lines += [f'  {name:<44} {reason}' for name, reason in sorted(plan.full.items())]
        lines.append('refusal log (rule order):')
        lines += [f'  {item["code"]:<22} {item["leaf"]:<44} {item["rule"]}' for item in plan.refused]
    return '\n'.join(lines)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Print the YOLO-PEFT style adapter placement plan for a config.')
    parser.add_argument('--config', type=Path, required=True, help='Experiment YAML (with an optional peft section).')
    parser.add_argument('--weights', default=None, help='Override the pretrained weights named in the YAML.')
    parser.add_argument('--nc', type=int, default=None, help='Class count; defaults to the Source dataset YAML.')
    parser.add_argument('--json', type=Path, default=None, help='Write the full plan as JSON.')
    parser.add_argument('--verbose', action='store_true', help='List every adapted leaf and every refusal.')
    return parser.parse_args(args)


def main(args=None):
    options = parse_args(args)
    plan, leaves = build_plan(options.config, options.weights, options.nc)
    print(format_report(plan, leaves, options.verbose))
    if options.json is not None:
        options.json.write_text(plan.to_json(), encoding='utf-8')
    return 0 if plan.status != 'REFUSE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 3: Check leaf extraction on the real model**

Run: `python scripts/peft_plan.py --config configs/dann/<an existing P2 config>.yaml --verbose`
Expected: status `UNVERIFIED` (YOLOv13), hook layer 1 listing 2 upstream layers and 2,832 trainable parameters, `E-GROUP-RANK` for layers 1 and 3, `E-HEAD-GEOMETRY` for `Detect.cv2`, `X-REINIT` for the new class-prediction convs when `nc != 80`. Without a `peft:` section the CLI uses defaults with `allow_unverified=True`.

- [ ] **Step 4: Commit**

```bash
git add yolo_peft/graph.py scripts/peft_plan.py
git commit -m "Add leaf extraction and plan report CLI"
```

---

### Task 3: LoRA Convolution, State And Merge

**Files:**
- Create: `yolo_peft/lora.py`
- Create: `tests/test_yolo_peft_lora.py`

**Interfaces:**
- Consumes: Task 1 `Plan` and `PeftConfig`.
- Produces: `LoRAConv2d`, `attach_adapters`, `split_adapters`, `check_adapter_meta`, `load_adapter_state`, `merge_adapters`, `apply_trainable_policy`, `regroup_optimizer`, `export_merged_checkpoint`.

- [ ] **Step 1: Create `lora.py`**

```python
from __future__ import annotations

import copy
import dataclasses
import math
import shutil
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics.yolo_peft.config import PeftConfig
from ultralytics.yolo_peft.planner import Plan


class LoRAConv2d(nn.Module):
    """Conv2d plus a rank-r update, y = base(x) + scale * B(A(dropout(x))).

    A reuses the base kernel, stride, padding and dilation; B is 1x1. B starts at zero, so a fresh adapter is an
    identity edit. The wrapper deliberately exposes no `weight`: code that would read or fuse the base weight while
    adapters are unmerged (for example `fuse_conv_and_bn`) fails loudly instead of silently dropping them.
    """

    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float = 0.0, rslora: bool = True) -> None:
        super().__init__()
        if type(base) is not nn.Conv2d:
            raise TypeError(f'LoRAConv2d wraps nn.Conv2d, received {type(base).__name__}')
        if base.groups != 1:
            raise ValueError('E-GROUP-RANK: grouped convolutions cannot take an adapter')
        if base.padding_mode != 'zeros':
            raise ValueError('E-SHAPE: only zero padding is supported')
        limit = min(base.out_channels, base.in_channels * base.kernel_size[0] * base.kernel_size[1])
        if not 1 <= rank <= limit:
            raise ValueError(f'E-SHAPE: rank {rank} must be within [1, {limit}]')
        self.base = base
        self.rank, self.alpha, self.rslora = int(rank), float(alpha), bool(rslora)
        self.scale = self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank
        options = {'device': base.weight.device, 'dtype': base.weight.dtype}
        self.lora_a = nn.Conv2d(
            base.in_channels, self.rank, base.kernel_size, base.stride, base.padding, base.dilation, bias=False,
            **options,
        )
        self.lora_b = nn.Conv2d(self.rank, base.out_channels, 1, bias=False, **options)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    in_channels = property(lambda self: self.base.in_channels)
    out_channels = property(lambda self: self.base.out_channels)
    kernel_size = property(lambda self: self.base.kernel_size)
    stride = property(lambda self: self.base.stride)
    padding = property(lambda self: self.base.padding)
    dilation = property(lambda self: self.base.dilation)
    groups = property(lambda self: self.base.groups)

    @property
    def weight(self):
        raise RuntimeError('unmerged LoRAConv2d has no single weight; call merge_adapters() before fusing or exporting')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale

    def extra_repr(self) -> str:
        return f'rank={self.rank}, scale={self.scale:.4f}'

    @torch.no_grad()
    def merged_conv(self) -> nn.Conv2d:
        """Plain Conv2d with W' = W + scale * (B @ A); the base bias is kept as is."""
        delta = torch.einsum('or,rihw->oihw', self.lora_b.weight[:, :, 0, 0].float(), self.lora_a.weight.float())
        merged = copy.deepcopy(self.base)
        merged.weight.copy_((self.base.weight.float() + delta * self.scale).to(self.base.weight.dtype))
        return merged


def adapter_modules(model: nn.Module) -> list[tuple[str, LoRAConv2d]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, LoRAConv2d)]


def adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [part.weight for _, module in adapter_modules(model) for part in (module.lora_a, module.lora_b)]


def adapter_state_dict(model: nn.Module, copy_tensors: bool = True) -> dict[str, torch.Tensor]:
    state = {}
    for name, module in adapter_modules(model):
        for part in ('lora_a', 'lora_b'):
            weight = getattr(module, part).weight
            state[f'{name}.{part}.weight'] = weight.detach().clone() if copy_tensors else weight
    return state


def _parent(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parent_path, _, attribute = dotted.rpartition('.')
    return (model.get_submodule(parent_path) if parent_path else model), attribute


def attach_adapters(model: nn.Module, plan: Plan, cfg: PeftConfig) -> list[str]:
    """Replace each planned Conv2d under `model.model` with a LoRAConv2d. Names in the plan are relative to model.model."""
    attached = []
    for name, rank in sorted(plan.adapt.items()):
        parent, attribute = _parent(model, f'model.{name}')
        setattr(parent, attribute, LoRAConv2d(parent._modules[attribute], rank, cfg.alpha(rank), cfg.dropout, cfg.rslora))
        attached.append(name)
    return attached


def split_adapters(weights) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    """Strip LoRAConv2d wrappers from a checkpoint model in place.

    Returns (adapter tensors, metadata by module name). A model without adapters yields two empty dicts, so the
    stock `BaseModel.load` sees plain parameter names and never silently drops base weights on resume.
    """
    model = weights['model'] if isinstance(weights, dict) else weights
    found = adapter_modules(model)
    state = adapter_state_dict(model)
    meta = {name: {'rank': m.rank, 'alpha': m.alpha, 'rslora': m.rslora} for name, m in found}
    for name, module in found:
        parent, attribute = _parent(model, name)
        setattr(parent, attribute, module.base)
    return state, meta


def check_adapter_meta(plan: Plan, meta: dict[str, dict]) -> None:
    expected = {f'model.{name}': rank for name, rank in plan.adapt.items()}
    found = {name: item['rank'] for name, item in meta.items()}
    if expected != found:
        different = sorted(set(expected.items()) ^ set(found.items()))[:6]
        raise ValueError(f'adapter layout in the checkpoint differs from the current plan, for example {different}')


def load_adapter_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Strict adapter restore: every key must match, nothing is skipped silently."""
    live = adapter_state_dict(model, copy_tensors=False)
    missing, unexpected = sorted(set(live) - set(state)), sorted(set(state) - set(live))
    if missing or unexpected:
        raise KeyError(f'adapter state mismatch: missing {missing[:3]}, unexpected {unexpected[:3]}')
    with torch.no_grad():
        for key, tensor in state.items():
            if live[key].shape != tensor.shape:
                raise ValueError(f'adapter {key}: checkpoint shape {tuple(tensor.shape)} != {tuple(live[key].shape)}')
            live[key].copy_(tensor.to(live[key].dtype))


def merge_adapters(model: nn.Module) -> int:
    """Fold every adapter into its base conv and remove the wrappers, in place. Returns how many were merged."""
    found = adapter_modules(model)
    for name, module in found:
        parent, attribute = _parent(model, name)
        setattr(parent, attribute, module.merged_conv())
    return len(found)


def dann_modules(model: nn.Module) -> list[nn.Module]:
    return [getattr(model, name) for name in ('domain_classifier', 'extra_domain_classifiers') if hasattr(model, name)]


def apply_trainable_policy(
    model: nn.Module, plan: Plan, cfg: PeftConfig, extra: Iterable[nn.Module] = ()
) -> dict[str, int]:
    """Set requires_grad for every parameter from the plan; call after the stock freeze block has run."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter_parameters(model):
        parameter.requires_grad_(True)
    for index in plan.full_layers:
        for parameter in model.model[index].parameters():
            parameter.requires_grad_(True)
    for name, reason in plan.full.items():
        if reason != 'FULL-LAYER':
            for parameter in model.model.get_submodule(name).parameters():
                parameter.requires_grad_(True)
    if cfg.bn_mode == 'affine':
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    if cfg.train_gates:
        for module in model.modules():
            if type(module).__name__ == 'FullPAD_Tunnel':
                module.gate.requires_grad_(True)
    for module in extra:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if '.dfl' in name:  # the stock trainer always freezes DFL
            parameter.requires_grad_(False)
    adapters = sum(parameter.numel() for parameter in adapter_parameters(model))
    return {
        'total': sum(parameter.numel() for parameter in model.parameters()),
        'trainable': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'adapters': adapters,
    }


def regroup_optimizer(optimizer: torch.optim.Optimizer, model: nn.Module, cfg: PeftConfig) -> dict[str, int]:
    """Drop frozen and adapter parameters from the stock groups and append one adapter group.

    Group 0 stays the bias group, which the stock warm-up treats specially (warmup_bias_lr); the adapter group is
    appended last so it follows the normal warm-up and schedule with lr = lr0 * lr_mult.
    """
    adapters = adapter_parameters(model)
    adapter_ids = {id(parameter) for parameter in adapters}
    for group in optimizer.param_groups:
        group['params'] = [p for p in group['params'] if p.requires_grad and id(p) not in adapter_ids]
    optimizer.add_param_group(
        {'params': adapters, 'lr': optimizer.defaults['lr'] * cfg.lr_mult, 'weight_decay': cfg.weight_decay}
    )
    return {
        'groups': len(optimizer.param_groups),
        'adapter_params': sum(parameter.numel() for parameter in adapters),
        'optimized_params': sum(p.numel() for group in optimizer.param_groups for p in group['params']),
    }


def _prediction(output) -> torch.Tensor:
    return output[0] if isinstance(output, (list, tuple)) else output


def export_merged_checkpoint(path, plan_json: str, cfg: PeftConfig, imgsz: int = 128, tolerance: float = 1e-4) -> dict:
    """Rewrite a training checkpoint as a plain, fusable, exportable model after verifying the merge.

    Keeps `<stem>_peft.pt` (the unmerged training checkpoint) and `<stem>_adapter.pt` (adapter tensors and plan).
    Raises before touching `path` if the merged model disagrees with the adapted one on a fixed probe.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = (checkpoint.get('ema') or checkpoint['model']).float().eval()
    if not adapter_modules(model):
        raise RuntimeError(f'{path} contains no LoRA adapters')
    probe = torch.randn(1, 3, imgsz, imgsz, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        before = _prediction(model(probe)).clone()
    adapters = adapter_state_dict(model)
    merged = merge_adapters(model)
    with torch.no_grad():
        after = _prediction(model(probe))
    error = float((before - after).abs().max() / (before.abs().max() + 1e-12))
    if error > tolerance:
        raise RuntimeError(f'merged model deviates from the adapted model: relative error {error:.3e} > {tolerance}')
    shutil.copyfile(path, path.with_name(f'{path.stem}_peft{path.suffix}'))
    torch.save(
        {'adapters': adapters, 'plan': plan_json, 'config': dataclasses.asdict(cfg)},
        path.with_name(f'{path.stem}_adapter{path.suffix}'),
    )
    checkpoint['ema'] = model.half()
    checkpoint['model'] = None
    checkpoint['optimizer'] = None
    checkpoint['peft'] = {'merged': merged, 'relative_error': error, 'plan': plan_json}
    torch.save(checkpoint, path)
    return {'merged': merged, 'relative_error': error}
```

- [ ] **Step 2: Write real-torch tests**

Create `tests/test_yolo_peft_lora.py` with at least these cases (use small `nn.Sequential` / hand-built modules; no GPU):

1. A fresh `LoRAConv2d` equals its base conv on random input (B is zero).
2. After randomising `lora_b`, `merged_conv()(x)` matches the wrapper output within `1e-5` for kernels 1 and 3, stride 1 and 2, with and without bias.
3. `LoRAConv2d(...).weight` raises `RuntimeError`; `fuse_conv_and_bn` on a `Conv` whose conv is wrapped fails loudly.
4. Grouped conv, non-zero padding mode and rank above the limit raise `ValueError`.
5. `attach_adapters` -> `split_adapters` -> `attach_adapters` -> `load_adapter_state` round-trips tensors; a missing or extra key raises `KeyError`; a changed rank in the plan raises `ValueError` from `check_adapter_meta`.
6. `apply_trainable_policy` leaves `.dfl` frozen, trains adapters, layers 0-1 and DANN modules, and nothing else under the default config.
7. `regroup_optimizer` keeps the stock group order, removes frozen and adapter parameters from the stock groups, and appends one adapter group with `lr == defaults['lr'] * lr_mult`.
8. `export_merged_checkpoint` on a small saved checkpoint: relative error below `1e-4`, wrappers gone, `<stem>_peft.pt` and `<stem>_adapter.pt` exist, `checkpoint['model'] is None`, and the merged EMA model passes `fuse()`.

- [ ] **Step 3: Run**

Run: `python -m pytest tests/test_yolo_peft_lora.py -q`
Expected: all pass. Fix `lora.py` if real PyTorch behaves differently from the authoring sandbox (for example dtype handling in `merged_conv`, or `state_dict` key order).

- [ ] **Step 4: Commit**

```bash
git add yolo_peft/lora.py tests/test_yolo_peft_lora.py
git commit -m "Add LoRA convolution adapters with strict state and verified merge"
```

---

### Task 4: Route The Training Script

**Files:**
- Modify: `scripts/train_yolov13_dann.py:13-14,24-29,61-66`
- Test: `tests/test_yolov13_dann.py` (next to the existing `select_trainer` / `build_trainer` tests)

**Interfaces:**
- Consumes: `load_peft_config`, `PEFTDANNDetectionTrainer`.
- Produces: `select_trainer(experiment, peft=False)`; `build_trainer` passes `peft_config=` only when the YAML has an enabled `peft:` section.

- [ ] **Step 1: Add the import**

After the existing `from ultralytics.models.yolo.detect.train_dann import ...` line add:

```python
from ultralytics.yolo_peft.config import load_peft_config
```

- [ ] **Step 2: Replace `select_trainer` and `build_trainer`**

```python
def select_trainer(experiment, peft=False):
    if experiment == 'e0':
        if peft:
            raise ValueError('peft is supported for e0bn, e05, and e1 only')
        return DetectionTrainer
    if experiment in {'e0bn', 'e05', 'e1'}:
        if peft:
            from ultralytics.yolo_peft.trainer import PEFTDANNDetectionTrainer

            return PEFTDANNDetectionTrainer
        return DANNDetectionTrainer
    raise ValueError('experiment must be one of e0, e0bn, e05, or e1')
```

```python
def build_trainer(config_path):
    overrides, dann_config = load_experiment_config(config_path)
    peft_config = load_peft_config(config_path)
    trainer_type = select_trainer(dann_config['experiment'], peft=peft_config is not None)
    if trainer_type is DetectionTrainer:
        return trainer_type(overrides=overrides)
    if peft_config is not None:
        return trainer_type(dann_config=dann_config, overrides=overrides, peft_config=peft_config)
    return trainer_type(dann_config=dann_config, overrides=overrides)
```

`load_experiment_config` ignores unknown top-level keys, so the `peft:` section does not disturb it. The lazy import keeps the module importable when `yolo_peft/trainer.py` is not needed.

- [ ] **Step 3: Add tests**

Following the existing `build_trainer` tests (monkeypatch the trainer classes in the script module): (a) `select_trainer('e1', peft=True)` returns the PEFT trainer and `select_trainer('e0', peft=True)` raises `ValueError`; (b) a YAML without `peft:` produces exactly the old kwargs `{'dann_config', 'overrides'}`; (c) a YAML with `peft:` produces kwargs including `peft_config` that is a `PeftConfig`; (d) `peft: {enabled: false}` behaves as (b); (e) the `e0` branch still passes only `overrides`.

- [ ] **Step 4: Run**

Run: `python -m pytest tests/test_yolov13_dann.py -q -k "trainer"`
Expected: pass, including the pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_yolov13_dann.py tests/test_yolov13_dann.py
git commit -m "Route PEFT experiments through the DANN training script"
```

---

### Task 5: Trainer Mixin

**Files:**
- Create: `yolo_peft/trainer.py`
- Create: `tests/test_yolo_peft_trainer.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: `PEFTTrainerMixin`, `PEFTDANNDetectionTrainer`, `PEFTPlanRefused`.

- [ ] **Step 1: Create `trainer.py`**

```python
from __future__ import annotations

from pathlib import Path

from ultralytics.models.yolo.detect.train_dann import DANNDetectionTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.yolo_peft import graph, lora
from ultralytics.yolo_peft.config import PeftConfig
from ultralytics.yolo_peft.planner import detect_family, plan_placement


class PEFTPlanRefused(RuntimeError):
    """The planner refused the request; the message lists the reason codes."""


class PEFTTrainerMixin:
    """Adapter-aware training hooks layered over a Detection trainer; place it first in the MRO."""

    def __init__(self, *args, peft_config: PeftConfig | None = None, **kwargs) -> None:
        if not isinstance(peft_config, PeftConfig):
            raise TypeError('peft_config must be a PeftConfig (DDP child processes cannot receive it)')
        self.peft_config = peft_config
        self.peft_plan = None
        super().__init__(*args, **kwargs)

    def _setup_train(self, world_size):
        if world_size > 1:
            raise NotImplementedError('PEFT training is validated on a single GPU only')
        super()._setup_train(world_size)

    def get_model(self, cfg=None, weights=None, verbose=True):
        # On resume `weights` is the saved EMA model with LoRA wrappers: strip them first so that the stock
        # BaseModel.load sees plain parameter names, then restore the adapters strictly.
        adapters, meta = ({}, {}) if weights is None else lora.split_adapters(weights)
        model = super().get_model(cfg, weights, verbose)
        layers = len(model.model)
        if any(index >= layers for index in self.peft_config.full_train_layers):
            raise ValueError(f'full_train_layers {self.peft_config.full_train_layers} exceed the {layers} model layers')
        loaded = graph.loaded_state_keys(model, weights)
        plan = plan_placement(
            graph.collect_leaves(model, loaded),
            graph.layer_parents(model),
            tuple(getattr(model, 'domain_feature_layers', ())),
            detect_family(graph.layer_classes(model)),
            weights is not None,
            self.peft_config,
        )
        if RANK in {-1, 0}:
            (Path(self.save_dir) / 'peft_plan.json').write_text(plan.to_json(), encoding='utf-8')
        for warning in plan.warnings:
            LOGGER.warning(f'peft: {warning}')
        if plan.status == 'REFUSE':
            raise PEFTPlanRefused(f'PEFT plan refused with {plan.codes}; see peft_plan.json and scripts/peft_plan.py')
        lora.attach_adapters(model, plan, self.peft_config)
        if adapters:
            lora.check_adapter_meta(plan, meta)
            lora.load_adapter_state(model, adapters)
        self.peft_plan = plan
        LOGGER.info(
            f'peft: status={plan.status} rank={plan.rank} adapted={plan.counts["adapted"]} '
            f'full={plan.counts["full"]} trainable_ratio={plan.counts["trainable_ratio"]}'
        )
        return model

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        # The stock freeze block in _setup_train has already run and would have undone any earlier freezing, so the
        # trainable set is applied here, immediately before the groups are built.
        target = de_parallel(model)
        counts = lora.apply_trainable_policy(target, self.peft_plan, self.peft_config, lora.dann_modules(target))
        optimizer = super().build_optimizer(
            model=model, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations
        )
        groups = lora.regroup_optimizer(optimizer, target, self.peft_config)
        LOGGER.info(
            f'peft: trainable {counts["trainable"]:,}/{counts["total"]:,} parameters '
            f'({counts["adapters"]:,} adapter), {groups["groups"]} optimizer groups, '
            f'{groups["optimized_params"]:,} optimized'
        )
        return optimizer

    def final_eval(self):
        if RANK in {-1, 0}:
            for checkpoint in (self.last, self.best):
                if Path(checkpoint).exists():
                    result = lora.export_merged_checkpoint(checkpoint, self.peft_plan.to_json(), self.peft_config)
                    LOGGER.info(
                        f'peft: merged {result["merged"]} adapters into {checkpoint} '
                        f'(relative error {result["relative_error"]:.2e})'
                    )
        super().final_eval()


class PEFTDANNDetectionTrainer(PEFTTrainerMixin, DANNDetectionTrainer):
    """DANN trainer (experiments e0bn, e05, e1) with LoRA adapters placed by the YOLO-PEFT style planner."""
```

- [ ] **Step 2: Write the tests**

Use a tiny stand-in base class (not the real trainer) that records the call order, to prove: `get_model` strips wrappers before the base `get_model`, writes `peft_plan.json`, raises `PEFTPlanRefused` on a REFUSE plan, and restores adapters on resume; `build_optimizer` applies the trainable policy *before* calling the base builder and regroups afterwards; `final_eval` exports merged checkpoints before the base `final_eval`; `world_size > 1` raises; a non-`PeftConfig` argument raises `TypeError`.

- [ ] **Step 3: Run**

Run: `python -m pytest tests/test_yolo_peft_trainer.py tests/test_yolo_peft_lora.py tests/test_yolo_peft_planner.py -q`
Expected: pass.

- [ ] **Step 4: Verify against the real trainer source**

Read `engine/trainer.py` and confirm: the freeze block in `_setup_train` runs before `build_optimizer`; `build_optimizer` is called with keyword arguments `model, name, lr, momentum, decay, iterations`; `final_eval` is the method that triggers validation of `best.pt`; `save_model` stores the EMA under `ema`. If any signature differs on the installed Ultralytics version, adjust the mixin, not the base class.

- [ ] **Step 5: Commit**

```bash
git add yolo_peft/trainer.py tests/test_yolo_peft_trainer.py
git commit -m "Add PEFT trainer mixin for DANN experiments"
```

---

### Task 6: DANN Reach And Diagnostics

**Files:**
- Modify: `yolo_peft/trainer.py` (logging only)
- Test: `tests/test_yolo_peft_trainer.py`

**Interfaces:**
- Consumes: `plan.hooks`, the trainable policy counts.

- [ ] **Step 1: Log reach at the start of training**

For every hook in `plan.hooks` log `layer, upstream layer count, trainable parameters, promoted leaves`. With the default config the P2 line must read: hook 1, 2 upstream layers, 2,832 trainable parameters.

- [ ] **Step 2: After `apply_trainable_policy`, assert reach on the live model**

For each DANN hook layer, compute the parameters of the upstream layers (`graph.layer_parents` closure) that have `requires_grad=True`; if the count is zero and `dann_reach != 'off'`, raise `RuntimeError` naming the hook. This guards against the plan and the live trainable set drifting apart.

- [ ] **Step 3: Test and commit**

Add a test with a synthetic model where the stem is frozen: the assertion fires. Run the three PEFT test files, then:

```bash
git add yolo_peft/trainer.py tests/test_yolo_peft_trainer.py
git commit -m "Log and enforce DANN hook reach under PEFT"
```

---

### Task 7: Experiment Configurations And Runbook

**Files:**
- Create: `configs/peft/ic_p2_r0_full_seed42.yaml` .. `ic_p2_r6_core_attn_seed42.yaml` (7 files)
- Create: `docs/experiments/yolov13-lora-peft-runbook.md`

**Interfaces:**
- Consumes: the experiment matrix in the design document.

- [ ] **Step 1: Read the effective optimizer from an existing run**

Open a previous P2 run log and copy the `optimizer: AdamW(lr=..., momentum=...)` line. Use those values as `optimizer`, `lr0`, `momentum` in every arm below, together with `warmup_bias_lr: 0.0` (that is what `optimizer=auto` forced).

- [ ] **Step 2: Create the seed 42 configs**

Copy every path field (`source_data`, `target_data`, `model`, `weights`, `project`) and the `train` / `dann` sections from `configs/dann/ic_p2_a0_closemosaic_seed42.yaml`, add the pinned optimizer keys to `train`, and set `train.name` per arm. R0 has no `peft:` section. The other arms add:

```yaml
# R1 (experiment: e1)
peft:
  profile: core
  rank: 8
  allow_unverified: true
# R2: same as R1 with experiment: e05
# R3: R1 with rank: 4
# R4: R1 with head_box: full
# R5: R1 with bn_mode: affine
# R6: R1 with profile: core_attn
```

Create seeds 43 and 44 only after seed 42 shows a non-catastrophic result for the arm.

- [ ] **Step 3: Dry-run every config**

Run: `python scripts/peft_plan.py --config configs/peft/<file>.yaml --json <arm>_plan.json`
Expected: R0 is not applicable (no `peft:`); R1-R6 print `UNVERIFIED` with the P2 hook reached; record the trainable ratio for each arm in the runbook.

- [ ] **Step 4: Write the runbook**

Record the run order (R0, R1, R2, then R3-R6), the decision rules from the design document (catastrophe gate and non-inferiority thresholds, confirmed with the owner before the first run), the command `python scripts/train_yolov13_dann.py --config <file>`, where `peft_plan.json` and the merged checkpoints appear, and the TP/FP/FN table to fill at conf=0.40.

- [ ] **Step 5: Commit**

```bash
git add configs/peft docs/experiments/yolov13-lora-peft-runbook.md
git commit -m "Add LoRA experiment configs and runbook"
```

---

### Task 8: Verification On Real PyTorch

**Files:** none created.

- [ ] **Step 1: Full test suite**

Run: `python -m pytest tests -q`
Expected: all pre-existing tests still pass, plus the new PEFT tests. Any failure in `graph.py`, `lora.py`, `trainer.py` or `scripts/peft_plan.py` is expected-class work: fix and re-run.

- [ ] **Step 2: Plan check on the real architecture**

Run `scripts/peft_plan.py` for the R1 config with `--verbose`. Confirm: layers 1 and 3 are `E-GROUP-RANK`; layer 0 and 1 are `X-FULL-LAYER`; every `Detect.cv2` conv is `E-HEAD-GEOMETRY`; the attention leaves are `X-PROFILE-ATTN`; the trainable ratio is below `budget_ratio`; `peft_plan.json` is identical on a second run (deterministic).

- [ ] **Step 3: One-epoch GPU smoke run**

Run the R1 config with `epochs: 1` into a throw-away project directory. Confirm in the log: `peft: status=UNVERIFIED`, the trainable parameter line, a non-zero domain loss for `e1`, a merged-checkpoint line with relative error below `1e-4`, and a final evaluation that completes (no `weight` error from fusing).

- [ ] **Step 4: Resume check**

Interrupt a two-epoch run after epoch 1 and resume it. Confirm the adapter layout check passes and the resumed loss continues from the saved value rather than restarting.

- [ ] **Step 5: Merge equivalence on a trained checkpoint**

Load `<stem>_peft.pt` and the merged `.pt`, run the same validation batch through both in eval mode, and confirm the outputs agree within tolerance.

- [ ] **Step 6: Report**

Record the results, any code changes needed to make the torch-dependent files pass, and the open inputs from the design document in the runbook.
