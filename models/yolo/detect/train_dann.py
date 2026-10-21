from __future__ import annotations

import json
import math
import os
from numbers import Real

import torch
from torch.nn import functional as F

from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.dann import DANNDetectionModel, dann_lambda
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOCAL_RANK, RANK


DEFAULT_DANN_CONFIG = {
    'target_data': None,
    'target_batch': None,
    'target_supervised': False,
    'source_style_aug': False,
    'source_style_prob': 0.50,
    'source_style_alpha': 0.30,
    'supervised_batch': None,
    'p5_layer': 31,
    'domain_feature_layers': None,
    'domain_hidden': 256,
    'domain_dropout': 0.1,
    'domain_loss_gain': 1.0,
    'lambda_max': 1.0,
    'gamma': 10.0,
    'warmup_fraction': 0.0,
}
DANN_CONFIG_ENV = 'ULTRALYTICS_DANN_CONFIG'


def _finite_number(name, value):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f'{name} must be a real scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def validate_dann_config(config):
    if not isinstance(config, dict):
        raise TypeError('dann_config must be a dictionary')
    merged = {**DEFAULT_DANN_CONFIG, **config}
    experiment = merged.get('experiment')
    if experiment not in {'e0bn', 'e05', 'e1'}:
        raise ValueError('experiment must be e0bn, e05, or e1 for DANNDetectionTrainer')
    if experiment in {'e05', 'e1'} and (
        not isinstance(merged['target_data'], (str, bytes)) or not merged['target_data']
    ):
        raise ValueError('target_data is required')
    if merged['p5_layer'] not in DANNDetectionModel.VERIFIED_DOMAIN_FEATURE_LAYERS:
        raise ValueError('p5_layer must be one of the verified YOLOv13 feature layers: 1, 23, 27, or 31')
    domain_feature_layers = merged['domain_feature_layers']
    if domain_feature_layers is None:
        domain_feature_layers = (merged['p5_layer'],)
    else:
        if not isinstance(domain_feature_layers, (list, tuple)):
            raise TypeError('domain_feature_layers must be a list or tuple')
        domain_feature_layers = tuple(domain_feature_layers)
        if not domain_feature_layers:
            raise ValueError('domain_feature_layers must not be empty')
        if any(not isinstance(layer, int) or isinstance(layer, bool) for layer in domain_feature_layers):
            raise TypeError('domain_feature_layers must contain integers')
        if len(set(domain_feature_layers)) != len(domain_feature_layers):
            raise ValueError('domain_feature_layers must contain unique layers')
        if any(layer not in DANNDetectionModel.VERIFIED_DOMAIN_FEATURE_LAYERS for layer in domain_feature_layers):
            raise ValueError('domain_feature_layers contains an unverified YOLOv13 feature layer')
        if 'p5_layer' in config and config['p5_layer'] != domain_feature_layers[0]:
            raise ValueError('p5_layer and domain_feature_layers conflict')
    merged['p5_layer'] = domain_feature_layers[0]
    merged['domain_feature_layers'] = domain_feature_layers
    for name in ('domain_hidden',):
        if not isinstance(merged[name], int) or isinstance(merged[name], bool) or merged[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    target_batch = merged['target_batch']
    if target_batch is not None and (
        not isinstance(target_batch, int) or isinstance(target_batch, bool) or target_batch <= 0
    ):
        raise ValueError('target_batch must be a positive integer or null')
    if not isinstance(merged['target_supervised'], bool):
        raise TypeError('target_supervised must be a bool')
    if not isinstance(merged['source_style_aug'], bool):
        raise TypeError('source_style_aug must be a bool')
    supervised_batch = merged['supervised_batch']
    if supervised_batch is not None and (
        not isinstance(supervised_batch, int) or isinstance(supervised_batch, bool) or supervised_batch <= 0
    ):
        raise ValueError('supervised_batch must be a positive integer or null')
    if experiment == 'e0bn' and supervised_batch is None:
        raise ValueError('supervised_batch is required for e0bn')
    dropout = _finite_number('domain_dropout', merged['domain_dropout'])
    gain = _finite_number('domain_loss_gain', merged['domain_loss_gain'])
    lambda_max = _finite_number('lambda_max', merged['lambda_max'])
    gamma = _finite_number('gamma', merged['gamma'])
    warmup_fraction = _finite_number('warmup_fraction', merged['warmup_fraction'])
    source_style_prob = _finite_number('source_style_prob', merged['source_style_prob'])
    source_style_alpha = _finite_number('source_style_alpha', merged['source_style_alpha'])
    if not 0.0 <= dropout < 1.0:
        raise ValueError('domain_dropout must be within [0, 1)')
    if gain < 0.0:
        raise ValueError('domain_loss_gain must be non-negative')
    if not 0.0 <= lambda_max <= 1.0:
        raise ValueError('lambda_max must be within [0, 1]')
    if gamma <= 0.0:
        raise ValueError('gamma must be greater than zero')
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError('warmup_fraction must be within [0, 1)')
    if not 0.0 <= source_style_prob <= 1.0:
        raise ValueError('source_style_prob must be within [0, 1]')
    if not 0.0 <= source_style_alpha <= 1.0:
        raise ValueError('source_style_alpha must be within [0, 1]')
    merged.update(
        domain_dropout=dropout,
        domain_loss_gain=gain,
        lambda_max=lambda_max,
        gamma=gamma,
        warmup_fraction=warmup_fraction,
        source_style_prob=source_style_prob,
        source_style_alpha=source_style_alpha,
    )
    return merged


class DANNDetectionTrainer(DetectionTrainer):
    def __init__(self, dann_config=None, *args, **kwargs) -> None:
        if dann_config is None:
            serialized = os.environ.get(DANN_CONFIG_ENV)
            if serialized is None:
                raise ValueError('dann_config is required')
            dann_config = json.loads(serialized)
        self.dann_config = validate_dann_config(dann_config)
        self.target_loader = None
        self._target_iterator = None
        self._target_epoch = None
        self._dann_step = 0
        super().__init__(*args, **kwargs)

    def train(self):
        previous = os.environ.get(DANN_CONFIG_ENV)
        os.environ[DANN_CONFIG_ENV] = json.dumps(self.dann_config)
        try:
            return super().train()
        finally:
            if previous is None:
                os.environ.pop(DANN_CONFIG_ENV, None)
            else:
                os.environ[DANN_CONFIG_ENV] = previous

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DANNDetectionModel(
            cfg,
            nc=self.data['nc'],
            verbose=verbose and RANK == -1,
            p5_layer=self.dann_config['p5_layer'],
            domain_feature_layers=self.dann_config['domain_feature_layers'],
            domain_hidden=self.dann_config['domain_hidden'],
            domain_dropout=self.dann_config['domain_dropout'],
            domain_loss_gain=self.dann_config['domain_loss_gain'],
        )
        if weights:
            model.load(weights)
        return model

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        e0bn_train = self.dann_config['experiment'] == 'e0bn' and mode == 'train'
        loader = super().get_dataloader(
            dataset_path, batch_size=batch_size * 2 if e0bn_train else batch_size, rank=rank, mode=mode
        )
        if e0bn_train:
            batch_sampler = getattr(getattr(loader, 'batch_sampler', None), 'sampler', None)
            if batch_sampler is None or not hasattr(batch_sampler, 'drop_last'):
                raise TypeError('e0bn requires a mutable batch sampler with drop_last support')
            batch_sampler.drop_last = True
            if hasattr(loader, 'reset'):
                loader.reset()
        return loader

    def _build_target_loader(self, target_data, target_batch):
        source_data = self.data
        try:
            self.data = target_data
            return self.get_dataloader(target_data['train'], batch_size=target_batch, rank=LOCAL_RANK, mode='train')
        finally:
            self.data = source_data

    @staticmethod
    def _target_batch_per_rank(target_batch, world_size):
        if world_size <= 1:
            return target_batch
        if target_batch < world_size:
            raise ValueError('target_batch must be at least the DDP world size')
        if target_batch % world_size:
            raise ValueError('target_batch must be divisible by the DDP world size')
        return target_batch // world_size

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        if self.dann_config['experiment'] == 'e0bn':
            supervised_batch = self._target_batch_per_rank(self.dann_config['supervised_batch'], world_size)
            optimizer_batch = self.batch_size // max(world_size, 1)
            if optimizer_batch != supervised_batch:
                raise ValueError('e0bn requires train batch to equal supervised_batch')
            self._e0bn_supervised_batch = supervised_batch
            self._dann_step = self.start_epoch * len(self.train_loader)
            return
        target_data = check_det_dataset(self.dann_config['target_data'])
        target_batch = self.dann_config['target_batch'] or self.batch_size
        target_batch = self._target_batch_per_rank(target_batch, world_size)
        self.target_loader = self._build_target_loader(target_data, target_batch)
        self._target_iterator = iter(self.target_loader)
        self._target_epoch = None
        self._dann_step = self.start_epoch * len(self.train_loader)

    def _reset_target_iterator(self) -> None:
        sampler = getattr(self.target_loader, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(self.epoch)
        if hasattr(self.target_loader, 'reset'):
            self.target_loader.reset()
        self._target_iterator = iter(self.target_loader)
        self._target_epoch = self.epoch

    def _next_target_batch(self):
        if self._target_epoch != self.epoch:
            self._reset_target_iterator()
        try:
            return next(self._target_iterator)
        except StopIteration:
            self._target_iterator = iter(self.target_loader)
            return next(self._target_iterator)

    @staticmethod
    def _target_style_source(source, target, probability, alpha):
        if probability == 0.0 or alpha == 0.0:
            return source
        dims = (-2, -1)
        source_mean = source.mean(dims, keepdim=True)
        source_std = source.std(dims, keepdim=True, unbiased=False).clamp_min(torch.finfo(source.dtype).eps)
        target = target.detach()
        target_mean = target.mean(dims, keepdim=True)
        target_std = target.std(dims, keepdim=True, unbiased=False)
        indices = torch.arange(source.shape[0], device=source.device) % target.shape[0]
        transferred = (source - source_mean) / source_std
        transferred = transferred * target_std[indices] + target_mean[indices]
        blended = source.lerp(transferred, alpha).clamp(0.0, 1.0)
        selected = torch.rand((source.shape[0], 1, 1, 1), device=source.device) < probability
        return torch.where(selected, blended, source)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self.dann_config['experiment'] == 'e0bn':
            images = batch['img']
            supervised_batch = self._e0bn_supervised_batch
            if images.shape[0] != 2 * supervised_batch:
                raise ValueError('e0bn batches must contain exactly 2 * supervised_batch images')
            batch_indices = batch.get('batch_idx')
            if not isinstance(batch_indices, torch.Tensor):
                raise TypeError('e0bn batches must contain tensor batch_idx annotations')
            supervised_targets = batch_indices < supervised_batch
            for key in ('batch_idx', 'cls', 'bboxes'):
                value = batch.get(key)
                if not isinstance(value, torch.Tensor) or value.shape[0] != supervised_targets.shape[0]:
                    raise ValueError(f'e0bn annotation tensor {key} must align with batch_idx')
                batch[key] = value[supervised_targets]
            batch['img'] = images[:supervised_batch]
            batch['target_img'] = images[supervised_batch:]
            batch['dann_enabled'] = False
            batch['dann_lambda'] = 0.0
            return batch
        target_batch = self._next_target_batch()
        if 'img' not in target_batch:
            raise KeyError('target batches must contain img')
        target_images = target_batch['img'].to(self.device, non_blocking=True).float() / 255
        if target_images.ndim != 4 or target_images.shape[0] == 0:
            raise ValueError('target images must be a non-empty 4D tensor')
        source_images = batch['img']
        if target_images.shape[1] != source_images.shape[1]:
            raise ValueError('source and target images must have matching channels')
        if target_images.shape[2:] != source_images.shape[2:]:
            target_images = F.interpolate(
                target_images, size=source_images.shape[2:], mode='bilinear', align_corners=False
            )
        if self.dann_config['source_style_aug']:
            source_images = self._target_style_source(
                source_images,
                target_images,
                self.dann_config['source_style_prob'],
                self.dann_config['source_style_alpha'],
            )
            batch['img'] = source_images

        total_steps = self.epochs * len(self.train_loader)
        progress = min(self._dann_step / max(total_steps - 1, 1), 1.0)
        warmup_fraction = self.dann_config['warmup_fraction']
        adversarial_progress = max(progress - warmup_fraction, 0.0) / (1.0 - warmup_fraction)
        batch['target_img'] = target_images
        batch['target_supervised'] = self.dann_config['target_supervised']
        if batch['target_supervised']:
            source_batch_size = source_images.shape[0]
            target_batch_indices = target_batch.get('batch_idx')
            if not isinstance(target_batch_indices, torch.Tensor):
                raise TypeError('supervised target batches must contain tensor batch_idx annotations')
            for key in ('cls', 'bboxes'):
                value = target_batch.get(key)
                if not isinstance(value, torch.Tensor) or value.shape[0] != target_batch_indices.shape[0]:
                    raise ValueError(f'supervised target annotation tensor {key} must align with batch_idx')
            batch['batch_idx'] = torch.cat((batch['batch_idx'], target_batch_indices + source_batch_size))
            batch['cls'] = torch.cat((batch['cls'], target_batch['cls']))
            batch['bboxes'] = torch.cat((batch['bboxes'], target_batch['bboxes']))
        batch['dann_enabled'] = self.dann_config['experiment'] == 'e1'
        batch['dann_lambda'] = dann_lambda(
            adversarial_progress,
            lambda_max=self.dann_config['lambda_max'],
            gamma=self.dann_config['gamma'],
        )
        self._dann_step += 1
        return batch

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = (
            'box_loss',
            'cls_loss',
            'dfl_loss',
            'domain_loss',
            'domain_acc',
            'grl_lambda',
        )
        return validator
