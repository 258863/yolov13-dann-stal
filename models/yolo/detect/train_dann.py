from __future__ import annotations

import math
from numbers import Real

import torch
from torch.nn import functional as F

from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.dann import DANNDetectionModel, dann_lambda
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOCAL_RANK, RANK


DEFAULT_DANN_CONFIG = {
    'target_batch': None,
    'p5_layer': 31,
    'domain_hidden': 256,
    'domain_dropout': 0.1,
    'domain_loss_gain': 1.0,
    'lambda_max': 1.0,
    'gamma': 10.0,
}


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
    if not isinstance(merged.get('target_data'), (str, bytes)) or not merged['target_data']:
        raise ValueError('target_data is required')
    if merged.get('experiment') not in {'e05', 'e1'}:
        raise ValueError('experiment must be e05 or e1 for DANNDetectionTrainer')
    if merged['p5_layer'] != 31:
        raise ValueError('p5_layer must be 31 in the first DANN version')
    for name in ('domain_hidden',):
        if not isinstance(merged[name], int) or isinstance(merged[name], bool) or merged[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    target_batch = merged['target_batch']
    if target_batch is not None and (
        not isinstance(target_batch, int) or isinstance(target_batch, bool) or target_batch <= 0
    ):
        raise ValueError('target_batch must be a positive integer or null')
    dropout = _finite_number('domain_dropout', merged['domain_dropout'])
    gain = _finite_number('domain_loss_gain', merged['domain_loss_gain'])
    lambda_max = _finite_number('lambda_max', merged['lambda_max'])
    gamma = _finite_number('gamma', merged['gamma'])
    if not 0.0 <= dropout < 1.0:
        raise ValueError('domain_dropout must be within [0, 1)')
    if gain < 0.0:
        raise ValueError('domain_loss_gain must be non-negative')
    if not 0.0 <= lambda_max <= 1.0:
        raise ValueError('lambda_max must be within [0, 1]')
    if gamma <= 0.0:
        raise ValueError('gamma must be greater than zero')
    merged.update(
        domain_dropout=dropout,
        domain_loss_gain=gain,
        lambda_max=lambda_max,
        gamma=gamma,
    )
    return merged


class DANNDetectionTrainer(DetectionTrainer):
    def __init__(self, dann_config, *args, **kwargs) -> None:
        self.dann_config = validate_dann_config(dann_config)
        self.target_loader = None
        self._target_iterator = None
        self._target_epoch = None
        self._dann_step = 0
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DANNDetectionModel(
            cfg,
            nc=self.data['nc'],
            verbose=verbose and RANK == -1,
            p5_layer=self.dann_config['p5_layer'],
            domain_hidden=self.dann_config['domain_hidden'],
            domain_dropout=self.dann_config['domain_dropout'],
            domain_loss_gain=self.dann_config['domain_loss_gain'],
        )
        if self.dann_config['experiment'] == 'e05':
            model.domain_classifier.requires_grad_(False)
        if weights:
            model.load(weights)
        return model

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        target_data = check_det_dataset(self.dann_config['target_data'])
        target_batch = self.dann_config['target_batch'] or self.batch_size
        target_batch = max(target_batch // max(world_size, 1), 1)
        self.target_loader = self.get_dataloader(
            target_data['train'], batch_size=target_batch, rank=LOCAL_RANK, mode='train'
        )
        self._target_iterator = iter(self.target_loader)
        self._target_epoch = None
        self._dann_step = self.start_epoch * len(self.train_loader)

    def _reset_target_iterator(self) -> None:
        sampler = getattr(self.target_loader, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(self.epoch)
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

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
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

        total_steps = self.epochs * len(self.train_loader)
        progress = min(self._dann_step / max(total_steps - 1, 1), 1.0)
        batch['target_img'] = target_images
        batch['dann_enabled'] = self.dann_config['experiment'] == 'e1'
        batch['dann_lambda'] = dann_lambda(
            progress, lambda_max=self.dann_config['lambda_max'], gamma=self.dann_config['gamma']
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
