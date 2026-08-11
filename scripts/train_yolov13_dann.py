from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.train_dann import DANNDetectionTrainer, validate_dann_config


def _required_string(config, key):
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{key} must be a non-empty string')
    return value


def select_trainer(experiment):
    if experiment == 'e0':
        return DetectionTrainer
    if experiment in {'e0bn', 'e05', 'e1'}:
        return DANNDetectionTrainer
    raise ValueError('experiment must be one of e0, e0bn, e05, or e1')


def load_experiment_config(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(raw, dict):
        raise TypeError('experiment config must contain a YAML mapping')
    experiment = _required_string(raw, 'experiment')
    select_trainer(experiment)
    source_data = _required_string(raw, 'source_data')
    model = _required_string(raw, 'model')
    train = raw.get('train', {})
    dann = raw.get('dann', {})
    if not isinstance(train, dict) or not isinstance(dann, dict):
        raise TypeError('train and dann sections must be mappings')

    overrides = {**train, 'data': source_data, 'model': model}
    weights = raw.get('weights')
    if weights is not None:
        if not isinstance(weights, str) or not weights:
            raise ValueError('weights must be null or a non-empty string')
        overrides['pretrained'] = weights

    dann_config = {**dann, 'experiment': experiment, 'target_data': raw.get('target_data')}
    if experiment != 'e0':
        dann_config = validate_dann_config(dann_config)
    return overrides, dann_config


def build_trainer(config_path):
    overrides, dann_config = load_experiment_config(config_path)
    trainer_type = select_trainer(dann_config['experiment'])
    if trainer_type is DetectionTrainer:
        return trainer_type(overrides=overrides)
    return trainer_type(dann_config=dann_config, overrides=overrides)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Train YOLOv13 with the P5 DANN experiment controls.')
    parser.add_argument('--config', type=Path, required=True, help='Path to an E0, E0-BN, E0.5, or E1 YAML.')
    return parser.parse_args(args)


def main(args=None):
    options = parse_args(args)
    build_trainer(options.config).train()


if __name__ == '__main__':
    main()
