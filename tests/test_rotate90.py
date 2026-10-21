from __future__ import annotations

import random

import numpy as np
import pytest

from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.data.augment import RandomFlip, RandomRotate90
from ultralytics.utils.instance import Instances


RECT = (100, 200, 140, 230)  # x1, y1, x2, y2: a 40 x 30 px object, deliberately off-centre
SHAPES = [(640, 640, False), (480, 640, False), (480, 640, True)]  # height, width, normalized


def make_labels(height: int, width: int, normalized: bool) -> dict:
    x1, y1, x2, y2 = RECT
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[y1:y2, x1:x2] = 255
    box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    segment = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    if normalized:
        box = box / np.array([width, height, width, height], dtype=np.float32)
        segment = segment / np.array([width, height], dtype=np.float32)
    # RandomPerspective leaves xyxy boxes behind, so RandomRotate90 must not assume any particular box format.
    return {'img': image, 'instances': Instances(box, segment, None, bbox_format='xyxy', normalized=normalized)}


def measured_box(image: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(image[..., 0])
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


def label_box(labels: dict) -> np.ndarray:
    instances = labels['instances']
    instances.convert_bbox(format='xyxy')
    height, width = labels['img'].shape[:2]
    box = instances.bboxes[0].astype(np.float64)
    return box * np.array([width, height, width, height]) if instances.normalized else box


def label_segment_box(labels: dict) -> np.ndarray:
    instances = labels['instances']
    height, width = labels['img'].shape[:2]
    points = instances.segments[0].astype(np.float64)
    if instances.normalized:
        points = points * np.array([width, height])
    return np.array([*points.min(axis=0), *points.max(axis=0)])


@pytest.mark.parametrize('k', [1, 2, 3])
@pytest.mark.parametrize(('height', 'width', 'normalized'), SHAPES)
def test_rotate90_boxes_and_segments_follow_the_rotated_image(k, height, width, normalized) -> None:
    labels = RandomRotate90(p=1.0, fixed_k=k)(make_labels(height, width, normalized))
    expected = measured_box(labels['img'])
    np.testing.assert_allclose(label_box(labels), expected, atol=1e-2)
    np.testing.assert_allclose(label_segment_box(labels), expected, atol=1e-2)


@pytest.mark.parametrize('k', [1, 2, 3])
def test_rotate90_fixed_k_matches_numpy_rot90(k) -> None:
    original = make_labels(480, 640, False)['img']
    labels = RandomRotate90(p=1.0, fixed_k=k)(make_labels(480, 640, False))
    np.testing.assert_array_equal(labels['img'], np.rot90(original, k))


def test_rotate90_random_choice_uses_each_of_the_three_rotations() -> None:
    original = make_labels(480, 640, False)['img']
    random.seed(0)
    seen = set()
    for _ in range(60):
        labels = RandomRotate90(p=1.0)(make_labels(480, 640, False))
        matches = [k for k in (1, 2, 3) if np.array_equal(labels['img'], np.rot90(original, k))]
        assert len(matches) == 1
        seen.add(matches[0])
        np.testing.assert_allclose(label_box(labels), measured_box(labels['img']), atol=1e-2)
    assert seen == {1, 2, 3}


def test_rotate90_zero_probability_is_a_noop_and_consumes_no_random_numbers() -> None:
    labels = make_labels(480, 640, False)
    state = random.getstate()
    result = RandomRotate90(p=0.0)(labels)
    assert random.getstate() == state
    assert result['img'].shape == (480, 640, 3)
    np.testing.assert_allclose(label_box(result), RECT)


def test_rotate90_then_flips_keep_labels_on_the_object() -> None:
    labels = RandomRotate90(p=1.0, fixed_k=1)(make_labels(480, 640, False))
    labels = RandomFlip(p=1.0, direction='vertical')(labels)
    labels = RandomFlip(p=1.0, direction='horizontal')(labels)
    np.testing.assert_allclose(label_box(labels), measured_box(labels['img']), atol=1e-2)


def test_rotate90_fixed_is_a_valid_training_argument() -> None:
    assert DEFAULT_CFG_DICT['rotate90'] == 0.0
    assert DEFAULT_CFG_DICT['rotate90_fixed'] is False
    args = get_cfg(overrides={'rotate90': 0.5, 'rotate90_fixed': True})
    assert args.rotate90 == 0.5
    assert args.rotate90_fixed is True
