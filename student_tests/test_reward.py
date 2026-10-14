import json

import pytest

from lab.reward import compute_reward

GT = [{"label": "pothole", "bbox": [100, 100, 200, 200]}]
PERFECT = {"label": "pothole", "bbox_2d": [100, 100, 200, 200]}


def test_perfect():
    assert compute_reward(json.dumps([PERFECT]), GT) == 1.0


def test_invalid_json():
    assert compute_reward("not a json list", GT) == 0.0


def test_wrong_class():
    assert compute_reward(json.dumps([{**PERFECT, "label": "crack"}]), GT) == 0.2


def test_extra_box_penalized():
    extra = {"label": "pothole", "bbox_2d": [400, 400, 500, 500]}
    assert compute_reward(json.dumps([PERFECT, extra]), GT) == pytest.approx(0.6)


def test_both_empty():
    assert compute_reward("[]", []) == 1.0


def test_empty_prediction_on_positive_image_is_not_perfect():
    assert compute_reward("[]", GT) == 0.2


def test_duplicate_prediction_cannot_match_truth_twice():
    assert compute_reward(json.dumps([PERFECT, PERFECT]), GT) == pytest.approx(0.6)


def test_giant_box_does_not_match_small_damage():
    giant = {"label": "pothole", "bbox_2d": [0, 0, 1000, 1000]}
    assert compute_reward(json.dumps([giant]), GT) == 0.2
