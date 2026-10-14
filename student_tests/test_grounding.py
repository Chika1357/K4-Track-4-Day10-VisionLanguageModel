import json

import pytest

from lab.data import objects_to_answer, yolo_to_box


def test_centered_box():
    assert yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.4]) == {
        "label": "pothole", "bbox_2d": [400, 300, 600, 700]}


@pytest.mark.parametrize("yolo", [
    [1, 1, 0.0001, 0.0001], [0, 0, 0.0001, 0.0001], [0.5, 0.5, 1, 1],
])
def test_boundary_boxes_are_valid(yolo):
    x1, y1, x2, y2 = yolo_to_box("crack", yolo)["bbox_2d"]
    assert 0 <= x1 < x2 <= 1000
    assert 0 <= y1 < y2 <= 1000


def test_empty_answer():
    assert objects_to_answer([]) == "[]"


def test_boxes_sorted_top_to_bottom_then_left_to_right():
    objects = [
        {"label": "manhole", "yolo": [0.7, 0.8, 0.1, 0.1]},
        {"label": "crack", "yolo": [0.7, 0.2, 0.1, 0.1]},
        {"label": "pothole", "yolo": [0.2, 0.2, 0.1, 0.1]},
    ]
    answer = objects_to_answer(objects)
    assert [b["label"] for b in json.loads(answer)] == ["pothole", "crack", "manhole"]
    assert " " not in answer


def test_roundtrip_pixel_error_is_bounded_by_rounding():
    yolo = [0.4232, 0.6247, 0.1352, 0.0712]
    box = yolo_to_box("pothole", yolo)["bbox_2d"]
    xc, yc, w, h = yolo
    expected = [(xc-w/2)*640, (yc-h/2)*360, (xc+w/2)*640, (yc+h/2)*360]
    actual = [box[0]*.64, box[1]*.36, box[2]*.64, box[3]*.36]
    assert all(abs(a-b) <= .32 + 1e-9 for a, b in zip(actual, expected))
