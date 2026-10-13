"""Test phần nhãn của lab10.py (không cần GPU). Chạy: python -m pytest -q test_lab10.py"""
import json

from lab10 import norm1000_to_pixel, objects_to_answer, yolo_to_box
from scripts.score import _gt_boxes, evaluate, parse_output


def test_yolo_to_box_rounds_and_clamps():
    assert yolo_to_box("crack", [0.5, 0.5, 0.2, 0.1]) == {"label": "crack", "bbox_2d": [400, 450, 600, 550]}
    assert yolo_to_box("pothole", [0.01, 0.99, 0.1, 0.1])["bbox_2d"] == [0, 940, 60, 1000]


def test_answer_sorted_top_to_bottom_then_left_to_right():
    objs = [{"label": "crack", "yolo": [0.8, 0.6, 0.1, 0.1]},
            {"label": "manhole", "yolo": [0.2, 0.6, 0.1, 0.1]},
            {"label": "pothole", "yolo": [0.5, 0.2, 0.1, 0.1]}]
    assert [b["label"] for b in json.loads(objects_to_answer(objs))] == ["pothole", "manhole", "crack"]
    assert objects_to_answer([]) == "[]"


def test_answer_parses_and_scores_perfectly():
    entry = {"image_id": "a", "width": 640, "height": 360, "objects": [
        {"label": "crack", "yolo": [0.3, 0.4, 0.05, 0.02]},
        {"label": "manhole", "yolo": [0.7, 0.8, 0.1, 0.08]}]}
    ans = objects_to_answer(entry["objects"])
    assert parse_output(ans) is not None
    res = evaluate([entry], {"a": ans}, n_boot=1)
    assert res["micro"]["f1"] == 1.0 and res["parse_valid_rate"] == 1.0


def test_roundtrip_to_pixels_matches_gt():
    entry = {"objects": [{"label": "pothole", "yolo": [0.4, 0.7, 0.15, 0.1]}]}
    box = json.loads(objects_to_answer(entry["objects"]))[0]["bbox_2d"]
    gt = _gt_boxes(entry)[0]["bbox"]
    px = norm1000_to_pixel(box, 640, 360)
    ref = norm1000_to_pixel(gt, 640, 360)
    assert all(abs(a - b) <= 0.64 for a, b in zip(px, ref))   # sai số làm tròn ≤ 0,5/1000 khung


def test_pixel_coordinates_score_badly_on_norm1000():
    entry = {"image_id": "a", "width": 640, "height": 360, "objects": [
        {"label": "crack", "yolo": [0.5, 0.5, 0.2, 0.2]}]}
    px = [round(v) for v in norm1000_to_pixel(yolo_to_box("crack", [0.5, 0.5, 0.2, 0.2])["bbox_2d"], 640, 360)]
    raw = json.dumps([{"label": "crack", "bbox_2d": px}])
    assert evaluate([entry], {"a": raw}, n_boot=1)["micro"]["f1"] == 0.0
    assert evaluate([entry], {"a": raw}, coord="pixel", n_boot=1)["micro"]["f1"] == 1.0
