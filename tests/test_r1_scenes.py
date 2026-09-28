"""oracle_mock_v1 world fixtures + DINO freeze helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from planner.problem_generator.init_generator.adapters.mock import (  # noqa: E402
    OracleMockAdapter,
)
from planner.r1.scenes import (  # noqa: E402
    detections_to_oracle_mock,
    mock_scene_file_for_world,
    nms_detections,
)


def test_nms_keeps_highest_score_on_overlap():
    dets = [
        {"name": "bolt1", "box": [0, 0, 10, 10], "score": 0.3},
        {"name": "hammer", "box": [0, 0, 10, 10], "score": 0.8},
        {"name": "scissors", "box": [50, 50, 60, 60], "score": 0.7},
    ]
    kept = nms_detections(dets, iou=0.5)
    names = [d["name"] for d in kept]
    assert names == ["hammer", "scissors"]


def test_detections_to_oracle_mock_filters_and_promotes_place(tmp_path: Path):
    dets = [
        {"name": "hammer", "box": [0, 0, 10, 10], "score": 0.8},
        {"name": "small_box", "box": [1, 1, 9, 9], "score": 0.9},
        {"name": "metal_tray", "box": [20, 20, 40, 40], "score": 0.6},
        {"name": "bolt1", "box": [80, 80, 90, 90], "score": 0.2},
    ]
    payload = detections_to_oracle_mock(
        dets,
        on_surface={"hammer": "table", "metal_tray": "table"},
        locations=["table"],
        place_locations=("metal_tray",),
        min_score=0.4,
        skip_names=("small_box",),
    )
    names = [o["name"] for o in payload["objects"]]
    locs = [loc["name"] for loc in payload["locations"]]
    assert payload["format"] == "oracle_mock_v1"
    assert names == ["hammer"]
    assert locs == ["table", "metal_tray"]
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    scene = OracleMockAdapter.load(path)
    assert {o.name for o in scene.objects} == {"hammer"}
    assert {loc.name for loc in scene.locations} == {"table", "metal_tray"}


def test_workshop_mock_loads_like_other_v3_worlds():
    path = mock_scene_file_for_world("workshop")
    assert path.name == "workshop.json"
    assert path.is_file()
    scene = OracleMockAdapter.load(path)
    names = {o.name for o in scene.objects}
    locs = {loc.name for loc in scene.locations}
    assert "hammer" in names
    assert "screwdriver" in names
    assert "wood_block" in names
    assert "small_box" not in names
    assert "metal_tray" not in names
    assert locs == {"table", "metal_tray"}
    assert all(o.location == "table" for o in scene.objects)
