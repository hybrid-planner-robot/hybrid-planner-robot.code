"""VLM inventory parse + image→SceneState wiring (no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from planner.image_scene import (
    execution_poses_plink0,
    perceive_image,
    scene_from_inventory_only,
    scene_to_oracle_mock,
)
from planner.problem_generator.init_generator.adapters.mock import OracleMockAdapter
from vlm.inventory import (
    InventoryError,
    apply_task_buckets,
    default_support_id,
    ensure_support_locations,
    parse_inventory,
)

_REPO = Path(__file__).resolve().parent.parent
_INV = _REPO / "tests" / "fixtures" / "image_hybrid" / "inventory.json"
_SCRIPT = _REPO / "scripts" / "run_image_hybrid.py"


def test_parse_canonical_inventory():
    inv = parse_inventory(_INV.read_text(encoding="utf-8"))
    assert inv.object_ids() == ("red_cup", "wood_cube")
    assert "table" in inv.location_ids()
    assert "shelf" in inv.location_ids()
    assert inv.objects[0].query == "red cup"


def test_parse_uniquifies_duplicate_object_ids():
    inv = parse_inventory(
        '{"objects": ['
        '{"id": "red_grapes", "query": "red grapes near the cube"}, '
        '{"id": "red_grapes", "query": "red grapes next to the bowl"}, '
        '{"id": "green_grapes", "query": "green grapes"}'
        '], "locations": ["cloth"]}'
    )
    assert inv.object_ids() == ("red_grapes", "red_grapes_2", "green_grapes")
    assert inv.objects[1].query == "red grapes next to the bowl"


def test_parse_string_list_and_promotes_table():
    inv = parse_inventory(
        '{"objects": ["red cup", "table"], "locations": []}'
    )
    inv = ensure_support_locations(inv)
    assert "red_cup" in inv.object_ids()
    assert "table" in inv.location_ids()
    assert "table" not in inv.object_ids()


def test_empty_locations_get_generic_support_not_table():
    inv = parse_inventory('{"objects": ["mug"], "locations": []}')
    inv = ensure_support_locations(inv)
    assert inv.location_ids() == ("support",)
    assert "table" not in inv.location_ids()


def test_default_support_prefers_tablecloth_over_notebook():
    """Task-typed notebook is a place target, not the current resting surface."""
    inv = parse_inventory(
        '{"objects": ["red_pen", "black_pen"], '
        '"locations": ['
        '{"id": "notebook", "query": "white notebook"}, '
        '{"id": "tablecloth", "query": "gray cloth"}'
        "]}"
    )
    assert default_support_id(inv) == "tablecloth"
    scene = scene_from_inventory_only(inv)
    assert {o.location for o in scene.objects} == {"tablecloth"}
    assert {loc.name for loc in scene.locations} == {"notebook", "tablecloth"}


def test_task_promotes_notebook_from_objects_to_location():
    """VLM often lists the place target as an object; the task bucket wins."""
    inv = parse_inventory(
        '{"objects": ["black_pen", "red_pen", "marker", "notebook"], '
        '"locations": ["table"]}'
    )
    inv = apply_task_buckets(
        inv, "put the black pen, red pen, and marker on the notebook"
    )
    assert "notebook" in inv.location_ids()
    assert "notebook" not in inv.object_ids()
    assert {"black_pen", "red_pen", "marker"} <= set(inv.object_ids())
    scene = scene_from_inventory_only(inv)
    assert "notebook" not in {o.name for o in scene.objects}
    assert "notebook" in {loc.name for loc in scene.locations}
    assert {o.location for o in scene.objects} == {"table"}


def test_task_promotes_book_for_must_be_on():
    """``must be on`` is place-on, even without put/place."""
    inv = parse_inventory(
        '{"objects": ["black_marker", "red_marker", "black_pen", "book"], '
        '"locations": ["blue_tablecloth"]}'
    )
    inv = apply_task_buckets(
        inv, "all writing tools must be on the book"
    )
    assert "book" in inv.location_ids()
    assert "book" not in inv.object_ids()
    assert {"black_marker", "red_marker", "black_pen"} <= set(inv.object_ids())


def test_task_does_not_promote_stack_bottom():
    inv = parse_inventory(
        '{"objects": ["wood_cube", "blue_box"], "locations": ["table"]}'
    )
    inv = apply_task_buckets(inv, "stack the wood cube on the blue box")
    assert "blue_box" in inv.object_ids()
    assert "blue_box" not in inv.location_ids()


def test_parse_promotes_bowl_and_drops_robot():
    inv = parse_inventory(
        '{"objects": ["red_grapes", "white_bowl", "robot_arm"], '
        '"locations": ["blue_tablecloth"]}'
    )
    assert "white_bowl" in inv.container_ids()
    assert "white_bowl" in inv.location_ids()
    assert "white_bowl" not in inv.object_ids()
    assert "robot_arm" not in inv.object_ids()
    assert "red_grapes" in inv.object_ids()


def test_parse_container_open_flag():
    inv = parse_inventory(
        '{"objects": ["mug"], "locations": ["table"], '
        '"containers": [{"id": "box", "query": "cardboard box", "open": "closed"}]}'
    )
    assert inv.containers[0].open is False
    assert inv.container_is_open("box") is False

    inv2 = parse_inventory(
        '{"objects": ["mug"], "locations": ["table"], '
        '"containers": [{"id": "box", "query": "cardboard box", "closed": true}]}'
    )
    assert inv2.containers[0].open is False


def test_scene_uses_vlm_open_flag_not_box_in_the_name(tmp_path):
    """A plastic box is not closed just because the id contains 'box'."""
    inv = parse_inventory(
        '{"objects": ["sponge_1"], "locations": ["table"], '
        '"containers": [{"id": "plastic_box", "query": "clear plastic box"}]}'
    )
    scene = scene_from_inventory_only(inv)
    box = next(loc for loc in scene.locations if loc.name == "plastic_box")
    assert box.type == "container"
    assert box.open is True

    inv_shut = parse_inventory(
        '{"objects": ["sponge_1"], "locations": ["table"], '
        '"containers": [{"id": "plastic_box", "query": "clear plastic box", '
        '"open": false}]}'
    )
    scene_shut = scene_from_inventory_only(inv_shut)
    shut = next(loc for loc in scene_shut.locations if loc.name == "plastic_box")
    assert shut.open is False

    inv_open = parse_inventory(
        '{"objects": ["sponge_1"], "locations": ["table"], '
        '"containers": [{"id": "kitchen_drawer", "query": "white drawer", '
        '"open": true}]}'
    )
    scene_open = scene_from_inventory_only(inv_open)
    drawer = next(loc for loc in scene_open.locations if loc.name == "kitchen_drawer")
    assert drawer.open is True


def test_scene_marks_bowl_as_open_container(tmp_path):
    inv = parse_inventory(
        '{"objects": ["red_grapes"], "locations": ["cloth"], '
        '"containers": [{"id": "white_bowl", "query": "white bowl"}]}'
    )
    scene = scene_from_inventory_only(inv)
    bowl = next(loc for loc in scene.locations if loc.name == "white_bowl")
    assert bowl.type == "container"
    assert bowl.open is True
    payload = scene_to_oracle_mock(scene)
    dumped = next(loc for loc in payload["locations"] if loc["name"] == "white_bowl")
    assert dumped["type"] == "container"
    assert dumped["open"] is True
    path = tmp_path / "s.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = OracleMockAdapter.load(path)
    bowl2 = next(loc for loc in loaded.locations if loc.name == "white_bowl")
    assert bowl2.type == "container"
    assert bowl2.open is True


def test_parse_rejects_empty_objects():
    with pytest.raises(InventoryError, match="no manipulable"):
        parse_inventory('{"objects": [], "locations": ["table"]}')


def test_list_objects_from_image_uses_planner_stub(synthetic_pil_image):
    from vlm.inventory import list_objects_from_image
    from vlm.planner import VLMPlanner

    class _Stub(VLMPlanner):
        def __init__(self):
            pass

        def _run_inference(self, messages):
            assert messages[0]["role"] == "system"
            assert "GroundingDINO" in messages[0]["content"]
            return (
                '{"objects": [{"id": "blue_box", "query": "blue box"}],'
                ' "locations": [{"id": "table", "query": "table"}]}'
            )

    inv = list_objects_from_image(synthetic_pil_image, _Stub(), task="pick the box")
    assert inv.object_ids() == ("blue_box",)
    assert inv.locations[0].id == "table"


def test_list_objects_from_image_custom_prompt(synthetic_pil_image, tmp_path):
    from vlm.inventory import list_objects_from_image
    from vlm.planner import VLMPlanner

    prompt = tmp_path / "inv.md"
    prompt.write_text("CUSTOM INVENTORY PROMPT\n", encoding="utf-8")

    class _Stub(VLMPlanner):
        def __init__(self):
            pass

        def _run_inference(self, messages):
            assert messages[0]["content"].startswith("CUSTOM INVENTORY PROMPT")
            return (
                '{"objects": [{"id": "red_pen", "query": "red pen"}],'
                ' "locations": [{"id": "notebook", "query": "notebook"}]}'
            )

    inv = list_objects_from_image(
        synthetic_pil_image,
        _Stub(),
        task="stack pens on the notebook",
        prompt_path=prompt,
    )
    assert inv.object_ids() == ("red_pen",)
    assert "notebook" in inv.location_ids()


def test_scene_uses_vlm_surface_not_sim_table():
    inv = parse_inventory(
        '{"objects": ["coffee_mug"], '
        '"locations": [{"id": "desk", "query": "wooden desk"}]}'
    )
    scene = scene_from_inventory_only(inv)
    assert {o.name for o in scene.objects} == {"coffee_mug"}
    assert {loc.name for loc in scene.locations} == {"desk"}
    assert scene.objects[0].location == "desk"
    payload = scene_to_oracle_mock(scene)
    assert payload["objects"][0]["location"] == "desk"
    assert [loc["name"] for loc in payload["locations"]] == ["desk"]


def test_oracle_mock_dump_loads(tmp_path):
    inv = parse_inventory(_INV.read_text(encoding="utf-8"))
    scene = scene_from_inventory_only(inv)
    payload = scene_to_oracle_mock(scene)
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = OracleMockAdapter.load(path)
    assert {o.name for o in loaded.objects} == {"red_cup", "wood_cube"}
    assert loaded.robot.gripper_empty is True


class _FakePerception:
    def __init__(self) -> None:
        self._last_detection = None

    def get_pose(self, name, image, k, cam_to_base, **kwargs):
        self._last_detection = {
            "name": name,
            "box": [10, 10, 40, 40],
            "score": 0.9,
        }
        return {"x": 0.3, "y": 0.1, "z": 0.02}


def test_perceive_image_with_fake_dino(synthetic_pil_image):
    inv = parse_inventory(_INV.read_text(encoding="utf-8"))
    scene, out_inv, dets = perceive_image(
        synthetic_pil_image,
        inventory=inv,
        perception=_FakePerception(),
        skip_dino=False,
    )
    assert out_inv.object_ids() == inv.object_ids()
    assert {o.name for o in scene.objects} == {"red_cup", "wood_cube"}
    assert all(d["score"] == 0.9 for d in dets)
    posed = [o for o in scene.objects if o.pose is not None]
    assert len(posed) == 2


def test_localize_queries_containers_for_execution_poses(synthetic_pil_image):
    inv = parse_inventory(
        '{"objects": ["red_grapes"], "locations": ["cloth"], '
        '"containers": [{"id": "white_bowl", "query": "white bowl"}]}'
    )
    scene, _, dets = perceive_image(
        synthetic_pil_image,
        inventory=inv,
        perception=_FakePerception(),
        skip_dino=False,
    )
    names = [d["name"] for d in dets]
    assert "red_grapes" in names
    assert "white_bowl" in names
    assert "white_bowl" not in {o.name for o in scene.objects}
    bowl = next(loc for loc in scene.locations if loc.name == "white_bowl")
    assert bowl.type == "container"
    poses = execution_poses_plink0(scene, dets)
    assert "red_grapes" in poses
    assert "white_bowl" in poses
    payload = scene_to_oracle_mock(scene, execution_poses=poses)
    assert payload["execution_poses"]["white_bowl"]["x"] == 0.3


def test_script_skip_dino_skip_plan(synthetic_scene_image):
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--image",
            str(synthetic_scene_image),
            "--task",
            "place the red cup on the shelf",
            "--inventory-json",
            str(_INV),
            "--skip-dino",
            "--skip-plan",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "red_cup" in result.stdout
    assert "not calling the planner" in result.stdout


def test_script_help_lists_llm_plan_planner():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--planner" in result.stdout
    assert "llm_plan" in result.stdout


def test_script_help_lists_execute():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--execute" in result.stdout
