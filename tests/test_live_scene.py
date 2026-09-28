"""D10 — live_scene extract: caches, honest fail, host re-export."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from planner import live_scene  # noqa: E402


def _load_host():
    spec = importlib.util.spec_from_file_location(
        "run_loop_host",
        _REPO / "scripts" / "run_loop_host.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_loop_host"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_scene_infra_matches_host_fd_filter():
    assert live_scene.SCENE_INFRA == frozenset(
        {
            "floor",
            "room",
            "ground_plane",
            "sun",
            "robot_pedestal",
            "overview_camera",
            "table",
            "workbench",
            "workshop_room",
            "central_workbench",
            "assembly_bench",
            "parts_shelf",
            "tool_cabinet_left",
            "back_pegboard",
            "parts_bin",
            "toolbox",
        }
    )


def test_world_caches_are_dicts():
    assert isinstance(live_scene._WORLD_SURFACES, dict)
    assert isinstance(live_scene._WORLD_PROP_CACHE, dict)


def test_acquire_live_scene_fails_honestly_without_capture(monkeypatch):
    monkeypatch.setattr(live_scene, "_capture", lambda _args: None)
    with pytest.raises(RuntimeError, match="mock-scene"):
        live_scene.acquire_live_scene(world="tabletop", do_pre_scan=False)


def test_host_reexports_dino_sweep_and_pre_scan():
    host = _load_host()
    assert host._run_dino_scene_sweep is live_scene._run_dino_scene_sweep
    assert host._pre_scan is live_scene._pre_scan
    assert host._capture is live_scene._capture
    assert host._infer_on_surface is live_scene._infer_on_surface
