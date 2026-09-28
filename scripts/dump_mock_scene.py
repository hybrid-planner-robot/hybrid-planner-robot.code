#!/usr/bin/env python3
"""
One-shot live DINO → ``oracle_mock_v1`` fixture for ``--mock-scene``.

Run this when a world changes (Gazebo + overview capture). Tests then use the
frozen JSON and do not load GroundingDINO.

    python scripts/dump_mock_scene.py --world workshop
    python scripts/dump_mock_scene.py --world workshop --min-score 0.40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main() -> int:
    from planner.live_scene import acquire_live_scene
    from planner.r1.scenes import (
        DEFAULT_PLACE_LOCATIONS,
        DEFAULT_SKIP_NAMES,
        detections_to_oracle_mock,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="workshop")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: tests/fixtures/llm_plan/<world>.json)",
    )
    parser.add_argument("--min-score", type=float, default=0.40)
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=None,
        help="If set, collapse overlapping boxes to the highest score.",
    )
    parser.add_argument(
        "--place-location",
        action="append",
        default=None,
        help="Promote a detected name to a place location (repeatable). "
        "Default per world: workshop → metal_tray, tabletop → shelf.",
    )
    args = parser.parse_args()

    place = tuple(args.place_location or DEFAULT_PLACE_LOCATIONS.get(args.world, ()))
    skip = DEFAULT_SKIP_NAMES.get(args.world, ())
    out = (
        Path(args.out)
        if args.out
        else _REPO / "tests" / "fixtures" / "llm_plan" / f"{args.world}.json"
    )

    print(f"[dump] live DINO sweep world={args.world} min_score={args.min_score}")
    result = acquire_live_scene(
        world=args.world,
        scene_source="dino",
        perception_only=True,
        do_pre_scan=False,
    )
    on_surface = {
        o.name: (o.location or "table") for o in result.scene.objects
    }
    # Hybrid base_locations always injects ``shelf``; only keep table + place dests.
    payload = detections_to_oracle_mock(
        result.detections,
        on_surface=on_surface,
        locations=["table"],
        place_locations=place,
        min_score=args.min_score,
        nms_iou=args.nms_iou,
        skip_names=skip,
        note=(
            f"Frozen from GroundingDINO overview on {args.world}.world "
            f"(min_score={args.min_score}). Not Gazebo GT."
        ),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    names = [o["name"] for o in payload["objects"]]
    locs = [loc["name"] for loc in payload["locations"]]
    print(f"[dump] {len(result.detections)} detections → {len(names)} objects")
    print(f"[dump] objects: {names}")
    print(f"[dump] locations: {locs}")
    print(f"[dump] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
