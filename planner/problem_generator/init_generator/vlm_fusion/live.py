"""
Live vision-VLM fusion client (Session 16).

Wraps the already-loaded planning VLM (typically Qwen3-VL) for a **single**
YELLOW disambiguation call. CI keeps ``MockVlmFusionClient``; inject
``complete_fn`` for unit tests without GPU.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from planner.problem_generator.init_generator.schema import SceneState

from .client import (
    VlmFusionRequest,
    build_prompt,
    parse_vlm_patch_json,
    patch_to_scene,
)

CompleteFn = Callable[[str, str, Sequence[Any]], str]


def _known_symbols(*scenes: SceneState | None) -> frozenset[str]:
    names: set[str] = set()
    for scene in scenes:
        if scene is None:
            continue
        names.update(o.name for o in scene.objects)
        names.update(loc.name for loc in scene.locations)
        for rel in scene.relations:
            names.update(rel.args)
        if scene.robot.holding:
            names.add(scene.robot.holding)
        if scene.robot.camera_aimed_at:
            names.add(scene.robot.camera_aimed_at)
    return frozenset(names)


class LiveVlmFusionClient:
    """
    ``VlmFusionClient`` backed by a local vision VLM (or injectable ``complete_fn``).

    Tracks ``call_count`` so ``HybridProblemSession`` can assert ≤1 call per
    YELLOW verification and 0 on GREEN/RED.
    """

    def __init__(
        self,
        *,
        complete_fn: CompleteFn | None = None,
        planner: Any | None = None,
    ) -> None:
        if complete_fn is not None and planner is not None:
            raise ValueError("pass only one of complete_fn or planner")
        if complete_fn is None and planner is None:
            raise ValueError("pass complete_fn or planner")
        self._complete_fn = complete_fn
        self._planner = planner
        self.call_count = 0
        self.last_raw: str | None = None
        self.last_error: str | None = None

    @classmethod
    def from_planner(cls, planner: Any) -> "LiveVlmFusionClient":
        """Reuse the loop's already-loaded ``VLMPlanner`` (no second weight load)."""
        return cls(planner=planner)

    def disambiguate(self, request: VlmFusionRequest) -> SceneState:
        self.call_count += 1
        self.last_error = None
        system, user = build_prompt(request)
        try:
            raw = self._complete(system, user, request.images)
            self.last_raw = raw
            known = _known_symbols(
                request.expected,
                request.observed,
                request.oracle_facts,
                request.dino_facts,
                request.tracker_facts,
            )
            return parse_vlm_patch_json(
                raw,
                known_symbols=known if known else None,
            )
        except Exception as exc:  # noqa: BLE001 — never crash the loop on YELLOW
            self.last_error = str(exc)
            print(f"[vlm-fusion] disambiguate failed: {exc}", flush=True)
            return patch_to_scene(relations=[])

    def _complete(
        self,
        system: str,
        user: str,
        images: Sequence[Any],
    ) -> str:
        if self._complete_fn is not None:
            return self._complete_fn(system, user, images)

        planner = self._planner
        assert planner is not None
        if getattr(planner, "_model", None) is None and not hasattr(
            planner, "_run_inference"
        ):
            raise RuntimeError("planner is not loaded (call load() first)")

        from PIL import Image as PilImage

        pil_images = []
        for img in images or []:
            if hasattr(planner, "_to_pil"):
                pil_images.append(planner._to_pil(img))
            elif isinstance(img, PilImage.Image):
                pil_images.append(img.convert("RGB"))
            else:
                pil_images.append(PilImage.open(img).convert("RGB"))

        # Qwen-VL chat template expects ≥1 image; text-only YELLOW uses a blank.
        if not pil_images:
            pil_images = [PilImage.new("RGB", (64, 64), color=(128, 128, 128))]

        messages = planner._build_messages(system, user, pil_images)
        return planner._run_inference(messages)
