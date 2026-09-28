"""
VLM hybrid fusion for YELLOW / relation ambiguity only.

The vision VLM never authors ``:init`` alone — it returns relation corrections
that merge into an already-fused SceneState (oracle / DINO / tracker).
"""

from .client import (
    VlmFusionClient,
    VlmFusionRequest,
    build_vlm_callback,
    parse_vlm_patch_json,
    patch_to_scene,
)
from .live import LiveVlmFusionClient
from .mock import MockVlmFusionClient

__all__ = [
    "LiveVlmFusionClient",
    "MockVlmFusionClient",
    "VlmFusionClient",
    "VlmFusionRequest",
    "build_vlm_callback",
    "parse_vlm_patch_json",
    "patch_to_scene",
]
