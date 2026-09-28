"""VLM scene inventory: names + DINO queries from an image (no action plan)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from planner.problem_generator.goal_generator.json_facts import extract_json_object
from prompts import DIR as PROMPTS_DIR

# Default: type each visible entity from the robot task (item vs location
# vs container). The older keyword-style prompt is inventory/keyword.txt.
INVENTORY_PROMPT_PATH = PROMPTS_DIR / "inventory" / "task_typed.txt"

_LOCATION_HINTS = frozenset(
    {
        "table",
        "shelf",
        "shelf_b",
        "counter",
        "workbench",
        "desk",
        "tray",
        "metal_tray",
        "floor",
        "drawer",
    }
)
# Workspace slabs items rest on *now*. Task destinations (notebook, tray, plate)
# may also be locations, but they are not the default :init support.
_STATIC_SURFACE_TOKENS = frozenset(
    {
        "table",
        "tablecloth",
        "cloth",
        "desk",
        "counter",
        "workbench",
        "floor",
        "shelf",
        "shelf_b",
        "support",
    }
)
# Last-token hints: these are destinations you put things into, not items.
_CONTAINER_HINTS = frozenset(
    {"bowl", "drawer", "bin", "container", "basket"}
)
_SKIP_OBJECT_TOKENS = frozenset(
    {"robot", "panda", "gripper", "camera", "franka"}
)


class InventoryError(ValueError):
    """VLM inventory JSON could not be trusted."""


@dataclass(frozen=True)
class NamedEntity:
    """One visible entity: PDDL ``id`` and GroundingDINO ``query``."""

    id: str
    query: str
    # Lid / cavity state from the photo. Only containers use this.
    # ``None`` = the VLM did not say; callers treat that as open.
    open: bool | None = None


@dataclass(frozen=True)
class SceneInventory:
    objects: tuple[NamedEntity, ...]
    locations: tuple[NamedEntity, ...]
    containers: tuple[NamedEntity, ...] = ()
    raw: str | None = None

    def object_ids(self) -> tuple[str, ...]:
        return tuple(e.id for e in self.objects)

    def location_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for ent in (*self.locations, *self.containers):
            if ent.id not in seen:
                seen.append(ent.id)
        return tuple(seen)

    def container_ids(self) -> tuple[str, ...]:
        return tuple(e.id for e in self.containers)

    def container_is_open(self, cid: str) -> bool:
        """Whether ``cid`` is an open vessel in the photo.

        Comes from the VLM ``open`` flag, not from the symbol name. If the
        model omitted it, the cavity is assumed visible (open): listing the
        entity as a container already means things go *into* it.
        """
        for ent in self.containers:
            if ent.id == cid:
                return True if ent.open is None else bool(ent.open)
        return True


def normalize_symbol(label: str) -> str:
    """Map a free-form name to a snake_case PDDL symbol."""
    name = label.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _id_tokens(symbol: str) -> set[str]:
    return {part for part in symbol.lower().split("_") if part}


def is_skipped_object_id(symbol: str) -> bool:
    """True for the robot / cameras, which are not PDDL items."""
    return bool(_id_tokens(symbol) & _SKIP_OBJECT_TOKENS)


def is_container_id(symbol: str) -> bool:
    """True when the symbol's last token is a vessel / openable box."""
    tokens = [part for part in symbol.lower().split("_") if part]
    return bool(tokens) and tokens[-1] in _CONTAINER_HINTS


def is_static_surface_id(symbol: str) -> bool:
    """True for table/cloth/desk/floor — not portable destinations."""
    tokens = _id_tokens(symbol)
    if tokens & _STATIC_SURFACE_TOKENS:
        return True
    return any(
        tok.endswith("cloth") or tok.endswith("table") for tok in tokens
    )


def parse_open_flag(item: dict[str, Any]) -> bool | None:
    """Read ``open`` / ``closed`` from a VLM entity dict. ``None`` if omitted."""
    if "open" in item:
        return _coerce_open(item.get("open"))
    if "closed" in item:
        closed = _coerce_open(item.get("closed"))
        if closed is None:
            return None
        return not closed
    return None


def _coerce_open(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "open", "opened", "1"}:
        return True
    if text in {"false", "no", "closed", "shut", "0"}:
        return False
    return None


def _as_entity(item: Any, *, fallback_kind: str) -> NamedEntity | None:
    open_flag: bool | None = None
    if isinstance(item, str):
        symbol = normalize_symbol(item)
        query = item.strip() or symbol.replace("_", " ")
    elif isinstance(item, dict):
        raw_id = item.get("id") or item.get("name") or item.get("label") or ""
        query = str(item.get("query") or item.get("phrase") or raw_id).strip()
        symbol = normalize_symbol(str(raw_id) or query)
        if not query:
            query = symbol.replace("_", " ")
        open_flag = parse_open_flag(item)
    else:
        return None
    if not symbol:
        return None
    if fallback_kind == "location" or symbol in _LOCATION_HINTS:
        pass
    return NamedEntity(id=symbol, query=query, open=open_flag)


def _parse_list(raw_list: Any, *, kind: str) -> list[NamedEntity]:
    if raw_list is None:
        return []
    if isinstance(raw_list, dict) and kind == "objects":
        # {"red_cup": "red cup", ...}
        items = [{"id": k, "query": v} for k, v in raw_list.items()]
    elif isinstance(raw_list, (list, tuple)):
        items = list(raw_list)
    else:
        raise InventoryError(f"{kind} must be a list")
    seen: set[str] = set()
    out: list[NamedEntity] = []
    for item in items:
        ent = _as_entity(item, fallback_kind=kind)
        if ent is None:
            continue
        symbol = ent.id
        if symbol in seen:
            n = 2
            while f"{symbol}_{n}" in seen:
                n += 1
            symbol = f"{symbol}_{n}"
            ent = NamedEntity(id=symbol, query=ent.query, open=ent.open)
        seen.add(symbol)
        out.append(ent)
    return out


# Any "... on/onto the <dest>" tail: put/place, must be on, belong on, …
_TASK_ON_DEST = re.compile(
    r"\bon(?:to)?\s+(?:the\s+)?"
    r"(?P<dest>[a-z0-9]+(?:[_\s][a-z0-9]+)*)\s*\.?\s*$",
    re.IGNORECASE,
)
_TASK_IN_DEST = re.compile(
    r"\b(?:put|place|drop)\b[\s\S]+?\bin(?:to)?\s+(?:the\s+)?"
    r"(?P<dest>[a-z0-9]+(?:[_\s][a-z0-9]+)*)\s*\.?\s*$",
    re.IGNORECASE,
)
_TASK_STACK = re.compile(r"^\s*stack\b", re.IGNORECASE)


def _match_named_entity(phrase: str, entities: tuple[NamedEntity, ...]) -> NamedEntity | None:
    """Map an NL destination phrase onto an inventory id (notebook, white_bowl)."""
    normalized = normalize_symbol(phrase)
    if not normalized:
        return None
    by_id = {ent.id: ent for ent in entities}
    if normalized in by_id:
        return by_id[normalized]
    phrase_words = normalized.replace("_", " ")
    ranked = sorted(entities, key=lambda ent: len(ent.id), reverse=True)
    for ent in ranked:
        variants = {ent.id, ent.id.replace("_", " ")}
        for variant in variants:
            if variant == normalized or variant == phrase_words:
                return ent
            if re.search(rf"\b{re.escape(variant)}\b", phrase_words):
                return ent
        name_words = ent.id.replace("_", " ")
        if re.search(rf"\b{re.escape(phrase_words)}\b", name_words):
            return ent
    return None


def apply_task_buckets(inventory: SceneInventory, task: str) -> SceneInventory:
    """Re-bucket a destination that the task uses as ON / INTO, not as a grasp.

    The typed inventory prompt already asks for this; VLMs still put
    ``notebook`` / ``plate`` in ``objects``. ``(on item notebook)`` then
    fails because ``on`` is ``item × location``.
    """
    text = str(task or "").strip()
    if not text:
        return inventory

    objects = list(inventory.objects)
    locations = list(inventory.locations)
    containers = list(inventory.containers)
    loc_ids = {e.id for e in locations}
    container_ids = {e.id for e in containers}

    def _take_object(ent: NamedEntity) -> NamedEntity | None:
        for i, obj in enumerate(objects):
            if obj.id == ent.id:
                return objects.pop(i)
        return None

    if not _TASK_STACK.match(text):
        on_m = _TASK_ON_DEST.search(text)
        if on_m:
            dest = _match_named_entity(on_m.group("dest"), tuple(objects))
            if dest is not None and dest.id not in loc_ids and len(objects) > 1:
                taken = _take_object(dest)
                if taken is not None:
                    locations.append(taken)
                    loc_ids.add(taken.id)

    in_m = _TASK_IN_DEST.search(text)
    if in_m:
        dest = _match_named_entity(in_m.group("dest"), tuple(objects))
        if dest is not None and dest.id not in container_ids and len(objects) > 1:
            taken = _take_object(dest)
            if taken is not None:
                containers.append(taken)
                container_ids.add(taken.id)
                if taken.id not in loc_ids:
                    locations.append(taken)
                    loc_ids.add(taken.id)

    if not objects:
        return inventory
    return SceneInventory(
        objects=tuple(objects),
        locations=tuple(locations),
        containers=tuple(containers),
        raw=inventory.raw,
    )


def parse_inventory(raw: str) -> SceneInventory:
    """
    Validate inventory JSON.

    Accepts the canonical shape plus a flat name→query object for ``objects``.
    Empty object lists are an error (the VLM must see *something*). Locations
    may be empty; the caller may add a generic ``support`` surface.
    """
    try:
        data = extract_json_object(raw)
    except (ValueError, TypeError) as exc:
        raise InventoryError(f"invalid inventory JSON: {exc}") from exc

    objects = _parse_list(data.get("objects"), kind="objects")
    locations = _parse_list(data.get("locations"), kind="locations")
    containers = _parse_list(data.get("containers"), kind="locations")

    loc_ids = {e.id for e in locations}
    container_ids = {e.id for e in containers}

    def _add_location(ent: NamedEntity) -> None:
        if ent.id not in loc_ids:
            locations.append(ent)
            loc_ids.add(ent.id)

    def _add_container(ent: NamedEntity) -> None:
        if ent.id not in container_ids:
            containers.append(ent)
            container_ids.add(ent.id)
        _add_location(ent)

    kept_objects: list[NamedEntity] = []
    for ent in objects:
        if is_skipped_object_id(ent.id):
            continue
        if is_container_id(ent.id) or ent.id in container_ids:
            _add_container(ent)
            continue
        if ent.id in _LOCATION_HINTS or ent.id in loc_ids:
            _add_location(ent)
            continue
        kept_objects.append(ent)
    objects = kept_objects

    for ent in list(containers):
        _add_location(ent)

    if not objects:
        raise InventoryError("inventory lists no manipulable objects")

    return SceneInventory(
        objects=tuple(objects),
        locations=tuple(locations),
        containers=tuple(containers),
        raw=raw,
    )


def list_objects_from_image(
    image: Image.Image | str | Path,
    planner: Any,
    *,
    task: str = "",
    prompt_path: str | Path | None = None,
) -> SceneInventory:
    """
    One VLM call: visible items + surfaces. Does not emit a plan.

    ``planner`` is a loaded ``VLMPlanner`` (Qwen-VL). Tests inject a stub
    with ``_to_pil`` / ``_build_messages`` / ``_run_inference``.
    ``prompt_path`` overrides the default task-typed inventory prompt.
    """
    path = Path(prompt_path) if prompt_path is not None else INVENTORY_PROMPT_PATH
    system = path.read_text(encoding="utf-8").rstrip()
    pil = planner._to_pil(image)
    user = "Name every visible object and support surface in this photo."
    if str(task).strip():
        user += f"\nThe robot will later be asked: {task.strip()}"
    messages = planner._build_messages(system, user, [pil])
    raw = planner._run_inference(messages)
    return apply_task_buckets(parse_inventory(raw), task)


def ensure_support_locations(inventory: SceneInventory) -> SceneInventory:
    """If the VLM named no support surface, add a generic ``support``."""
    skip = set(inventory.container_ids())
    if any(ent.id not in skip for ent in inventory.locations):
        return inventory
    extra = NamedEntity(id="support", query="surface the objects rest on")
    return SceneInventory(
        objects=inventory.objects,
        locations=(extra,) + tuple(inventory.locations),
        containers=inventory.containers,
        raw=inventory.raw,
    )


def default_support_id(inventory: SceneInventory) -> str:
    """Surface for ``(on item loc)`` when we do not measure stacking.

    Prefer a static workspace slab (table, cloth, …) over a task destination
    that was typed as a location (notebook, tray). Being a legal *place*
    target must not mean every item already rests on it.
    """
    inv = ensure_support_locations(inventory)
    skip = set(inv.container_ids())
    candidates = [ent for ent in inv.locations if ent.id not in skip]
    if not candidates:
        return inv.locations[0].id
    static = [ent for ent in candidates if is_static_surface_id(ent.id)]
    if static:
        return static[0].id
    return candidates[0].id
