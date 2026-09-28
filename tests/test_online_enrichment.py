"""Session 28 — domain completeness + closed-catalog enricher contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.hybrid_runtime import select_domain_template
from planner.online_enrichment import (
    ENV_ONLINE_ENRICHMENT,
    REFUSE_MESSAGE,
    DomainCompleteness,
    EnrichmentRequest,
    EnrichmentStatus,
    StubOnlineDomainEnricher,
    detect_needed_enrichment_skills,
    format_refuse_message,
    resolve_domain_for_task,
    resolve_online_enrichment,
    select_domain,
)
from planner.skill_catalog import (
    CATALOG_SKILLS,
    ENRICHMENT_CANDIDATE_SKILLS,
    enrichment_candidates_for_domain,
    skills_missing_from_domain,
)


# ── Flag default OFF ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("off", False),
        ("1", True),
        ("true", True),
        ("on", True),
        ("enrichment", True),
    ],
)
def test_resolve_online_enrichment(raw, expected):
    assert resolve_online_enrichment(raw) == expected
    if raw is None:
        assert resolve_online_enrichment(env={}) is False
        assert resolve_online_enrichment(env={ENV_ONLINE_ENRICHMENT: "1"}) is True


# ── Skill catalog registry ───────────────────────────────────────────────────


def test_enrichment_candidates_missing_from_all_fixed_domains():
    """pour/tilt/stir/cut are catalog skills not in any of the four templates."""
    for template in (
        "manipulation_base",
        "manipulation_stacking",
        "containers_manipulation",
        "navigation_manipulation",
    ):
        missing = enrichment_candidates_for_domain(template)
        assert missing == ENRICHMENT_CANDIDATE_SKILLS
        assert missing <= skills_missing_from_domain(template)
        assert missing <= CATALOG_SKILLS


def test_catalog_does_not_invent_open_vocab_skills():
    assert "write" not in CATALOG_SKILLS
    assert "fly" not in CATALOG_SKILLS
    assert enrichment_candidates_for_domain("manipulation_base") == {
        "pour",
        "tilt",
        "stir",
        "cut",
    }


# ── Completeness selection ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "command,template,completeness,needed",
    [
        ("place red_cup on shelf", "manipulation_base", DomainCompleteness.COMPLETE, ()),
        ("stack blue_block on red_block", "manipulation_stacking", DomainCompleteness.COMPLETE, ()),
        ("put pen into the drawer", "containers_manipulation", DomainCompleteness.COMPLETE, ()),
        ("navigate to the kitchen", "navigation_manipulation", DomainCompleteness.COMPLETE, ()),
        (
            "pour from the bottle into the cup",
            "manipulation_base",
            DomainCompleteness.INCOMPLETE,
            ("pour",),
        ),
        (
            "tilt the pan over the bowl",
            "manipulation_base",
            DomainCompleteness.INCOMPLETE,
            ("tilt",),
        ),
        (
            "stir the soup in the pot",
            "manipulation_base",
            DomainCompleteness.INCOMPLETE,
            ("stir",),
        ),
        (
            "cut the apple on the board",
            "manipulation_base",
            DomainCompleteness.INCOMPLETE,
            ("cut",),
        ),
        (
            "pour the juice into the glass then place glass on table",
            "manipulation_base",
            DomainCompleteness.INCOMPLETE,
            ("pour",),
        ),
        (
            "pour water from bottle to mug",
            "manipulation_base",
            DomainCompleteness.INCOMPLETE,
            ("pour",),
        ),
    ],
)
def test_select_domain_completeness(command, template, completeness, needed):
    result = select_domain(command)
    assert result.template == template
    assert result.completeness == completeness
    assert result.needed_skills == needed
    # Template picker stays in sync with Session 22 helper.
    assert result.template == select_domain_template(command)


def test_detect_needed_enrichment_skills_closed_set():
    assert detect_needed_enrichment_skills("pour water") == ("pour",)
    assert detect_needed_enrichment_skills("mix the batter") == ("stir",)
    assert detect_needed_enrichment_skills("slice the bread") == ("cut",)
    assert detect_needed_enrichment_skills("place cup on table") == ()
    # Open-vocab verbs must not invent catalog entries.
    assert detect_needed_enrichment_skills("write a letter") == ()
    assert detect_needed_enrichment_skills("fly to the moon") == ()


# ── Routing: complete never calls enricher; incomplete does (when enabled) ───


def test_complete_task_never_calls_enricher_when_enabled():
    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "place red_cup on shelf",
        enrichment_enabled=True,
        enricher=stub,
        scene_symbols=("red_cup", "shelf", "table"),
    )
    assert res.selection.completeness == DomainCompleteness.COMPLETE
    assert res.enrichment is not None
    assert res.enrichment.status == EnrichmentStatus.SKIPPED
    assert stub.calls == []
    assert res.used_enricher is False


def test_incomplete_task_calls_enricher_when_enabled():
    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "pour water from bottle to mug",
        enrichment_enabled=True,
        enricher=stub,
        scene_symbols=("bottle", "mug", "table"),
    )
    assert res.selection.completeness == DomainCompleteness.INCOMPLETE
    assert res.selection.needed_skills == ("pour",)
    assert len(stub.calls) == 1
    req = stub.calls[0]
    assert isinstance(req, EnrichmentRequest)
    assert req.candidate_skills == ("pour",)
    assert req.scene_symbols == ("bottle", "mug", "table")
    assert res.enrichment is not None
    assert res.enrichment.status == EnrichmentStatus.STUB
    assert res.used_enricher is True
    assert res.refused is False


def test_incomplete_task_does_not_call_enricher_when_flag_off():
    """Default path unchanged — incomplete goals still use fixed template only."""
    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "pour water from bottle to mug",
        enrichment_enabled=False,
        enricher=stub,
    )
    assert res.selection.completeness == DomainCompleteness.INCOMPLETE
    assert res.enrichment_enabled is False
    assert stub.calls == []
    assert res.enrichment is not None
    assert res.enrichment.status == EnrichmentStatus.SKIPPED
    assert res.template == "manipulation_base"


def test_fixed_domain_template_selection_unchanged():
    """Session 22 regression: select_domain_template still picks fixed templates."""
    assert select_domain_template("place red_cup on shelf") == "manipulation_base"
    assert select_domain_template("stack a on b") == "manipulation_stacking"
    assert select_domain_template("put pen into the drawer") == "containers_manipulation"
    assert select_domain_template("go to the table") == "navigation_manipulation"
    assert select_domain_template("pour the can into the glass") == "manipulation_base"


# ── Enrichment host remap (smallest capable template) ────────────────────────


def test_pour_into_glass_enriches_on_manipulation_base():
    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "pour the can into the glass",
        enrichment_enabled=True,
        enricher=stub,
        scene_symbols=("can", "glass", "table"),
    )
    assert res.selection.template == "manipulation_base"
    assert res.selection.completeness == DomainCompleteness.INCOMPLETE
    assert res.selection.needed_skills == ("pour",)
    assert len(stub.calls) == 1
    assert stub.calls[0].template == "manipulation_base"
    assert stub.calls[0].candidate_skills == ("pour",)


def test_llm_richer_template_remaps_enrichment_onto_smallest_host():
    """Any over-rich host remaps to the smallest template that can enrich."""

    class _RichHostSelector:
        backend = "llm"

        def select(self, command, *, scene_symbols=(), world=None):
            from planner.online_enrichment import DomainSelectionResult

            return DomainSelectionResult(
                template="containers_manipulation",
                completeness=DomainCompleteness.INCOMPLETE,
                reason="into suggests a container",
                needed_skills=("pour",),
                backend="llm",
            )

    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "pour the can into the glass",
        enrichment_enabled=True,
        selector=_RichHostSelector(),
        enricher=stub,
        scene_symbols=("can", "glass", "table"),
    )
    assert res.selection.template == "manipulation_base"
    assert "enrichment host remapped" in res.selection.reason
    assert stub.calls[0].template == "manipulation_base"


def test_false_complete_coverage_veto_recovers_cued_skill():
    """complete + uncovered goal → incomplete with cue-recovered catalog skill."""

    class _FalseCompleteSelector:
        backend = "llm"

        def select(self, command, *, scene_symbols=(), world=None):
            from planner.online_enrichment import DomainSelectionResult

            return DomainSelectionResult(
                template="containers_manipulation",
                completeness=DomainCompleteness.COMPLETE,
                reason="objects look like containers",
                needed_skills=(),
                backend="llm",
            )

    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "pour the can into the glass",
        enrichment_enabled=True,
        selector=_FalseCompleteSelector(),
        enricher=stub,
        scene_symbols=("can", "glass", "table"),
    )
    assert res.selection.completeness == DomainCompleteness.INCOMPLETE
    assert res.selection.needed_skills == ("pour",)
    assert res.selection.template == "manipulation_base"
    assert "coverage veto" in res.selection.reason
    assert len(stub.calls) == 1
    assert stub.calls[0].candidate_skills == ("pour",)


def test_false_complete_revision_supplies_catalog_skill():
    """When cues miss, optional selector revision can name a catalog skill."""

    class _RevisingSelector:
        backend = "llm"

        def select(self, command, *, scene_symbols=(), world=None):
            from planner.online_enrichment import DomainSelectionResult

            return DomainSelectionResult(
                template="containers_manipulation",
                completeness=DomainCompleteness.COMPLETE,
                reason="misread as place-into",
                needed_skills=(),
                backend="llm",
            )

        def revise_after_uncovered(self, command, previous, *, scene_symbols=(), world=None):
            from planner.online_enrichment import DomainSelectionResult

            return DomainSelectionResult(
                template="manipulation_base",
                completeness=DomainCompleteness.INCOMPLETE,
                reason="gap needs liquid transfer",
                needed_skills=("pour",),
                backend="llm",
            )

    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "get me something to drink",
        enrichment_enabled=True,
        selector=_RevisingSelector(),
        enricher=stub,
        scene_symbols=("bottle", "glass", "table"),
    )
    assert res.selection.completeness == DomainCompleteness.INCOMPLETE
    assert res.selection.needed_skills == ("pour",)
    assert res.selection.template == "manipulation_base"
    assert len(stub.calls) == 1


def test_preferred_host_is_smallest_capable_template():
    from planner.online_enrichment import preferred_enrichment_host_template

    assert preferred_enrichment_host_template(("pour",)) == "manipulation_base"
    assert preferred_enrichment_host_template(("pour", "tilt")) == "manipulation_base"
    assert preferred_enrichment_host_template(()) is None


# ── Empty-catalog refuse veto when the template already covers the goal ──────


class _IncompleteEmptySelector:
    backend = "llm"

    def select(self, command, *, scene_symbols=(), world=None):
        from planner.online_enrichment import DomainSelectionResult

        return DomainSelectionResult(
            template="manipulation_base",
            completeness=DomainCompleteness.INCOMPLETE,
            reason="selector unsure",
            needed_skills=(),
            backend="llm",
        )


def test_empty_skills_skipped_when_template_already_covers_place():
    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "the wooden cube belongs on the shelf, not the table",
        enrichment_enabled=True,
        selector=_IncompleteEmptySelector(),
        enricher=stub,
        scene_symbols=("wood_cube", "shelf", "table"),
    )
    assert res.selection.completeness == DomainCompleteness.COMPLETE
    assert stub.calls == []
    assert res.used_enricher is False
    assert (
        "already expresses the goal" in res.selection.reason
        or "skipped empty-catalog refuse" in res.selection.reason
    )


def test_empty_skills_skipped_for_kitchen_place_on_tray():
    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "place the mug on the tray",
        enrichment_enabled=True,
        selector=_IncompleteEmptySelector(),
        enricher=stub,
        scene_symbols=("mug", "table", "target_tray"),
    )
    assert res.selection.completeness == DomainCompleteness.COMPLETE
    assert stub.calls == []
    assert res.used_enricher is False


def test_split_scene_symbols_treats_trays_as_locations():
    from planner.online_enrichment import split_scene_symbols

    objects, locations = split_scene_symbols(
        ("mug", "table", "tray", "target_tray", "metal_tray", "hammer")
    )
    assert "mug" in objects and "hammer" in objects
    assert {"table", "tray", "target_tray", "metal_tray"} <= set(locations)


def test_split_scene_symbols_treats_bowl_as_location():
    from planner.online_enrichment import split_scene_symbols

    objects, locations = split_scene_symbols(
        ("red_grapes", "white_bowl", "yellow_cube")
    )
    assert "red_grapes" in objects and "yellow_cube" in objects
    assert "white_bowl" in locations


def test_quantified_put_in_bowl_covers_containers_template():
    from planner.online_enrichment import template_covers_command

    assert template_covers_command(
        "put the edible items in the bowl",
        "containers_manipulation",
        objects=("red_grapes", "green_grapes", "yellow_cube"),
        locations=("blue_tablecloth", "white_bowl"),
    )
    assert not template_covers_command(
        "put the edible items in the bowl",
        "manipulation_base",
        objects=("red_grapes", "yellow_cube"),
        locations=("blue_tablecloth", "white_bowl"),
    )


def test_all_writing_tools_on_book_covers_base_and_skips_enricher():
    from planner.online_enrichment import DomainSelectionResult, template_covers_command

    command = "all writing tools must be on the book"
    symbols = (
        "black_marker",
        "red_marker",
        "black_pen",
        "book",
        "blue_tablecloth",
    )
    assert template_covers_command(command, "manipulation_base", symbols)

    class _IncompleteBase:
        backend = "llm"

        def select(self, command, *, scene_symbols=(), world=None):
            return DomainSelectionResult(
                template="manipulation_base",
                completeness=DomainCompleteness.INCOMPLETE,
                reason="place-in-container is needed to put items on the book",
                needed_skills=(),
                backend="llm",
            )

    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        command,
        enrichment_enabled=True,
        selector=_IncompleteBase(),
        enricher=stub,
        scene_symbols=symbols,
    )
    assert res.selection.template == "manipulation_base"
    assert res.selection.completeness == DomainCompleteness.COMPLETE
    assert res.selection.needed_skills == ()
    assert stub.calls == []
    assert res.enrichment is not None
    assert res.enrichment.status == EnrichmentStatus.SKIPPED


def test_complete_put_edible_in_bowl_skips_enricher():
    class _CompleteContainers:
        backend = "llm"

        def select(self, command, *, scene_symbols=(), world=None):
            from planner.online_enrichment import DomainSelectionResult

            return DomainSelectionResult(
                template="containers_manipulation",
                completeness=DomainCompleteness.COMPLETE,
                reason="put in bowl",
                needed_skills=(),
                backend="llm",
            )

    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "put the edible items in the bowl",
        enrichment_enabled=True,
        selector=_CompleteContainers(),
        enricher=stub,
        scene_symbols=(
            "red_grapes",
            "white_bowl",
            "yellow_cube",
            "blue_tablecloth",
        ),
    )
    assert res.selection.completeness == DomainCompleteness.COMPLETE
    assert res.selection.template == "containers_manipulation"
    assert "coverage veto" not in res.selection.reason
    assert stub.calls == []


def test_spurious_pour_does_not_override_covering_containers():
    """LLM pour on put-in-bowl must not remap to manipulation_base."""

    class _PourOnPutIn:
        backend = "llm"

        def select(self, command, *, scene_symbols=(), world=None):
            from planner.online_enrichment import DomainSelectionResult

            return DomainSelectionResult(
                template="containers_manipulation",
                completeness=DomainCompleteness.INCOMPLETE,
                reason="requires pouring",
                needed_skills=("pour",),
                backend="llm",
            )

    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "put all you see to eat in the bowl",
        enrichment_enabled=True,
        selector=_PourOnPutIn(),
        enricher=stub,
        scene_symbols=(
            "red_grapes",
            "green_grapes",
            "white_bowl",
            "blue_tablecloth",
        ),
    )
    assert res.selection.template == "containers_manipulation"
    assert res.selection.completeness == DomainCompleteness.COMPLETE
    assert res.selection.needed_skills == ()
    assert stub.calls == []
    assert res.enrichment is not None
    assert res.enrichment.status == EnrichmentStatus.SKIPPED


def test_empty_skills_still_refuses_ungenerable_solder():
    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "solder the wire",
        enrichment_enabled=True,
        selector=_IncompleteEmptySelector(),
        enricher=stub,
        scene_symbols=("wire", "table"),
    )
    assert res.selection.completeness == DomainCompleteness.INCOMPLETE
    assert res.refused is True
    assert len(stub.calls) == 1
    assert stub.calls[0].candidate_skills == ()


def test_native_place_in_container_does_not_refuse_belongs_to():
    """Live mix-up: LLM names a template action; v4 implicit place stays complete."""

    class _NativePlaceIn:
        backend = "llm"

        def select(self, command, *, scene_symbols=(), world=None):
            from planner.online_enrichment import DomainSelectionResult

            return DomainSelectionResult(
                template="containers_manipulation",
                completeness=DomainCompleteness.INCOMPLETE,
                reason="no action in the template can place an item into a container",
                needed_skills=("place_in_container",),
                backend="llm",
            )

    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "the cube belongs to the container",
        enrichment_enabled=True,
        selector=_NativePlaceIn(),
        enricher=stub,
        scene_symbols=("white_cube", "light_blue_bowl", "gray_tablecloth"),
    )
    assert res.selection.completeness == DomainCompleteness.COMPLETE
    assert res.selection.needed_skills == ()
    assert res.refused is False
    assert stub.calls == []
    assert "treated as complete" in res.selection.reason


def test_complete_belongs_to_is_not_regex_refused():
    """Regex miss without a pour/cut cue must not become ungenerable."""

    class _CompleteNoRevise:
        backend = "llm"

        def select(self, command, *, scene_symbols=(), world=None):
            from planner.online_enrichment import DomainSelectionResult

            return DomainSelectionResult(
                template="manipulation_base",
                completeness=DomainCompleteness.COMPLETE,
                reason="implicit place",
                needed_skills=(),
                backend="llm",
            )

    stub = StubOnlineDomainEnricher()
    res = resolve_domain_for_task(
        "the cube belongs to the container",
        enrichment_enabled=True,
        selector=_CompleteNoRevise(),
        enricher=stub,
        scene_symbols=("white_cube", "light_blue_bowl", "gray_tablecloth"),
    )
    assert res.selection.completeness == DomainCompleteness.COMPLETE
    assert res.refused is False
    assert stub.calls == []
    assert "coverage veto" not in res.selection.reason


# ── Refuse contract ──────────────────────────────────────────────────────────


def test_refuse_message_contract():
    msg = format_refuse_message(
        template="manipulation_base",
        command="levitate the cup",
        needed_skills=(),
    )
    assert REFUSE_MESSAGE in msg
    assert "manipulation_base" in msg
    assert "levitate the cup" in msg
    assert "Refusing rather than inventing" in msg


def test_stub_refuses_when_candidate_set_empty():
    stub = StubOnlineDomainEnricher()
    # Force handoff with empty candidates (no catalog skill for the gap).
    selection = select_domain("place red_cup on shelf")
    # Manually build an incomplete-like request with no candidates.
    from planner.online_enrichment import DomainSelectionResult

    fake = DomainSelectionResult(
        template="manipulation_base",
        completeness=DomainCompleteness.INCOMPLETE,
        reason="forced empty candidates for refuse test",
        needed_skills=(),
    )
    outcome = stub.enrich(
        EnrichmentRequest(
            template=fake.template,
            command="do something impossible",
            scene_symbols=("red_cup",),
            candidate_skills=(),
            reason=fake.reason,
            selection=fake,
        )
    )
    assert outcome.status == EnrichmentStatus.REFUSED
    assert outcome.refuse_message is not None
    assert REFUSE_MESSAGE in outcome.refuse_message
    # selection unused var silence
    assert selection.completeness == DomainCompleteness.COMPLETE


def test_stub_rejects_non_catalog_skill_names():
    stub = StubOnlineDomainEnricher()
    selection = select_domain("pour water")
    outcome = stub.enrich(
        EnrichmentRequest(
            template=selection.template,
            command="pour water",
            scene_symbols=(),
            candidate_skills=("teleport", "fly"),  # not in catalog
            reason="test",
            selection=selection,
        )
    )
    assert outcome.status == EnrichmentStatus.REFUSED
