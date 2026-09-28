"""R1 closed-catalog enricher: affordance predicates + object assignment.

Fork of the R0 online enricher (``planner.domain_llm.LLMDomainEnricher``).
Do not import R1 from the R0 default path; the host opt-in flag is
``--enrichment-profile r1``.
"""

from .enricher import R1DomainEnricher
from .init_facts import apply_init_facts, apply_init_facts_from_additions
from .parse import (
    R1AssignmentError,
    R1EnrichmentPayloadError,
    parse_r1_assignment,
    parse_r1_enrichment,
)
from .prompts import R1_ASSIGN_SYSTEM_PROMPT, R1_ENRICH_SYSTEM_PROMPT

__all__ = [
    "R1_ASSIGN_SYSTEM_PROMPT",
    "R1_ENRICH_SYSTEM_PROMPT",
    "R1AssignmentError",
    "R1DomainEnricher",
    "R1EnrichmentPayloadError",
    "apply_init_facts",
    "apply_init_facts_from_additions",
    "parse_r1_assignment",
    "parse_r1_enrichment",
]
