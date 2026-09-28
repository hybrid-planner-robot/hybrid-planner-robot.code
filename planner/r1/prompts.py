"""R1 system prompts, loaded from ``prompts/r1/`` (do not change R0 enrich)."""

from prompts import load_prompt

R1_ENRICH_SYSTEM_PROMPT = load_prompt("r1", "enrich.md")
R1_ASSIGN_SYSTEM_PROMPT = load_prompt("r1", "assign.md")
