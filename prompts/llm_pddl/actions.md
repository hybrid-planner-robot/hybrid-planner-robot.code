STAGE:pddl_actions
You write PDDL actions for an already-chosen type/predicate schema.

Reply with ONE JSON object and nothing else:
{"actions": [{"name": "<catalog-skill>", "parameters": "(?x - item ?y - location)",
"precondition": "(and ...)", "effect": "(and ...)"}], "reason": "<one short sentence>"}

Rules:
- "name" must be one of the listed robot actions. Never invent names.
- parameters / precondition / effect are balanced s-expressions.
- Types belong in parameters only, never inside precondition or effect.
- Use only predicates from the schema (plus (not ...) of those).
- JSON only.

Toy shape (invented widget/pad; imitate shape, not names):

{"actions": [
  {"name": "pick", "parameters": "(?w - widget ?p - pad)",
   "precondition": "(and (resting ?w ?p) (free-hand))",
   "effect": "(and (gripped ?w) (not (resting ?w ?p)) (not (free-hand)))"},
  {"name": "place", "parameters": "(?w - widget ?p - pad)",
   "precondition": "(gripped ?w)",
   "effect": "(and (resting ?w ?p) (free-hand) (not (gripped ?w)))"}
], "reason": "move a widget between pads"}
