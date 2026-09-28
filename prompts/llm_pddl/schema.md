STAGE:pddl_schema
You propose PDDL types and predicates for a compact scene and a task.
Do not write actions, :init, or :goal yet.

Reply with ONE JSON object and nothing else:
{"types": ["item", "location"], "predicates": ["(on ?x - item ?y - location)",
"(holding ?x - item)", "(hand-empty)"], "reason": "<one short sentence>"}

Rules:
- Predicates are s-expressions with types on variables.
- Invent names that fit this scene; do not copy a stock repository domain.
- JSON only.

Toy shape (invented widget/pad vocabulary — imitate the shape, not the names
if this task needs different fluents):

{"types": ["widget", "pad"], "predicates": ["(resting ?w - widget ?p - pad)",
"(gripped ?w - widget)", "(free-hand)"], "reason": "support pick and place"}
