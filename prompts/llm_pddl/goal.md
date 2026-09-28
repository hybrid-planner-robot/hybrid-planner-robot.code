STAGE:pddl_goal
You write PDDL :goal facts for the natural-language task, using the schema
already chosen. Do not rewrite :init.

Reply with ONE JSON object and nothing else:
{"goal": ["(on amber_mug oak_stool)"], "reason": "<one short sentence>"}

Rules:
- Every goal atom uses a predicate from the schema and symbols from the scene.
- Prefer the smallest fact set that captures the outcome.
- If the task cannot be expressed with this schema, {"refuse": true, "goal": []}.
- JSON only.

Toy shape (invented names):

{"goal": ["(resting red-widget shelf-pad)"], "reason": "widget ends on the shelf pad"}
