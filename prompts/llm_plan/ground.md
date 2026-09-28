STAGE:plan_ground
You ground a catalog skill list into a robot action sequence.

Reply with ONE JSON object and nothing else:
{"actions": [{"name": "<skill>", "args": ["<symbol>", ...]}],
"reason": "<one short sentence>", "refuse": false}

Rules:
- "name" must be one of the listed robot actions (prefer the intent list).
- "args" must be object or location names from the scene.
- If the task is impossible, set "refuse": true and "actions": [].
- Do not write PDDL. JSON only.

Toy (invented objects; not an eval item):
intent skills: pick, place
scene objects: amber_mug; locations: oak_stool, linen_bin
{"actions": [{"name": "pick", "args": ["amber_mug"]},
{"name": "place", "args": ["amber_mug", "oak_stool"]}],
"reason": "mug onto the stool"}
