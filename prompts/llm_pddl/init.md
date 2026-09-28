STAGE:pddl_init
You write a PDDL :objects list and :init facts from a compact scene, using
the schema already chosen. Do not write :goal.

Reply with ONE JSON object and nothing else:
{"objects": {"amber_mug": "item", "oak_stool": "location"},
"init": ["(on amber_mug oak_stool)", "(hand-empty)"],
"reason": "<one short sentence>"}

Rules:
- Object names must match symbols in the scene dump (exact spelling).
- Every init atom uses a predicate from the schema.
- Encode gripper_empty / holding / on-relations from the scene.
- JSON only.

Toy shape (invented names):

{"objects": {"red-widget": "widget", "table-pad": "pad", "shelf-pad": "pad"},
 "init": ["(resting red-widget table-pad)", "(free-hand)"],
 "reason": "widget starts on the table pad"}
