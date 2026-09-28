STAGE:pddl_refuse
You decide whether a robot catalog can model a natural-language task in PDDL.

Reply with ONE JSON object and nothing else:
{"refuse": false, "reason": "<one short sentence>"}

Rules:
- "refuse": true only when no listed robot action can produce the outcome.
- "refuse": false when the task is modelable. Do not write domain or problem
  text in this step.
- JSON only.

Toy (invented; not an eval item):
task: "gild the amber mug"
{"refuse": true, "reason": "no catalog skill applies gold leaf"}

task: "the amber mug should sit on the oak stool"
{"refuse": false, "reason": "pick and place can move a mug onto a stool"}
