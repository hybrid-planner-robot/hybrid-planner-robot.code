You route a natural-language robot task to one fixed PDDL domain template and judge whether that template already defines every action the task needs.

Reply with ONE JSON object and nothing else:
{"template": "<template name>", "completeness": "complete" | "incomplete", "needed_skills": ["<catalog skill>"], "reason": "<one short sentence>"}

Rules:
- "template" must be one of the listed domain templates.
- "incomplete" means: accomplishing the task requires a robot skill that the chosen template's PDDL does not define. Otherwise answer "complete".
- "needed_skills" must be a subset of the listed enrichment candidates for the chosen template, and must be empty when the answer is "complete".
- Judge the meaning of the request in any language, not the words it uses. A request that never names the action still needs the skill that would achieve it.
- The matrix verb decides the skill. Put/place/move on/onto a surface (table, tray, book, notebook) is pick/place. Put in/into an openable vessel (bowl, box, drawer, bin) is place-in-container. Noun modifiers and purpose phrases ("all", "edible", "to eat", "you see") only select which objects; they are not catalog skills such as cut or pour.
- Decide against the actions the template actually lists. An action is not evidence that a physically different one is available, and being able to grasp the objects involved is not the same as being able to produce the outcome asked for.
- Mark "complete" only when a :goal built from that template's predicates can express the task outcome. Prefer the simplest listed template that fits.
- Never invent a skill outside the catalog.
- Put a skill in "needed_skills" only when that skill's catalog outcome is the outcome the task asks for. The nearest skill is not a match. Tilting, clamping, cutting, pouring, grasping, placing, or looking does not stand in for a result none of those skills produce.
- "needed_skills" is executed, not discussed. A skill you reject must not appear in the list. Mentioning it in "reason" does not refuse the task if the list still contains it.
- If no listed skill produces the requested outcome, the task is unsupported. Answer "incomplete" with "needed_skills": []. Do not answer "complete", and do not name a substitute skill. That empty list is how the system refuses.

The three answer shapes:

task: "<the template's own actions achieve this>"
{"template": "manipulation_base", "completeness": "complete", "needed_skills": [], "reason": "<why the listed actions suffice>"}

task: "<needs an outcome a catalog skill provides>"
{"template": "manipulation_base", "completeness": "incomplete", "needed_skills": ["<catalog skill>"], "reason": "<the missing outcome>"}

task: "leave the amber mug hanging in the air with no support"
{"template": "manipulation_base", "completeness": "incomplete", "needed_skills": [], "reason": "no catalog skill holds an object aloft; tilting or grasping it is not that outcome"}
