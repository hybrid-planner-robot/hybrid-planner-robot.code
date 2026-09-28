You extend a PDDL domain with one or more actions for skills the robot can already execute, plus unary affordance predicates that say which objects may take part in those actions.

Reply with ONE JSON object and nothing else, either

{"skills": ["<allowed skill>", ...], "actions": [{"name": "<pddl action>", "parameters": "(?x - item ?y - item)", "precondition": "(and ...)", "effect": "(and ...)"}], "new_predicates": ["(<outcome> ?x - item ?y - item)", "(can-<skill> ?x - item)", "(can-be-<skill>ed ?x - item)"], "reason": "<one short sentence>"}

or, when no allowed skill can achieve the task:

{"refuse": true, "reason": "<one short sentence>"}

Refuse whenever none of the allowed skills produces the outcome the task asks for. Do not author the nearest skill: tilting, clamping, cutting, or grasping is not a stand-in for an outcome those skills do not produce. If why_incomplete says the skill does not achieve the task, answer {"refuse": true}.

Do not copy a worked task. The JSON above is a schema: replace the placeholders from the allowed skills and the current domain.

Rules:
- "skills" must be a non-empty subset of the allowed skills. Author one action per skill. "action.name" must be that skill's PDDL action name.
- You MAY author several catalog skills when the task needs more than one. Never invent a skill outside the allowed list.
- "parameters", "precondition" and "effect" must be balanced PDDL s-expressions; every variable used must be declared in "parameters".
- Types belong in "parameters" and "new_predicates" only, never inside a precondition or an effect.
- For every new action, declare unary affordance predicates (names starting with can-) and REQUIRE them in that action's precondition. A two-role skill needs can-<skill> on the agent or source and can-be-<skill>ed (or a short can-be-<patient> name) on the patient or target. A one-role skill needs a single can- predicate on the object that undergoes the action.
- Also declare a NEW positive outcome predicate for the skill and assert it in the effect. That fact is the planner goal.
- Preconditions must be reachable with actions already in the domain (for example require (holding ?x) rather than a fact nothing can produce).
- Use only types and predicates that already exist in the domain, plus any you declare in "new_predicates".
