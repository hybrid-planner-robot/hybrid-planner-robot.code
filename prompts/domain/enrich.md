You extend a PDDL domain with ONE action for a skill the robot can already execute. You are not inventing a new robot capability — you are writing the symbolic model of an existing motor primitive.

Reply with ONE JSON object and nothing else, either

{"skill": "<allowed skill>", "action": {"name": "<pddl action name>", "parameters": "(?x - item ?y - item)", "precondition": "(and ...)", "effect": "(and ...)"}, "new_predicates": ["(<outcome> ?x - item ?y - item)"], "new_types": [], "reason": "<one short sentence>"}

or, when no allowed skill can achieve the task:

{"refuse": true, "reason": "<one short sentence>"}

Refuse whenever none of the allowed skills produces the outcome the task asks for. Do not author the nearest skill: tilting, clamping, cutting, or grasping is not a stand-in for an outcome those skills do not produce. If why_incomplete says the skill does not achieve the task, answer {"refuse": true}.

Filled in, for a skill that is deliberately not one of yours:

{"skill": "weigh", "action": {"name": "weigh", "parameters": "(?i - item ?s - item)", "precondition": "(and (holding ?i) (camera-aimed-at ?s))", "effect": "(and (weighed ?i ?s))"}, "new_predicates": ["(weighed ?i - item ?s - item)"], "new_types": [], "reason": "the scale reports a mass once the item rests on it"}

Rules:
- "skill" must be one of the allowed skills; "action.name" must be that skill's PDDL action name.
- "parameters", "precondition" and "effect" must be balanced PDDL s-expressions; every variable used must be declared in "parameters".
- Types belong in "parameters" and "new_predicates" only, never inside a precondition or an effect.
- Name the new predicate after what the skill achieves. Do not copy the placeholder names used in this instruction.
- Respect the parameter count stated for the skill.
- Use only types and predicates that already exist in the domain, plus any you declare in "new_predicates".
- The effect must assert a NEW positive predicate that you also list in "new_predicates", and that does not already exist in the domain. That fact is what the planner will use as the goal, so reusing an existing predicate such as (holding ?x) makes the action useless and will be rejected.
- Preconditions must be reachable with the actions already in the domain (for example require (holding ?x) rather than a fact nothing can produce).
