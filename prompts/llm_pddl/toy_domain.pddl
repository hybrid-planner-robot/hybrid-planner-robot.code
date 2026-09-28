; Invented few-shot toy for llm_pddl (not a repository template).
; Two types, three predicates, two actions.

(define (domain toy-table)
  (:requirements :strips :typing)
  (:types widget pad)
  (:predicates
    (resting ?w - widget ?p - pad)
    (gripped ?w - widget)
    (free-hand)
  )
  (:action pick
    :parameters (?w - widget ?p - pad)
    :precondition (and (resting ?w ?p) (free-hand))
    :effect (and (gripped ?w)
                 (not (resting ?w ?p))
                 (not (free-hand)))
  )
  (:action place
    :parameters (?w - widget ?p - pad)
    :precondition (gripped ?w)
    :effect (and (resting ?w ?p)
                 (free-hand)
                 (not (gripped ?w)))
  )
)
