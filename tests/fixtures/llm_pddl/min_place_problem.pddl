; Matching minimal problem for llm_pddl CI smoke (invented; not a repo template).

(define (problem place-cube)
  (:domain toy-place)
  (:objects
    wood_cube - widget
    table shelf - pad
  )
  (:init
    (resting wood_cube table)
    (free-hand)
  )
  (:goal (resting wood_cube shelf))
)
