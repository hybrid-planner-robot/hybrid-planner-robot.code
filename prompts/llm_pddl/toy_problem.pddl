; Matching toy problem for the llm_pddl few-shot (invented; not a repo template).

(define (problem toy-demo)
  (:domain toy-table)
  (:objects
    red-widget - widget
    table-pad shelf-pad - pad
  )
  (:init
    (resting red-widget table-pad)
    (free-hand)
  )
  (:goal (resting red-widget shelf-pad))
)
