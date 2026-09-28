; Domain template 1: Base tabletop manipulation
; Use when: flat surfaces only, no stacking, no containers, no navigation.
; Primitives covered: pick, place, look_at
; Session 23: dropped vacuous (reachable ?o) — it was always asserted, never computed.

(define (domain manipulation-base)
  (:requirements :strips :typing :equality :conditional-effects)

  (:types
    item     - object   ; graspable objects (cup, box, tool, ...)
    location - object   ; fixed surfaces (table, shelf, floor, ...)
  )

  (:predicates
    (on ?i - item ?l - location)         ; item rests on a surface
    (clear ?i - item)                    ; nothing on top - always true in this template
    (holding ?i - item)                  ; gripper holds this item
    (gripper-empty)                      ; gripper is free
    (camera-aimed-at ?i - item)          ; wrist camera oriented toward item
  )

  (:action pick
    :parameters (?i - item ?l - location)
    :precondition (and (on ?i ?l) (clear ?i) (gripper-empty)
                       (camera-aimed-at ?i))
    :effect (and (holding ?i)
                 (not (gripper-empty))
                 (not (on ?i ?l)))
  )

  (:action place
    :parameters (?i - item ?l - location)
    :precondition (holding ?i)
    :effect (and (on ?i ?l)
                 (clear ?i)
                 (gripper-empty)
                 (not (holding ?i)))
  )

  ; Camera aims at exactly one item: looking at ?i retracts any previous target.
  (:action look-at
    :parameters (?i - item)
    :precondition (gripper-empty)
    :effect (and (forall (?prev - item)
                   (when (not (= ?prev ?i))
                     (not (camera-aimed-at ?prev))))
                 (camera-aimed-at ?i))
  )
)
