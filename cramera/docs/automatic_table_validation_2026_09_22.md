# Automatic table placement — 2026-09-22

## Reported failure and correction

The Builder plan was Park arms → Move torso HIGH → Transport milk with a semantic
Table destination and no named table. Pickup succeeded, but destination grounding
was still running after 25 minutes. The provider exhausted each table's candidates
in annotation order, starting with coffee and bedside tables about 13.5 metres
away, before reaching the dining table about 2.5 metres from the object.

Placement candidates now share a world-distance ordering across all matching
surfaces. Exact named selections, physical support checks, occupancy checks and
collision avoidance retain their previous behavior. The Builder calls the option
`automatic (nearest first)` and reports when it is searching for a reachable
placement.

## Reproduction

The original generated plan was rerun with its original robot state and object
poses. All 13 actions succeeded, including pickup, two navigation actions, placement
and final arm parking. The final object is detached from the gripper.

The recording has 4,918 frames and lasts 98.05 seconds:
[Replay the original Builder plan](http://localhost:8712/index.html?scene=cramera_pr2_automatic_table_2026_09_22).

The regression input is
`test/cramera_test/dataset/builder_first_found_transport.json`.

## Checks

- Six new surface-ordering cases: four failed before the fix; all six pass after it.
- All 23 placement tests pass; the placement provider has 100% statement coverage.
- Six Builder status/state/restart/pose tests pass, including placement search,
  resumed motion, missing snapshots and terminal results.
- The generated unnamed-table transport passes in 107.56 seconds. It checks
  carrying, collision avoidance, release and physical tabletop support.
- The existing named-table acceptance completes the transport, but one run fails
  its exact two-navigation assertion because it records three navigation attempts.
  That assertion is unchanged. Surface sampling is stochastic; the named-table
  candidate ordering is unchanged by this correction.
- A diagnostic repeat of that named-table test records two successful navigation
  actions, then stops on a gripper-fingertip/milk collision (14.6 mm penetration).
  This additional case remains unresolved; these results do not establish reliable
  transport for every sampled placement and starting state. Collision checks were
  not disabled or relaxed.
