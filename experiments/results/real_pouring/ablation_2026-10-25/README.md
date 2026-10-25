# Closed-loop ablation on Tracy, 2026-10-25 — no bank / fills only, corn and styrofoam

The feedback ladder of the goal sweep (`../sweep_2026-10-24/`), one pour per goal: the same
controller, settings and uncalibrated built drain (repose 0°, discharge 0.2, rate 0.8 /s), but

- **no bank** (`*_nobank`): the live bank is off; the controller integrates its own drain and
  stops when its believed fill reaches the goal. No camera count is recorded (the stream
  detector runs inside the bank), so the "told" fill is the controller's own belief, read from
  `transfer_control_cycles.npz` (`world_positions`, receiver fill-level DOF).
- **fills only** (`*_fillsonly`): the bank runs and tells the controller the measured receiver
  and source fills, but its drain corrections are withheld (`corrects_drain: false`,
  commit a8174bd1d4); the controller pours on the drain as built. `live_bank_trace.json`
  holds what the bank saw and which hypothesis it would have committed to.
`transfer_ending.txt` says how each pour ended: `goal_reached`, `time_limit` (60 s), or
`stopped_by_operator` — the operator halted the pour by hand (Ctrl+C) because the fill no longer
progressed and the robot's motion had become unstable while the controller held the stall.

- **bank** = the sweep of 2026-10-24 (`../sweep_2026-10-24/dashboard/pours.json`), two pours
  per goal, same goals.

Code on the robot: `pouring-constraint-inspector` at fadce9b74b (corn, both conditions;
styrofoam no bank) and 46bd8fa685 (styrofoam fills only); both after the sweep's 918254f00e
by the fills-only switch (a8174bd1d4), the 60 s transfer time limit with Ctrl+C ending
recorded in `transfer_ending.txt` (46bd8fa685), and the reset of the believed fills after
the pre-tilt (fadce9b74b, needed because the uncalibrated drain integrates a phantom stream
during the pre-tilt when no bank overrides the fills). The wall recalibration 2d1bff5063
came after these bags (14:31); they use the lifted-receiver calibration, as the sweep did.
Styrofoam ran 12:47–13:24, corn 13:37–14:00. Every pour weighed.

- `goalSweep-*/`: one folder per pour, unpacked from `~/Downloads/goalSweep-corn-fillsonly.zip`,
  `goalSweep-corn-noBank.zip`, `goalSweep-styrodoam-fillsonly.zip`, `goalSweep-Styrofoam-noBank.zip`
  (mcap, pour_settings.json, weighed_particles.json, transfer_control_cycles.npz,
  transfer_ending.txt, live_bank_trace.json where the bank ran).
- `analysis/`: `rows.py` (per-pour rows), `ladder_rows.{json,csv}` (the 22 ablation pours plus
  the 22 sweep pours at the same goals), figures and their script.

## Headline (percentage points of the receiver)

| | none (no bank) | fills only | fills + bank (10-24) |
|---|---|---|---|
| corn, goals 0.1–0.5 | n=5, mean −23.9, MAE 23.9, 0 within 5; every pour "goal reached" by the controller's own belief in 6–19 s, weighed 0.00–0.14 | n=5, mean −10.0, MAE 10.0, 1 within 5; 4 of 5 stalled below the goal and were ended by the operator (15–41 s without flow at 68–88°), 1 hit the 60 s limit | n=10, mean −1.2, MAE 2.6, 9 within 5 |
| styrofoam, goals 0.1–0.6 | n=6, mean −32.2, MAE 32.2, 0 within 5; "goal reached" in 3–15 s, weighed ≤ 0.06 | n=6, mean −11.6, MAE 11.6, 1 within 5; all 6 ended by the operator, stalled at 35–81° | n=12, mean +20.8, MAE 20.8, 1 within 5 |

The controller part (told − goal) is ≈ 0 for every no-bank pour: the controller stopped
exactly where it believed the goal, and the belief was wrong by the whole error. In fills only
the told fill is the camera's, the controller part is −2 … −20 pp: told the truth, the built
drain predicts flow at the held tilt, the predicted fill sits at the goal, and nothing flows —
the stall of the particle-simulation table (fill feedback with a too-early onset), on hardware.
