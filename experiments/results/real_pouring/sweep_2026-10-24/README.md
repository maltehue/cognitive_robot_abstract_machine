# Goal sweeps on Tracy, 2026-10-24 — corn and styrofoam

Code state on the robot: branch `pouring-constraint-inspector` at `918254f00e` (2026-10-24 13:59).
Same controller and settings for both materials: uncalibrated drain, live hypothesis bank,
3°/s approach, 10°/s return cap, 30°/s² tilt-acceleration limit, upright-in-place turn-back,
a full source (Jeroen) cup; corn poured into the translucent cup (fill read through the wall),
styrofoam into a Jeroen cup (fill counted at the opening). Every pour weighed afterwards.

- `bags_corn/`, `bags_styrofoam/`: one folder per pour (mcap, pour_settings.json,
  live_bank_trace.json, weighed_particles.json, control-cycle recording), unpacked from
  `~/Downloads/goalSweep-24-10-corn.zip` and `goal-sweep-24-10-styrofoam.zip`.
- `dashboard/`: the summary published as the "Corn vs Styrofoam Goal Sweeps" dashboard
  (https://claude.ai/artifact/YHh9Dj7VzqmWbFGZBiBU4u): `pours.json` (one row per pour),
  `series.json` (every pour at 0.2 s), `turnbacks.json` (one row per turn-back), `index.html`.
- `analysis_scripts/`: the scripts that built those files from the bags (scratch quality).
- `learned_rate/`: the learned outflow-rate study behind the note "Effect models that improve
  with use" (https://claude.ai/artifact/KRcghtyDfR7ztLdGhc94Vc): dataset, models, leave-one-pour-out
  predictions, logs. 28 corn pours of 2026-10-16 … 10-23; open-loop scoring only.
- `analysis_2026-10-23/`: replays of the 2026-10-23 pours (forecasts, motion, controller cost).

Headline (goals ≤ 0.6, two pours each): corn 10 pours, MAE 2.6 pp, 9/10 within 5 pp, spill 1 g;
styrofoam 12 pours, mean overshoot +20.8 pp (controller part +23 pp), spill up to 17 % of the
receiver past 100°. Copied from the analysis session's scratchpad on 2026-10-24; not yet committed.
