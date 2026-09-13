# Constraint enforcement investigation

Handoff note for an agent picking this up. Written 2026-09-13 on branch `constraint-inspector`.

The investigation started from bumpy motion in `demo_pouring_transfer.py` and ended
somewhere more general: **giskardpy's inequality constraints do not reliably hold a
bound during a motion, and the failure is structural rather than a tuning problem.**
The pouring demo is where it was measured, not where it lives.

Read section 3 (what is confirmed, F7 in particular) and section 4 (what was falsified)
before proposing anything. Four plausible fixes were tried and three of them made the robot worse.

---

## 1. The open questions this hands over — both answered on 2026-09-13

1. **Does this affect any motion that combines several constraints, not just pouring?**
   **Yes.** `HeightGoal` and `AngleGoal` both let their quantity leave its band under a
   plain joint-goal disturbance, with no fill task and no terminal-prediction row taking
   part. `DistanceGoal` fails differently (see F8). Tests:
   `test_pouring.py::TestFeatureGoalGuardsHeldCupWhileWristRotates`, three
   `xfail(strict=True)` tests, one per feature goal.

   The scenario: on the `tracy_transfer_world` fixture a feature goal first brings its
   quantity into a band from outside; **only once the goal observes the quantity inside
   the band** (a `start_condition`) does a `JointPositionList` rotate `left_wrist_3` by
   `_WRIST_ROTATION` at `_WRIST_SPEED = 0.3 rad/s`, which alone would push the quantity
   out. The safety contract asserted is the loosened one: *once inside the band, never
   leave it again* (`BandTrace.worst_excursion_after_entry() == 0.0`). The same contract
   now applies to the pouring clearance test (excursion there: 6.4 mm).

   Note `_WRIST_ROTATION = 2.5` from the fixture's wrist position 4.29 rad exceeds the
   joint's 2π limit; `JointPositionList.apply_limits_to_target` clips it, so the
   effective rotation is ~2.0 rad.

   | guard | band | disturbance alone would move it to | excursion after entry | ticks outside |
   |---|---|---|---|---|
   | `HeightGoal` (rim centre above receiver rim) | [0.03, 0.08] m | ≈ −0.06 m | **3.4 mm** below | 142 / 476 |
   | `AngleGoal` (cup tilt from upright) | [0.3, 0.5] rad | ≈ 2.3 rad | **104 mrad** above | 301 / 615 |
   | `DistanceGoal` (planar rim-to-rim) | [0.15, 0.20] m | ≈ 0.01 m | wrist goal never converges | — |

2. **Is `TerminalStatePredictionStrategy` implicated, or does it reproduce without it?**
   **It reproduces without it.** None of the tests above contain a fill task; the only
   rows are `IntegralStrategy` rows (the guard's inequality and the wrist's equality).
   The defect is in `IntegralStrategy` alone; the terminal prediction is a red herring.
   The pouring case is worse than the feature-goal cases because its disturbance, the
   fill row, is itself a front-loaded row pulling on the same joints (F4).

3. **Can it be tuned away with weights and reference velocities?** **No — see F7.**

---

## 2. The system under investigation

| thing | where |
|---|---|
| constraint → QP rows | `giskardpy/src/giskardpy/qp/enforcement_strategy.py` |
| constraint builders | `giskardpy/src/giskardpy/qp/constraint_collection.py` |
| fill/clearance tasks | `giskardpy/src/giskardpy/motion_statechart/tasks/pouring.py` |
| transfer gate (physics) | `semantic_digital_twin/.../semantic_annotations/mixins.py` (`_geometric_transfer_gate`) |
| pouring ODEs | `semantic_digital_twin/.../physics/equations/pouring_equations.py` |
| the demo | `demo_pouring_transfer.py` (talks to a separate Giskard process) |
| live controller config | `giskardpy/.../scripts/iai_robots/tracy/tracy_standalone.py` |

Three enforcement strategies exist:

- **`IntegralStrategy`** — one row for the whole horizon, constraining the *integral* of
  the expression's derivative. The jacobian is repeated **identically** across every
  velocity block. Used by `add_inequality_constraint`, therefore by every task
  inequality: pouring clearance, collision avoidance, all feature functions.
- **`VelocityStrategy`** — constrains the derivative directly **at every timestep**
  (one row and one slack per constraint per step). Used by `add_velocity_constraint`.
- **`TerminalStatePredictionStrategy`** — for the fill ODE; scales blocks by
  `horizon_normalized_weights` so earlier velocities carry more of the predicted change.

---

## 3. Confirmed findings

### F1 — `IntegralStrategy` silently discarded slack limits (fixed)

`create_slack_variables` hardcoded `±LargeNumber` instead of reading
`c.lower_slack_limit` / `c.upper_slack_limit`. `VelocityStrategy` reads them correctly.
So `add_inequality_constraint`'s documented "how much the lower error can be violated"
parameters were accepted, stored on the constraint, and dropped.

Fixed in `enforcement_strategy.py`; covered by
`test/giskardpy_test/test_qp/test_integral_strategy.py` (2 tests). **No behavioural
effect today** — no caller passes non-default limits, so it computes what it computed
before. It matters only once someone tries to bound a violation.

### F2 — the core defect: an integral row is satisfiable without ever acting

This is the important one.

`IntegralStrategy` asks "will the quantity have recovered *somewhere* in the 178-block
horizon". The QP satisfies that with a plan whose motion sits in later blocks (cheaper
against the velocity/jerk regularizer at the current state), executes only the first
block, and re-plans the identical deferral next cycle. **The row reports zero violation
while the quantity it guards stands still.**

Measured on hardware (`/tmp/control_cycles/goal_3.npz`, the 2026-09-12 23:33 run,
`minimum_clearance=0.08`):

```
   t   clearance   needed   planned over 2.2s   plan avg m/s   commanded now m/s
6.89     0.0353   +0.0447        +0.0447          +0.0201           -0.0071
8.14     0.0358   +0.0442        +0.0442          +0.0199           -0.0039
8.76     0.0351   +0.0449        +0.0449          +0.0202           -0.0007
```

Slack was **exactly 0.00000** for the whole window. The arm used 0.3–3% of its available
joint velocity; nothing was at a limit. The clearance sat 56% below its floor for five
seconds with a constraint that considered itself satisfied.

The contrast that proves the mechanism: the fill row uses
`TerminalStatePredictionStrategy`, which front-loads, and it *does* move the robot.
Same problem, two strategies, only one forces near-term action.

### F3 — `weight` does not mean priority

Effective slack weight is `weight / (reference_velocity² · control_horizon)`
(`enforcement_strategy.py:41`). The `DefaultWeights` tier ladder is therefore not the
priority ordering. From the 2026-09-12 20:39 recording (`prediction_horizon=120`):

| task | tier set in the demo | ref. velocity | effective weight |
|---|---|---|---|
| `FillByTransferTask` | `ABOVE_COLLISION_AVOIDANCE` (2500) | 0.03 | **23540** |
| `KeepProjectileInReceiver` | `MAXIMUM` (10000) | 0.10 | **8475** |
| `KeepSourceRimAboveReceiverRim` | `ABOVE_COLLISION_AVOIDANCE` (2500) | 0.20 | **530** |
| `AlignPlanes` | `ABOVE_COLLISION_AVOIDANCE` (2500) | 0.50 | **85** |

Three tasks set to the *same* tier span 278×, and the one task explicitly marked
`WEIGHT_MAXIMUM` ends up 2.8× **below** a task set a tier lower. The normalization itself
is defensible (its docstring says it makes different units comparable); the problem is
that `weight` is documented as "Priority weight relative to other tasks" and a caller
has no way to see the real ordering from the call site.

### F4 — the transfer gate and the clearance task constrain the same scalar

`KeepSourceRimAboveReceiverRim` (`pouring.py`) constrains
`(source.liquid_exit_point − receiver.opening_point) · Z`.
`_geometric_transfer_gate` (`mixins.py`) builds
`height_gate = logistic(source_exit.z − receiver_opening.z, sharpness)` from the
identical two points. So the fill rate is `logistic(clearance) · overlap_gate · outflow`,
and the fill row's gradient necessarily contains `g′(clearance) · ∂clearance/∂q`.

`DEFAULT_GATE_SHARPNESS = 80`, so `g′` is negligible above ~0.07 m and large below it:

| clearance | gate | g′ | cos(J_fill, J_rim) |
|---|---|---|---|
| 0.102 | 1.000 | 0.02 | +0.13 |
| 0.048 | 0.978 | 1.68 | **+0.997** |
| 0.035 | 0.943 | 4.29 | **+0.996** |
| 0.076 | 1.000 | 0.02 | −0.13 |

When the lip is in the logistic's knee the two rows become one direction in joint space
(only 6–11% of the rim gradient independent), with a 44× weight difference deciding
where along it the solution sits.

**Practical rule:** the floor must sit outside the knee, i.e.
`minimum_clearance ≳ 4.6 / height_gate_sharpness` ≈ **0.057 m** at sharpness 80.
`minimum_clearance=0.03` is inside the knee; sim showed the rims actually crossing there
(−3.0 mm at floor 0.03, −9.6 mm at 0.02).

### F5 — the gate's gradient was load-bearing, not redundant

**Do not remove the height gate from the control gradient.** It looks redundant with the
clearance task. It is not, because the clearance task does not work (F2). The gate's
`g′(clearance)·∂clearance/∂q` term was the only thing actually lifting the lip.

Hardware proof (2026-09-13 02:01 run, `height_gate_drives_control=False`):

```
   t   clearance  frozen_gate  |J_fill|  fill_weight  fill_slack  landing_error
 4.39    -0.0081       0.3442      0.00      15605.5      5.3400       0.0463
 6.39    -0.0198       0.1703      0.00      15605.5      5.3400       0.0502
 6.89    -0.0118       0.2807      0.00      15605.5      5.3400       0.0589
```

Whenever the frozen gate fell below 0.5 (198 cycles) `|J_fill|` was **median 0.000** —
the row had no gradient at all while its slack was pinned at 5.34 against weight 15605.
Clearance went negative for 13% of the motion (min −20.6 mm, rims interpenetrating), and
the landing error reached **59 mm against a 35 mm opening radius** — the pour missed the
cup.

Two distinct errors caused this:
1. Freezing the gate as a *multiplicative constant* attenuates every term in the row —
   tilt and aiming included — by the gate's value, so the row loses authority exactly
   when the gate is closing.
2. Removing the compensation exposed F2's latent breakage. Runaway: lip sinks → gate
   closes → fill gradient scaled to zero → nothing lifts the lip → sinks further.

### F6 — the same constraint form guards every threshold in the system

`collision_avoidance.py:245` builds its distance constraint exactly like the rim task:
`add_inequality_constraint` with `lower_error = buffer_zone_distance - contact_distance`,
default `IntegralStrategy`, default unbounded slack, `WEIGHT_COLLISION_AVOIDANCE` (50,
the lowest tier). The feature functions do the same.

So F2 is not pouring-specific. Note also that collision avoidance pairs its constraint
with a `_CancelBecauseCollisionViolated(CancelMotion)` monitor — the design already does
not trust the constraint to guarantee the distance, and aborts the motion instead.

### F7 — the deferral is untunable: invariant to weight, reference velocity and disturbance speed, linear in the horizon

Sweep on the `HeightGoal` scenario of section 1 (sim, 80 Hz, wrist at 1.0 rad/s unless
stated). Excursion below the 0.03 m floor after the band was entered:

| horizon | weight 1 | weight 10000 | rv 0.05 | rv 0.1 | rv 0.2 | rv 0.5 |
|---|---|---|---|---|---|---|
| 60 (0.75 s) | 1.1–1.3 mm | 1.1–1.3 mm | 1.1 | 1.3 | 1.2 | 1.2 |
| 120 (1.5 s) | 3.3–3.4 mm | 3.3–3.5 mm | 3.4 | 3.3 | 3.4 | 3.4 |
| 180 (2.25 s) | 5.5–5.7 mm | 5.7–5.9 mm | 5.7 | 5.6 | 5.5 | 5.7 |

- **Weight: no effect** over four orders of magnitude (1 → 10000, i.e. effective
  62500× after F3). Expected from F2: the row's slack is ~0 (≤ 1e-5 at weight 10000),
  so its price is never paid and the weight never enters the solution.
- **Reference velocity: no effect** from 0.05 to 0.5 m/s. It caps the demanded change
  at `rv · dt · horizon` (`IntegralStrategy._apply_cap`) and scales the slack weight, and
  neither touches a row that is satisfied by deferral.
- **Disturbance speed: no effect** — wrist at 0.1 / 0.3 / 1.0 rad/s gives 3.8 / 3.4 /
  3.4 mm. A slower disturbance just spends more ticks in the same deferred state.
- **Prediction horizon: linear.** ≈ 1.2 mm per 60 blocks. The excursion is the amount of
  correction the optimizer can park beyond the executed block, and a longer horizon
  gives it more room. This is the direct fingerprint of F2 and the reason W1 (horizon
  120 → 180) traded smoothness for a larger steady violation.

`AngleGoal` shows the same deferral at a slow disturbance and *another* failure on top
at a fast one:

| wrist speed | excursion above 0.5 rad | max slack |
|---|---|---|
| 0.1 rad/s | 59 mrad | 3e-6 |
| 0.3 rad/s | 104 mrad | 8e-6 |
| 1.0 rad/s | 544 mrad | **2e+1** |

At 1.0 rad/s the slack is real: compensating a wrist_3 rotation while holding the cup
tilt needs the forearm to re-orient, and at that speed the compensating joints saturate
their 1.0 rad/s velocity limits. That is an honest, priced violation, not deferral —
which is why the tests fix `_WRIST_SPEED = 0.3` so they measure deferral only. Note the
angle excursion (half the 0.2 rad band) is far larger than the height one: the deferred
correction is a fraction of the horizon's reachable change, and in angle units the
horizon reaches much farther.

**Joint velocity limits are the one lever that does move it — and it is not a task
parameter.** This branch removed `Tracy._setup_velocity_limits`, which tightened every
Tracy degree of freedom to 0.2 rad/s; here the joints run at their URDF limits
(1.0 rad/s). Running the identical `HeightGoal` scenario (box cup, horizon 120) with both
configurations:

| joint velocity limits | excursion after entry |
|---|---|
| 1.0 rad/s (this branch) | 1.84 mm |
| 0.2 rad/s (`main`, via `tighten_dof_velocity_limits_proportionally`) | 0.13 mm |

Every number in this section and in section 1 was measured at 1.0 rad/s. The
disturbance speed itself does not matter (above), so the effect runs through the
compensating joints: tight box limits shrink what the horizon can reach, and with it the
room the optimizer has to park the correction. Tight limits do not remove the deferral —
0.13 mm is still above the 0.1 mm tolerance and still grows with the horizon (0.29 mm at
180). The standalone reproduction on branch `constraint-enforcement-deferral` therefore
sets every Tracy joint to 1.0 rad/s itself and measures 5.2 mm at horizon 180, the same
regime as this branch.

**Conclusion:** holding a bound with an `IntegralStrategy` row cannot be made reliable by
tuning. Every knob a caller has (`weight`, `reference_velocity`/`maximum_velocity`) is
invisible to a row whose slack is zero, and the one knob that moves the excursion is the
horizon, in the wrong direction for smoothness. The fix has to change what the row
constrains (section 6).

### F8 — `DistanceGoal`'s "stability" rows are a damping term that starves competing goals

`DistanceGoal.build_artifacts` adds, per axis of the tip-to-reference vector, an
inequality with `lower_error = upper_error = 0` — a soft equality demanding *zero change*
of that component over the horizon. That is not a distance constraint; it is a
regularizer holding the tip still, at the goal's full weight, on all three axes.

Consequences measured in the section-1 scenario (24 configurations: weights 1 and
10000, rv 0.05–0.5, horizons 60/120/180):

- **The wrist goal never converges** — every configuration hit the 4000-tick (50 s)
  timeout. Three damping rows at weight 2500 (or even weight 1, matching the wrist goal)
  outvote one wrist row.
- In a confounded earlier run (wrist rotating *before* the band was entered) the
  distance was the only guard that "held" — because the damping rows resisted the
  rotation itself, not because the band row worked. When the band row does have to act
  alone (weight 1 rows, horizon 60/120) the excursion is 16–36 mm, worse than height.

So the comment `# An extra constraint that makes the execution more stable` is exactly
the empirical patch section 1 suspected: it masks F2 by damping all motion, and it makes
`DistanceGoal` unusable in parallel with any goal that needs to move the tip. The test
`test_distance_goal_keeps_the_rims_apart` is `xfail(strict=True, raises=TimeoutError)`
for this reason; once the damping rows go, it should flip to failing on the band
contract like the other two, and the marker needs changing accordingly.

---

## 4. Falsified hypotheses — do not retry these

| hypothesis | result | why it failed |
|---|---|---|
| Raise the fill task's `reference_velocity` to widen the bound cap | **worse, monotonically** | `reference_velocity` sets both the cap *and* the slack weight (`weight/rv²`). A wider cap licenses a bigger demanded change per cycle, so the reversal gets larger. peak accel 0.644 → 0.999 across 0.03→0.30; QP infeasible at 0.12. |
| Same, with `weight` raised by `(rv/0.03)²` to compensate | **infeasible above rv=0.05** | Restores stiffness and over-constrains the QP. |
| Bound the clearance row's slack (`lower/upper_slack_limit`) | **no effect whatsoever** | Slack is *already exactly zero* during the violation (F2). There is no price being paid, so capping the price changes nothing. Sim: min clearance 0.07352 → 0.07350. |
| Freeze the whole transfer gate out of the control gradient | **catastrophic on hardware** | See F5. Sim did not reproduce it — sim clearance never got near zero in that variant, so the sweep was blind to the failure. |
| Freeze only the height gate, keep overlap live | untested in isolation | Better founded, but sim cannot distinguish it from the full freeze (landing error 0.0006 vs 0.0007). Superseded by the front-loading fix. |

**Methodological warning:** the sweep that blessed the whole-gate freeze measured
clearance, gradient alignment, `|J_fill|`, fill level, velocity sign changes and peak
acceleration — but **not landing error**, the one quantity the change removed from the
gradient. Decide the metric from what the change touches, not from what is convenient.

---

## 5. What actually worked

### W1 — prediction horizon 120 → 180 (landed by the user in `tracy_standalone.py`)

Sim, fill task only:

| horizon | lookahead | peak wrist_3 accel | AlignPlanes cost integral | bound saturated |
|---|---|---|---|---|
| 120 | 1.48 s | 0.644 | 16651 | 58% |
| 180 | 2.22 s | 0.456 | **93** | 38% |
| 240 | 2.98 s | 0.358 | 162 | 31% |
| 320 | 3.97 s | — | — | never converged (97% saturated) |

Usable range 180–240; 320 over-predicts and the pour crawls.

### W2 — `FrontLoadedIntegralStrategy` (in the tree, opt-in)

Weights the horizon blocks by a ramp (unit mean) so early velocities carry more of the
demanded change, making "move now" the cheapest way to satisfy the row. Enabled per-task
via `KeepSourceRimAboveReceiverRim.demands_immediate_recovery=True`.

Hardware comparison, all at `minimum_clearance=0.03`:

| | run B (gate frozen, flat) | **run C (gate steers, front-loaded)** |
|---|---|---|
| min clearance | −0.0206 | **+0.0131** |
| cycles with rims crossed | 198 (13%) | **0** |
| landing error median / max | 0.0028 / 0.0590 | **0.0008 / 0.0268** |
| `\|J_fill\|` zero-gradient cycles | 198 | **0** |
| cycles aligned ≥0.99 | 0 | **0** (max 0.946) |
| final fill | — | **0.6981** (goal 0.70) |
| duration | 18.6 s | **14.7 s** |

**Caveats.** The floor is still not held: 25% of cycles below 0.03, settling at 0.0311,
excursion to 0.0131, rim cost integral 17.5 (i.e. still reporting almost no violation
while under its floor). Peak joint acceleration rose 0.449 → 0.557. And
`early_block_emphasis = 4.0` is **uncalibrated** — a linear ramp decaying to zero
(178:1) made the QP infeasible, 4:1 did not. That is the whole justification.

---

## 6. The structural fix that was never tried

Front-loading reduces deferral; it does not prevent it, and it needs a tuned ratio. The
standard structural answer is a **control barrier function at the velocity level**:

```
∂c/∂q · q̇  ≥  −α · (c − c_min)
```

The permitted rate of decrease shrinks to zero as the margin closes, making `c ≥ c_min`
forward-invariant — deferral is impossible because the row binds the velocity block that
is actually executed, every cycle. `α` is an approach-rate bound in 1/s, not a gain
traded against stability.

giskardpy already has the machinery: `VelocityStrategy` applies constraints to the
derivative at every timestep, is reached via `add_velocity_constraint`, and already
honours slack limits (so the row can be made near-hard rather than priced).

Honest caveat: forward invariance is a continuous-time result with exact dynamics.
Discretisation, the linearised jacobian and soft slack erode it. The claim is
"structurally cannot defer", not "provably never violated".

The cheaper companion, matching existing practice: wire
`KeepSourceRimAboveReceiverRim`'s existing `artifacts.observation` to a `CancelMotion`
the way collision avoidance does. That aborts on violation rather than holding the
margin, but it is one node and no tuning.

---

## 7. Working tree state

Production changes, all opt-in and default-off unless noted:

| file | change | tested |
|---|---|---|
| `qp/enforcement_strategy.py` | F1 slack-limit fix; new `FrontLoadedIntegralStrategy` (`early_block_emphasis=4.0`) | F1 unit-tested; strategy validated on hardware only |
| `qp/constraint_collection.py` | `add_inequality_constraint(enforcement_strategy=...)`, defaults to `IntegralStrategy` | no test |
| `tasks/pouring.py` | `KeepSourceRimAboveReceiverRim.demands_immediate_recovery`; `FillByTransferTask.height_gate_drives_control`; `TerminalFillConstraintTask._control_velocity` hook + `_compile_against_world` helper | **no test for either flag** |
| `physics/equations/pouring_equations.py` | `TransferGate` dataclass; `GatedInflowEquation.height_gate` / `.overlap_gate` / `.ungated_symbolic_velocity()` | no test |
| `semantic_annotations/mixins.py` | `_geometric_transfer_gate` returns `TransferGate` | no test |
| `demo_pouring_transfer.py` | `minimum_clearance=0.03`, `demands_immediate_recovery=True` | — |

Tests:

- `test/giskardpy_test/test_qp/test_integral_strategy.py` — new, 2 tests, passing.
- `test_pouring.py::TestRimClearanceDuringTransfer::test_clearance_stays_within_the_configured_band`
  — **`xfail(strict=True)`**, documents F2. It asserts the real contract (lip stays above
  `minimum_clearance`, read off the task). It should start passing when F2 is genuinely
  fixed; if it does, remove the marker.
- `test_pouring.py::TestFeatureGoalGuardsHeldCupWhileWristRotates` — three
  **`xfail(strict=True)`** tests (`HeightGoal`, `AngleGoal`, `DistanceGoal`) documenting
  F7/F8 with `IntegralStrategy` rows only (section 1). The feature goals have no
  `enforcement_strategy` parameter, so these are the tests to turn a fix for
  `IntegralStrategy` (or `add_inequality_constraint`'s default) against. The distance
  one expects a `TimeoutError` (F8) and must be re-marked once the damping rows change.
- `BandTrace` (same file) is the loosened contract: first sample inside the band, then
  no sample beyond the band ± tolerance afterwards. The pouring clearance test uses it
  too.
- Existing tests only asserted clearance `> 0.0` (rims don't cross), never the floor,
  which is why F2 survived.

**`height_gate_drives_control=False` is dangerous — do not enable it** (F5). The flag is
kept only because the split is a prerequisite for any future attempt, and because
`_control_velocity` is a reasonable seam.

Two **pre-existing breakages** from the main merge, unrelated to this work:

1. `World.set_positions_1DOF_connection` was removed (commit `bf817b6e0`, replaced by
   `JointState.from_mapping(...).apply_to(world)`). 9 call sites repaired in
   `giskardpy_test`; **17 remain** in `test/coraplex_test`,
   `test/semantic_digital_twin_test` and `learned_pouring`.
2. `HasRootBody.create_with_new_body_in_world()` no longer accepts `active_axis`,
   breaking `single_cup_world.py` at fixture setup → 9 errors in `test_pouring.py`.

---

## 8. How to read a control-cycle recording

The highest-value reusable technique. Recordings land in `/tmp/control_cycles/goal_N.npz`
(enable with `record_control_cycles=True` in `tracy_standalone.py`); `goal_3` is the
transfer. Inspect interactively with
`python -m giskardpy.qp.constraint_inspector /tmp/control_cycles/goal_3.npz`, or read the
arrays directly:

- `metadata` → JSON with `structure.row_names`, `control_horizon`,
  `model_predictive_control_time_step`.
- `row_lower_bounds[cycle, row]` **is the error the row is asking for**, already capped.
  For `KeepSourceRimAboveReceiverRim` it is `minimum_clearance − clearance`, so
  `clearance = minimum_clearance − lb`. For `KeepProjectileInReceiver` the first two rows
  are the per-axis landing error (the third is zero by construction), so the aiming error
  is `norm(lb[:, [1,2]])`.
- `row_slacks` → the violation. **Zero slack with a persistent position error is the
  signature of F2.**
- `row_sensitivities[cycle, row]` → `∂expression/∂q`. Cosine similarity between two rows
  reveals collinearity (F4); its norm collapsing to zero reveals a dead row (F5).
- `row_contributions[cycle, row]` → the change planned over the **whole horizon**.
  Compare against `row_sensitivities[cycle,row] @ velocities[cycle]`, the rate commanded
  **now** — the gap between them is the deferral.
- `row_weights` → the *effective* weights, i.e. after the F3 normalization. This is the
  only place the real priority ordering is visible.

Environment: `workon cram2`, run from the repo root. The headless sim used for A/B tests
reuses `test/giskardpy_test/test_motion_statechart/test_pouring.py`'s
`tracy_transfer_world` fixture (call the fixture functions via `.__wrapped__`) plus
`_build_transfer_motion`, driven by an `Executor` with a `ControlCycleRecorder` attached.
Rebuild the world per run — deep-copying it does not re-bind the coupled inflow
equation's cached symbolic expressions. One run is ~20–40 s.

**The sim is not the robot.** It reproduces mechanisms (collinearity, gate slope, rim
crossing) but understates severity, and it entirely missed the F5 failure. Validate
anything load-bearing on hardware.

---

## 9. The predicted-value prototype on the pouring transfer (2026-09-13, not committed)

The prototype from the standalone branch's section 8 (`GeometricScheduledStrategy`: nine
predicted-value rows per constraint at steps 0, 1, 3, 7, 15, 31, 63, 127 and the last
block, slack weight shared, gap closed proportionally over the horizon) was made the
default of `add_inequality_constraint` here via a working-tree switch
(`constraint_collection.INEQUALITY_ENFORCEMENT_STRATEGY`; the rim task's non-front-loaded
path reads the same switch). Sim, `tracy_transfer_world`, horizon 120, 80 Hz.

| floor / aim weight | integral: min clearance, excursion after entry | scheduled: min clearance, excursion after entry | fill reached |
|---|---|---|---|
| 0.08, `MAXIMUM` (the strict-xfail config) | 0.0735, 6.4 mm | **0.0800, 0.0 mm** (0 of 887 ticks outside) | both |
| 0.07, `MAXIMUM` | 0.0650, 4.9 mm | 0.0675, 2.4 mm | both |
| 0.06, `MAXIMUM` | 0.0560, 3.9 mm | 0.0593, **13.4 mm above the band** | both |
| 0.05, default | 0.0443, 5.7 mm | 0.0440 — **QP fails at tick 177** | integral only |
| 0.03, `MAXIMUM` (demo) | **−0.0301 (rims cross)**, 60 mm | 0.0267 — **QP fails at tick 315** | integral only |

Tick time 13.0 vs 12.8 ms. Peak joint acceleration 0.97 vs 0.65 in every run that
finished — the braking rows stop the lip harder at the band edge.

Module run with the switch on: the three strict xfails flip to passing (clearance band at
0.08, the height guard, and the distance guard, whose wrist goal converges here), and one
real regression: `test_pouring_lip_stays_above_receiver_rim` (floor 0.05) dies with the
QP failure below.

**Why the QP fails below floor 0.06 — and it is not the new rows.** Offline replay of the
failing problem (PIQP status −1, max iterations, unchanged at 5000): dropping all nine
clearance rows and their slacks still fails; dropping *only the fill row* (the
`TerminalStatePredictionStrategy` row, weight 23540) solves in 23 iterations. Per-tick
logging shows the fill row's largest coefficient at 0.0125 for the entire integral run,
but 0.16–0.38 in the scheduled run while the lip dips into the gate's knee (ticks
110–140, clearance 0.044–0.05) and 0.59 at the failing tick, 47× normal. Mechanism: the
much heavier fill row (F3) pushes the lip below the floor along the collinear direction
(F4); the integral row lets it settle 5 mm under and the gate gradient stays small; the
scheduled rows lift it back, the lip crosses the knee twice, the gate's logistic slope
enters the fill row's gradient and the QP becomes unsolvable. Solver settings do not
help (`max_iter` 2000/5000, `reg_lower_limit` 1e-8 → same; 1e-6 → primal infeasible;
multistage KKT → fails at tick 11). Softening the rows (unit slack coefficient, i.e.
6400× softer) "solves" it but the lip then leaves the band upward by 9 cm; five rows
instead of nine finish but sink 14.5 mm under the floor.

**Reading.** In the pouring transfer the prototype does what it did in the standalone
scene — it stops the deferral — but it can only hold what the fill row lets it hold. The
fill row outweighs it 3× (F3) and is collinear with it in the knee (F4), and once the
clearance rows are stiff enough to push back, the fill row's gate gradient breaks the
solver. So on this demo the prototype is only usable with the floor outside the knee
(F4's rule, ≥ 0.06–0.07 m), which is also where it holds. Making floor 0.03 work needs the
fill row fixed first: bound or normalise the gate gradient in the fill row, or lower the
gate sharpness, or give the clearance rows priority over the fill row rather than a
weight. The harder braking (peak acceleration +50%) is the other cost to look at before
hardware.

Probes: `test/giskardpy_test/test_motion_statechart/test_probe_transfer_strategies.py`
(untracked) runs the table above; env vars `PROBE_STRATEGIES`, `PROBE_CONFIGS`,
`PROBE_MAX_ITER`, `PROBE_REG`, `PROBE_KKT`, output `PROBE_OUT`.


---

## 10. What landed (2026-09-13)

- `PredictedValueStrategy` in `giskardpy/qp/enforcement_strategy.py` is the default of
  `add_inequality_constraint`: nine predicted-value rows per constraint at steps
  0, 1, 3, 7, 15, 31, 63, 127 and the last block and the proportional approach schedule
  of section 9. Unit tests in `test/giskardpy_test/test_qp/test_predicted_value_strategy.py`.
  Equality constraints keep the integral row.
- **Weight scale.** The rows' slack weight is normalized over the horizon exactly like an
  integral row's and then shared across the rows, so a bound keeps today's weight
  relative to the goals it competes with. The prototype of section 9 skipped the horizon
  division and was therefore ~H× stiffer at equal nominal weight; that made the
  collision-avoidance buffer effectively hard against an equal-weight goal and failed
  the two `test_collision_avoidance_tasks` tests that pin down the soft-buffer semantics.
  With the horizon-normalized weight those pass unchanged, the guards still hold (the
  deferral fix does not depend on the weight), and the pouring clearance needs the
  `WEIGHT_MAXIMUM` tier to hold its floor against the fill row — at
  `ABOVE_COLLISION_AVOIDANCE` the fill row pushed the lip 17 mm out of the band and then
  broke the solver even at floor 0.08. `KeepSourceRimAboveReceiverRim` therefore
  defaults to `WEIGHT_MAXIMUM` and the demo sets it explicitly. This is F3 in action: the
  clearance row is now genuinely pushed instead of deferred, so its weight matters.
- Removed as superseded: `FrontLoadedIntegralStrategy` and
  `KeepSourceRimAboveReceiverRim.demands_immediate_recovery` (W2),
  `FillByTransferTask.height_gate_drives_control` with its `_control_velocity` seam and
  `_compile_against_world` helper (the F5 experiment), the `enforcement_strategy`
  parameters of the constraint builders, and `GatedInflowEquation.ungated_symbolic_velocity`.
  The gate split into height and overlap factors stays.
- The clearance floor is 0.07 m in the demo and as the task's default, following F4's
  rule and section 9's measurements: 0.05 and 0.03 sit in the gate's knee, where the
  fill row breaks the solver once the clearance rows push back. Making a lower floor work
  needs the fill row's gate gradient bounded first.
- Tests: the clearance band test (floor 0.08) and the height and distance guards are
  ordinary tests now; the angle guard stays a strict xfail for the residual excursion
  that is not deferral (section 8 of the standalone branch's write-up).
- Still open: hardware validation of the demo at floor 0.07 (the simulation understated
  every effect so far), the fill row's conditioning in the knee, the F3 weight story,
  and `DistanceGoal`'s damping rows (F8).
