# Inequality constraints do not hold their bound during a motion

Write-up of an investigation from 2026-09-13. Everything needed to understand the problem
and to reproduce it is on this branch:

- this document, and
- `test/giskardpy_test/test_motion_statechart/test_feature_goal_band_holding.py`, three
  tests that show the failure against `main` with nothing but a joint goal and a
  feature goal.

The problem was first noticed as bumpy motion in a cup-to-cup pouring demo. It turned out
to be general: **giskardpy's inequality constraints do not reliably keep a quantity
inside its bounds while another goal is moving the robot, and the cause is how the
constraint is written into the QP, not how it is tuned.** The pouring demo is where it
was measured, not where it lives. Section 6 summarises what the demo taught, so nobody
has to read that branch.

---

## 1. The problem in plain words

An inequality constraint such as "keep the cup's rim at least 3 cm above the table"
becomes one row in the QP. That row does not say "stay above 3 cm now". It says: *the
sum of all planned velocity steps over the whole prediction horizon must bring the
quantity back inside the bound.* It says nothing about *when* inside the horizon that
happens.

The optimizer is free to put the correction into the late blocks of the horizon, and it
prefers to, because moving later is cheaper against the velocity and jerk regularizer
than moving now. The resulting plan is perfectly legal: at the end of the horizon the
quantity is back inside, the slack is zero, the row reports no violation.

But the controller only executes the first block of that plan. In the next cycle it
replans from the same state, and the same deferral is again the cheapest legal plan. So
the correction is permanently *scheduled* and never *executed*. The quantity sits outside
its bound, and the constraint that is supposed to guard it considers itself satisfied
the whole time.

Three consequences follow, and all three were measured (section 4):

- **A longer horizon makes it worse.** More blocks means more room to park the correction
  beyond the executed one. The excursion grows linearly with the horizon.
- **Weight and reference velocity cannot help.** They only price the slack, and the slack
  is zero, because the row is genuinely satisfied on its own terms.
- **Joint velocity limits shrink it but do not remove it.** Tight limits shrink what the
  late blocks can promise, so less can be deferred. The mechanism is unchanged.

The fix therefore has to change *what the row constrains* (section 7): either make
"move now" the cheapest legal plan by weighting early blocks, or constrain the executed
step directly.

---

## 2. How a constraint becomes a QP row today

`giskardpy/src/giskardpy/qp/enforcement_strategy.py` holds three ways of turning a
constraint into rows:

- **`IntegralStrategy`** — one row for the whole horizon, constraining the *integral* of
  the expression's derivative. The jacobian is repeated identically across every
  velocity block. This is what `add_inequality_constraint` uses, and therefore every task
  inequality in the system: collision avoidance, all feature functions, the pouring
  clearance. **This is the row described in section 1.**
- **`VelocityStrategy`** — constrains the derivative directly *at every timestep* (one row
  and one slack per constraint per step). Reached via `add_velocity_constraint`.
- **`TerminalStatePredictionStrategy`** — built for the pouring fill equation; scales the
  blocks so earlier velocities carry more of the predicted change. It is the one strategy
  that *does* force near-term action, and that contrast is what exposed the problem.

Two properties of the integral row matter for the discussion:

- Its demanded change is clamped to `reference_velocity · dt · horizon`
  (`IntegralStrategy._apply_cap`), so `reference_velocity` is a cap on the correction, not
  a speed the robot moves at.
- Its slack weight is `weight / (reference_velocity² · control_horizon)`. The
  `DefaultWeights` tiers are therefore *not* the priority order a caller sees: in the
  pouring demo three tasks set to the same tier ended up spanning 278× in effective
  weight, and the task marked `WEIGHT_MAXIMUM` sat 2.8× *below* one a tier lower. This is
  a documentation and API problem in its own right, independent of the deferral.

The same form guards every threshold in the system. `collision_avoidance.py` builds its
distance constraint exactly like a feature goal (`add_inequality_constraint`, default
`IntegralStrategy`, unbounded slack), and pairs it with a `CancelMotion` monitor — the
design already does not trust the constraint to hold the distance, and aborts instead.

---

## 3. How to see it: the test on this branch

`test_feature_goal_band_holding.py::TestFeatureGoalGuardsHeldCupWhileWristRotates`

**Scene.** Tracy holds a box-shaped cup upright in its left gripper above a table with a
reference cup on it. Every Tracy joint is set to a 1.0 rad/s velocity limit (Tracy's
default tightens them to 0.2 rad/s; see section 4 for why that matters).

**Motion.** A feature goal — the *guard* — bounds one quantity of the held cup to a band
and first brings it into that band from outside. **Only once the guard observes the
quantity inside the band** does a `JointPositionList` start rotating `left_wrist_3` by
2.0 rad at 0.3 rad/s. On its own that rotation would push the quantity far out of the
band; the rest of the arm has to compensate, and it can — the final configuration
satisfies both goals every time.

**Contract.** Deliberately the loose one: a motion may start outside the band and be
brought in, but *once inside, the quantity must never leave again* (`BandTrace` in the
test file: first sample inside the band, then no sample beyond the band ± 0.1 mm or
0.1 mrad). Each test is `xfail(strict=True)`, so the markers flip to failures the moment
the deferral stops.

**Result** (simulation, 80 Hz, prediction horizon 180):

| guard | band | rotation alone would move it to | excursion after entering the band |
|---|---|---|---|
| `HeightGoal` — rim centre above reference rim | [0.03, 0.08] m | ≈ −0.06 m | **5.2 mm** below the floor |
| `AngleGoal` — cup tilt from upright | [0.3, 0.5] rad | ≈ 2.3 rad | **136 mrad** above the ceiling |
| `DistanceGoal` — planar rim-to-rim distance | [0.15, 0.20] m | ≈ 0.01 m | wrist goal never converges (section 5) |

For the height and angle guards the row's slack stayed below 1e-5 throughout (measured
on the horizon-120 runs of this scenario): a persistent position error with zero slack
is the fingerprint of the deferral.

Run it with the markers ignored to see the numbers:

```
pytest test/giskardpy_test/test_motion_statechart/test_feature_goal_band_holding.py --runxfail
```

---

## 4. What was measured: nothing a caller can set removes it

Sweep on the `HeightGoal` scenario (box cup, wrist at 1.0 rad/s, joints at 1.0 rad/s).
Excursion below the 0.03 m floor after the band was entered:

| horizon | weight 1 | weight 10000 | reference velocity 0.05 / 0.1 / 0.2 / 0.5 m/s |
|---|---|---|---|
| 60 blocks (0.75 s) | 1.1–1.3 mm | 1.1–1.3 mm | 1.1 / 1.3 / 1.2 / 1.2 mm |
| 120 blocks (1.5 s) | 3.3–3.4 mm | 3.3–3.5 mm | 3.4 / 3.3 / 3.4 / 3.4 mm |
| 180 blocks (2.25 s) | 5.5–5.7 mm | 5.7–5.9 mm | 5.7 / 5.6 / 5.5 / 5.7 mm |

- **Weight: no effect** over four orders of magnitude. The row's slack is ~0 (≤ 1e-5 even
  at weight 10000), so its price is never paid and the weight never enters the solution.
- **Reference velocity: no effect** from 0.05 to 0.5 m/s. It caps the demanded change and
  scales the slack weight; neither touches a row that is satisfied by deferral.
- **Speed of the disturbance: no effect.** Wrist at 0.1 / 0.3 / 1.0 rad/s gives 3.8 / 3.4 /
  3.4 mm at horizon 120. A slower disturbance just spends more ticks in the same deferred
  state.
- **Prediction horizon: linear**, ≈ 1.2 mm per 60 blocks. The excursion is the amount of
  correction the optimizer can park beyond the executed block.
- **Joint velocity limits: the one thing that moves it, and it is not a task parameter.**
  The same scenario at horizon 120 gives 1.84 mm with joints at 1.0 rad/s and 0.13 mm
  with Tracy's default 0.2 rad/s tightening. Tight box limits shrink what the horizon can
  reach and with it the deferrable amount; the 0.13 mm is still a violation and still
  grows with the horizon (0.29 mm at 180).

`AngleGoal` shows the same deferral and, at a fast disturbance, a second failure on top
of it:

| wrist speed | excursion above 0.5 rad | max slack |
|---|---|---|
| 0.1 rad/s | 59 mrad | 3e-6 |
| 0.3 rad/s | 104 mrad | 8e-6 |
| 1.0 rad/s | 544 mrad | **2e+1** |

At 1.0 rad/s the slack is real: holding the cup's tilt while wrist 3 rotates needs the
forearm to re-orient, and at that speed the compensating joints saturate their velocity
limits. That is an honest, priced violation, not deferral, which is why the tests rotate
at 0.3 rad/s: they measure deferral only. (The angle excursion is much larger than the
height one because, in angle units, the horizon reaches much farther, and the deferred
correction is a fraction of that reach.)

**Conclusion.** Holding a bound with an `IntegralStrategy` row cannot be made reliable by
tuning. Every knob a caller has — `weight`, `reference_velocity`/`maximum_velocity` — is
invisible to a row whose slack is zero, and the one knob that changes the excursion is
the horizon, in the wrong direction for smoothness.

---

## 5. A side finding: `DistanceGoal`'s "stability" rows

`DistanceGoal.build_artifacts` adds, per axis of the tip-to-reference vector, an extra
inequality with `lower_error = upper_error = 0` under the comment
`# An extra constraint that makes the execution more stable`. That is a soft equality
demanding *zero change* of the component over the horizon: not a distance constraint but
a damping term holding the tip still, at the goal's full weight, on all three axes.

Measured in the section-3 scenario over 24 configurations (weights 1 and 10000,
reference velocities 0.05–0.5, horizons 60/120/180): the wrist goal *never* converges —
three damping rows outvote one joint row even at equal weight — and the motion times out
after 50 s. When the band row was forced to act alone (weight 1, horizons 60/120) the
excursion was 16–36 mm, worse than the height goal's.

So the comment describes an empirical patch for the deferral: it masks it by damping all
motion, and it makes `DistanceGoal` unusable in parallel with any goal that needs to move
the tip. The distance test on this branch is therefore `xfail(strict=True,
raises=TimeoutError)`; once the damping rows go it should fail on the band contract like
the other two, and the marker needs changing.

---

## 6. Where it was found: the pouring demo, in brief

The demo (branch `constraint-inspector`, not needed to follow this document) pours from a
cup held by Tracy into a cup on a table. Among its tasks: a fill task driving the
receiver's predicted fill level, an aiming task, and a *clearance task* keeping the
pouring lip at least `minimum_clearance` above the receiver's rim — an ordinary
`add_inequality_constraint`, i.e. an `IntegralStrategy` row.

**What was seen on hardware.** With the floor at 0.08 m, the lip sat at 0.035 m — 56%
below its floor — for five seconds. In every cycle the clearance row planned the full
+0.045 m recovery over the 2.2 s horizon, commanded a *downward* lip velocity for the
executed step, and reported exactly zero slack. Nothing was at a joint limit; the arm was
using 0.3–3% of its velocity range. The fill row, built with the front-loading
`TerminalStatePredictionStrategy`, did move the robot: same problem, two strategies, only
one forces near-term action.

**Why it was worse than the generic case.** The physics gate that lets liquid flow is a
logistic of the very same clearance the task bounds. Below ~0.06 m the two rows' gradients
become collinear (cosine 0.997), so a 44× effective-weight difference decides where along
one joint-space direction the lip sits — and the fill row's gate term turned out to be the
only thing actually lifting the lip, because the clearance row did not work. Removing
that term from the control gradient ("it is redundant with the clearance task") made the
rims interpenetrate and the pour miss the cup by 59 mm on hardware. Lesson: a constraint
that is being masked by another row looks fine until the other row is changed.

**What was tried and did not help** — do not retry:

| hypothesis | result |
|---|---|
| Raise the fill task's `reference_velocity` to widen its cap | worse, monotonically: it also lowers the slack weight, so the reversal grows; infeasible at 0.12 |
| Same, with `weight` scaled by the velocity² to compensate | infeasible above 0.05 |
| Bound the clearance row's slack (`lower/upper_slack_limit`) | no effect at all — the slack is already zero, there is no price to cap (found in passing: `IntegralStrategy` was ignoring those limits entirely; fixed on that branch, no behavioural effect since no caller sets them) |
| Take the gate out of the control gradient | catastrophic on hardware, see above; the simulation did not reproduce it |

**What helped, partially.**

- Prediction horizon 120 → 180 smoothed the motion (peak wrist acceleration 0.64 → 0.46,
  a task cost integral 16651 → 93) — and, by section 4, made the steady clearance
  violation larger. Both effects are real and pull against each other.
- A **front-loaded integral row** (`FrontLoadedIntegralStrategy` on that branch): the
  horizon blocks are weighted by a ramp with unit mean so early velocities carry more of
  the demanded change, making "move now" the cheapest way to satisfy the row. On hardware
  with the floor at 0.03 m: rims never crossed (before: 13% of cycles), landing error
  median 0.8 mm (before 2.8 mm), pour finished in 14.7 s. But the floor was still not
  held — 25% of cycles below it, settling on it, excursion to 0.013 m — and the ramp's
  4:1 emphasis is uncalibrated (a ramp decaying to zero made the QP infeasible, 4:1 did
  not; that is the whole justification). Front-loading *reduces* deferral; it does not
  prevent it.

**The simulation understates all of this.** It reproduces the mechanisms but not their
severity, and it entirely missed the hardware failure above. Anything load-bearing has
to be validated on the robot.

---

## 7. Proposed fix

The structural answer is a **control barrier function at the velocity level**:

```
∂c/∂q · q̇  ≥  −α · (c − c_min)
```

The permitted rate of *decrease* of the margin shrinks to zero as the margin closes, so
`c ≥ c_min` becomes forward-invariant: deferral is impossible because the row binds the
velocity block that is actually executed, every cycle. `α` is an approach-rate bound in
1/s, not a gain traded against stability.

giskardpy already has the machinery: `VelocityStrategy` applies constraints to the
derivative at every timestep, is reached via `add_velocity_constraint`, and honours slack
limits, so the row can be made near-hard rather than priced.

Caveat: forward invariance is a continuous-time result with exact dynamics.
Discretisation, the linearised jacobian and soft slack erode it. The claim is
"structurally cannot defer", not "provably never violated".

Two cheaper companions, matching existing practice:

- Front-loading the integral row (section 6), if a calibrated emphasis can be found.
- Wiring the constraint's observation to a `CancelMotion`, the way collision avoidance
  does. That aborts on violation instead of holding the margin, but it is one node and no
  tuning.

The three tests on this branch are the acceptance test for any of these: they have no
`enforcement_strategy` parameter to opt into, so a fix has to reach them through
`add_inequality_constraint`'s default, and the strict markers flip the moment it does.
