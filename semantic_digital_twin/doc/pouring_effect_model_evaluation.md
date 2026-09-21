# Evaluating the pouring effect model

## Why this note exists

The controller is built from effect models: a task states what an action is supposed to
do to the world, and the optimizer is handed the model that predicts it. Pouring is the
first such model built far enough to be worth testing, so it is the specimen for the
question the work actually asks — *does an effect model capture the true behaviour well
enough, and where it does not, can estimating its parameters close the gap without
giving up the semantics that make the model readable?*

This note fixes what that question is being asked about: which parameters the pouring
model has, which of them have an independent ground truth, what can be measured while a
pour runs, and which experiments separate a parameter problem from a structural one.

## What the model says

The drain is a Torricelli law on the head of liquid standing above the pouring lip. For
a container of inner height `A` whose lip sits `r` from the tilt axis, at tilt `α` and
normalized fill `h`:

```
φ(h) = atan2(A − h·A, r)                 lip angle at this fill
d(α, h) = max(0, L(h) · sin(α − φ(h)))   head above the lip
ḣ = −k · d(α, h) / A                     normalized drain
v = C_d · sqrt(2 g d)                    stream speed at the lip
```

The stream then flies ballistically to the receiver's opening plane, and a pair of
logistic gates — one on the source lip's clearance above the receiver, one on how far
the landing point falls from the opening centre — multiply the drain, so liquid leaves
the source only while it would arrive somewhere.

Two properties matter for what follows. The head is clamped at zero, so the model is an
*equilibrium* law and not only a rate law: every tilt has a fill `h = A − r·tan α` that
it drains to and then stops at. And the drain is stated in normalized fill, so moving
liquid between two containers requires converting through each one's capacity.

Code: `semantic_digital_twin/physics/equations/pouring_equations.py`, and the coupling in
`semantic_digital_twin/semantic_annotations/mixins.py`.

## The parameters

### Geometry — has ground truth, currently derived rather than estimated

| parameter | meaning | where its value comes from | what it decides |
| --- | --- | --- | --- |
| `container_height` | inner height `A` | `root.collision.height` | onset tilt, normalization |
| `container_width` | inner width | `root.collision.width` | default lip offset |
| `lip_offset` | lip's horizontal distance from the tilt axis `r` | half the width, or a spout's outlet | onset tilt, head |
| `capacity` | volume held at full fill | the collision box, or stated | the transfer ratio |
| `opening_radius` | receiver mouth | receiver collision | the overlap gate |
| exit point, exit direction | where the stream leaves | forward kinematics | landing point |

These are the semantically clear ones: each names something a ruler or a measuring jug
can settle independently of any pour. That is what makes them useful as evidence — an
estimate can be compared against a measurement, not only against a fit.

Two caveats on their current values. They are read from the *collision bounding box*, so
for a container with walls they describe the outside, not the cavity: for the demo
receiver the box holds 1.51× the true cavity. And the equations carry only two extents,
so an equation built without an annotation falls back to reading the depth as the width.

### Coefficients — no independent measurement, genuinely open

| parameter | meaning | current value | what it decides |
| --- | --- | --- | --- |
| `outflow_rate_constant` `k` | head to normalized drain, in 1/s | 1.0, never estimated | how fast it drains |
| `discharge_coefficient` `C_d` | Torricelli loss at the lip | 0.3 default, 0.2 cup, 0.7 spout | stream speed, landing point |
| `exit_speed` | fallback stream speed | 0.2 | unused once a head model exists |

`k` and `C_d` describe the same physical process — liquid crossing the lip — but are two
independent numbers, so the model can predict a fast stream that drains nothing. A
consistent model would derive the drain from the exit speed and the film's cross-section.
They are also excited by different measurements: `k` shows up in the fill, `C_d` only in
where the stream lands.

### Guards and smoothing — not physics, but they move the prediction

`MINIMUM_POUR_HEAD` (0.01 m) floors the head *inside* the exit-speed expression, so a
container that is provably not pouring still predicts a 0.44 m/s stream and a landing
point for it. `MINIMUM_DROP_HEIGHT` (0.01 m) does the same for flight time. The two gate
sharpnesses (80) multiply the drain itself, so they sit in the effect model's output and
not only in the optimizer's gradients.

### What has no parameter at all

- **The substance.** Every parameter above describes a container or a loss coefficient.
  Water and rice differ in this model only through `k` and `C_d`. This is the deepest of
  the findings below: the largest observed error is a quantity with nowhere to live.
- **Receiver overflow.** `InflowEquation` does not read the receiver's own fill.
- **Path dependence.** The model is a state relation; granular contents may not be.

## What can be measured while a pour runs

| quantity | in simulation | on hardware | which error it observes |
| --- | --- | --- | --- |
| source contents | particle count | wrist force/torque | `k`, and the onset |
| receiver contents | particle count | scale under the receiver | the transfer ratio |
| inflow rate | `MeasuredInflowRate` | derivative of either | `k` |
| onset — is a stream flowing | count change | threshold on the same signal | lip geometry, repose |
| tilt | exact | proprioception | — |
| landing point, spill | particle positions | vision | `C_d`, exit direction |

The useful coincidence: the model's largest error is the *onset tilt*, and onset is the
cheapest and most robust thing to measure — a threshold on a load-cell derivative, no
calibration needed. Observers exist in `semantic_digital_twin/physics/particles.py`
(`MeasuredFillLevel`, `MeasuredInflowRate`); they read the simulator, and the hardware
column is what they would be re-backed by.

## Findings so far

All from the MuJoCo particle scene: spherical grains of 5 mm radius in a cup of 35 mm
inner radius and 60 mm height, contacts at sliding friction 0.6.

Findings 1, 2, 5, 6 and 7 were re-measured after two defects were found in how the scene
was set up, and the corrections are larger than the findings they replace. Findings 8, 9
and 10 were measured before those fixes and have not been repeated; they are marked
where they stand.

### 0. Two defects in the measurement, not in the model

**The source was half as full as the code believed.** `HollowCylinder.particle_capacity`
counted seats in the spawn packing. A packing holds its grains clear of each other, so it
stands looser than what they settle into: a cup packed to its rim settled to 0.50 of its
cavity while the model was initialised at 1.00. Every equilibrium point, every transient
and every closed-loop run of the first round was measured against a fill the model had
wrong by a factor of two. The count is taken by volume now, a packing may stand above the
rim and drop in as the grains below it compact, and the demo seeds the model's fill from
what the contents settle to rather than from what was asked for.

**Two of the three friction coefficients never reached the solver.** Both the world
geometry and the particles wrote a three-coefficient friction without asking for a contact
resolved in enough dimensions to use it, so the engine kept the sliding coefficient and
discarded the torsional and rolling ones. Four different rolling coefficients produced
settled piles identical to the last digit. The grains were frictionless ball bearings,
and `ContactParameters.create_for_grasped_object` — which raises both coefficients
deliberately, to stop a held object spinning or rolling between the pads — had never had
any effect either.

Both are fixed. Neither was a defect in the effect model, and both changed what the model
was being judged against.

### 1. Tilt sets how much is left, and the model's curve is in the right place

Gravity walked around a fixed cup, which is the same as tilting it for a quasi-static
equilibrium and leaves out the arm, the grasp and the swing. Settled fill, as the share of
capacity the retained grains take up:

| tilt | 0–60° | 65° | 70° | 75° | 80° | 90° | 100° |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fill held | 0.63 | 0.62 | 0.60 | 0.55 | 0.45 | 0.22 | 0.00 |

Nothing moves until about 70°, which is what the first round saw. What changed is the
comparison: the model was being asked to predict this from a starting fill of 1.00 while
the cup stood at 0.63.

### 2. With the fill counted honestly, the model fits at its true geometry

Least squares of the model's equilibrium against the curve above:

| free parameters | repose | lip offset | rms |
| --- | --- | --- | --- |
| repose, geometry pinned to truth | **36.8°** | 35 mm | **0.007** |
| repose and lip offset | 32.5° | 30 mm | 0.006 |

Freeing the lip buys nothing: the error moves by 0.001 and the lip moves 5 mm, where the
first round's fit demanded 155 mm on a cup whose rim is 35 mm out. The structure
represents this equilibrium with every geometric parameter at its measured value, and it
tracks the falling limb and not only the onset — the descent spans 0.63 in fill and the
error over all fourteen points is 0.007.

This reverses the first round's sharpest conclusion, that the structure could not
represent the equilibrium with honest geometry. That conclusion was drawn from a fit to a
curve the model was being shown from the wrong starting state.

### 3. Feedback on a wrong model rescues liveness, not accuracy

`CalibratedDrainScale` (`semantic_digital_twin/physics/drain_calibration.py`) holds a
multiplicative factor on the drain at whatever makes the prediction match the measured
inflow. It is still the difference between nothing pouring and grains being delivered,
and it still overshoots: against a goal of 42 grains the nominal model delivers 0 without
it and 77 with it (finding 7).

The reason is structural. A gain on the drain cannot move the tilt at which the drain
*starts*, and the onset is what the nominal model has wrong — it puts it at 27° where the
grains need 70°. Scaling a rate that is zero leaves it zero, so the correction can only
act once the pour is already under way, by which point the cup is past 100°.

### 4. A quantified error in a semantically clear parameter, now fixed

(unchanged — the capacity conversion, corrected in `d9e813c88`.)

### 5. The contents are a state relation, reached about six seconds after the tilt

Holding 85° for twenty seconds, reached either straight from the settled cup or through a
staircase of five lower tilts, fill retained:

| path | 1 s | 2 s | 4 s | 8 s | 12 s | 20 s |
| --- | --- | --- | --- | --- | --- | --- |
| direct | 0.489 | 0.472 | 0.403 | 0.341 | 0.295 | **0.261** |
| staircase | 0.494 | 0.477 | 0.420 | 0.358 | 0.307 | **0.250** |

**The two paths converge**, to within one grain. The settled amount at a tilt does not
depend on how the cup got there, so the contents are a state relation and the model's
form survives. That part of the first round holds unchanged.

The rate does not. Fitting a single relaxation gives **6.35 s** on the direct path and
7.80 s on the staircase, against the first round's 2.96 s and 4.03 s.

The first round divided its 2.96 s into the model's own time constant and concluded
`outflow_rate_constant` should be near 3. That reasoning does not survive either, for a
reason independent of the fill bug: **the rate constant is not separably estimable from
the repose angle.** The model's time constant at 85° and `outflow_rate_constant = 1`
depends strongly on the repose it is evaluated with —

| model | time constant at k=1 | k needed for the measured 6.35 s |
| --- | --- | --- |
| repose 0° | 11.47 s | 1.81 |
| repose 36.8° | **1.50 s** | **0.24** |

— because the repose shifts the operating point close to onset, where the head is small
and `∂f/∂h` is steep. A rate constant is only meaningful alongside the repose it was
fitted with. The self-consistent estimate for this scene is **repose 36.8° with k = 0.24**.

### 6. The parameter that makes the model fit is not the quantity it is named after

The angle of repose is measurable without any cup: drop the same grains on a flat plate
and read the slope of the pile they make. Across a 150× sweep of the only coefficient
that controls it, 200 grains released from a column:

| rolling friction | apex | spread | interior slope |
| --- | --- | --- | --- |
| 0.0001 (production) | 4.7 mm | 2064 mm | no pile at all |
| 0.001 | 7.4 mm | 179 mm | ~0° |
| 0.002 | 12.2 mm | 110 mm | ~0° |
| 0.008 | 13.0 mm | 115 mm | ~0° |
| 0.015 | 17.9 mm | 131 mm | ~0° |

The crest profile at the strongest setting, in millimetres of height against radius:

```
r:   5.5  16.5  27.5  38.5  49.5  60.5  71.5  82.5  93.5  104.5
z:  12.6  15.4  17.9  13.0  12.5  12.5  12.5  11.9  11.7    4.7
```

That is a flat-topped plateau with a sharp edge, not a cone. A material with an angle of
repose builds a cone; these build a puddle. **The measured repose angle is about zero.
The fitted one is 36.8°.**

This is what the semantic-distance criterion exists for, and it is the one place in this
investigation where it has paid. The model's structure is adequate (finding 2) and one
parameter calibrates it to rms 0.007 — but that parameter's independent measurement
contradicts its fitted value, so it is absorbing something else. The most plausible
candidate is jamming in a cavity seven grains wide, which is a property of the container,
not of the contents.

The practical consequence: the calibration is **local**. It cannot be obtained by
measuring the material and carried to another container, because it is not a property of
the material. A black-box model reaching rms 0.007 would have looked like success, and the
second error signal — distance from the independently measured value — is the only thing
that says otherwise.

### 7. Estimating the parameters turns a controller that delivers nothing into one that hits its goal

Goal 42 of 111 grains, one run per variant, the simulation being deterministic:

| variant | gain | delivered | left in source | spilled | peak tilt | cycles |
| --- | --- | --- | --- | --- | --- | --- |
| nominal, k=1, repose 0° | off | **0** | 111 | 0 | 38.1° | timed out |
| nominal | on | 77 | 17 | 17 | 102.6° | 401 |
| mixed, k=1.5, repose 36.8° | off | 33 | 65 | 13 | 83.3° | timed out |
| mixed | on | 66 | 15 | 30 | 100.4° | 404 |
| estimated, k=0.24, repose 36.8° | off | 78 | 25 | 8 | 99.5° | 193 |
| estimated | on | 95 | 0 | 16 | 104.6° | 357 |

**The onset parameter is decisive.** The nominal model puts the onset at 27°, so at 38° it
believes it is pouring, the terminal-state row is satisfied in prediction, and the
controller stops tilting while no grain has moved. It sits there until the tick limit.
Adding the fitted repose moves the model's onset to 64° against a true 70°, and the same
controller with no feedback at all begins to deliver.

This reverses the first round's headline, that a better-fitting model controls worse. That
comparison was between three parameter sets all fitted to the distorted curve.

**The rate constant trades undershoot against overshoot**, and the honestly estimated one
overshoots: k=0.24 delivers 78 where the wrong k=1.5 delivers 33. That is not a property
of the model — see finding 11.

### Findings 8 to 10 — measured before the fixes of finding 0

The three that follow were all measured with the source half as full as the model
believed and the grains rolling frictionlessly. They are kept because their
*mechanisms* are still informative and two of them record experiments whose code was
removed, but none of their numbers should be quoted. In particular the peak tilt of
97° to 98° that finding 10 flags as an unexplained invariant across fourteen runs is
no longer an invariant: the same configurations now reach 38° to 105°, and the tilt
tracks the model's onset parameter. Whatever held it was in the scene, not the
controller.

### 8. Counting what is committed removes the lag, and most of the overshoot is not lag

The receiver's measurement now reports what it is committed to becoming rather than what
has landed: contents standing above its opening that have left the source will arrive
whatever the controller does next (`MeasuredCommittedFillLevel`). Same goal of 48 grains:

| variant | reporting | delivered | left in the source | spilled |
| --- | --- | --- | --- | --- |
| nominal + gain | landed | 61 | 22 | 22 |
| nominal + gain | committed | 70 | 22 | 13 |
| honest estimate + gain | landed | 73 | 0 | 32 |
| honest estimate + gain | committed | 92 | 0 | 13 |
| fitted geometry | landed | 98 | 0 | 7 |
| fitted geometry | committed | **79** | **16** | 10 |
| fitted geometry + gain | committed | 79 | 18 | 8 |

Read the delivered column alone and the change looks harmful, but it is the wrong column.
How much left the source is what the controller decided; how much of that landed rather
than spilled is the aim, and it moves by ten grains or more between runs that differ in
any way, since which grain clips the rim is chaotic. The column that answers whether the
controller stopped pouring is what stayed behind.

By that column the change does one thing: **the accurate model now stops.** It retains 16
to 18 grains where it previously emptied the cup, while the nominal and the honest
estimate retain exactly what they did before. Removing the transport lag only helps a
controller whose model is good enough to act on the earlier signal — which is the first
sign of model fidelity paying for itself anywhere in this investigation.

It is also nowhere near sufficient. 89 of 105 grains still leave the source against a
target of about 48, so the lag was the smaller part: most of the overshoot is genuine
commitment, made while the cup is past the angle at which its contents stay in. That is
what a bound on the predicted or the recoverable fill would have to address, and it is
the case for building one.

One control check worth recording: with the rate measured in the same units as the level,
the gain now changes nothing for the accurate model (79 delivered either way). A
correction with nothing to correct should be inert, and it is.

### 9. Capping the predicted fill changes nothing, because the overshoot is not in the prediction

`TerminalStateCapConstraint` refuses joint velocities that would take the predicted
terminal fill past a value, where `TerminalStatePredictionConstraint` drives it to one.
`FillByTransferTask.overfill_allowance` puts the cap at the goal. Same accurate model,
capped and uncapped, across three grain microstates produced by nudging the settling
time, goal 48 grains:

| microstate | uncapped | capped |
| --- | --- | --- |
| +0 ms | 79 delivered, 16 left | 67, 21 |
| +7 ms | 71, 21 | 79, 19 |
| +13 ms | 72, 22 | *QP infeasible* |

The cap's effect is smaller than the spread between microstates and its sign is not
even consistent: at one settling it delivers twelve fewer, at the next eight more. The
twelve-grain improvement that the first batch appeared to show was noise. Two of six
capped runs ended in `InfeasibleException`, reproducing the failure mode the earlier
spill-cap experiment hit.

The mechanism is visible in a column that does not move at all: **the peak tilt is
97.0° to 98.0° in every run, capped or not.** The cap barely changes the trajectory,
and the reason is that it is nearly redundant with the constraint beside it. The
equality already holds the predicted terminal fill at the goal, so a bound at the same
value is inactive except where the prediction would pass it — and the prediction does
not pass it. The prediction lands on the goal; reality delivers twice it.

So the overshoot is not a prediction overshoot, and no bound on the predicted fill can
address it. What is irreversible is not the receiver's level but **the source's tilt**:
at 97° the cup's equilibrium is nearly empty, so its contents will leave whatever the
controller does next, and the receiver's fill row has no way to say so.

That names the constraint that would work, and it is the mirror of the one built here
rather than something new: a **floor under the source's predicted terminal fill**, with
the bound on the lower side and the upper left open, applied to the source's fill
connection instead of the receiver's. Bounding how empty the source may get bounds the
tilt directly, which is the quantity that cannot be taken back. For this scene the
source starts at 0.585 of its capacity and should give up 0.267 of it, so the floor
sits at 0.318.

### 10. Bounding the source instead makes it consistently worse, and never moves the tilt

Finding 9 named the quantity that cannot be taken back as the source's tilt, so the
bound went there: a floor under the source's predicted terminal fill, set at run time
to the measured starting fill less what the goal asks the source to give up (0.31 of
0.58). Same accurate model, same three microstates:

| microstate | no bound | ceiling on the receiver | floor under the source |
| --- | --- | --- | --- |
| +0 ms | 79 delivered, 16 left | 67, 21 | **83, 15** |
| +7 ms | 71, 21 | 79, 19 | **85, 13** |
| +13 ms | 72, 22 | *infeasible* | **88, 11** |

Unlike the ceiling, the floor has an effect outside the microstate spread — the three
floor runs deliver 83 to 88 against 71 to 79 without it, and they do not overlap. It is
just the wrong way round: more is delivered and less is retained, every time. The
motion also runs longer (138 to 151 cycles against 130 to 134), so the extra delivery is
extra time spent pouring.

And the column the bound was aimed at does not move: peak tilt is 97.1° to 98.4°, the
same 97° to 98° every other configuration produces. The row changes the outcome without
changing the quantity it constrains.

Two explanations fit and this run does not separate them. The row may simply be
outweighted: it carries the fill task's own weight while the aim
(`KeepProjectileInReceiver`) runs at `WEIGHT_MAXIMUM`, and the spill-aware
investigation concluded that the max-weight aim chase saturates the arm's kinematic
budget and blocks exactly this kind of correction. Or the bound's sign or scaling is
wrong in a way the derivation hides — it is the linearized recursion's
``minimum − x_free`` on the lower side, which is right on paper and untested against a
live solve.

The diagnostic that separates them is to record whether the row is active and what it
contributes to the commanded tilt, before changing anything. Tuning weights against an
unverified row would be fitting noise.

Both bounds were removed from the tree once measured. What they leave behind is the
observation that survives them: **peak tilt is 97° to 98° in every configuration tried**
— nominal, estimated and fitted parameters, gain on and off, landed and committed
reporting, ceiling and floor. Fourteen runs, one tilt. Whatever holds it there is not
the fill goal, since nothing done to the fill goal moves it, and it should be identified
before another constraint is written.

### 11. The prediction window, not the rate constant, caused the overshoot

`TerminalFillConstraintTask.prediction_duration` is a fixed 1.5 s. Against a measured
relaxation of 6.35 s that is a quarter of a time constant: the fill barely moves inside
the horizon, the controller reads its tilt as ineffective, and it asks for more. Finding 5
of the first round had already written the mechanism down; nothing had connected it to the
overshoot, because the rate constant was being blamed instead.

Same estimated model, same grains, no feedback gain, only the window changed:

| prediction window | delivered (goal 42) | spilled | peak tilt | cycles |
| --- | --- | --- | --- | --- |
| 1.5 s (the default) | 78 | 8 | 99.5° | 193 |
| **3.0 s** | **42** | **3** | 91.7° | 243 |
| 6.0 s | 37 | 14 | 84.3° | 1828 |

Monotonic in the window, with the optimum near half the measured time constant. At 3.0 s
the controller hits the goal exactly, spills three grains, and converges — the best run of
the investigation, and the first one to land on the goal rather than pass through it.

**`prediction_duration` should be derived from the measured time constant of the contents,
not left at a constant.** It is currently a controller tuning parameter standing in for a
property of the material, which is the same category error the repose angle makes in the
other direction.

One caution before this is leaned on. The 3.0 s was found by trying three values, so it is
a tuned number, not a measured one; what is measured is the trend and the mechanism. And
the same caveat as finding 6 applies to the whole set — repose and rate were both fitted
to this cup's own curves, so this is a well-calibrated controller for this scene, not a
demonstration that the physics transfers.


## Experiments this sets up

**E1 — Is the contents' behaviour a state relation at all?** Run and answered: yes. The
two paths to one tilt converge to within one grain over twenty seconds (finding 5), so the
settled amount is a function of the tilt and the model's form survives. What it needs is
the right target and the right rate, both of which are existing parameters.

**E2 — Does the semantically right parameter beat a semantically empty one?** Estimate an
*onset* parameter (the repose angle, or equivalently the lip offset) and compare against
`CalibratedDrainScale` on identical runs.

Run and answered, in findings 7 and 11, and answered *for* the hypothesis once the scene
defects of finding 0 were fixed. The onset parameter is the difference between a
controller that delivers nothing and one that delivers; with the prediction window set
from the measured time constant it lands exactly on the goal, open loop, with three
grains spilled, where the bare gain overshoots by 83% and empties the source.

The first round answered the opposite, from the same experiment run against a half-full
cup. Worth repeating across goals and scenes before it is leaned on — one run per
variant, deterministic but a single scenario.

**E3 — Identify the coefficients from transients.** Held tilts carry no information about
`k` or `C_d`; they only set how fast the equilibrium is approached. Estimating them needs
tilt steps and the landing point, and they should be fitted separately from the geometry
rather than thrown into one fit.

Partly run, in finding 5, with one result that changes how it should be done: `k` and the
repose angle are **not separably estimable**. The model's time constant at one tilt moves
7.6× between repose 0° and repose 36.8°, so a rate fitted against the wrong onset is
wrong by that factor. They have to be fitted jointly, or the onset fixed first from the
equilibrium curve and the rate fitted against it.

**E5 — Does the calibration transfer?** The open question finding 6 leaves. Repeat the
equilibrium fit in a second container of a different width with the same grains. If the
fitted repose angle follows the container it is a jamming parameter and has to be
re-estimated per vessel; if it stays put it is a property of the contents after all and
the plate measurement is the thing that is wrong. This is the experiment that decides
whether any of this is physics or bookkeeping, and it is cheap.

**E4 — Repeat against hardware.** The measurement column above is chosen so that the same
observers re-back onto a wrist force/torque sensor and a scale. Until then every number
here is a statement about the simulator's contact model.

## Open questions

- Whether the container parameters should stay derived from the collision geometry or
  become estimated quantities in their own right. They currently describe the outside of
  a walled container; the model wants the cavity.
- Whether `k` and `C_d` should remain independent, given they describe one process.
- Whether the gate sharpnesses belong to the effect model or to the controller. They
  multiply the drain today, so they are in the prediction either way.
- Whether the guard floors should be stated as model parameters, since
  `MINIMUM_POUR_HEAD` makes the model predict a stream from a container that is not
  pouring.
- Whether `prediction_duration` belongs to the task at all. Finding 11 makes it a
  function of the contents' time constant, which is a property of the material the task
  is acting on rather than of the controller acting on it.
- What the fitted repose angle is actually measuring, given the contents have none
  (finding 6). If it is jamming against the walls, the model has no parameter for the
  thing that dominates its onset, and naming that parameter honestly matters more than
  fitting it well.
