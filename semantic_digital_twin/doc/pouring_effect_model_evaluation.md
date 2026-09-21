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

All from the MuJoCo particle scene: 105 spherical grains of 5 mm radius in a cup of
35 mm inner radius and 60 mm height, contacts at sliding friction 0.6 with no rolling
resistance.

### 1. Tilt sets how much is left, and the model's curve is in the wrong place

Tilting the cup one step at a time and letting it settle at each, without refilling
between steps, grains still in the source:

| measured tilt | after 1 s | after 2 s | after 4 s | model drains to |
| --- | --- | --- | --- | --- |
| 64° | 105 | 105 | 105 | 0 |
| 68° | 105 | 105 | 105 | 0 |
| 72° | 103 | 102 | 100 | 0 |
| 76° | 95 | 91 | 91 | 0 |
| 80° | 84 | 80 | 76 | 0 |
| 84° | 66 | 64 | 59 | 0 |

The model empties this cup at 59.7° and predicts that a *full* cup pours at any tilt at
all. The grains have not started at 68°. The model's whole useful range lies below the
tilt at which anything happens.

### 2. No estimate of the existing parameters rescues it; one new parameter nearly does

Least squares of the model's equilibrium against the six observations:

| free parameters | lip offset | repose | RMS error (fill units) |
| --- | --- | --- | --- |
| none, geometry as built | 35 mm | — | 0.87 |
| lip offset | hits bound | — | 0.87 |
| repose angle, geometry pinned to truth | 35 mm | 59.3° | 0.092 |
| both | 111 mm | 71.1° | 0.015 |

Freeing the lip offset changes nothing, and not because the optimum is weak: at full
fill the head is `r·sin α`, positive for *any* tilt and *any* lip offset, so "nothing
happens until 68°" is unreachable by estimating the parameters the model has. Adding an
angle of repose — a real, named, independently measurable property, and the first
parameter of the *contents* rather than the container — drops the error ninefold with
every container parameter left truthful.

The last row is the case the semantic-clarity criterion exists for. It fits six times
better again, and does so by putting the lip 111 mm from the tilt axis on a cup whose rim
is 35 mm out. The fit improved while a parameter with a ground truth walked away from it.
That second error signal — distance from the independently measured value — is what
separates "the parameters were unknown" from "the structure is wrong", and it is the
argument for insisting on semantically clear parameters. A black-box model reaching 0.015
would have looked like success.

Caveat: this curve is a property of the contact parameters chosen for the simulation, not
of a measured material. A 47–68° effective repose angle is far above any real granular
material (sand is about 34°), which suggests jamming in a coarse discrete packing — six
grains across the cup — rather than a continuum property. The numbers are evidence about
the *shape* of the mismatch, not calibration targets.

### 3. Feedback on a wrong model rescued liveness, not accuracy

`CalibratedDrainScale` (`semantic_digital_twin/physics/drain_calibration.py`) holds a
multiplicative factor on the drain at whatever makes the prediction match the measured
inflow. It was the difference between nothing pouring at all and grains being delivered
— and against a goal of 32 grains it empties the source completely, landing 97 in the
receiver and spilling the other 8.

The reason is structural, and it is the hypothesis this work should state: **feedback
compensates model error only in the directions the adapted parameters span.** A gain on
the rate spans "how fast", while the true error is in "at what tilt does it start". Once
the controller tilts past the real onset, the gain keeps reporting "still not enough"
until it commands a tilt that empties the cup — here 107.5°.

Note also that this factor is the *wrong kind* of parameter for the evaluation: it has no
units, no referent and no measurable truth, so it can absorb any modelling error and
therefore reports nothing about whether the model is right. It is a usable controller
crutch; it is not evidence.

### 4. A quantified error in a semantically clear parameter, now fixed

`outflow_volume_rate` converted normalized fill to a volume with
`half_cross_section_area = (width / 2) · height`, which is an *area*: dimensionally wrong
for a volume, and scaling linearly with the container's width where a volume scales
quadratically. It therefore cancelled only between containers of equal width.

- Cup-to-cup in the demo (35 mm into 60 mm inner radius): the receiver's level rose 1.64×
  faster than it should.
- The faucet path, where the inflow is a genuine volume rate, was off by 3.9× for the
  demo receiver, and dividing a volume rate by an area.

Replaced by an explicit `capacity` — the volume held at full fill, in cubic metres — read
from the collision box, statable when known. The scaling law is now right and only a
constant shape factor remains, which cancels between containers of the same shape.

Nothing in any fit would have revealed this. Checking a parameter against its definition
did, which is the same argument as finding 2 from the other side.

### 5. The contents are a state relation, reached about three times slower than the model thinks

Holding the same tilt (83.9°) for twenty seconds, reached either straight from a full
cup or through the staircase of every lower tilt, grains still in the source:

| path | 1 s | 2 s | 4 s | 8 s | 12 s | 20 s |
| --- | --- | --- | --- | --- | --- | --- |
| direct, from full | 84 | 74 | 67 | 61 | 49 | **44** |
| staircase | 66 | 64 | 59 | 50 | 49 | **45** |

**The two paths converge**, to within one grain. The settled amount at a tilt does not
depend on how the cup got there, so the contents are a state relation after all, and the
four-to-nine grain gap visible at four seconds was elapsed time rather than history.

What separates the model from the measurement is not the form but the rate. Fitting a
single relaxation gives 2.96 s on the direct path and 4.03 s on the staircase, while the
model's own time constant — `−1/(∂ḣ/∂h)`, read straight off
:meth:`symbolic_ode_jacobians` — is 9.41 s at `outflow_rate_constant = 1`. Since that
time constant is inversely proportional to the rate constant, matching the observation
needs `outflow_rate_constant` near 3. It has never been estimated and sits at its
default of 1.

This is also why the closed loop overshoots, without any structural explanation being
needed. The fill row predicts over a 1.5 s window, which at the model's 9.41 s is 16% of
a time constant: the fill barely moves inside the horizon, so the controller reads its
tilt as ineffective and asks for more. At the measured 3 s the same window is half a time
constant and the predicted response is roughly three times larger.

Two limits worth recording. A single relaxation is a good but not exact description — two
timescales fit better (rms 0.015 against 0.043 on the direct path) — and the settling is
intermittent rather than smooth: the direct trace loses more between 8 and 12 s than
between 4 and 8 s, which is an avalanche, not a decay. A first-order ODE tracks the
envelope, not the stick-slip. For a pour lasting one to three seconds the fast component
is what matters, so this is a known limit rather than a blocking one.

### 6. Estimating the parameters does not rescue the model, and a third definition disagrees

With the converged curve of finding 5 as the data, fitting the model's equilibrium:

| free parameters | repose | lip offset | RMS |
| --- | --- | --- | --- |
| none, model as it stands | — | 35 mm | 0.704 |
| repose, geometry true | 45.4° | 35 mm | 0.180 |
| repose and lip offset | 70.9° | **155 mm** | 0.020 |

Sharper than finding 2 and pointing the other way. With the lip where it actually is,
the best repose angle still leaves 0.180 on a quantity bounded in `[0, 1]`: it predicts
0.76 where the cup holds 0.96 and 0.46 where it holds 0.22. The observed curve is far
steeper than `1 − (r/A)·tan(α − θ)` can be at the true lever arm, whatever the angle.
Only moving the lip to 4.4× its measured distance fits. So for granular contents the
structure cannot represent the equilibrium with honest geometry, and the criterion fires
negative.

The closed-loop half of the comparison could not be run, for a reason worth more than the
comparison would have been. Every variant — nominal, honest estimate, best fit, with and
without the gain — empties the source against a goal of 32 of 105 grains:

| variant | gain | delivered | spilled | peak tilt |
| --- | --- | --- | --- | --- |
| nominal, k=1, repose 0 | on | 97 | 8 | 107.7° |
| honest estimate, k=0.05, repose 45.4° | off | 88 | 17 | 109.5° |
| honest estimate | on | 88 | 17 | 111.4° |
| fitted geometry, k=0.1, repose 70.9°, lip 155 mm | off | 87 | 18 | 112.0° |
| fitted geometry | on | 92 | 13 | 112.3° |

A model fitting the grains to rms 0.020 controls no better than one at rms 0.704, which
says the comparison is measuring something else. It is: **the goal is unreachable in the
model's own units.** Draining the entire source raises the receiver's modelled fill by
0.223, against a goal of 0.3, so the controller saturates its tilt whatever its
parameters are.

The cause is a third quantity with two definitions. `MeasuredFillLevel` reports the share
of *the contents* standing in a container, so all the grains in the receiver reads 1.0.
The fill DOF the model integrates is the share of *that container's capacity*, where the
same state reads 0.223. Perception overwrites the DOF with the first while the ODE
integrates the second; for a 35 mm source pouring into a 60 mm receiver they differ by
4.5×.

This is also the real cause of the overshoot, rather than anything in the drain model,
and it is why fixing the capacity made the closed loop slightly worse: the old conversion
put the reachable receiver fill at 0.366, just above the goal, and the corrected one puts
it at 0.223, below it. The controller went from chasing a barely reachable target to an
impossible one.

### 7. A better-fitting model controls worse

With fill level meaning one thing (finding 6 fixed: the share of the container's own
capacity, counts converted through the measured volume one grain occupies) the goal is
reachable and the comparison runs. Goal 48 of 105 grains, one run per variant, the
simulation being deterministic:

| variant | equilibrium fit | gain | delivered | spilled | peak tilt |
| --- | --- | --- | --- | --- | --- |
| nominal, k=1, repose 0 | rms 0.704 | off | **0** | 0 | 49.9° |
| nominal | rms 0.704 | on | **61** | 22 | 97.8° |
| honest estimate, k=0.05, repose 45.4° | rms 0.180 | off | 64 | 41 | 113.8° |
| honest estimate | rms 0.180 | on | 73 | 32 | 111.3° |
| fitted geometry, k=0.1, repose 70.9°, lip 155 mm | rms 0.020 | off | **98** | 7 | 103.0° |
| fitted geometry | rms 0.020 | on | 99 | 6 | 104.6° |

Two things, and the second is the more important.

**The gain rescues liveness, as finding 3 said.** The nominal model without it delivers
nothing at all: at 49.9° the model believes it is pouring freely, the terminal-state row
is satisfied in prediction, and the controller stops tilting while the grains have not
begun to move. The gain is what breaks it out, and it is the best performer in the table.

**Model accuracy runs the other way from control quality.** Ordering the variants by how
well they predict the grains orders them inversely by how close they land: rms 0.704
delivers 61, rms 0.180 delivers 64 to 73, rms 0.020 delivers 98 — twice the goal. The
best model is the worst controller.

The mechanism is visible in the cycle counts: the accurate model reaches the goal in 158
control cycles against the nominal one's 373. A model that predicts the pour correctly
asks for it sooner, and pouring cannot be undone. By the time the goal is crossed the
cup is near 103°, the grains past the lip are committed, and reversing the tilt takes
over a second. Accuracy buys a faster approach, and a faster approach commits more.

So the limit here is not the effect model at all. The task says reach a fill level; it
does not say that over-delivering cannot be taken back. Nothing in the formulation
distinguishes arriving at the goal from passing through it, so a model good enough to
arrive quickly is punished for it. That is a statement about the task, not about pouring.

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

## Experiments this sets up

**E1 — Is the contents' behaviour a state relation at all?** Run and answered: yes. The
two paths to one tilt converge to within one grain over twenty seconds (finding 5), so the
settled amount is a function of the tilt and the model's form survives. What it needs is
the right target and the right rate, both of which are existing parameters.

**E2 — Does the semantically right parameter beat a semantically empty one?** Estimate an
*onset* parameter online (the repose angle, or equivalently the lip offset) from the same
measurements that currently drive the gain, and compare against `CalibratedDrainScale` on
identical runs. Prediction from finding 3: the onset parameter recovers the goal where the
gain overshoots, without the model becoming more correct in any other respect.

Run and answered, in finding 7, and answered against the hypothesis: the semantically
clear estimate does not beat the bare gain, and the parameters that fit best control
worst. The bottleneck is that the task does not express the irreversibility of pouring,
so better prediction only buys a faster commitment. Worth repeating across goals and
scenes before it is leaned on — one run per variant, deterministic but a single
scenario.

**E3 — Identify the coefficients from transients.** Held tilts carry no information about
`k` or `C_d`; they only set how fast the equilibrium is approached. Estimating them needs
tilt steps and the landing point, and they should be fitted separately from the geometry
rather than thrown into one fit.

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
