# Thesis Experiment Feasibility Report

Audit of five candidate thesis experiments against this codebase (branch `bmp`, 2026-08-12;
section 5 added 2026-08-13).
Companion to the hostile-examiner analysis of the thesis draft in `~/Documents/phdThesis`.

**TL;DR:** The runtime regime switch and the verification mode are days of glue code on
machinery that already exists and is tested. The container sweep splits: a rectangular-geometry
sweep is ~1 day, a bottle/watering can is a 2–4 week modeling project. For new VDJM domains,
wiping is feasible but chapter-sized; cutting has no foundation here and should not be attempted.
Re-implementing knowledge-based servoing on the SDT + EQL and coupling it to the VDJM is the
largest item (chapter-sized) and the strongest single contribution: it is the only candidate that
closes the Adaptivity Gap endogenously and the only one that attacks the
precision–expressivity trade-off instead of illustrating it.

Recommended order:

1. **Verification mode** (~1 week) — fills the conclusion chapter's admitted gap, yields a
   confusion-matrix table.
2. **Regime switch** (~3–5 days) — the switching *mechanism*, empirically characterized. Note it
   does not by itself close the Adaptivity Gap (see the scope caveat in section 1 and section 5).
3. **Rectangular container sweep** (~1 day), then the **frustum extension** (~1 week) for a
   "strongly different containers" claim.
4. **Servoing–VDJM coupling** (staged: ~1.5 weeks for the demonstrator, 4–6 weeks for the full
   claim) — the synthesis contribution. Start it in parallel with 1–3 if the timeline allows,
   since its stage A subsumes the driver written for item 2.
5. **Wiping VDJM** (3–4 weeks) only as a deliberate fourth-domain contribution. Skip cutting.

---

## 1. Runtime constraint-regime switch — highly feasible, ~3–5 days

### What already exists

The statechart life cycle already gates constraint rows in and out of the solved QP per
control cycle:

- `ConstraintCollection.link_to_motion_statechart_node`
  (`giskardpy/src/giskardpy/qp/constraint_collection.py:133`) multiplies every constraint's
  `quadratic_weight` by `if_eq(node.life_cycle_variable, RUNNING, 1, 0)`.
- `QPData.apply_filters` (`giskardpy/src/giskardpy/qp/qp_data.py:136-177`) drops all rows whose
  weight evaluated to 0, every cycle. A NOT_STARTED / PAUSED / DONE task contributes **zero rows**
  and activates in the next cycle when its start condition fires — no recompile, no goal restart.
- Pause propagates to all descendants (`motion_statechart/graph_node.py:637, 673, 582`).
- External-signal nodes exist: `WaitForMessage`, `TopicSubscriberNode`
  (`motion_statechart/ros2_nodes/topic_monitor.py`), `ForceImpactMonitor`
  (`ros2_nodes/force_torque_monitor.py`). The ROS executor spins in a background thread, so topic
  callbacks are delivered mid-motion.
- Clearance-like constraints exist: `DistanceGoal` (`tasks/feature_functions.py:227`, planar),
  `KeepSourceRimAboveReceiverRim` (`tasks/pouring.py:395`, the template for a band-inequality
  task), `CartesianVelocityLimit` family (`tasks/cartesian_tasks.py`), collision-avoidance goals
  with per-cycle symbolic buffer distances (`goals/collision_avoidance.py`;
  `UpdateTemporaryCollisionRules` at `:289` is a ready-made runtime collision-regime switch).
- Runtime-tunable numeric parameters have precedent: register a `FloatVariable` in `build()`,
  rewrite it in `on_tick` (`CartesianPositionTrajectory`, `binding_policy.py`).
- Instrumentation is built in: `ActionFeedbackPublisher` streams life-cycle/observation changes;
  `GoalGanttChartPlotter` renders the life-cycle-over-time figure that documents a regime switch.

### Experiment shape (single goal, no restart)

```python
Parallel([
    transfer_task, no_spill, keep_above, keep_plane,          # baseline regime
    strict_spill := KeepProjectileInReceiver(threshold=0.005, weight=WEIGHT_MAXIMUM),
    inject  := WaitForMessage(topic_name="/regime/inject",  msg_type=Empty),
    retract := WaitForMessage(topic_name="/regime/retract", msg_type=Empty),
    hold    := BoolTopicMonitor(topic_name="/human/grasp"),  # to be written
])
strict_spill.start_condition = inject.observation_variable    # (a) inject
strict_spill.end_condition   = retract.observation_variable   # (b) retract
motion.pause_condition       = hold.observation_variable      # (c) pause + release
```

### What must be written

| Work | Where | Size |
|---|---|---|
| `BoolTopicMonitor` (observation follows `msg.data`; `WaitForMessage` latches TRUE forever, so pause could never be released) | `motion_statechart/ros2_nodes/topic_monitor.py` | ~25 lines |
| `KeepOutSphere` task: 3-D `‖p_tip − p_center‖ ≥ r` inequality + observation; center/radius as registered variables | new `motion_statechart/tasks/keep_out.py` | ~120 lines |
| Experiment driver (build chart, `execute_async`, publish signals, log feedback) | repo root or `experiments/src/experiments/` | ~200 lines |
| Tests | `test/giskardpy_test/test_motion_statechart/` (pattern: `test_force_torque_nodes.py`) | ~150 lines |

Not touched: `executor.py`, `qp/*`, `motion_server.py`, `control_loop.py`, `motion_statechart.py`,
`graph_node.py`. Regenerate ORM if new serializable node classes are added.

### Caveats to state honestly

- Every switchable constraint must be **pre-declared** in the statechart. A genuinely unforeseen
  constraint requires a new goal → robot stops, chart recompiles.
- Mid-motion **world-model** changes abort the goal by design
  (`WorldModelModifiedDuringMotionError`, `control_loop.py:96`) — keep-out geometry must pre-exist;
  only state (poses, float variables) may change during a motion.
- Condition scope: trigger monitors must be siblings of the gated task
  (`MotionStatechart._validate_condition_scope`).
- Inject soft high-weight rows, not hard ones — a hard row dropped into a committed pour can make
  the QP infeasible (the repo's own collision code warns about this, `collision_avoidance.py:98`).
- Pause zeroes the modelled fill velocity; a physical liquid would keep moving.
- "Instant" pause = one control cycle to command change, jerk-limited physical stop over a few
  cycles.

Related: `semantic_safety_filter_concept.md` already specs a CBF/superquadric keep-out layer on
exactly these hooks (unimplemented); the `KeepOutSphere` task would be phase 3 of that plan.

---

## 2. New VDJM effect domains — wiping feasible (chapter-sized), cutting not

### What is genuinely generic already

- Passive DOFs: `LiquidConnection(PrismaticConnection, HasUpdateState)`
  (`semantic_digital_twin/.../world_description/connections.py:1393`) — `active_dofs=[]` so the QP
  can never command it; the post-cycle integration hook `World.step_physics(dt)`
  (`world.py:2561`) steps every `HasUpdateState` connection and is domain-agnostic.
- The terminal-state prediction row is **fully generic** over `(f, ∂f/∂x, ∂f/∂q)`:
  `ConstraintCollection.add_terminal_state_prediction_constraint`
  (`qp/constraint_collection.py:328`) and
  `qp/terminal_state_prediction_strategy.py` contain zero pouring vocabulary.
- Task pattern: `TerminalFillConstraintTask` (`tasks/pouring.py:43`) does all the work; concrete
  tasks are 40–60 lines each once the base class is generalized (currently typed
  `LiquidConnection`, hooks named `_fill_*` — ~1 hour rename/retype).

### What is fill-specific (i.e. a new domain copies the pattern, it does not plug in)

`FillEquation`/`FillContext`/`LiquidConnection`'s two named equation slots, `fill_position` /
`tilt_expression` context, `HasFillLevel` factory (`semantic_annotations/mixins.py:1298`).
`DifferentialEquation` is an 11-line empty ABC — there is no generic
`VirtualEffectConnection` with a list of attached ODEs.

### Wiping — feasible, two real research obstacles

1. **No contact-area representation exists.** Collisions expose points/normals/distances
   (`collision_variable_managers.py` provides per-cycle *symbolic* contact symbols — the pattern
   to copy), never area. Build a differentiable soft overlap of the tool footprint against the
   surface plane × a logistic penetration gate — same construction style as
   `_geometric_transfer_gate` / `_logistic` (`mixins.py:1712-1804`). ~80–150 lines.
2. **Coverage is path-dependent.** The strategy assumes `ẋ = f(x, q)` (differentiates w.r.t.
   positions only), but coverage rate ∝ tool speed, and scrubbing one spot forever must not count.
   Options, increasing cost: (i) control **progress along a reference sweep path** `s(q)` — a
   genuine function of configuration, preserves all existing QP machinery (try first);
   (ii) `add_velocity_constraint` for the rate, coverage DOF monitoring-only (gives up terminal
   prediction); (iii) sibling strategy placing `∂f/∂q̇` in the row (~150–250 lines + tests).
3. **One terminal-state row per statechart** (`MultipleTerminalStateConstraintsError`) — a lumped
   scalar coverage (or sequential per-cell goals) is the honest deliverable, not an N-cell field.
   The `(1−c)` saturation in `ċ = k·A_contact(q)·v_tan·(1−c)` gives `∂f/∂c < 0`, the
   well-conditioned case for the linearization.

Head start available: `HasSupportingSurface.calculate_supporting_surface` (`mixins.py:764`)
extracts surface `Region`s; `_2d_surface_sample_space_excluding_objects` builds a 2-D
random_events Event of the surface minus objects (numeric coverage ground truth); `Sponge`
annotation; `planar_raster_xy`/`build_surface_path` (`coraplex/.../tool_paths.py`) as reference
sweep; `WipingAction` as plan-level wrapper.

Surface estimate: `wiping_equations.py` ~250–350, `CoverageConnection` ~130, `HasSurfaceCoverage`
mixin ~200–300, base-task generalization ~50 edits, `CoverageTask` + `KeepToolOnSurface` ~150,
effects/enums ~30, optional `WipingMSCModel` ~100, ORM regen, tests at the pouring ratio
(pouring has ~3,500 test lines). **~1,000–1,400 source + 1,500–2,500 test lines** — comparable to
the whole pouring domain. This is a contribution, not an experiment.

### Cutting — recommend against

- Cut *depth* as a passive DOF is expressible (configuration-determined;
  `build_cutting_path` in `tool_paths.py:769` already computes the needed geometry), but the
  stated dynamics — force, material properties — have **no representation**: no material model,
  no symbolic force in the QP, no compliance model. Force exists only as ROS threshold monitors.
  A kinematic `ḋ = k·v_blade` surrogate is physically vacuous; a learned surrogate has no dataset.
- **Separation is the wall:** a topology change of the world model, with no mesh split/boolean,
  no deformables, no soft-body simulator backend. Categorically outside what a virtual joint can
  express. (Also: cutting is hysteretic — `d = max over history` — the same velocity-dependence
  wrinkle as wiping plus a ratchet the connection clamp does not express.)

---

## 3. Container sweep — two different experiments under one name

### (a) Rectangular parameter sweep — very feasible, ~1 day

- Geometry is a plain `(A, r)` pair on the equation, decoupled from the body (the paper's
  single-cup world runs the mug-geometry equation on a 1 m box body).
- Parametric box-cup builder exists: `_box_cup_body(name, height, width)`
  (`test/giskardpy_test/test_motion_statechart/test_pouring.py:1089`); scaled STL cup variants
  exist (`semantic_digital_twin/resources/mjcf/jeroen_cups.xml`).
- Sweep precedent: the perception-noise `@pytest.mark.parametrize` at `test_pouring.py:1365`
  writing per-condition `.npz` traces; plots via `learned_pouring/make_paper_plots.py`.
- Learned arm: loop the existing CLI
  `python -m semantic_digital_twin.physics.equations.head_surrogate_training --container-height H
  --container-width W --checkpoint p.pt` per geometry. Watch: the 1e-6 geometry-match guard
  (`_validate_geometry_matches_reference`), one l4casadi C build per geometry
  (`~/.cache/semantic_digital_twin/l4casadi/`).
- Caveat for the writing: the surrogate is distilled from the analytic head, so the learned arm
  demonstrates provenance-indifference across geometry, not physical generalization.
- Metrics/table infra exists but is unwired: `experiments/src/experiments/experiment_definitions.py`
  (`ExperimentResult`, `ExperimentsTable`, `TypstRenderer`, `MeanAndStandardDeviation`).

### (b) Bottle / watering can / wine glass — 2–4 weeks of modeling, currently fails *silently*

"Container geometry" is the **AABB of the collision mesh**: `A` = full outer bbox height,
`r` = half the bbox y-extent (`shape_collection.py:220-249`; consumed in
`HasFillLevel.initialize_fill_level`, `opening_radius`, `rim_point`, `_rim_exit_point`,
`liquid_exit_direction` — `mixins.py:1298, 1756, 1778, 1608, 1638`). Measured on the in-repo
coke bottle (`coraplex/resources/objects/Static_CokeBottle.stl`, non-watertight): opening radius
comes out ~0.047 m vs a real neck of ~0.011 m (4×), lip at the body half-width, jet direction
along body +Z. No crash — physically wrong numbers.

What must be built, in dependency order:

1. **Cavity + opening extraction from mesh** (net new, ~300–500 lines): `r(z)` profile from
   trimesh z-slices, opening detection (rim ring), inner-vs-outer height. Reuse
   `calculate_supporting_surface` idioms; `ObstacleContainmentChecker`
   (`experiments/src/experiments/free_space_volume_estimation.py:53`) handles non-watertight
   meshes.
2. **Profile geometry equation class**: refactor `RectangularContainerGeometry`
   (`pouring_equations.py:104`) into an interface; smooth CasADi-differentiable `r(h)` (spline /
   low-order polynomial, no lookup tables); gated/ungated/JSON variants; relax two `isinstance`
   gates (`mixins.py:1589`, `learned_pouring_equations.py:417`); ORM regeneration.
3. **Fix fill↔volume mapping**: `fill ∈ [0,1]` currently maps to volume via a **constant**
   `half_cross_section_area` — and note it is the 2-D side rectangle `r·A` (m²), so
   cross-geometry transfer conserves a physically wrong quantity even between box cups of
   different sizes. Touches `mixins.py:1569-1577`, `pouring_equations.py:403-413`, goal semantics.
4. **Azimuth/spout kinematics**: `tilt_expression_from_fk` is `acos(R_zz)` — no azimuth. A spout
   or single lip needs 2-DOF tilt + lip-azimuth-aware exit point/direction; touches no-spill tasks.
5. **Ground truth**: nothing in the repo validates a non-rectangular head model — no liquid sim,
   no volumetric fill. Options: Monte-Carlo "cavity volume below a tilted plane" on
   `MonteCarloFreeSpaceSampler`, external SPH, or real-robot scale.
6. **Assets**: no watering can / wine glass / pot mesh exists; bottle mesh needs watertight
   repair; `apartment_bowl.stl` is in millimetres.
7. **Learned head**: widen `HeadSurrogate` to `(tilt, fill, A, r)` (input dim 2 is hardcoded),
   relax the geometry guard, retarget distillation onto item 5's ground truth. The gap is
   acknowledged in `learned_pouring/report.md:228-231`.
8. **Harness**: parametrized pytest + `.npz` is cheapest; the `giskardpy/experiment/` framework
   exists only on branch `bmp-deployment-template` (commit `9e278d274`) and its test on `bmp` is
   broken.

**Middle option (recommended): tapered/frustum profile** — linear `r(z)` + real opening
extraction covers wine glass, wide pot, scaled-mug family at ~1/3 the cost (items 1-opening, 2, 3
only; no azimuth/spout work). Bottle-neck and spout cases are then honestly labelled as the
characterized boundary of the analytic model — which strengthens the geometry-conditioned
surrogate argument in the thesis.

---

## 4. Verification mode against external motions — **IMPLEMENTED 2026-08-12**

> Status: implemented as the self-contained package
> `experiments/src/experiments/bmp_verification/` (scope validator, typed-verdict
> pipeline, labeled perturbation harness, Claude-CLI generator, verify-repair loop,
> experiment drivers) plus the runner `experiments/scripts/run_bmp_verification.py`
> and 29 tests under `test/experiments_test/test_bmp_verification/` (all passing;
> CLI calls mocked). No library code was modified. Smoke results: perturbation study
> over all 27 apartment containers = 270 labeled cases, confusion-matrix accuracy
> 1.000; live LLM smoke (full and minimal information) verified on first attempt
> (~$0.11/loop). Observation: minimal information may need harder ablation (e.g. not
> naming the container kind) to induce failures — the model's household priors are
> strong. Run under the `cram2` virtualenv (`workon cram2`); the `experiments`
> package was installed editable there.

### What already exists

- Predicates: `Causes`, `SatisfiesRequest` in
  `semantic_digital_twin/src/semantic_digital_twin/reasoning/bmp_predicates.py`;
  `MotionStatechartCanPerform` in `coraplex/src/coraplex/body_motion_problem/predicates.py`
  (costmap-sampled base poses × end effectors × Giskard `CartesianPose` sequence as feasibility
  oracle). BMP instance = conjunctive EQL query (canonical form:
  `test/coraplex_test/test_body_motion_problem.py:1055`).
- **The verification direction (fixed τ, open goal) is implemented and tested**:
  `test_infer_effects_and_tasks_from_given_motion` (`:1070`, hand-written tilt list),
  `test_query_task_and_effect_satisfying_motion` (`:912`, `:1186`). `MotionTrajectory.converged`
  docstring explicitly anticipates hand-written trajectories.
- τ is **object-centric**: `MotionTrajectory.data: dict[Connection, list[float]]`
  (`world_description/motion.py`) — container-DOF positions, not robot joints. `CanPerform`
  *synthesizes* the robot motion from τ; an externally supplied robot-joint trajectory has
  nowhere to go in the current design. State this plainly in the chapter.
- Sandboxed replay: `Causes._map_motion_to_effect` (`bmp_predicates.py:120-143`) inside
  `World.reset_state_context()` (state-array copy, not a deep copy — cheap).
- External-generator seam: `PhysicsModel` (one-method ABC); the adapter already exists as a test
  double `_RecordedTrajectoryModel`
  (`test/semantic_digital_twin_test/test_reasoning/test_bmp_predicates.py:44`).
- Scenario builder: `_extend_world` / `ContainerScenario` / `ContainerSelection`
  (`test_body_motion_problem.py:221`) — the container sweep over a world.
- The two-kitchen × three-robot evaluation was **deleted**; recover `get_world(use_kitchen=...)`
  and `present_results` from commit `32303f169` (removed at `05d5b07f9`/`8c926b184`).

### What is missing (the scientific payload)

1. **Repair first**: ~8 call sites of removed `set_positions_1DOF_connection` in
   `test_body_motion_problem.py` (also `learned_pouring/test_pouring_learned_paper.py:168`,
   `test_debug_expression_publisher.py:159`) → migrate to
   `JointState.from_mapping(...).apply_to(world)`. Nothing runs until this is done.
2. **Typed verdicts do not exist** — failure is a bare `False`. Worse, the discriminating
   exceptions are collapsed at `coraplex/.../predicates.py:298-306`
   (`CollisionViolatedError | LocalMinimumReachedError | QPSolverException → False`). Build a
   verdict hierarchy (SEMANTIC / CAUSAL / EMBODIMENT / OUT_OF_SCOPE / VERIFIED, house style:
   `DataclassException` subclasses) + an ordered-evaluation function that attributes the first
   false — the conjunctive query gives set membership, not attribution.
3. **OUT_OF_SCOPE has no detector**: replay writes positions with **no DOF-limit check**
   (`JointState.apply_to` → raw state write). Pre-replay validator: per-sample limit check
   against `connection.active_dofs[0].limits`, step-magnitude plausibility, does-τ-track-known
   -connections. This is the easiest missing piece and the whole out-of-scope class.
4. **Perturbation harness**: pure function `MotionTrajectory → MotionTrajectory` — noise,
   temporal warp, truncation, sign flip, target-swap (τ for the wrong container), limit
   overshoot — each labelled with its intended failure class → confusion matrix
   (precision/recall per class) via the existing `ExperimentsTable`/`TypstRenderer`.
5. **Trajectory import**: `MotionTrajectory.to_json/from_json` keyed by connection name (mirror
   `JointState.from_str_dict`) + npz/CSV loader. `MotionDAO`/`MotionTrajectoryDAO` already exist
   for DB round-trips.
6. Promote `_RecordedTrajectoryModel` to library code as the official external-generator adapter.

### Semantics traps for the experiment design

- `Causes.__call__` returns False if the effect is **already achieved** pre-replay — ensure
  non-achieved pre-states or this shows up as spurious causal failures.
- Runtime cost is dominated by `CanPerform` (10+ base samples × end effectors × up to 2000 ticks
  per candidate); cache feasibility per (robot, container, base pose) since perturbed τ variants
  share the reachability question.

Surface: ~5 new/modified library files, ~2 new files in `experiments/`, ~4 touched files for
exception preservation + test repair.

### Choice of external generator: LLM proposing object-level trajectories (recommended)

Two-tier design:

- **Tier 1 (statistical backbone): perturbation harness** — controlled ground truth, clean
  precision/recall confusion matrix with known failure labels. Defends against "the detector
  doesn't work"; vulnerable alone to "you built the errors you detected."
- **Tier 2 (headline): LLM as generator** — prompt an LLM with the verbalized scene (the EQL
  predicates already self-verbalize via `verbalize_expression`; SDT annotations give the scene
  description) to emit container-DOF waypoint trajectories as structured JSON → parse to
  `MotionTrajectory`. This matches the object-centric τ interface exactly (LLMs are good at
  parameterized object-level output and bad at robot joint trajectories — which `CanPerform`
  synthesizes anyway). Hallucinations produce *genuine* typed failures in all four classes:
  wrong handle (semantic), already-open door (causal — the correct verdict there), DOF-limit
  overshoot (out-of-scope), Stretch-infeasible arcs (embodiment).

**The high-impact ablation: typed vs. binary feedback in the repair loop.** Condition A returns
only "failed, retry"; condition B returns the typed verdict. Measure retries-to-success across
container × robot × kitchen. If B wins, the value of typed diagnosis — the Law's core selling
point — is demonstrated with data instead of argument. This also converts the thesis's
foundation-model future-work section into a preliminary result and positions against
Brunke 2025 (LLM→CBF) from the complementary, verification direction.

Ruled out: learned policies/VLAs (robot-action output → needs a new simulate-and-read-out
component; weeks of infrastructure), human teleop (no recording infra, weaker story),
off-the-shelf planners (robot-joint output, "planner verifies planner" impresses nobody).

Extra cost on top of the verification infrastructure: prompt template, JSON→`MotionTrajectory`
parser, repair-loop driver, transcript archiving — **~3–5 days**.

### Harness: Claude Code CLI headless mode (flags verified 2026-08-12)

The generator can run through the locally authenticated Claude Code installation — no API key
management. Verified capabilities:

- Single shot: `claude -p "<prompt>" --output-format json` → envelope with `result`,
  `session_id`, `total_cost_usd`.
- Structured output: `--json-schema '<schema>'` → parsed `structured_output` field
  (best-effort model-side; keep a validation + one-retry fallback).
  Schema: `{connection_name: str, waypoints: [float], time_step: float}`.
- Repair loop is native: `--resume <session-id> -p "<feedback>"` continues the conversation;
  condition A (binary "failed, retry") vs condition B (verbalized typed verdict — the EQL
  predicates self-verbalize) is just a different feedback string. Cap at k retries.
- **Tool lockdown (validity-critical): `--tools ""`** removes all tools from context — without
  it the generator could read URDFs/world-model source, look up DOF limits, and the
  OUT_OF_SCOPE failure class silently vanishes. State this in the methodology.
- System prompt: `--system-prompt "<generator role>"` (full replacement).
- Reproducibility: CLI exposes no temperature/seed. Mitigate: pin exact model id via `--model`,
  archive every JSON envelope + session JSONLs
  (`~/.claude/projects/<cwd>/<session-id>.jsonl`), report n repetitions per condition.

Module: `experiments/src/experiments/bmp_verification/llm_generator.py` —
scene verbalizer (SDT → text) → generator subprocess call → `structured_output` →
`MotionTrajectory` → out-of-scope validator → ordered predicates → typed verdict →
repair loop → `ExperimentsTable`/`TypstRenderer`.

Design knobs to fix before implementing:

1. **Information ablation** as a controlled variable: withhold DOF limits → out-of-scope
   failures; withhold opening state → causal failures (already-open); full information →
   mostly embodiment failures. 2–3 information levels turn hallucination rates into a designed
   independent variable.
2. **CLI subprocess vs Python Agent SDK**: SDK is cleaner at scale (no per-call process spawn,
   programmatic sessions, budget cap) and uses the same authentication; CLI is simpler and
   adequate for a few hundred calls. Prototype with CLI, migrate only if volume grows.
3. **Model-tier sweep** (Haiku/Sonnet/Opus, same protocol): proposal quality varies, detection
   performance should not — evidence that the verifier is generator-agnostic.

---

## 5. Knowledge-based servoing on SDT + EQL, coupled to the VDJM — the synthesis contribution

A modern re-implementation of Case Study 1 (`ch-casestudy.typ`, `huerkamp2025.pdf`) on today's
stack — qualitative theory as an EQL rule tree over a Semantic Digital Twin, feeding the motion
statechart instead of a discrete primitive-to-twist bridge, with the VDJM supplying the numeric
effect model. This is the largest item in this report and the one with the highest argumentative
return.

### Why this is a contribution and not a port

Three claims become testable that no other candidate experiment in this report can support.

1. **The precision–expressivity trade-off is an artifact of the primitive interface, not of
   symbolic causal reasoning.** `ch-casestudy.typ:349-351` presents the two case studies as
   bracketing a trade-off: ~6 % final goal error for the symbolic instantiation against <1 % for
   numeric fill-level control, and the chapter attributes the concession to the *plugged-in
   theory* ("nothing prevents a precision-oriented rule set"). The sharper diagnosis is
   architectural: precision is lost in `eq:cs1:twist`, where the reasoner's ten Boolean
   primitives become a fixed-gain bang-bang twist. Replace that bridge with the reasoner
   *selecting and parameterizing statechart tasks* whose fill row is the VDJM terminal-state
   prediction, and the symbolic layer keeps its semantics while the QP keeps its precision. The
   published 6 % is a **real, citable baseline to beat under matched expressivity** — an unusually
   clean comparison to have available inside one's own thesis.
2. **It is the only candidate that closes the Adaptivity Gap endogenously.** Section 1's regime
   switch supplies the switching *mechanism* but takes its trigger from a hand-published ROS
   topic, so the context→constraint decision sits outside the system. The defeasible reasoner is
   exactly that missing decision layer: at 10 Hz it decides which constraint regime holds, the
   statechart executes at controller rate. "Medication ⇒ strict no-spill" then follows from an SDT
   annotation through a rule, not from the experimenter's keyboard.
3. **It repairs two delimitations the thesis already concedes in writing.**
   `ch-casestudy.typ:331`: the published system "predates the SDT machinery … there is no
   persistent property map `P` supporting physics instantiation, and states are grounded per cycle
   rather than maintained as a twin." Same paragraph: `Π_avoid` (spilling) is "handled reactively
   … rather than certified against in advance" — which the VDJM's predictive rows plus
   `KeepProjectileInReceiver` do certify in advance. A combined system converts both admitted
   limitations into resolved ones, and `ch-casestudy.typ:359`'s first limitation (fixed velocity
   gains causing corrective spillage) dissolves with the twist bridge that produced it.

### What already exists

The mapping from the paper's five-equation loop (`eq:cs1:loop`) onto machinery in this tree is
closer than the chapter's "predates the SDT" caveat suggests:

| Paper component | Modern realization in this tree |
|---|---|
| Defeasible theory `(R_th, >)`, superiority `r' > r` | `Refinement` conclusion selector (`krrood/src/krrood/entity_query_language/rules/conclusion_selector.py:162`) — literal "yields left unless the right side produces values", i.e. except-if. `Alternative`, `Next` alongside it |
| `SCHMOD` (schemas/affordances from facts) | MCRDR annotation inference: `WorldReasoner.infer_semantic_annotations` (`semantic_digital_twin/.../reasoning/world_reasoner.py:41`), `world_rdr/world_semantic_annotations_mcrdr.py` |
| `FWDMOD` (what to expect / what to query next) | `BackwardInferenceIndex` / `ConclusionSufficientConditionSets` (`krrood/.../entity_query_language/rdr/backward_inference.py:80`) — for a target conclusion it returns every sufficient condition set, which *is* the query-derivation step |
| `SatisfiesRequest` as a query over a fact base | EQL conjunctive queries over the twin; `bmp_predicates.py`, `coraplex/.../body_motion_problem/predicates.py`; queries self-verbalize (`entity_query_language/verbalization/`) |
| Fact base `Y` grounded per cycle | The SDT itself — persistent property map, no per-cycle re-grounding |
| Affordance/schema vocabulary | Partly present as annotations: `HasFillLevel`, `HasSupportingSurface` (Support), container annotations (Containment), `Connection` types (Linkage) |
| Primitives → twist → QP | Motion statechart tasks + life-cycle gating (section 1), `Sequence`/`Parallel`, feature-function tasks |
| `Causes` numerically | VDJM terminal-state prediction row (`qp/constraint_collection.py:328`, `qp/terminal_state_prediction_strategy.py`) |
| Evaluation platforms | `physics_simulators/src/physics_simulators/mujoco_simulator.py`; `resources/mjcf/pr2_kinematic_tree.xml`, `collision_configs/{pr2,hsrb}.srdf` |

Note what is *absent*: no image-schema, affordance-schema, or defeasible-rule code exists anywhere
in this tree (verified by grep over `.py`/`.md`). The CS1 implementation is not present here at
all, so the theory layer is net-new authoring — against existing formalism support, not from
scratch.

### What must be written

| Work | Where | Size |
|---|---|---|
| Schema/affordance annotations (Containment, Support, Linkage, `canPourTo`-style affordances) as `SemanticAnnotation` subclasses over existing mixins; ORM regen | `semantic_digital_twin/.../semantic_annotations/` | ~200–300 |
| Qualitative transfer theory as an EQL rule tree with `Refinement` exceptions (pouring first; draining/scraping only for the modularity claim) | new `semantic_digital_twin/.../reasoning/substance_transfer/` | ~300–500 |
| Expectation layer: target conclusion → sufficient condition sets → the facts to check this cycle, plus retraction on refutation | same package, on top of `BackwardInferenceIndex` | ~150–250 |
| **Reasoner→statechart interface** (the research core): symbolic conclusion ⇒ task gating + `FloatVariable` rewriting, replacing `eq:cs1:twist` | new `giskardpy/.../motion_statechart/` binding module | ~200–400 |
| Two-rate driver (reasoner ~10 Hz alongside the control loop) + transcript logging of inference chains | `experiments/src/experiments/` | ~250 |
| MuJoCo particle scenario matched to the paper's setup for the 6 %-baseline comparison | `experiments/`, reusing `mujoco_simulator.py` | ~200 |
| Tests | `test/semantic_digital_twin_test/`, `test/giskardpy_test/test_motion_statechart/` | ~800–1200 |

**~1,300–1,900 source + ~800–1,200 test lines; 4–6 weeks** for the full claim.

### Staging (this is what makes it plannable)

- **Stage A — demonstrator, ~1.5 weeks.** Pouring only. Reasoner writes *only* pre-declared task
  life-cycle conditions and float variables; no new annotations beyond what exists; grounding read
  from world state, not perception. Yields the headline figure: a semantic rule, not a topic
  publish, flips the constraint regime mid-pour, at VDJM precision. Reuses section 1's driver
  wholesale — build section 1's experiment so its trigger source is swappable.
- **Stage B — the numbers, +1.5 weeks.** MuJoCo particle scenario, PR2/HSR, matched gains and
  metrics, n runs against the published 6 % under matched expressivity (fill goal + spill reaction
  + retained substance).
- **Stage C — the modularity claim, +2–3 weeks.** Draining and scraping absorbed by rule-set
  extension with every other component unchanged — the CS1 claim re-run on the new stack, now
  with the twin and predictive `Π_avoid`.

Stage A alone is publishable as a thesis section; B makes it a result; C makes it a chapter.

### Caveats to state honestly

- **Hard blocker: the reasoner cannot modify the world model during a motion.**
  `WorldReasoner.reason` runs inside `with self.world.modify_world():`
  (`world_reasoner.py:47`), and `ControlLoop.apply_world_updates` aborts on
  `self.world.world_is_being_modified`
  (`giskardpy/src/giskardpy/middleware/ros2/control_loop.py:100-107`,
  `WorldModelModifiedDuringMotionError`). So MCRDR annotation inference cannot run concurrently
  with an active goal as written. Three options, needing a developer decision:
  (i) restrict in-motion inference to a path that writes only DOF state and float variables
  (Stage A's scope — cheapest, and adequate for the headline claim);
  (ii) run annotation inference only at motion boundaries (loses the closed loop that is the whole
  point of `FWDMOD`);
  (iii) admit annotation-only modifications during motion — needs an argument that adding an
  annotation cannot invalidate compiled symbolic expressions, which is *false* in general, since a
  new annotation is precisely what would imply a new task.
  Do not paper over this: it is the same pre-declaration boundary section 1 hits, and it is the
  honest scope statement for both.
- **Rate is unverified.** The paper needs ~10 Hz for the reasoner. EQL/MCRDR inference latency over
  a populated apartment twin has not been measured, and `CanPerform`-style queries with costmap
  sampling are far slower than that. Measure this in Stage A before committing to B/C — if it
  lands at 1–2 Hz, the contribution survives but the claim becomes "reasoning at task-event rate",
  not "servoing".
- **`Refinement` is not Antoniou defeasible logic.** It is except-if refinement over an RDR tree:
  conclusions are overridden by more specific rules rather than defeated via an explicit
  superiority relation over a rule set, and there is no `p`/`−p` objection machinery. The mapping
  is defensible and arguably the better engineering, but it must be *argued* in the chapter, not
  assumed — an examiner who knows defeasible logic will ask.
- **Perception is still factored out.** This does not repair `ch-casestudy.typ:359`'s fourth
  limitation (simulation-only, semantic-feature perception unvalidated). Grounding facts from the
  twin is arguably *further* from real perception than the paper's geometric grounding, since the
  twin is assumed populated. State it; do not claim a hardware contribution.
- **Framing risk.** Described as "re-implementation on the current stack" this reads as
  engineering. It must be presented as the three claims above — trade-off dissolution, endogenous
  adaptivity, predictive `Π_avoid` — with the published 6 % as the baseline. A port without the
  comparison is not worth 4–6 weeks.
- **Thesis placement is a real question, not a detail.** It instantiates neither case study
  cleanly: it is CS1's theory on CS2's executive. Options: a synthesis section in @ch:casestudy2
  after the analytic/learned spectrum; a short chapter of its own between the case studies and the
  conclusion; or the lead item of Ch. 7's forward-looking section if only Stage A is completed.
  The choice changes how much of B/C is required.

Relation to the other items: this **subsumes section 1's decision layer** while depending on
section 1's mechanism, and it is independent of sections 2–4. If both are done, section 1 should be
written as the mechanism characterization (switch latency, injection-phase feasibility envelope)
and this as the closed semantic loop that uses it.
