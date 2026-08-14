# Knowledge-servoing spike findings

Unsupervised session, branch `knowledge-servoing-plan`. Started 2026-08-13 22:38 UTC.

Each section records what was tried, the `file:line` evidence, the answer, and whether it
**confirms** or **contradicts** the plan (`doc/knowledge_servoing/implementation_plan.md`).

---

## Toolchain baseline (before any change)

Ran the briefing's verification command from the repo root:

```
uv run pytest test/giskardpy_test/test_motion_statechart/test_pouring.py -x -q
```

**Result: cannot collect.** The whole `test/giskardpy_test` tree fails at conftest import —
`test/giskardpy_test/conftest.py:9` imports `GiskardTester`, which reaches
`giskardpy/src/giskardpy/middleware/ros2/giskard.py:9` → `import rclpy`, and `rclpy` is not
installed in this environment (`ModuleNotFoundError: No module named 'rclpy'`). This blocks *every*
giskardpy test here, not only the HSRB/Justin robot tests the briefing warned about.

Task 1 lives entirely under `test/krrood_test/`, a separate tree with its own conftest that does
not import ROS. That tree runs cleanly:

```
uv run pytest test/krrood_test/test_ripple_down_rules -q
=> 44 passed, 36 skipped, 0 failed   (12 s)
```

**Decision (reversible):** proceed with Task 1 (krrood-only, toolchain works). Task 2 edits shared
QP machinery and *must* keep the three named giskardpy test files green, which cannot be verified
here because the giskardpy tree cannot be collected. That makes Task 2's stop condition ("keeping
the baseline green needs verification") unmeetable in this environment — see the Task 2 section.

Note: running the krrood tree in isolation regenerates
`test/krrood_test/dataset/ormatic_interface.py` (drops the coraplex/SDT imports that are absent when
only krrood is importable). That is an environment artifact, not a code change; it was reverted with
`git checkout --` and never committed.

---

## Task 1 — MCRDR spike (plan §4.1.1)

Toy MCRDR built by hand over a three-boolean case, nothing to do with pouring. Conclusions are three
frozen "regime decision" mimic classes; one top rule carries a stop rule; one top rule conditions on
a conclusion an earlier rule made. Evidence tests:
`test/krrood_test/test_ripple_down_rules/test_knowledge_servoing_spike.py`; mimics in
`test/krrood_test/dataset/knowledge_servoing_case.py`.

### Item 1 — does `copy_case` survive a frozen dataclass?

`MultiClassRDR._classify` (`krrood/.../rdr.py:1497`) does `case_cp = copy_case(case)` then writes each
firing rule's conclusion into `case_cp` via `update_case_with_conclusion_output`
(`rdr.py:1522`, `helpers.py:161` → `utils.py:353`).

`copy_case` (`krrood/.../utils.py:865-889`) shallow-copies the object, then for every *iterable*
attribute tries `setattr(case_copy, attr, copy(attr_value))` and **skips on `AttributeError`**
(`utils.py:886-888`). `dataclasses.FrozenInstanceError` subclasses `AttributeError`, so on a frozen
dataclass that per-attribute deep-copy is silently skipped.

Measured behaviour (three distinct case shapes):

| Case shape | `copy_case` result | Writing a conclusion into the copy |
|---|---|---|
| frozen, **no** conclusion field | new equal frozen instance | **`AttributeError`** — `update_case_in_case_query` does `getattr(case, attribute_name)` with no default (`utils.py:370`); the attribute does not exist → crash |
| frozen, **with** a mutable `set` field | new instance **sharing the original's set object** (the per-attr copy was skipped) | writes succeed **but mutate the original's set** — the conclusion **leaks back into the "immutable" object** |
| mutable (non-frozen), with a `set` field | new instance with a **distinct** set (per-attr copy ran) | writes land only on the copy; original untouched |

**Answer: a frozen dataclass does *not* work cleanly as an MCRDR `Case`.** It either crashes (no
accumulator) or silently leaks conclusions into the shared object (with an accumulator). The leak is
the dangerous case: it is exactly the object §2.3 wants to hand between threads immutable, and
`copy_case`'s shallow copy makes the reasoner's classification mutate it in place.

**Relation to the plan (§4.1.1, decision-day item 1): CONFIRMS the concern; selects the fallback.**
The plan already anticipated this and named two fallbacks. The evidence says the frozen-object route
is not merely awkward but unsafe (shared-set leak), so a fallback is required, not optional. Of the
two:

- *mutable working copy* — ground/derive a **non-frozen** working case for the classifier; it copies
  cleanly and never leaks. Demonstrated working.
- *thin mutable `Case` adapter* — wrap the frozen situation in a mutable holder before `classify`.

Recommended, most-reversible reading of §2.3 + §4.1.1: keep `TransferSituation` a **frozen**
dataclass for the thread hand-off (so §2.3's immutable-crossing guarantee holds), and inside
inference — reasoner-side, never shared — build a mutable working copy for `classify`. This satisfies
both the thread-safety design and the engine. It does **not** contradict the plan.

### Item 2 — do stop rules and multi-conclusion behave as the plan claims?

In-memory `MultiClassRDR.classify` over the toy theory:

- **multiple top rules contribute in one pass** — `engage ∧ restrict` ⇒ `{EngageRegime,
  RestrictRegime, EscalateRegime}`.
- **a stop rule blocks the parent and adds nothing** — `MultiClassStopRule` (`rules.py:543-566`)
  sets `top_rule.fired = False`; with the defeater true, `engage` ⇒ `{}` (Engage blocked, nothing
  substituted).
- **intra-pass chaining** — a later top rule conditioned on `EngageRegime ∈ case.conclusions` fires
  in the *same* pass, because `_classify` writes each conclusion back into `case_cp` before
  evaluating the next top rule (`rdr.py:1518-1528`). `engage` alone ⇒ `{EngageRegime,
  EscalateRegime}`; `restrict` alone ⇒ `{RestrictRegime}` (no chaining without the antecedent).

**Answer: all three behave exactly as §4.1.1 claims. CONFIRMS the plan.** The "being wrong twice"
check (§9) passes: the MCRDR engine really does supply multi-conclusion, defeaters-as-stop-rules and
single-pass forward chaining, so WP1's stratification reduces to top-rule ordering rather than a
hand-built pipeline.

### Item 3 — generated code vs in-memory tree at control rate

Both paths produce **identical** conclusions for every toy input (multi-conclusion, stop, chaining).
Latency on this machine (2000 iterations, `MutableSituation`, warm):

| Path | per `classify` | single-situation rate | 4 situations / control cycle |
|---|---|---|---|
| in-memory tree | ~117 µs | ~8 500 Hz | ~476 µs ⇒ ~2 100 Hz |
| generated code (`_write_to_python` + import) | ~35 µs | ~28 500 Hz | ~9 µs×4 ⇒ ~7 000 Hz |

**Answer: both clear the plan's ~10 Hz over a handful of situations (§8.2) by 200×–2800×.** The
in-memory tree alone is more than fast enough; generated code is a ~3.3× constant-factor win, not a
requirement. **CONFIRMS §8.2**, and answers §4.1.1 decision-day item 2: run the in-memory tree unless
a later profile shows a real budget problem — the generated path is an optimization, not a
correctness need.

**One caveat the plan should record.** The generated form is only equivalent when produced by the
*current* code writer. `MultiClassTopRule.get_conclusion_as_source_code` (`rules.py:674-691`) emits
`update_case_and_conclusions_with_rule_output(case, conclusions, …)`, which writes conclusions back
into the case and therefore *does* support chaining and stop rules. But the committed world
annotation MCRDR
(`semantic_digital_twin/.../world_rdr/world_semantic_annotations_mcrdr.py:27-51`) is an *older* flat
form (`conclusions.update(make_set(conclusion_X(case)))`) that never writes back into the case — a
theory relying on intra-pass chaining would silently lose it if shipped as that stale generated form.
A theory that uses chaining must be regenerated with the current writer, or run from the in-memory
tree. Also note the generated `classify` copies only `Case` objects, not plain dataclasses
(`create_case` returns dataclasses as-is, `rdr.py:1189-1192`), so a dataclass case passed to
generated code is mutated in place — the same item-1 hazard, now on the caller's object.

**Net: Task 1 does not invalidate the plan's design.** All three items confirm §4.1.1/§8.2; item 1
only forces the already-documented fallback. Task 2 was therefore in scope, but is blocked by the
environment (see below).

---

## Task 2 — symbolic `goal_value`

**Implemented, on explicit user instruction to "continue without running the tests"** after ROS
could not be installed here (see below). The change is the type-widening slice of plan §3.1 plus its
tick-time and JSON consequences. What could be verified here was; the two giskardpy pouring
regression suites could not be run, so their green-ness is **unverified in this environment** and
must be confirmed in a ROS-capable one.

### What was changed

- `TerminalStatePredictionConstraint.goal_value`, `TerminalFillConstraintTask.goal_value` and
  `ConstraintCollection.add_terminal_state_prediction_constraint`'s `goal_value` widened from `float`
  to `sm.ScalarData`. The bound `sm.Scalar(constraint.goal_value) - free_response()`
  (`terminal_state_prediction_strategy.py:275`) already accepted a symbolic value, so no bound change
  was needed.
- `TerminalFillConstraintTask.on_tick` now resolves the goal to a live float via a new
  `_current_goal_value` (`FloatVariableData.get_value` for a `FloatVariable`, `evaluate()` for a
  general symbolic value, `float()` otherwise), and `_fill_goal_reached` takes that float. This is the
  §3.1 fix for the `<=`/`>=` sites at `tasks/pouring.py:241,281` that would otherwise raise
  `HasFreeVariablesError` on a symbolic goal. Plain-float goals behave exactly as before.
- JSON guard (committed separately): `DataclassJSONSerializer.to_json` now raises
  `SymbolicValueNotSerializableError` when a field holds a symbolic value with free variables, so a
  symbolic-goal task fails loudly instead of round-tripping to goal 0
  (`json_serializer.py`, `adapters/exceptions.py`). The guard triggers only on free variables, so
  constant serialization is unchanged.

The guard at `tasks/pouring.py:468` (`minimum_clearance <= 0.0`) was **not** touched: it belongs to
`KeepSourceRimAboveReceiverRim` and only breaks if `minimum_clearance` becomes symbolic, which is
plan §3.3, not the §3.1 `goal_value` widening. Changing it would be out-of-scope drive-by work.

### What was verified, and how

`rclpy` is only needed by giskardpy's ROS2 middleware; the QP and task modules import without it, so
the changes were exercised by importing the test modules directly (bypassing the ROS conftest) and
running the test methods, plus the runnable SDT gate:

| Check | How | Result |
|---|---|---|
| retargeted bound == float-goal bound; tracks live changes; goal is a free var of the bound | `test_qp/test_terminal_state_symbolic_goal.py` run standalone | pass |
| `_current_goal_value` resolves a live symbolic goal; `_fill_goal_reached` uses it without raising; float goal unchanged | `test_motion_statechart/test_terminal_fill_symbolic_goal.py` run standalone | pass |
| symbolic-goal task refuses JSON serialization | same file, standalone | pass |
| generic JSON guard + no serialization regression | `test/krrood_test/.../test_symbolic_field_serialization_guard.py` + existing json tests under pytest | 2 + 21 pass |
| no krrood regression from the guard import | `test_ripple_down_rules`, `test_symbolic_math`, `test_utils` under pytest | 363 passed, 36 skipped |
| SDT pouring-equations gate still green | `test_pouring_equations.py` under pytest | 7 passed |

The §3.2 single-terminal-row regression the briefing asked for **already exists** as
`test_terminal_state_prediction_strategy.py::TestTerminalStateConstraintValidation::test_two_terminal_constraints_raise_dedicated_error`,
so it was not duplicated.

### What remains unverified (the honest gap)

- `test/giskardpy_test/test_motion_statechart/test_pouring.py` and `..._learned.py` — the two named
  regression gates that exercise full pouring motions through `on_tick` — **could not be run** (see
  the rclpy blocker below). The `_fill_goal_reached`/`on_tick` change preserves float-goal behavior by
  construction and the standalone task tests confirm the resolution path, but the end-to-end pour has
  not been re-run here. **Run both in a ROS-capable environment before relying on this.**
- The new giskardpy-tree tests cannot be collected here either (same conftest blocker); they were
  verified by direct import. They will run under pytest in CI.
- **ORM not regenerated.** The `sm.ScalarData` type means ormatic emits no column for `goal_value`
  (§3.1, decided). Regeneration was **not** run: in this container ormatic misfires (it drops
  unrelated coraplex/SDT imports because those packages are not fully importable), so
  `scripts/regenerate_all_orm.py` would produce a corrupt mass-diff, not the clean "column
  disappears" diff. The consequence is documented in the field docstrings; **regenerate ORM in a
  fully provisioned environment** and commit that as its own change.

### The three regression gates

| Regression gate | Tree | Status here |
|---|---|---|
| `test/semantic_digital_twin_test/test_physics/test_pouring_equations.py` | SDT | **7 passed** (before and after the change) |
| `test/giskardpy_test/test_motion_statechart/test_pouring.py` | giskardpy | **cannot collect** (rclpy) |
| `test/giskardpy_test/test_motion_statechart/test_pouring_learned.py` | giskardpy | **cannot collect** (rclpy) |

Both giskardpy gates die at conftest import: `test/giskardpy_test/conftest.py:9` →
`giskardpy/.../utils_for_tests.py:10` → `giskardpy/.../ros2/giskard.py:9` does a hard `import rclpy`
(plus the full ROS2 action stack) at module load, before any mock is installed. `rclpy` is not
installed in this container. A mock exists (`semantic_digital_twin/.../utils.py:213 MockedRCLPY`, and
`dataclasses.py` installs it) but only on the SDT import path — it never reaches giskardpy's
top-level `import rclpy`.

### Attempt to install ROS with the repo's script — blocked by egress policy

The repo ships `scripts/setup_ros_workspace.sh` → `.github/docker/setup_workspace.py`. Two facts make
it unusable in *this* container:

1. It presupposes a ROS Jazzy **base image**. It only `apt install`s `ros-jazzy-*` *overlay*
   packages and sources `/opt/ros/jazzy/setup.bash` to `colcon build`; it never installs `ros-base`
   or `rclpy` themselves. This container has no `/opt/ros` at all, no `rclpy` anywhere on disk, and no
   ROS apt source configured (Ubuntu Noble, root, `apt` present).
2. The ROS package mirrors needed to create that base are **denied by this session's egress policy**
   (proxy `403 host_not_allowed`, which `/root/.ccr/README.md` says to report, not route around):
   `packages.ros.org` → 403, `snapshots.ros.org` → 403, and even the `ros-apt-source` GitHub release
   → 403. `github.com` itself is reachable, but the base packages are not on it.

So ROS cannot be installed here regardless of the script. Unblocking Task 2 requires either a
container built **from** a `ros:jazzy` base image (then the repo script applies as intended), or an
egress-policy change that allows `packages.ros.org`. Neither is doable from inside this session.

Tooling note: `scripts/format_docstrings.py` could not run — `docformatter` is not installed in this
container (`black` is). All new/changed files were formatted with `black` only. A fully provisioned
environment should re-run `scripts/format_docstrings.py` on them.

---

## WP1 — reasoning framework (slice 1, post-spike)

Built the plan's domain-agnostic framework interface layer (§2.5.1) plus a single concrete theory, on
the reasoning side only — no controller, no QP, no pouring vocabulary, no ROS. Fully runnable and
verified here.

New package `semantic_digital_twin/.../reasoning/knowledge_servoing/`:

- `interfaces.py` — `Situation` (frozen), `ControlDecision` → `RegimeDecision`/`ParameterDecision`
  (the two write channels), `DecisionSet` (with `of_type`/`contains_type`/`from_conclusions`),
  `SituationGrounding` and `SymbolicTheory` (the latter declares `decision_types` for the future
  binding policy's build-time checks).
- `general_rdr_theory.py` — `GeneralRDRTheory` is the **only** theory adapter. A `GeneralRDR`
  composes one sub-classifier per decision family and re-runs them to a fixpoint, so a rule in one
  family can chain on a conclusion another family reached (§2.5, §4.1.1's `GeneralRDR` note). A
  `GeneralRDR` with a single sub-classifier is exactly a `MultiClassRDR`, so this one engine subsumes
  the multi-class case — the earlier separate `MultiClassRDRTheory` adapter and its base class were
  removed as needless surface area (**decision, at the user's request: keep it simple**). It
  classifies each frozen situation through a mutable working copy carrying one accumulator per family
  (built from the classifier's family names via `make_dataclass`), because `general_rdr_classify`
  writes each family's conclusions back into the case between rounds. This is the direct application
  of the spike's item 1: a frozen dataclass cannot be classified in place (it crashes or leaks), so
  the situation is wrapped in a mutable case whose `copy_case` isolates the accumulators and leaves
  the frozen situation untouched. Rules read facts as `case.situation.<fact>` and other families'
  conclusions as `case.<family>`.

Verified by `test/semantic_digital_twin_test/test_reasoning/test_knowledge_servoing_framework.py`
against a domain-free mimic gauge theory (`knowledge_servoing_mimic.py`, a regime family and a
parameter family): both write channels, a defeater cascading across families, intra-family and
cross-family chaining, aggregation across situations, `DecisionSet` channel filtering, and that
inference does not mutate the frozen situation — **9 passed**. The 9 failures elsewhere in
`test_reasoning` (`test_bmp_predicates.py`) are pre-existing stale-API breakage
(`create_with_new_body_in_world` no longer takes `active_axis`), unrelated to this change and left
untouched.

Two things recorded for the next session:

- **ORM:** ormatic auto-discovers these dataclasses (it wanted `DecisionSetDAO`,
  `ControlDecisionDAO`, …). They are transient per-cycle reasoning objects and almost certainly
  should not be persisted; whether to add them to `generate_orm.py`'s `ignore_classes` or map them is
  an open decision. No regen was committed (the container regenerates ORM incorrectly). No runnable
  test gates on this, so nothing here is red because of it.
- **Rule-authoring convention:** rules read facts as `case.situation.<fact>` and earlier conclusions
  as `case.conclusions`. That is the explicit-wrapper convention; the world annotation RDR instead
  classifies the domain object directly. WP1's real `TransferSituation` theory should settle which
  convention it uses (wrapper vs a flat delegating case) — a reversible choice, wrapper chosen here
  for clarity.

---

## WP2 — statechart binding (slice 1, post-spike)

Connected the theory to the controller. New package
`giskardpy/.../motion_statechart/knowledge_servoing/` wires a `SymbolicTheory`'s decisions to the two
write channels (§2.2):

- `SymbolicTheoryNode` — on each control tick grounds the world, runs the theory, publishes the
  decision set to a `DecisionSlot`, and applies the parameter decisions to their float variables.
- `ConcludedMonitor` — observation TRUE while the latest decision set contains a given decision type,
  FALSE after an inference without it, UNKNOWN before the first (channel 1: gate a task's start
  condition on it).
- `DecisionBindingPolicy` — the declarative map from decision types onto task activations and
  float-variable writes, validated at `build()` (unbound type, double binding, wrong channel,
  unregistered target all raise). This is the "pluggable part of the controller" (§2.5.2).
- `DecisionSlot` — the single-writer hand-off between node and monitors.

**Decision, at the user's request (keep it simple): the reasoner runs synchronously on the control
tick, not on a separate thread.** The spike measured inference at ~35–120 µs over a handful of
situations — far inside a 10 Hz (or 50 Hz) control budget — so the plan's §2.3 thread-safety
apparatus is unnecessary: §2.3 existed to protect the reasoner's *off-thread reads* of live world
state, and with grounding+inference on the one thread that owns the world there is no cross-thread
read at all. The frozen-situation / mutable-working-copy hygiene is kept (cheap, and keeps the door
open). A reasoner thread can return later if a much heavier theory ever exceeds the budget — a
reversible change, since `SymbolicTheoryNode` is the only place that would move.

Verified end to end without a robot or ROS (`test_knowledge_servoing_binding.py`, run by importing
the module directly, as the giskardpy conftest needs `rclpy`): a fake grounding + theory drive a
tick that publishes decisions, flips a `ConcludedMonitor` to TRUE, and lands a parameter decision's
value in its float variable; the build-time checks all raise — **10 passed**. These run under pytest
in a ROS-capable CI.

What remains for WP2 (needs a ROS-capable env to exercise fully): assembling a real `MotionStatechart`
that gates actual tasks off `ConcludedMonitor`s with a `SymbolicTheoryNode` sibling, ticked by a real
`Executor`; the `DecisionTranscript` (§5); and wiring the deferred symbolic `goal_value` as the
`RetargetFillLevel` `ParameterDecision`'s target (this is exactly where WP0's `goal_value` widening
reconnects).

---

## What the next session should do

1. **Confirm the Task 2 change in a ROS-capable environment.** Collect and run
   `test/giskardpy_test/test_motion_statechart/test_pouring.py` and `..._learned.py` (the two gates
   that could not run here) and the two new files
   (`test_qp/test_terminal_state_symbolic_goal.py`,
   `test_motion_statechart/test_terminal_fill_symbolic_goal.py`). These verify the `goal_value`
   widening did not regress the end-to-end pour.
2. **Regenerate ORM** with `scripts/regenerate_all_orm.py` in a fully provisioned environment and
   commit it on its own — `goal_value`'s `ScalarData` type drops its column, which cannot be
   regenerated correctly in this container.
3. **Re-run `scripts/format_docstrings.py`** on all files this branch added/changed once
   `docformatter` is available (they are `black`-clean already):
   `test/krrood_test/dataset/knowledge_servoing_case.py`,
   `test/krrood_test/test_ripple_down_rules/test_knowledge_servoing_spike.py`,
   `test/krrood_test/test_utils/test_symbolic_field_serialization_guard.py`,
   `krrood/src/krrood/adapters/{exceptions,json_serializer}.py`,
   `giskardpy/src/giskardpy/motion_statechart/tasks/pouring.py`,
   `giskardpy/src/giskardpy/qp/{terminal_state_prediction_strategy,constraint_collection}.py`,
   and the two new giskardpy test files.
4. **Author the real substance-transfer theory on top of the slice-1 framework.** `TransferSituation`
   subclasses `Situation` (frozen), and `SubstanceTransferTheory` is a `GeneralRDRTheory` — the
   mutable-working-copy handling (spike item 1) is already done by the framework, so this is rule
   authoring plus the grounding, not engine plumbing. Reuse the mimic-theory test as the pattern.
5. **Run the substance-transfer theory from the in-memory MCRDR tree** at control rate; item 3 shows
   ~10 Hz is cleared by ~200×–2800×, so the generated-code path is an optimization, not a
   requirement. If it is ever shipped as generated code, regenerate with the *current* writer (the
   committed world-annotation MCRDR is an older flat form that silently drops intra-pass chaining).
6. **Continue WP0 §3.3** (symbolic `minimum_clearance`/`clearance_band`, per-task weight variable,
   etc.); the `minimum_clearance <= 0.0` guard at `tasks/pouring.py:468` needs the same live-value
   treatment as `_fill_goal_reached` once that field is widened.
