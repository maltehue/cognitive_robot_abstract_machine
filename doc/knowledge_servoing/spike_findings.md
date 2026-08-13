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
only forces the already-documented fallback. Task 2 may proceed (subject to its own environment
blocker below).

---

## Task 2 — symbolic `goal_value`

(pending)

---

## What the next session should do

(to be finalized at end of session)
