# Briefing for an unsupervised session

You are working on the knowledge-based-servoing implementation described in
`doc/knowledge_servoing/implementation_plan.md`. Read that plan before doing anything, along with
the repository's `AGENTS.md`, which overrides any default behaviour you would otherwise apply.

Nobody is watching this run. That changes what "doing well" means: a small amount of verified,
committed work with an honest write-up beats a large amount of unverified work, and stopping at a
genuine blocker beats routing around it.

## What this run is for

**Two spikes, in order.** They are the cheapest things that can invalidate the plan's two largest
design decisions, so they are worth more than any amount of implementation built on top of
assumptions that might be wrong.

Do **not** attempt to implement the plan. It is 5–7 weeks of work. Do not start the theory content,
the MuJoCo bridge, arm A, or the experiment drivers.

## Environment

- Run everything with `uv run` from the repository root. Do not use a virtualenv under `~/venvs`.
- Tests are pytest. Reuse fixtures from the relevant `conftest.py`.
- **Known-failing before you start:** HSRB and Justin robot tests fail here for missing ROS
  packages. They are not your fault and not your problem. If you see them fail, note it and move on.
- Never edit `ormatic_interface.py` files by hand. If ORM breaks, run
  `scripts/regenerate_all_orm.py`.
- Run `scripts/format_docstrings.py` on files you modify.

## Hard rules

- **Test-driven.** Every behaviour gets a failing test before its implementation. Never modify an
  existing test to make it pass.
- **Never push to `mainmain`** (`cram2/cognitive_robot_abstract_machine`). Never open, comment on or
  modify anything upstream. Push only to `origin` (`maltehue/...`), only to this branch.
- **Commit incrementally**, after each green step. This container may die; anything uncommitted is
  lost. Small commits with honest messages.
- Commits are authored as the repository user. No `Co-Authored-By` trailer for an assistant, no
  assistant identity as author. A plain `Made with the help of Claude` line in the body is fine.

## Stop conditions

Working around a blocker is the main way an unsupervised run destroys value. If any of these occur,
**stop that task, write up what you found in `doc/knowledge_servoing/spike_findings.md`, commit, and
move to the next task** — do not invent a workaround and build on it:

- A spike's answer contradicts the plan's design decision (that is a *successful* spike — report it).
- A change would require editing an existing test.
- A change would require touching `executor.py`, `qp/qp_data.py`, `motion_statechart.py`,
  `graph_node.py` or `control_loop.py`. The plan explicitly does not touch these.
- You cannot get a test to pass without weakening what it asserts.
- More than roughly a third of the run has gone into one task.

## Output protocol

The transcript will not be read. Your findings only exist if they are in the repository.

Maintain `doc/knowledge_servoing/spike_findings.md` as you go, with one section per task:
what you tried, the file:line evidence, the answer, and — most importantly — **whether it confirms
or contradicts the plan**, naming the plan section. End the file with a short "what the next session
should do" list.

---

# Task 1 — MCRDR spike (do this first)

Plan reference: §4.1.1. The plan decides that the substance-transfer theory is authored as a
`krrood.ripple_down_rules.MultiClassRDR` over flat case objects. Three things were reasoned about
from source but never executed. Find out.

1. **Does `copy_case` survive a frozen dataclass?** `MultiClassRDR._classify` (`rdr.py:1491-1533`)
   copies the case and writes conclusions into the copy via
   `update_case_with_conclusion_output`. The plan's thread-safety design (§2.3) requires the object
   crossing between threads to be an immutable frozen dataclass. Determine whether a frozen
   dataclass works as an RDR `Case`, and if not, which of the plan's two fallbacks is right: a
   mutable working copy created inside grounding, or a thin mutable `Case` adapter.
2. **Do stop rules and multi-conclusion behave as the plan claims?** Build a minimal MCRDR over a
   toy situation (two or three boolean fields, nothing to do with pouring) and verify by test that:
   several top rules can contribute conclusions in one pass; a `MultiClassStopRule` blocks a parent
   conclusion and adds nothing; a later top rule can condition on a conclusion an earlier one made
   (the intra-pass chaining the plan relies on to avoid a hand-built stratified pipeline).
3. **Generated code or in-memory tree at control rate?** The world's annotation RDR ships as
   generated source (`semantic_digital_twin/.../world_rdr/world_semantic_annotations_mcrdr.py`).
   Measure classification latency both ways on the toy theory and report numbers. The plan needs
   ~10 Hz over a handful of situations (§8.2).

Deliverable: tests under `test/krrood_test/` following that package's rules — **`krrood` must stay
self-contained**, so use mimic classes in `test/krrood_test/dataset`, never an import from another
workspace package. Plus the findings section.

If item 2 contradicts the plan, say so plainly. The plan's §4.1 documents the EQL alternative and
§9 records that an earlier revision got this wrong once already; being wrong twice is fine, not
noticing is not.

# Task 2 — Symbolic `goal_value` (only if Task 1 did not invalidate the design)

Plan reference: §3.1, and the first three rows of §7's test table. This is real WP0 work, small and
well-bounded.

Widen the terminal-state prediction goal from a Python float to `sm.ScalarData` so a registered
`FloatVariable` can retarget it at runtime with no recompile:

- `TerminalStatePredictionConstraint.goal_value` (`qp/terminal_state_prediction_strategy.py:176`)
- `ConstraintCollection.add_terminal_state_prediction_constraint` (`qp/constraint_collection.py:328`)
- `TerminalFillConstraintTask.goal_value` (`motion_statechart/tasks/pouring.py:53`)

The mechanism is expected to work already — the bound at
`terminal_state_prediction_strategy.py:275` is `sm.Scalar(constraint.goal_value) - free_response()`
and the QP compiles once with float variables as parameters. The work is in the three places that
break:

- `_fill_goal_reached` (`tasks/pouring.py:241` and `:281`) and the guard at `:468` compare with
  `<=`/`>=`, which on a symbolic value raises `HasFreeVariablesError`. Read live values through
  `float_variable_data.get_value`, or move the comparison into a compiled observation.
- ORM: regenerate and accept that the column silently disappears (decided; §8.0 item 2). Note it in
  the field docstring.
- Add a guard so a symbolic-valued task raises on JSON serialization rather than silently
  deserializing to goal 0 (`krrood/adapters/json_serializer.py:490-558`).

Write the failing tests first: rewriting the goal variable mid-motion retargets the row; the bound
equals what the strategy computes for that goal; `_fill_goal_reached` reads the live value without
raising. Also add the regression that pins the plan's §3.2 design constraint — two terminal-fill
tasks in one chart must raise `MultipleTerminalStateConstraintsError` at compile.

# Task 3 — only if 1 and 2 are both green and time remains

Stop and write up instead of starting anything larger. A clean `spike_findings.md` plus two green
spikes is a complete, useful night's work.
