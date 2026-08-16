# Knowledge-Based Servoing × VJDF — Implementation Plan

Implementation plan for section 5 of `experiment_feasibility_report.md`: re-implementing the
knowledge-based servoing loop of `huerkamp2025.pdf` on the Semantic Digital Twin + EQL +
motion-statechart stack, with the virtual-joint dynamics framework (VJDF — passive
`LiquidConnection` DOF + fill equation + terminal-state prediction row) supplying the numeric
effect model in place of the paper's fixed-gain twist bridge.

Branch `bmp`, 2026-08-13. Revision 3, after a code-grounded review and a first implementation pass;
every claim below was checked against the files it cites. §9 lists what earlier revisions got wrong.

> **Status, 2026-08-16 — Stage A is built and running.** The framework, both instantiations, the
> binding layer, the decision transcript and the demonstration exist on branch
> `knowledge-servoing-plan` and are green. What the plan below still describes as future work is
> Stage B onward. §0 records what was built and, more usefully, the four places where building it
> contradicted the plan.

---

## 0. What Stage A actually built, and where it contradicted this plan

### 0.1 Built

| Piece | Where |
|---|---|
| Framework interfaces (`Situation`, `ControlDecision` → `RegimeDecision`/`ParameterDecision`, `DecisionSet`, `SituationGrounding`, `SymbolicTheory`) | `semantic_digital_twin/.../reasoning/knowledge_servoing/` |
| Theory engine adapter over `GeneralRDR` | same, `general_rdr_theory.py` |
| Statechart binding (`SymbolicTheoryNode`, `ConcludedMonitor`, `DecisionBindingPolicy`, `DecisionSlot`, `DecisionTranscript`) | `giskardpy/.../motion_statechart/knowledge_servoing/` |
| Symbolic `goal_value` (WP0 §3.1) | `qp/terminal_state_prediction_strategy.py`, `qp/constraint_collection.py`, `tasks/pouring.py` |
| Substance-transfer instantiation | `semantic_digital_twin/.../reasoning/substance_transfer/` |
| Contextual-safety instantiation (§2.5.4 demonstration 2) | `semantic_digital_twin/.../reasoning/contextual_safety/` |
| Demonstration, Gantt chart and transcripts | `experiments/.../knowledge_servoing/`, `experiments/scripts/run_knowledge_servoing_transfer.py` |

The demonstration reaches a theory-set fill goal of 0.4 at 0.444 and ends because the theory
concludes the transfer is finished. Its transcript:

```
substance transfer
cycle 1:   +AlignSourceOverReceiver
cycle 11:  +PourIntoReceiver, RetargetFillLevel
cycle 171: +ConcludeTransfer  −AlignSourceOverReceiver, PourIntoReceiver, RetargetFillLevel
contextual safety
cycle 1:   +EnforceCaution
```

### 0.2 Four things building it contradicted

**The near-goal rule does not dissolve (contradicts §1.3).** §1.3 claims the paper's `25a > 18c` —
near-goal ⇒ decrease tilt — dissolves entirely under the terminal-state row's anticipation. Measured:
it dissolves *only while actuation is fast relative to flow*. At the shared fixture's coupling the
source empties in under two seconds, the arm needs ~40 cycles to close the geometric outflow gate,
and the MPC horizon is far shorter than that, so the transfer overshot by ~47 % and no controller
could have done better. Slowing the coupling to a rate the actuation can track brought overshoot to
~3 %. **The honest claim is conditional**: MPC anticipation subsumes the near-goal rule when the
control horizon exceeds the actuation time constant, and the symbolic rule earns its keep when it
does not. That is a more interesting statement than the flat one and it is now backed by a
measurement.

**Concluding a transfer must actively command the source upright.** The outflow gate is geometric,
so a statechart that merely stops driving the fill leaves the source tilted and aimed and the
receiver keeps filling. `ConcludeTransfer` gates a return-to-upright task; without it the run has no
way to stop pouring. This is a real property of the effect model, not an implementation detail, and
any new VJDF domain will have the analogous question.

**The reasoner runs synchronously, so §2.3's threading design is not built.** Inference measured at
tens of microseconds, so grounding, inference and application all run on the control thread and the
cross-thread hand-off §2.3 designs against is unnecessary. The frozen-situation discipline is kept —
it costs nothing and keeps the door open — but the thread-safety analysis in §2.3 now describes a
contingency, not the system. The node infers every fifth control cycle rather than every one.

**Grounding must bind its compiled expressions to the world's state array.** Compiling without
parameters produces a function that evaluates at an all-zero configuration and *fails silently*,
returning plausible-looking wrong facts rather than raising. Both groundings use
`VariableParameters.from_lists(world.state.position_float_variables)` with
`bind_args_to_memory_view`. Any future grounding must do the same; it is the single easiest way to
get a theory that reasons confidently about a world it cannot see.

### 0.3 Two repository defects found while verifying

Neither is caused by this work and both were hiding evidence:

- **`pytest test/giskardpy_test/test_motion_statechart/` collected zero tests and exited zero.** A
  module-scope `importorskip` in `test_pouring_learned.py` aborts collection for everything
  requested alongside it. Fixed with a `collect_ignore` conftest; the directory now collects 273.
- **The pouring suite could not build its fixtures**, so 18 tests errored in setup before asserting
  anything, from three accumulated API drifts (`set_positions_1DOF_connection` removed,
  `create_with_new_body_in_world` signature changed, and a `PrefixedName` nested inside another so
  any error message naming the body raised `TypeError`). Migrated; the suite is green.

---

## 1. The claim the implementation has to support

The paper's loop (`eq:cs1:loop`, p. 890) ends in a lossy step. `INVMOD` produces ten Boolean motion
primitives, and `eq:cs1:twist` (p. 891) collapses them into a fixed-gain bang-bang task-frame twist

```
ξ_d = ( α(x₊−x₋), α(y₊−y₋), α(z₊−z₋), n_t(βt₊−γt₋) + n_z β(r₊−r₋) )
```

with α = 0.02, β = 0.03, γ = 1 held constant for every run. The measured ~6 % final goal error and
the corrective spillage the paper attributes to "fixed velocity gains" (p. 893, and the first
limitation in `ch-casestudy.typ:359`) are properties of *that bridge*. Everything upstream — image
schemata, defeasible rules, the affordance vocabulary — is innocent of the precision problem.

The implementation replaces the bridge, and nothing else, with a reasoner that **selects and
parameterizes motion-statechart tasks** whose effect row is the VJDF terminal-state prediction
constraint (`giskardpy/src/giskardpy/qp/constraint_collection.py:328`).

### 1.1 The experiment is a within-system ablation

Both bridges are implemented and driven by the *same* reasoner, the *same* theory, the *same* scene
and robot:

| Arm | Reasoner | Bridge | Expected |
|---|---|---|---|
| **A (paper replication)** | qualitative theory | `CommandedTaskFrameTwist`, fixed α/β/γ | large error, late reaction, high spill risk |
| **B (this contribution)** | *same* qualitative theory | VJDF task binding | sub-1 % error, anticipatory |
| **C (numeric baseline)** | none — fixed statechart | VJDF task binding | ceiling; isolates the reasoner's cost |

Comparing arm B against the published 6 % alone would confound the bridge with the simulator,
scene, robot, particle model and metric definition. A and B in one harness remove every confound but
the one under test.

Since the MuJoCo bridge is out of scope (§8.0.1), **this within-system ablation carries the claim
by itself** — the published number is context, not a comparison, because it counts particles and
nothing here does. The second-order threat that creates, and the model-mismatch sweep that answers
it, are in §6.2.1.

### 1.2 What the "semantics survives" claim can honestly say

The claim is *not* that the theory is untouched. Partitioning the 68 active rules of the original
theory (`doc/knowledge_servoing/original_defeasible_theory.txt`) by what the architecture change
does to them:

| | by rule count | by decision pattern |
|---|---|---|
| unchanged (state/schema/affordance derivation) | ~41 % | ~26 patterns |
| retargeted (primitive → task-regime decision) | ~18 % | ~12 patterns |
| dissolved (the QP now solves it) | ~35 % | ~7 patterns |
| added (decisions the old interface could not express) | ~6 % | ~6 patterns |

The gap between the two columns is the whole argument. 17 of the ~24 dissolved rules are the
per-direction duplicate families — `23a–23d`, `28b–28e`, `29b–29f`, `30a–30d` — that the theory file
itself apologises for: *"At the moment, there can be only one conclusion so splitting the motion
command in different rules."* They exist to compensate for the bridge. Their **conditions** survive
as one retargeted rule each; only their **direction choice** disappears, into the constraint
gradient.

So the defensible form is: *the derivational semantics survives intact; what disappears is
precisely the part of the theory that existed to compensate for the bridge.* State the rule-count
number too — an examiner will compute it.

### 1.3 One casualty to declare up front — now measured, and conditional

**The paper's flagship superiority pair `25a > 18c` dissolves in arm B only when the control horizon
outlasts the actuation.** Near-goal decrease-tilt overriding slow-flow increase-tilt *is* MPC
anticipation, and the terminal-state row does it continuously ("the optimizer eases off before
overshooting rather than reacting late", `tasks/pouring.py:132-134`).

Stage A measured the boundary (§0.2). When the pour outruns the arm — the source emptying in under
two seconds against ~40 cycles to close the geometric gate — the row's horizon is far too short and
the transfer overshoots by ~47 %. At a coupling the actuation can track, overshoot falls to ~3 % and
the rule is genuinely redundant. So the claim to make is that **the symbolic near-goal rule is a
horizon extension**, subsumed by MPC exactly when the horizon is long enough and load-bearing when
it is not. Arm A keeps the pair verbatim regardless; a mismatch or flow-rate sweep (§6.2.1) turns
this from an anecdote into the curve where the two layers trade off.

Arm B's defeasible interactions are different and real, and the chapter must substitute them
explicitly rather than claim the old pair survived:
- `SpillRisk` refining `PourIntoReceiver` into `RestrictSpill` — a weight/clearance override, not a stop;
- `goalReached` / `overflows` / receiver-not-upright defeating `canPourTo` — a regime exit;
- transfer-stalled ⇒ `HoldTransfer` — see §4.4, a failure mode the fixed-gain bridge could not have.

### 1.4 Pluggability is the paper's contribution, so the framework must be domain-agnostic

The paper's first stated contribution is "a modular control framework that integrates symbolic
reasoning and control, enabling the execution of tasks through **pluggable symbolic theories**"
(p. 887), and its §2.1 lists the requirements a theory must meet to be plugged in. Pouring is the
*demonstration*, not the framework. A re-implementation that hard-wires pouring into the servoing
loop reproduces the paper's example while discarding its claim.

So the deliverable is two things with a hard boundary between them:

- a **framework** — grounding, theory, decision, binding, expectation interfaces — that contains
  no pouring vocabulary whatsoever, and
- the **substance-transfer instantiation**, which is the only theory the thesis carries to
  experimental depth.

The boundary is enforced by a test, not by discipline (§7). This costs ~2–3 days over a
pouring-only design, almost all of it interface work plus one throwaway mimic theory in the test
dataset — and it is what lets the chapter state the pluggability claim as a property of the code
rather than an aspiration.

Two claims that the paper somewhat conflates, and that this plan keeps apart:

| Claim | What demonstrates it | Cost |
|---|---|---|
| **Rule-set extension** — same theory type, more rules, new task variation | draining and scraping (paper §6.3/6.4), Stage C | weeks |
| **Theory pluggability** — different theory, different situation type, different decision vocabulary, same framework | a second theory over non-pouring tasks (§2.5.4) | ~3 days |

The second is the stronger claim and the cheaper demonstration. The paper only ever showed the
first.

**Correction, 2026-08-16.** As built, "theory pluggability" is narrower than stated above: the
statechart's *action vocabulary* is fixed by whoever assembles it, and the binding policy enforces
the fit — `validate()` raises when a theory declares a decision no pre-declared node enacts. The
contextual-safety theory reads as a clean demonstration only because its remedy, a velocity limit,
was put in the chart by the assembler who already knew what it would conclude. What is genuinely
pluggable is the **decision layer**: arbitrary facts, arbitrary rule structure, arbitrary decision
vocabulary, choosing among constraints that already exist. §1.5 states the resolution.

---

### 1.5 Where this sits in the thesis arc

Everything above serves one claim, and it is worth stating before the work packages, because it
decides which of them matter.

**The contribution is the target language for synthesis** — an interface expressive enough to
specify both what a task means and how its effect must evolve, and constrained enough that what
comes out of it can be checked before it runs. Not the reasoner, not the effect model, not the
language model that will eventually write into it.

#### 1.5.1 The two halves are failures of interface width in opposite directions

| | narrow domain | general domain |
|---|---|---|
| **narrow interface** | — | the paper: any theory reducible to a 6-D end-effector twist. Adapts to context, cannot say *how much* |
| **rich interface** | the VJDF: says exactly how a quantity evolves, but only where someone wrote the effect model | **this thesis: theories declare their own regime *and* effect constraints** |

The paper's expressiveness is bounded by the twist; the VJDF's generality is bounded by hand-written
effect models. Letting a theory declare the constraints it needs moves expressiveness into the
constraint vocabulary, which extends without touching either the reasoner or the controller. That is
also the resolution of §1.4's correction: the chart is assembled *from* the theories rather than the
theories having to fit a chart, so `SymbolicTheory` gains a way to state the constraints it requires
alongside the decisions it concludes.

#### 1.5.2 Effect-model structure is hand-built; its parameters are open

The honest boundary is not "effect models are hand-built" but "their *structure* is". Every effect
model in this tree is already a parameterized family —
`ArticulatedPouringEquation(container_height, container_width, outflow_rate_constant,
discharge_coefficient)` — and those parameters have three distinct consumers:

- **tuned**, by hand or by an experiment sweep (§6.2.1's mismatch sweep is exactly this);
- **learned**, by distillation — the head surrogate path exists, and `test_pouring_learned` shows
  the controller is agnostic to whether the head model is analytic or learned;
- **reasoned about**, by a theory concluding a parameter the way it already concludes a fill goal.

Only the third is missing: `goal_value` is a live float variable, but equation parameters are baked
in when the coupling is built. Making them symbolic is the WP0 §3.1 pattern applied to the equations
rather than the tasks.

#### 1.5.2.1 The demonstration: a theory that bounds the head

The crispest single demonstration of the combination is a theory concluding **a bound on the
allowed head above the lip** — the hydrostatic head that drives the pour
(`PouringEquation.head_above_lip`, `pouring_equations.py:293`, which feeds both the Torricelli exit
speed and the outflow rate).

Three properties make it the right example, and none of them holds for a parameter-identification
example such as inferring a substance's viscosity:

1. **It is a control decision, not a measurement.** Viscosity is a property of the world that
   perception or annotation supplies; bounding the head is something the reasoner *decides*.
2. **It is stated at the abstraction level of the effect, not the robot.** The decision names no
   end effector, no direction, no velocity, not even a tilt. The optimizer finds whatever arm motion
   keeps the head within bound — which, since head is a function of fill *and* tilt, yields a
   fill-dependent tilt limit for free: the container is tilted less when fuller, without anyone
   writing that rule.
3. **It is therefore embodiment-independent.** Compare the same semantic decision — "be careful" —
   realized at three levels: joint velocity limits (robot-specific), a tool-speed cap (what the
   safety theory does today, still end-effector-specific and not actually a bound on the pour), and
   a head bound (physically meaningful and portable across embodiments). Only the last one says
   something true about the task rather than about this robot.

It is also where the declared-constraint idea (§1.5.1) and the effect model meet: the theory
declares an *inequality over an effect-model quantity*, using both channels for one decision — the
regime channel gates the constraint, the parameter channel supplies the bound.

And it is the necessity argument in one figure (§1.5.3): a theory alone has no quantity to bound,
an effect model alone has no reason to bound it. The transfer theory's terminal-state row then plans
against the restricted dynamics and still reaches its goal, only more slowly — so the two theories
interact **through the physics rather than through the robot**.

Implementation is small: a task constraining `head_above_lip` from above, with the bound as a
`ScalarData` so channel 2 can write it, plus a safety rule concluding it. The parameter surface also
gives the expectation layer's model-audit job (§4.5) something to *do*: a measured flow diverging
from the predicted one is a parameter to correct, not only a transfer to abandon.

#### 1.5.3 The need: tasks neither half can execute

The running example has to require both from the first chapter, or the combination reads as
engineering. It requires both when a quantity must be hit precisely *and* which constraints hold
depends on context: *"transfer 50 ml of reagent B into the flask; it is corrosive, so keep clear of
the balance and pour slowly."* Theories alone give ~6 % on the quantity; effect models alone need
someone to have foreseen the corrosive case when the goal was authored.

Drawing the example set from real laboratory protocols rather than authoring it makes the input
distribution independent of the person being evaluated, and the lab assets already in this tree make
that cheap. Note the current two-theory demonstration is *nearly* this but not yet: its safety
theory changes the speed, not the quantity. Making it change the quantity — a smaller aliquot or a
tighter tolerance when the substance is hazardous — turns the necessity argument into one figure.

#### 1.5.4 "Correct by construction" is two claims, and only one is by construction

- **Well-formedness by construction.** Already true and already enforced: an unbound decision type,
  a decision bound to the wrong channel, or a parameter target that was never registered all raise
  at build time. A synthesized theory cannot produce an ill-formed controller. This is a property of
  the interface and it is what makes synthesis into it safe.
- **Semantic correctness by verification.** Not by construction — a language model can emit a
  well-formed theory that is wrong. This is where the companion report's §4 pays off: `Causes`,
  `SatisfiesRequest`, the typed-verdict hierarchy and the repair loop are already implemented. **The
  final chapter is the marriage of report §4 and §5**, not a new component: synthesize into this
  interface, check with those predicates, repair on the typed verdict. §4's designed
  typed-versus-binary feedback ablation becomes the evaluation of the synthesis loop.

#### 1.5.5 Boundaries to state, in the order an examiner will find them

1. **Pre-declaration.** Constraints must exist before the motion compiles, so this is
   plug-in-at-assembly, not mid-motion (§8.1). Accepted.
2. **Effect-model structure.** Synthesis selects and parameterizes a model from the library; it does
   not write a new one. Say which is claimed (§1.5.2).
3. **The constraint vocabulary is a library with finite coverage.** A theory may declare constraints
   of forms someone implemented, so "any task" means "any task expressible in the available
   vocabulary". That makes coverage an *empirical* claim — of N protocol sentences, how many are
   expressible and what do the failures share? — which is a better result than an unbounded
   assertion.

#### 1.5.6 What the arc still needs evidentially

In order: the **necessity figure** (§1.5.3, a task needing both halves); the **precision result**
(arms A vs B, §6.2, in progress); **synthesis tractability** into this interface; and the
**coverage number** from §1.5.5.

---

## 2. Architecture

### 2.1 Layer map

| Paper (`eq:cs1:loop`) | This implementation | Thread / rate |
|---|---|---|
| `X[k+1] = ENV(X[k], U[k])` | `World.apply_control_commands` → `step_physics` (`world.py:2540-2572`) | control, 50 Hz |
| `Y[k+1] = SEMINT(X[k+1], Q[k])` | `TransferSituationGrounding` — compiled expressions over world state | **control thread**, 10 Hz |
| `S[k] = SCHMOD(Y[k])` | affordance facts folded into grounding (§3.2) | control thread |
| `U[k] = INVMOD(S[k], G)` | `SubstanceTransferTheory.infer` → `TransferDecisionSet` | **reasoner thread** |
| `Q[k] = FWDMOD(S[k], U[k], G)` | `ExpectationLayer` — persistence check + model audit (§4.5) | reasoner thread |
| primitives → twist → QP | **`ReasonerBinding` → task gating + float variables → QP** | control, 50 Hz |

### 2.2 The two write channels

Exactly two write channels exist from the symbolic layer into the running controller.

**Channel 1 — activation.** A decision gates a pre-declared task's life cycle.
`link_to_motion_statechart_node` multiplies every constraint's weight by a RUNNING indicator
(`qp/constraint_collection.py:134-146`) and `QPData.apply_filters` (`qp/qp_data.py:136-177`) drops
zero-weight rows each solve, so a NOT_STARTED task contributes zero rows and activates with no
recompile. **Verified better than assumed:** `Executor.tick` runs `motion_statechart.tick`
(observation, then life cycle) *before* `compute_command` (`executor.py:196-209`), so a monitor
flipping TRUE activates the gated task's rows in the **same** control cycle, not the next.

**Channel 2 — parameterization.** Decisions carry numeric arguments written into registered
`FloatVariable`s. The QP compiles once with float variables as parameters
(`qp/qp_controller.py:55-69, 110-124`) and `Executor.tick` passes `float_variable_data.data` into
`compute_command` every cycle (`executor.py:205-209`). This is what replaces α, β, γ: the reasoner
sets *what the constraint means*, the QP decides *how fast to move*.

**No third channel.** The reasoner never modifies the world model during a motion (§7.1).

### 2.3 Data flow, and why grounding runs on the control thread

> **Superseded by measurement, 2026-08-16.** Inference costs tens of microseconds, so the built
> system runs grounding, inference and application synchronously on the control thread, every fifth
> cycle. There is no reasoner thread and therefore no cross-thread read at all. What follows remains
> the design of record for the day a theory is heavy enough to need its own thread — and the
> hazards it catalogues are real, so anything that moves inference off-thread must satisfy it. The
> frozen-situation discipline is kept in the built system precisely to keep that door open.

The obvious design — reasoner thread reads the twin — is **unsafe**, and this is the largest
correction the review produced. The reasoner thread's *reads* race the 50 Hz control loop in at
least four places:

- `ForwardKinematicsManager.compute_np` is `@memoize`d into an unlocked dict
  (`forward_kinematics.py:164-165`) and returns *views* into a preallocated buffer that
  `recompute()` overwrites in place on every state change (`forward_kinematics.py:108-115, 184-200`;
  `CompiledFunction.evaluate` returns `self._out`, `symbolic_math.py:332-338`). Torn 4×4 transforms
  are possible every cycle.
- Instantiating any spatial-relation predicate mutates the global `SymbolGraph` singleton
  (`krrood/src/krrood/symbol_graph/symbol_graph.py:72-78, 210, 292-301`), unlocked.
- `contact` drives the shared bullet detector whose query construction is `@memoize`d
  (`pybullet_collision_detector.py:413-424`) and whose transforms are synced from the state-writing
  thread (`collision_detector.py:176-194`).
- `FloatVariableData`'s own docstring says it is not thread-safe.

**The design inverts the flow instead of adding locks.** Grounding is cheap (compiled CasADi
evaluations over the state vector) and inference is expensive (EQL tree traversal), so:

```
control thread, every N ticks:   ground -> immutable TransferSituation  ──> reasoner queue
reasoner thread:                 EQL inference over TransferSituation   ──> decision slot (lock-guarded)
control thread, every tick:      read decision slot -> write float variables, drive observations
```

Only frozen dataclasses cross the thread boundary. Nothing that touches FK, the symbol graph, the
collision detector or the world state ever runs off the control thread. This also forces
`TransferSituation` into existence, which §4.2 shows is required by the EQL API anyway — the safety
fix and the API constraint converge on the same object.

**Blacklist, to be enforced by a test:** no predicate that mutates world state may appear in
grounding. `Causes._map_motion_to_effect` replays a whole trajectory into the live world
(`reasoning/bmp_predicates.py:118-141`), and `reset_state_context` writes `state._data` twice
(`world.py:135-158`). These are for the verification experiment (report §4), never for servoing.

### 2.4 Node topology

```
QualitativeTheoryNode          (top level, start_condition constant TRUE, owns the reasoner thread)
Parallel([
    FillByTransferTask         gated by  ConcludedMonitor(PourIntoReceiver)
    KeepProjectileInReceiver   gated by  ConcludedMonitor(AlignSourceOverReceiver)
    KeepSourceRimAboveReceiverRim, AlignPlanes
    ConcludedMonitor(...)      ← must be siblings of what they gate
])
```

Two constraints on the topology, both verified:

- `MotionStatechart._validate_condition_scope` (`motion_statechart.py:631-649`) lets a condition
  reference only the owner or a same-parent sibling. A decision consumed at two depths (gating a
  task *inside* the `Parallel` and pausing the `Parallel` itself) needs **two** monitor nodes. The
  flat one-node-per-decision-type sketch is wrong as drawn.
- `on_tick` fires only while a node is RUNNING. `SymbolicTheoryNode` must be top level with a
  constant-true start condition and must never sit inside the `Parallel` it feeds, or its parameter
  writes freeze exactly when the regime is paused.

### 2.5 Framework and instantiation

#### 2.5.1 The five interfaces

Everything the framework needs is generic over two type parameters: the situation type a theory
grounds, and the decision types it concludes. Nothing else about a task is framework business.

```python
SituationType = TypeVar("SituationType", bound=Situation)


@dataclass(frozen=True)
class Situation(ABC):
    """One immutable snapshot of the facts a theory reasons over, for one subject of reasoning."""


@dataclass(frozen=True)
class ControlDecision(ABC):
    """A conclusion a theory reached, addressed to exactly one of the two write channels."""


@dataclass(frozen=True)
class RegimeDecision(ControlDecision, ABC):
    """A decision that activates, pauses or ends constraints (channel 1)."""


@dataclass(frozen=True)
class ParameterDecision(ControlDecision, ABC):
    """A decision that supplies numeric values to registered float variables (channel 2)."""


@dataclass
class SituationGrounding(Generic[SituationType], ABC):
    """Produces a theory's situations from the world; runs on the control thread."""

    @abstractmethod
    def ground(self, world: World) -> Sequence[SituationType]: ...


@dataclass
class SymbolicTheory(Generic[SituationType], ABC):
    """A pluggable symbolic theory: situations in, decisions out. Runs off the control thread."""

    @abstractmethod
    def infer(self, situations: Sequence[SituationType]) -> DecisionSet: ...
```

The decision hierarchy splits by **write channel, not by domain** — that is the axis the interface
determines, and it keeps `RegimeDecision`/`ParameterDecision` meaningful for a theory about
cutting, wiping or door opening without a single edit.

#### 2.5.2 The binding policy is the pluggable part of the controller

The paper's §2.1 requires that "the controller has to be able to execute each primitive alone or as
a composition of multiple ones" and to "smoothly switch from one primitive to another". Here that
becomes a declarative registry, built once at `build()` time:

```python
@dataclass
class DecisionBindingPolicy:
    """Maps a theory's decision types onto statechart activations and float-variable writes."""

    def activate(self, decision_type: type[RegimeDecision], node: MotionStatechartNode) -> None: ...

    def parameterize(
        self,
        decision_type: type[ParameterDecision],
        read_value: Callable[[ParameterDecision], float],
        target: SymbolicMathType,
    ) -> None: ...
```

Made hard to misuse (`AGENTS.md`), which mostly means raising at build rather than misbehaving at
run time: a decision type the theory can conclude but the policy does not bind raises
`UnboundDecisionTypeError`; a type bound to both channels raises; a `ParameterDecision` bound to a
float variable that was never registered raises. All three are build-time checks over the theory's
declared decision types — which is why `SymbolicTheory` must declare them, not just conclude them.

#### 2.5.3 Package layout, and where the boundary runs

| Package | Contents | May import |
|---|---|---|
| `semantic_digital_twin/.../reasoning/knowledge_servoing/` | `Situation`, `ControlDecision`, `SituationGrounding`, `SymbolicTheory`, `DecisionSet`, `ExpectationLayer` | SDT + krrood only |
| `giskardpy/.../motion_statechart/knowledge_servoing/` | `SymbolicTheoryNode`, `ConcludedMonitor`, `DecisionBindingPolicy`, `DecisionTranscript` | the above + giskardpy |
| `semantic_digital_twin/.../reasoning/substance_transfer/` | `TransferSituation`, `TransferDecision` and subclasses, `TransferSituationGrounding`, `SubstanceTransferTheory` | everything |

Neither framework package may import `HasFillLevel`, `LiquidSource`, `LiquidConnection`,
`tasks/pouring.py` or anything under `substance_transfer/`. That is checkable and therefore tested
(§7) — the same discipline `AGENTS.md` already imposes on `krrood`'s self-containment.

`SymbolicTheoryNode` becomes fully generic: it owns *a* grounding, *a* theory and *a* binding
policy, runs the thread, and applies whatever comes back. Every pouring-specific line moves into the
instantiation. `ConcludedMonitor` is already generic — it is parameterized by decision type.

#### 2.5.4 Proving pluggability without spending a chapter

Two demonstrations, in increasing cost:

1. **A mimic theory in the test dataset** (~half a day). A trivial non-pouring theory over a
   two-field situation, driving a `JointPositionList` task, exercising both channels. It proves the
   framework runs a theory it has never heard of, and it is the regression that stops pouring
   vocabulary from leaking back into the framework. This is the repo's established mimic-class
   pattern.
2. **A second theory alongside the transfer theory** (~3 days, optional but recommended). The
   working candidate is a *contextual safety* theory — the name is this plan's, not the paper's.

   Its defining property is what makes it a pluggability test rather than another pouring variant:
   its facts come from **semantic annotations and spatial relations, not from physics**, it has
   **no effect model of its own**, and it drives tasks that already exist. Shape:

   - *Situation*: the manipulated object, what it contains, what it is currently over, who is
     nearby — grounded from SDT annotations plus `Above`/`InsideOf`/`is_supported_by`
     (`reasoning/predicates.py:433-558, 252`), no fill equation involved.
   - *Rules*: object annotated as hazardous or hard to clean up ⇒ stricter no-spill regime;
     carried container currently over an electronics-annotated surface ⇒ keep-out regime plus a
     velocity cap; human hand in the workspace ⇒ `HoldTransfer`. Each is a two-to-four-condition
     rule over facts a populated apartment twin already carries.
   - *Decisions*: regime decisions gating collision-avoidance goals, a `CartesianVelocityLimit`
     (`tasks/cartesian_tasks.py:802`), and the keep-out task sketched in report §1; parameter
     decisions scaling their weights.

   It validates three things at once: two theories coexisting in one statechart, a theory with no
   VJDF effect domain driving purely geometric tasks, and decision-type disjointness enforced by the
   binding policy (§2.5.2). Rhetorically it is the endogenous-adaptivity claim in its cleanest
   form, because the trigger is a fact in the knowledge base rather than a fill level or a person
   publishing a topic — and under §8.0 it is the best available source of a mid-motion regime flip.

   It also connects to work already staged in this repo: `semantic_safety_filter_concept.md`
   specifies a CBF/superquadric keep-out layer on exactly these hooks (unimplemented), and
   `brunke2025.pdf` is its source. That concept synthesizes semantic constraints with an LLM; doing
   it with a defeasible theory instead is the same target from the interpretable direction, which is
   a positioning the thesis can use. **The subject matter is a free choice** — any theory meeting
   the defining property above works equally well as the pluggability demonstration; safety is
   proposed because its facts are already in the twin and its tasks already exist.

   **Build it in two phases, and build phase 1 as production code rather than a demo.** Once a fact
   source is wired (§2.6) this theory becomes a second *evaluation* domain, not just a pluggability
   proof — so its situation type, rules and decisions belong in
   `semantic_digital_twin/.../reasoning/contextual_safety/` from the start, with three or four real
   rules, not in the test dataset. The throwaway mimic theory (demonstration 1) already covers the
   regression job; this one should be written to grow.

   *Phase 1, now (~3 days):* annotation-driven facts from the populated twin, regime flips
   demonstrated end to end, no perception.
   *Phase 2, once a fact source exists (future work):* the same theory, unchanged, on perceived
   facts.

   The reason phase 2 is worth more than a second pouring evaluation: **safety facts have no
   analytic model, so perception is the only way to obtain them.** Fill level can be integrated from
   an ODE; "there is a hand in the workspace" or "that surface holds electronics" cannot. This
   theory is therefore the one part of the work that genuinely *requires* a fact source rather than
   merely benefiting from one, which makes it the natural evaluation ground the day the wiring lands
   — and the only route by which this line of work repairs the perception delimitation (§8.4).

#### 2.5.5 The limit of the pluggability claim — state it explicitly

Theory pluggability is cheap because the interface is narrow. **Effect-model pluggability is not**,
and the two must not be conflated in the writing. A new theory can plug in today and drive existing
tasks — Cartesian, alignment, collision, joint — with no library work at all. But a new theory that
wants *VJDF precision* needs a passive-DOF effect model in its own domain, and report §2 measured
what that costs: wiping is ~1,000–1,400 source lines and chapter-sized, cutting is out of reach.
`DifferentialEquation` is an 11-line empty ABC; there is no generic `VirtualEffectConnection` with
attached ODEs.

So the honest formulation, and the one to defend: **the reasoning-to-control interface is
domain-agnostic and demonstrated so; the effect-model layer beneath it is domain-specific and
pouring is the only domain instantiated.** A theory plugged in without its own effect model gets
symbolic control over existing tasks — the paper's capability. Getting the precision claim too
requires the effect model, and that is per-domain work. Note also that a second theory wanting its
own terminal-state row in the same chart collides with §3.2's single-row limit.

### 2.6 Fact sources are state writers, not grounding variants

In the original paper MuJoCo is not the plant under study — it stands in for **the real robot and
its perception**, supplying the facts (particle counts as fill level, spilling) that a real system
would perceive. That makes the fact source a swappable component, and it is worth being precise
about *which* seam it swaps, because the obvious answer is wrong.

The obvious answer is "write a second `SituationGrounding`". That is wrong, because grounding is not
the only path physical facts take into the loop. The QP's own task expressions read world state
directly — `_geometric_transfer_gate`, `_fill_goal_reached`, the terminal-state row's state variable
— so a grounding that read perceived values while the tasks read modelled ones would run the
reasoner and the controller on two disagreeing fact bases.

**The rule, therefore: a fact source writes the twin; the twin is the single source of truth; both
the reasoner's grounding and the QP's task expressions read the twin.** A MuJoCo particle observer
or a RoboKudo perception pipeline becomes a state writer on the `HasUpdateState` path
(`world_description/connections.py:50`, stepped by `World.step_physics`, `world.py:2561`), exactly
where `LiquidConnection` already lives. Nothing in the theory, the decisions, the binding policy or
the tasks changes when it is added.

Two consequences worth stating now, because they are cheap to honour and expensive to retrofit:

1. **Everything this plan builds is fact-source agnostic already**, and that is the reason the
   analytic-only decision (§8.0) costs so little. Validating theory, binding and wiring against
   analytic facts validates them against perceived facts too, up to the quality of the facts.
2. **One piece of wiring is not free**, and it should be named rather than discovered later:
   `LiquidConnection` *self-integrates* its fill DOF in `step_physics`. A perceived or
   particle-counted fill is a competing estimate of the same quantity. Wiring a real fact source
   means deciding whether it replaces the integration (measurement overwrites the DOF), corrects it
   (observer/filter), or runs alongside it as a second fact the theory can compare — the third
   option being the one that makes §4.5's model-audit job real. This is a design question for
   whenever that wiring happens; it needs no answer now.

---

## 3. WP0 — Library changes (~4–6 days, was "1–2 days")

Each item gets a failing test first.

### 3.1 Symbolic `goal_value` (channel 2's prerequisite)

The mechanism is real: the bound is `sm.Scalar(constraint.goal_value) - free_response()`
(`terminal_state_prediction_strategy.py:275`), `sm.Scalar` accepts symbolic input
(`symbolic_math.py:844`), `capped_bound` uses `sm.limit` (`enforcement_strategy.py:299-311`), and
the QP compiles once with float variables as parameters. But it is **not** a few lines:

- **Tick-time Python comparisons crash.** `_fill_goal_reached` (`tasks/pouring.py:241, 281`) and the
  guard at `:468` compare with `<=` / `>=`, which triggers `Scalar.__bool__` →
  `HasFreeVariablesError` (`symbolic_math.py:877-900`). All three sites must read live values via
  `float_variable_data.get_value` or move into a compiled observation expression.
- **ORM silently drops the field.** `TerminalFillConstraintTaskDAO.goal_value: Mapped[float]`
  (`giskardpy/src/giskardpy/orm/ormatic_interface.py:10369`), `KeepSourceRimAboveReceiverRimDAO`
  clearance columns (`:10336-10338`). Ormatic does not crash on `ScalarData` — it emits no column
  (precedent: `GiskardConstraintDAO` at `:10587-10613` has no columns for any `ScalarData` field).
  **Decided 2026-08-14: accept the loss.** Run `scripts/regenerate_all_orm.py`, persist no numeric
  default, and say so in the field's docstring so a future reader is not surprised by a task that
  round-trips without its goal.
- **The remote JSON path corrupts symbolic fields.** A `Scalar` serializes to a bare type marker and
  deserializes as `Scalar()` == 0 (`krrood/adapters/json_serializer.py:490-558`,
  `attribute_introspector.py:82-86`). WP2's nodes are local-only anyway, but these *tasks* stay
  remotely usable, and a symbolic-goal task shipped through `GiskardWrapper.execute` would pour to
  goal 0. Add a build-time guard that raises when a symbolic-valued task is serialized.

### 3.2 The single-terminal-row limit — a design constraint, not a bug to fix

`TerminalStatePredictionStrategy._constraint` raises `MultipleTerminalStateConstraintsError` when
`len(self.constraints) != 1` (`terminal_state_prediction_strategy.py:210-215`). Equality constraints
are grouped by strategy *type* across all nodes (`constraint_collection.py:49-59`) at **compile
time**, before any life-cycle gating applies. So two terminal-fill tasks cannot coexist in one
chart even if only one is ever RUNNING.

**Decision: do not generalize the strategy. Re-parameterize the single row instead** — which is
exactly what §3.1 buys. One terminal-fill task per chart, retargeted at runtime through its goal
variable, covers every case in this plan: pouring drives the receiver's fill, draining drives the
source's fill, and they are separate motions. What this *does* kill is the Stage C wording
"draining absorbed with every other component unchanged **in one chart**" — draining is a
rule-set extension plus its own chart. Say that; it is still the modularity claim.

If a later experiment genuinely needs simultaneous terminal rows, generalizing `_constraint` and
`_state_model` to per-constraint lists is a ~150–250-line WP of its own. Out of scope here.

### 3.3 Remaining WP0 items

| Change | File | Note |
|---|---|---|
| `minimum_clearance` / `clearance_band` accept `sm.ScalarData` | `tasks/pouring.py:420, 429` | same crash/ORM caveats as §3.1 |
| Per-task `FloatVariable` for `weight` | `graph_node.py:961` and each task | `quadratic_weight` is already `sm.ScalarData` (`qp/constraint.py:37`), but every task takes `weight: float` and bakes it in at build. Small, but per-task, not zero |
| Expose `_rim_exit_point` publicly | `mixins.py:1608` | it *is* `hasLowestOpeningCorner`, computed continuously from the tilt direction — a rename, not new code |
| `near` predicate | `reasoning/predicates.py` | **does not exist** (only `compute_euclidean_planar_distance`, `:216`). Rule 15a's own threshold (distance ≤ source's larger dimension) makes it a ~20-line predicate |
| Cache compiled predicate expressions | `reasoning/predicates.py:405-429` | every directional predicate builds a fresh `CompiledFunction` per call (`symbolic_math.py:622-640`). ~0.1–1 ms each; fine at 10 Hz but wasteful, and `InsideOf` calls the uncached `combined_mesh` three times (`shape_collection.py:133-145`), which is ms–tens of ms |
| `BoolTopicMonitor` | `ros2_nodes/topic_monitor.py` | shared with report §1; `WaitForMessage` latches TRUE forever so it cannot release a pause. ~25 lines |

`normalization_factor` (`qp/constraint.py:47`) stays a float: it participates in row normalization,
so changing it mid-motion changes the meaning of every bound.

---

## 4. WP1 — Theory layer (~2 weeks, was "3 days")

### 4.1 EQL is a single-pass deducer, not a forward chainer

This is the finding that most changes the plan. The original theory needs conditions over *derived*
facts, negation of derived facts, and multi-step chaining (`canPourTo` derived, then consumed).
None of that exists:

- Conclusions never become queryable facts. `Add._evaluate__` writes into the current solution row's
  bindings (`rules/conclusion.py:96-99`); conclusions fire at the active conditions root only after
  all conditions evaluate (`core/base_expressions.py:473-514`). There is no working memory, no fact
  assertion and no fixpoint loop anywhere in `krrood` — grep for fixpoint / forward-chain / assert
  returns nothing.
- `not_` (`factories.py:249`, `operators/core_logical_operators.py:46-66`) inverts truth per binding
  row. Usable over grounded facts; never over a rule conclusion.
- `near(x,y) ⇒ near(y,x)` and any transitive closure are inexpressible as rules.
- `Alternative` is else-if: two matching rules with no refinement between them means the
  first-declared silently wins (`conclusion_selector.py:242` → `core_logical_operators.py:125-149`)
  — no error. `Next` fires all branches but as separate solution rows, not one decision set. Two
  `Add`s to the same variable are last-write-wins in set iteration order.
- `SufficientConditionSet.evaluate_against` takes **one** shared variable and one case
  (`rdr/backward_inference.py:61-76`) — built for single-case-variable RDR, not for rules ranging
  over pairs `(?s, ?d)`.

All of the above is true of **raw EQL rule trees**. It is *not* true of the classic RDR engine in
the same library, which changes the recommendation — see §4.1.1.

### 4.1.1 MCRDR is the better engine for a defeasible theory

`krrood.ripple_down_rules.MultiClassRDR` (`rdr.py:1448`) supplies, natively, three things the
plan was working around:

**Multiple simultaneous conclusions.** `_classify` (`rdr.py:1491-1533`) walks the whole top-rule
chain and accumulates `self.conclusions` across every rule that fires. `PourIntoReceiver` *and*
`RestrictSpill` in one pass, no workaround. This kills the "one rule tree per decision family"
constraint of §8.3.

**Stop rules are literally defeaters.** `MultiClassStopRule` (`rules.py:543`) concludes a `Stop`
category "meant to stop the parent conclusion from being made", and its `evaluate_next_rule` sets
`self.top_rule.fired = False` (`:558-566`). A rule whose only effect is to block another rule's
conclusion *is* Antoniou's defeater. So `−canPourTo` (theory rules 10, 20a, 20b) is a first-class
stop rule attached to the `canPourTo` top rule — not the "fold the defeaters into procedural
grounding" workaround §8.3 proposed. `MultiClassFilterRule` (`rules.py:576`) additionally refines a
parent conclusion instead of killing it, which is exactly `SpillRisk` turning `PourIntoReceiver`
into `RestrictSpill`.

**Intra-pass chaining.** After each firing rule, `_classify` calls
`update_case_with_conclusion_output(case_cp, ...)` and the loop evaluates the next rule against that
same `case_cp`. Later top rules therefore *see earlier conclusions*. This is genuine forward
chaining within one classification pass, ordered by the top-rule chain. §4.1's "no chaining"
finding stands for EQL and falls for MCRDR — and the stratification WP1 needs becomes **top-rule
ordering**, which the engine already gives, rather than hand-built layers with materialization
between them. That is the single largest simplification available to WP1.

MCRDR also wants exactly the case shape §4.2 already arrived at: one flat object per subject of
reasoning. `TransferSituation` *is* an RDR `Case`. The EQL advantage that motivated the rule-tree
route — variables and joins over the twin — has already been spent in grounding by the time
inference runs.

#### What it costs

- **Backward inference is EQL-side.** `BackwardInferenceIndex` (`rdr/backward_inference.py:194`)
  operates on EQL `SymbolicExpression` trees, and the expectation layer's job 1 (§4.5 — persistence
  checks and retraction explanations) depends on it. A classic-RDR rule tree is walkable
  (`Rule.parent`, `.refinement`, `.alternative`, `.conditions`, `rules.py:99-135`), so an equivalent
  index is ~100–150 lines of new code. Budget it, or give up retraction explanations.
- **Verbalization is EQL-side** too. Classic RDR conditions are `CallableExpression`s; they generate
  readable source (`write_condition_as_source_code`, `rules.py:307`) but do not self-verbalize into
  the natural-language form the traceability claim leans on.
- **Case mutability.** `_classify` copies the case (`copy_case`) and writes conclusions into the
  copy. Verify this works with a frozen `TransferSituation` — ground frozen, let the engine copy —
  before committing; it is the one place the thread-safety design (§2.3) and the engine could
  collide.
- **Bypass `CaseReasoner`/`WorldReasoner`.** Call `rdr.classify(situation)` directly. The wrapper
  enters `modify_world` (`world_reasoner.py:47`), which is the §8.1 blocker; the engine itself does
  not touch the world.

#### The methodological argument this unlocks

RDR's payoff is not expressiveness, it is **knowledge acquisition and maintenance**, and the paper's
own §6.3 narrative is an RDR story told without the name: *"Once such a limitation is detected, the
design of our framework allows us to extend the symbolic theory… we added a rule that negates that
pouring is possible when a small amount of particles is left in the pot. By assigning it a higher
priority than the rule that makes pouring possible, we can successfully handle the discovered
limitation."* That is precisely adding an MCRDR stop rule as a refinement after observing a
misclassification.

Expressing the theory as an MCRDR converts that anecdote into a mechanism with two properties the
hand-authored theory cannot claim: the priority structure is **derived from the order of
correction** rather than hand-specified (the paper's `25a > 18c` and `28f > 23g` become tree
positions, not asserted pairs), and each correction is checked against stored cornerstone cases, so
a fix cannot silently break a case the theory previously got right. `fit_case` /
`stop_wrong_conclusion_else_add_it` (`rdr.py:1535+`) with an `Expert` is the acquisition loop, and
the world's own annotation reasoner already runs this way
(`reasoning/reasoner.py:65` uses `GeneralRDR`; `world_rdr/world_semantic_annotations_mcrdr.py` is
its generated output).

`GeneralRDR` composes per-attribute RDRs, which is the natural engine-level counterpart of §2.5's
multiple coexisting theories.

#### Decision, 2026-08-14

**The substance-transfer theory is authored as a `MultiClassRDR` over `TransferSituation` cases.**
EQL keeps the grounding-side queries over the twin. Retraction explanations are kept, so the
backward index is re-implemented over the RDR tree (~100–150 lines) rather than reusing
`BackwardInferenceIndex`. WP1 is correspondingly smaller than the stratified-pipeline design it
replaces, because top-rule ordering and stop rules do work the pipeline was doing by hand.

Two things to settle in the first day of WP1, both cheap to test and both able to invalidate the
above if they go the wrong way:

1. `copy_case` against a frozen `TransferSituation` — `_classify` copies the case and writes
   conclusions into the copy (`rdr.py:1512-1533`). If frozen dataclasses do not survive that, either
   the situation becomes a mutable working copy created inside `ground()` (still never shared across
   threads, so §2.3 holds) or a thin mutable `Case` adapter wraps it.
2. Whether the generated-code path (`RDRWithCodeWriter`) or the in-memory tree is the one to run at
   control rate. The world's annotation RDR ships as generated source
   (`world_rdr/world_semantic_annotations_mcrdr.py`), which is likely faster and is the pattern to
   follow if latency (§8.2) is tight.

### 4.2 `TransferSituation` — the case object

Forced by both §2.3 (thread safety) and §4.1 (`evaluate_against`'s single case variable). One
frozen dataclass per coupled source/receiver pair, holding the grounded facts, produced on the
control thread, consumed by everything else. The theory ranges over
`variable(TransferSituation, domain=...)` — the same case-object pattern the backward-inference
tests use (`test/krrood_test/test_eql_rdr/test_backward_inference.py:32-46`).

### 4.3 Grounding — the predicate inventory

The ontological shift to state first: **the analytic world has no liquid entity.** Fill is a scalar
passive DOF (`LiquidConnection`, `world_description/connections.py:1393-1447`; `active_dofs` is
empty at `:1431`). Liquid is a substance quantity, not an object with parts. Every predicate
treating liquid as an individual (`partOf`, `inside(?l,?y)`, `willBeAt(?p,?o)`) is reinterpreted or
dropped. That is a deliberate simplification, not a gap — it is what makes the effect model
differentiable.

**Direct** (existing API): `Container` → `HasFillLevel` (`mixins.py:1223`); `contains` →
`fill_level > 0` (`:1804`); `hasOpening` → `opening_point`/`opening_radius`/`rim_point`
(`:1761/1756/1778`); `SourceRole`/`DestinationRole` → the coupling itself
(`receive_outflow_from`, `:1345`; `LiquidTransferCoupling.source_id`, `connections.py:1353`) —
structural, cheaper and more stable than the paper's per-cycle derivation; `poursTo` → receiver
inflow velocity > 0, already compiled and evaluated per tick at `tasks/pouring.py:174-175`
(*measured*, not derived through rules 4/13/24); `movesUp` → sign of the same value; `isTilted` →
`pour_tilt_expression` (`:1651`); `above`/`leftOf`/`rightOf`/`behind`/`inFrontOf` →
`predicates.py:433-501`, **which require a `point_of_view` transform** — the paper resolves
direction from the gripper's perspective (p. 892), so grounding must pass the gripper frame;
`contact` → `predicates.py:68-87`; `goalReached` → the task's own `_fill_goal_reached`
(`tasks/pouring.py:118`) — theory and task **must read the same goal float variable** or they
disagree at the tolerance boundary; `supports`/`Solid` → `is_supported_by` (`predicates.py:252`).

**Derivable** (small computation over existing APIs): `canPourTo`/`hasOpeningWithin`/`within` → the
two factors of `_geometric_transfer_gate` (`mixins.py:1712-1754`) above threshold — strictly
stronger than the paper's opening-over-opening because it is projectile-arc aware; `farAbove` → the
clearance expression `KeepSourceRimAboveReceiverRim` already builds (`tasks/pouring.py:473-475`);
`slowFlowFrom`/`fastFlowFrom` → thresholds over `current_outflow_velocity` (`mixins.py:1579`),
surviving as *monitoring* only; `overshoot`/`undershoot` → sign of (landing point − opening point)
projected on the horizontal exit direction; `overflows` → `fill_level ≥ 1 − ε` with inflow > 0 (the
hook `pouring_equations.py:407-409` already anticipates); `almostGoalReached` → goal minus a band;
`hasLowestOpeningCorner`/`aligned` → `_rim_exit_point` (`mixins.py:1608`) and
`KeepProjectileInReceiver`'s own observation (`tasks/pouring.py:366-368`).

**`isSpilling` — the honest case.** The analytic world **cannot spill.** The source's drain is
gated by the same `_geometric_transfer_gate` as the receiver's inflow
(`pouring_equations.py:340-361`), so volume is conserved and nothing ever lands outside; an ungated
tilted source just loses mass to nowhere via the clamped fill DOF (`connections.py:1500-1520`).
What does exist is a *predictive risk* fact: head above lip (`pouring_equations.py:293-311`) while
the gate is below threshold — "outflow is imminent and would not land in the receiver". Rename
accordingly: **`SpillRisk` is the control fact** (analytic, predictive, arm B); **`isSpilled` is an
experiment metric** — and since the MuJoCo bridge is out of scope (§8.0), `isSpilled` is not
available at all, so `SpillRisk` and the geometric proxies in §6.2 are the only spill evidence the
work will have. `SpillRisk` is strictly better than the paper's predicate — it fires before loss
rather than after — but Stage A's headline figure must be triggered by something the analytic world
can produce, not by observed spillage.

**Dropped, with reasons:** `partOf` over liquid (no parts in a scalar model; every use reduces to
fill-rate and landing-point facts); `maxTiltReached` (no twin notion exists; under arm B it is
vacuous — joint limits live in the QP bounds, paper eq. 2 — keep only for arm A);
`passThrough`/rules 6–8, 13 (model-subsumed: outflow exists only through `liquid_exit_point` by
construction); `hasPath`/`PathRole`/`Trajector` (the path *is* the projectile arc; reify only for
the transcript); `willBeAt` in general (but keep its three model-backed instantiations:
`SpillRisk`, a `CartesianPose` goal, and the terminal-state row itself, which *is*
`willBe(fill, goal)`); `hasDestination`/`LocationRole` (goals live in decision arguments).

### 4.4 Decision vocabulary

`semantic_digital_twin/src/semantic_digital_twin/reasoning/substance_transfer/decisions.py`.
Changes from revision 1: `ApproachReceiver` added (rule 19 — the protocol starts 50–100 cm out, and
approach is a distinct regime from alignment); `AbandonTransfer` split into `ConcludeTransfer`
(success, rules 20/22/10a/10b) and `AbandonTransfer(defeat)` (rules 10/20b) because they are
different exits with different motion; `RestrictSpill` carries both mechanisms it was conflating.

```python
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto

from semantic_digital_twin.reasoning.knowledge_servoing.decisions import (
    ControlDecision,
    ParameterDecision,
    RegimeDecision,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, LiquidSource


class TransferDefeat(Enum):
    """Why a transfer was defeated, recorded on :class:`AbandonTransfer`."""

    RECEIVER_WOULD_OVERFLOW = auto()
    RECEIVER_NOT_UPRIGHT = auto()
    TRANSFER_STALLED = auto()
    EXPECTATION_REFUTED = auto()


@dataclass(frozen=True)
class TransferDecision(ControlDecision, ABC):
    """A single conclusion the substance-transfer theory reached in one inference cycle."""

    source: LiquidSource
    """The container liquid leaves during the transfer."""

    receiver: HasFillLevel
    """The container liquid enters during the transfer."""


@dataclass(frozen=True)
class TransferRegimeDecision(TransferDecision, RegimeDecision, ABC):
    """A transfer decision that gates constraints rather than supplying values."""


@dataclass(frozen=True)
class TransferParameterDecision(TransferDecision, ParameterDecision, ABC):
    """A transfer decision that supplies values to registered float variables."""


# %% regime activation (channel 1)

@dataclass(frozen=True)
class ApproachReceiver(TransferRegimeDecision):
    """Move the source into pouring range; gates the pre-pour Cartesian approach."""


@dataclass(frozen=True)
class AlignSourceOverReceiver(TransferRegimeDecision):
    """Hold the pour geometry; gates the landing-point and rim-clearance constraints."""


@dataclass(frozen=True)
class PourIntoReceiver(TransferRegimeDecision):
    """Transfer substance now; gates the terminal-fill task driving the receiver to its goal."""


@dataclass(frozen=True)
class ConcludeTransfer(TransferRegimeDecision):
    """Finish a successful transfer; ends the pour regime and gates the return to upright."""


@dataclass(frozen=True)
class HoldTransfer(TransferRegimeDecision):
    """Suspend the transfer without abandoning it; pauses the transfer regime's tasks."""


@dataclass(frozen=True)
class AbandonTransfer(TransferRegimeDecision):
    """Abort the transfer because its affordance was defeated."""

    defeat: TransferDefeat
    """The defeater that removed the pour affordance."""


# %% parameterization (channel 2)

@dataclass(frozen=True)
class RetargetFillLevel(TransferParameterDecision):
    """Set the receiver's goal fill level; written into the terminal-fill goal variable."""

    goal_fill_level: float
    """Normalized target fill level in ``[0, 1]``."""


@dataclass(frozen=True)
class RestrictSpill(TransferParameterDecision):
    """Tighten the no-spill regime in response to spill risk."""

    minimum_rim_clearance: float
    """Lower bound on the source-lip-above-receiver-rim clearance, in metres."""

    landing_weight_scale: float
    """Multiplier on the landing-point constraint's quadratic weight."""
```

The two intermediate bases exist so a decision's write channel is a *type*, not a convention: the
binding policy's build-time checks (§2.5.2) are then plain `issubclass` tests, and a decision that
belongs to neither channel cannot be written.

Draining (paper §6.3) adds **one** decision and zero new twin machinery — the source-side task
already exists (`PouringTask`, `tasks/pouring.py:183`, drains the source's own fill toward
`goal_value`). That one line is the modularity claim:

```python
@dataclass(frozen=True)
class RetainSubstance(TransferDecision):
    """Drain the source while retaining a residue; parameterizes the source-side terminal-fill task."""

    retained_fill_level: float
    """Normalized fill level that must remain in the source, in ``[0, 1]``."""
```

Scraping (§6.4) adds one decision plus one new task (the second controlled task frame, the paper's
eq. 3) and is the only variant that breaks the substance-quantity model — its content is a rigid
`Body` grounded via `is_supported_by`. That is why it costs a Stage C and the other two do not.

**A genuinely new decision the old interface could not express:** the fixed-gain bridge guaranteed
motion; the QP can settle in a local optimum with the gate closed and zero inflow — tilted,
`poursTo` false, nothing infeasible, nothing moving. Rule: tilted ∧ ¬poursTo ∧ ¬goalReached,
persistent over N cycles ⇒ `HoldTransfer` / `AbandonTransfer(TRANSFER_STALLED)`. The symbolic layer
is the only component positioned to see this.

### 4.5 The expectation layer — what FWDMOD actually becomes

The paper's `Q_` rules direct *perception*: sense only what the theory will need. Against a
persistent twin, grounding a fact is evaluating a pre-compiled expression over the state vector —
microseconds, ~15 facts per pair. **FWDMOD-as-perception-director degenerates to an optimization,
and not even the load-bearing one**; revision 1's claim that expectations are "the only place the
10 Hz budget is won or lost" was wrong. The budget is spent in EQL tree evaluation. Say so, and let
Stage A measure it.

Two defensible jobs remain, and the second is stronger than what it replaces:

**Job 1 — decision persistence with retraction explanations.** For each published decision,
`BackwardInferenceIndex` (`rdr/backward_inference.py:211-229`; index built once per tree and cached,
valid because the theory is frozen during a motion) returns the sufficient condition sets. Next
cycle, re-evaluate the *previously satisfied* set via `evaluate_against` (`:61-76`) instead of
re-running the forward tree: O(#guards). When it fails, **the specific `GuardCondition` that broke
is the retraction explanation** — "the pour was retracted because `opening_within` became false" —
machine-generated, straight into WP2's `DecisionTranscript`. The paper's FWDMOD had no retraction
explanations at all.

**Job 2 — model audit.** The QP embeds a numeric forward model (the terminal-state row) and trusts
it unconditionally. The expectation layer is the only component positioned to check that model
against the measured world: after `PourIntoReceiver` the theory expects the fill to rise —
analytically, and under MuJoCo, in particles actually arriving. Divergence (gate open, ODE predicts
inflow, particles clumping at friction 3) is invisible to the QP and refutes the *effect model*, not
the rule ⇒ `AbandonTransfer(EXPECTATION_REFUTED)`. This upgrades FWDMOD from "cheaper perception" to
"the symbolic layer audits the numeric layer", which the paper could not claim because its bridge
had no model to audit.

---

## 5. WP2 — Statechart binding (~4 days, the research core)

New module `giskardpy/src/giskardpy/motion_statechart/knowledge_servoing/`.

Every class here is generic over the theory (§2.5); none of it mentions pouring.

- **`SymbolicTheoryNode`** — top-level, constant-true start. Holds a `SituationGrounding`, a
  `SymbolicTheory` and a `DecisionBindingPolicy`. Owns the reasoner thread and the decision slot;
  calls `grounding.ground(world)` every N ticks *on the control thread* and enqueues the result; on
  every tick reads the latest decision set and applies the policy's parameter writes via
  `context.float_variable_data.set_value`. Registers the policy's float variables in `build()` and
  runs its build-time validation there; the registration pattern is `CartesianPositionTrajectory`
  (`tasks/cartesian_tasks.py:252-254, 372-402`).
- **`ConcludedMonitor`** — parameterized by a decision type; observation TRUE while the latest
  decision set contains one, FALSE otherwise, UNKNOWN before the first inference. One instance per
  (decision type × statechart level) — see §2.4.
- **`DecisionBindingPolicy`** — the declarative registry of §2.5.2. A chart may hold several
  `SymbolicTheoryNode`s with disjoint decision types (§2.5.4's second demonstration); the policy
  raises on overlap rather than letting two theories fight over one task.
- **`DecisionTranscript`** — every published decision set with its inference trace, its retraction
  explanations (§4.5 job 1) and the control cycle it became visible on. This is the artifact behind
  the paper's Figure 3 and behind the traceability claim.

**Correction to revision 1:** this is *not* reuse of an established pattern. `ThreadPayloadMonitor`
(`graph_node.py:1009-1074`) overrides neither `on_tick` nor supplies an observation expression, and
its `compute_observation` has no caller in `src/` — `MotionStatechart._update_observation_state`
calls only `on_tick` (`motion_statechart.py:749-762`). As a statechart node its observation never
reaches life-cycle transitions. `ThreadedPredicateMonitor` (`payload_monitors.py:139`) does work but
is one-shot per `on_start`. A repeating reasoner node is perfectly expressible — own thread in
`__post_init__`, `on_tick` reads the slot — but it is new code.

Both node types carry unserializable state, so like `ThreadedPredicateMonitor` they work only in a
locally ticked statechart. The pouring tests already do this (`test_pouring.py:320, 471, 577`
construct `Executor` directly); the `demo_pouring_transfer.py` route through
`GiskardWrapper.execute` to a remote Giskard process is **not** available.

---

## 6. WP3/WP4 — Arms and scenarios

### 6.1 Arm A, the twist bridge (~3 days)

- **`CommandedTaskFrameTwist`** — commands a 6-D task-frame twist. **Correction:**
  `add_velocity_constraint` (`constraint_collection.py:251-289`) *bounds*; commanding needs
  `add_velocity_eq_constraint` (`:291-326`), which has **zero callers anywhere** — an unexercised QP
  path. `VelocityStrategy` does support equality bounds (`enforcement_strategy.py:336-460`) so it
  should work, but budget derisking tests for that path itself. ~120 lines for the task, plus tests.
- **`MotionPrimitiveDecision`** subclasses (`MoveLeft`, `IncreaseTilt`, …) and a `TwistBridge`
  evaluating eq. (1) with configurable α/β/γ. The *same* theory emits both vocabularies; only which
  decisions the arm's binding consumes differs. Arm A needs the view-dependent predicates and
  `maxTiltReached` that arm B drops (§4.3).

### 6.2 Stage B is analytic only — decided 2026-08-14

**There is no twin↔MuJoCo bridge at all.** `physics_simulators` imports nothing from
`semantic_digital_twin`, giskardpy has no MuJoCo coupling, and no giskardpy test drives MuJoCo; the
Executor integrates the world kinematically itself (`world.py:2540-2572`). Revision 1's "~200 lines
plus tuning" omitted streaming QP commands into MuJoCo, syncing state back at 50 Hz, and — the real
problem — reconciling the analytic fill DOF with particle ground truth. As designed, arm B's
terminal row would servo the *analytic* fill (which keeps integrating regardless of the particles)
while the metric counts particles: the headline sub-1 % claim would be a model tracking itself.

**Stage B1 — analytic ablation (~1.5 weeks, including §6.2.1).** Arms A/B/C in the analytic world,
which all three drive identically, so the bridge comparison is internally valid and decisive on its
own terms. Metrics, in two groups:

- *precision*: final fill error, time to goal, control effort, overshoot past the goal;
- *spill proxies* (since spilling cannot be measured, §8.0.1): integrated `SpillRisk`,
  landing-point excursion beyond the receiver's `opening_radius`, rim-clearance band violations;
- *reasoner*: decision count, decisions per second, inference latency distribution, retraction
  count.

Conditions: arm × starting distance × starting yaw × goal fill × model mismatch (§6.2.1), n
repetitions. Reuse `experiments/src/experiments/experiment_definitions.py` (`ExperimentResult`,
`ExperimentsTable`, `MeanAndStandardDeviation`, `TypstRenderer` — present, unwired).

**Stage B2 — dropped.** No twin↔MuJoCo bridge is built. §8.0.1 records what this costs the claims.

#### 6.2.1 The one thing an independent fact source was providing, and how to stand in for it

With analytic grounding the fill *fact* comes from the same ODE the terminal-state row predicts
with, so the loop is closed on the model itself. Arm B therefore runs with a **perfect forward
model**, which flatters it in a way an examiner will spot immediately. Arm A is not symmetrically
affected, since the twist bridge uses no model at all: the comparison as it stands partly measures
"having a correct model" rather than "using a model well". This — not external validity in general
— is the specific thing an independent fact source (§2.6) would supply, because a particle count or
a perceived level can disagree with the model.

The stand-in is cheaper than wiring one and is a better *controlled* experiment than it would have
been, since the disagreement becomes a dial rather than whatever the simulator happens to produce:
**sweep model mismatch as an independent variable.** Perturb the plant's `outflow_rate_constant` and
`discharge_coefficient` away from the values the task's equation uses — the coupling already keeps
these as explicit parameters (`initialize_fill_level`, `mixins.py:1298`;
`ArticulatedPouringEquation`, `pouring_equations.py`), so this is a fixture change, not new
machinery. Report goal error for both arms across a mismatch range from zero to well beyond
plausible.

This turns the weakness into a result. Arm B's advantage should decay with mismatch and the decay
rate is itself informative — it measures how much of the precision gain is anticipation rather than
model fidelity, which is the question the sub-1 % headline actually raises. It also gives the
expectation layer's model-audit job (§4.5, job 2) something real to detect, since large mismatch is
exactly the condition under which the theory should conclude
`AbandonTransfer(EXPECTATION_REFUTED)`. Budget ~2 days inside B1.

Also dropped with B2: the PR2/HSR portability arm, which was a MuJoCo result in the paper. Do not
assert cross-robot generality from kinematics alone; HSRB tests fail in this environment regardless.

---

## 7. WP5 — Tests

Every behaviour gets a failing test first (`AGENTS.md`).

| Layer | Location | Pins |
|---|---|---|
| Symbolic goal | `test/giskardpy_test/test_motion_statechart/test_pouring.py` | rewriting the goal variable mid-motion retargets the terminal row; the bound equals what the strategy computes for that goal; `_fill_goal_reached` reads the live value without raising |
| Single-row limit | same | two terminal-fill tasks in one chart raise `MultipleTerminalStateConstraintsError` at compile — pins §3.2's design constraint so a later change cannot silently violate it |
| Serialization guard | same | a symbolic-valued task raises on JSON serialization rather than silently becoming goal 0 |
| Grounding | `test/semantic_digital_twin_test/test_reasoning/test_substance_transfer/` | each fact's threshold against a world with known geometry; `TransferSituation` is fully frozen |
| Thread discipline | same | grounding touches no blacklisted predicate; a `TransferSituation` shares no mutable state with the world (`SpillRisk` and every field is a plain float/bool) |
| Theory | same | each layer's rules fire on a hand-built `TransferSituation`; the refinement overrides the default; `ConcludeTransfer` and `AbandonTransfer` are reached by the right defeaters |
| Expectations | same | the guard that broke is reported as the retraction explanation; a refuted model audit yields `EXPECTATION_REFUTED` |
| Framework isolation | `test/giskardpy_test/test_motion_statechart/test_knowledge_servoing/` | neither framework package imports `HasFillLevel`, `LiquidSource`, `LiquidConnection`, `tasks/pouring.py` or `substance_transfer` — walk the module's transitive imports and assert. This is the pluggability claim as a regression |
| Framework genericity | same | a mimic theory in the test dataset (non-pouring situation, two facts, driving `JointPositionList`) runs end to end through `SymbolicTheoryNode` and exercises both channels |
| Binding validation | same | an unbound decision type raises `UnboundDecisionTypeError` at build; a type bound to both channels raises; a parameter bound to an unregistered variable raises; two theories with overlapping decision types raise |
| Binding | same | a gated task contributes zero QP rows before its decision and its rows in the *same* cycle after; the parameter channel changes the solved command with no recompile |
| Two-rate | same | a deliberately slow theory never stalls `Executor.tick`; the loop runs on the last published decision set |
| Closed loop | same | a full pour in the analytic world reaches the goal fill within tolerance, driven only by reasoner decisions |

The analytic closed-loop test is the CI-safe regression: `LiquidConnection` + `step_physics`, no
simulator, no ROS, on the existing `single_cup_world.py` fixtures. MuJoCo runs are experiment
drivers under `experiments/`, not CI tests.

---

## 8. Decisions

### 8.0 Taken, 2026-08-14

| # | Decision | Consequence |
|---|---|---|
| 1 | **Deferred** — single terminal row vs. N-row generalization | Safe to defer: "accept the limit" is the zero-work branch, so WP0 proceeds under it and nothing is foreclosed. Revisit only if an experiment needs two simultaneous terminal rows |
| 2 | **Accept ORM column loss.** Symbolic fields lose their column; no numeric default is persisted | No DB round-trip work in WP0. Note it in the docstring so a future reader is not surprised |
| 3 | **MCRDR** (`krrood.ripple_down_rules.MultiClassRDR`) is the theory engine | §4.1.1's recommendation adopted. Stop rules, multi-conclusion and top-rule-ordered chaining are native; WP1b shrinks |
| 3b | **Keep retraction explanations** | Build the backward index over the RDR tree (~100–150 lines, §4.1.1). The interpretability claim keeps its machine-generated evidence |
| 4 | **In-motion inference stays read-only** (option i) | `SCHMOD` is frozen for the duration of a motion; annotation inference runs at motion boundaries only (§8.1) |
| 5 | **No twin↔MuJoCo bridge for now.** Stage B2 is out of scope | The largest schedule risk is removed and the plan is analytic end to end. MuJoCo is a *fact source* standing in for robot and perception (§2.6), not the plant, so this is deferral rather than exclusion: the seam is a state writer and nothing built here forecloses it |

### 8.0.1 What deferring the simulator fact source costs

The architecture makes this cheap (§2.6) — but it is not free for the evidence base, and the
writing has to reflect that until a real fact source is wired.

- **Particle-based comparison to Table 1 is deferred.** Arm A cannot be validated against the
  published goal-error and spilling numbers while those are particle counts and nothing here counts
  particles. The published 6 % is context, not a comparison. **The within-system A-vs-B ablation
  carries the claim**, which is why §1.1 was built that way; external validity is explicitly the
  work that wiring a fact source back in would do.
- **`isSpilled` is unavailable; only `SpillRisk` exists.** The analytic world conserves volume by
  construction (§4.3), so spilling can be predicted but not measured. Arm A's headline failure mode
  in the paper — corrective spillage from fixed gains — is reproduced as its *cause* instead:
  integrated `SpillRisk`, landing-point excursion outside the receiver opening, rim-clearance
  violations. Report those and name them as proxies.
- **No spill-triggered regime flip in the demo.** Stage A's headline figure must be driven by a fact
  the analytic world can produce — goal proximity, `SpillRisk`, `overflows`, transfer-stalled, or an
  annotation (open item 6). Not by observed spillage.
- **Robot portability is untested.** The PR2/HSR arm of the paper's evaluation was a MuJoCo result.
  Drop the cross-robot claim rather than assert it from kinematics alone; HSRB tests fail in this
  environment regardless.

None of these is a design debt — each is closed by the same later step, wiring a fact source per
§2.6. Open item 6 becomes more valuable meanwhile, since it is the cleanest available source of a
mid-motion regime flip that needs no particles.

### 8.0.2 Taken since, 2026-08-16

| # | Decision | Consequence |
|---|---|---|
| 6 | **The contextual safety theory is in scope**, and is built as production code (`reasoning/contextual_safety/`) | Pluggability is demonstrated rather than asserted: two theories with different situation types run over one motion and meet only at their binding policies. Phase 2 (the same theory on perceived facts) needs no rewrite |
| 7 | **The reasoner runs synchronously**, every fifth control cycle | §2.3's threading design becomes a contingency (see its note). Reversible: `SymbolicTheoryNode` is the only place that would move |
| 8 | **MCRDR confirmed as the engine** by the Stage A spike | Stop rules, multi-conclusion and intra-pass chaining all behave as §4.1.1 claims; `copy_case` does *not* survive a frozen dataclass, so the theory builds a mutable working copy internally — the fallback §4.1.1 anticipated |

### 8.0.3 Still open

9. **Thesis placement** (§8.5) — determines how much of Stage C is required.
10. **WP0 §3.3** — symbolic `minimum_clearance`/`clearance_band` and per-task weight variables. The
    parameter channel currently reaches exactly one knob (the fill goal), which is why the safety
    theory is regime-only. Widening it is what lets a theory tune a constraint rather than only
    switch it.
11. **Arm A** (§6.1) and the **B1 ablation with the mismatch sweep** (§6.2, §6.2.1) — the experiment
    proper, now unblocked.

### 8.1 In-motion inference — the hard blocker

`WorldReasoner.reason` runs inside `with self.world.modify_world():` (`world_reasoner.py:47, 64`),
and `ControlLoop.apply_world_updates` aborts on `world_is_being_modified` *or* a buffered
structural-change message (`middleware/ros2/control_loop.py:100-111`). Note the argument is stronger
than revision 1 stated: `WorldReasoner.reason` bumps the model version and clears the memo cache
**even when nothing changed** (`WorldModelUpdateContextManager.__exit__`, `world.py:285-314, 481`),
so even a no-op reasoning pass trips the abort.

**This plan takes option (i):** in-motion inference is EQL over frozen `TransferSituation` objects
plus the two write channels — all read-only with respect to the world model. No
`infer_semantic_annotations`, no `add_semantic_annotation`, during a motion; annotation inference
runs at motion boundaries only.

The cost is that `SCHMOD` is fixed for the duration of a motion. Not binding for pouring — the
affordances that matter are established before the goal starts. Binding the moment a rule should
introduce a *new* object relationship mid-motion. That is the honest scope statement for this and
for report §1's regime switch alike.

### 8.2 Rate is unverified

The paper needs 10 Hz. EQL tree evaluation over a stratified pipeline has never been measured here.
Stage A measures it; if it lands at 1–2 Hz the contribution survives but the claim becomes
"reasoning at task-event rate", not "servoing". Grounding is not the risk (§4.5); tree evaluation
and the number of `TransferSituation` pairs are.

### 8.3 `Refinement` is not Antoniou defeasible logic — and the mismatch is sharper than stated

`Refinement` (`conclusion_selector.py:162`) is except-if specificity over a tree. The subtlety
revision 1 missed: encoding `r' > r` as a refinement of `r` **narrows `r'` to `r`'s situations**,
and `25a`'s conditions are not a superset of `18c`'s (`almostGoalReached` does not imply
`slowFlowFrom`). Here that is behaviourally harmless — under fast flow `18d` concludes decrease
anyway — but the general pattern must be argued: a refinement means "within `r`'s situations, `r'`
wins", which is what the paper's two superiority pairs actually mean, but is not Antoniou-equivalent
in general.

Under MCRDR (§4.1.1) the mismatch is **narrower than revision 2 claimed**, and two of its stated
limitations were wrong:

- **Negative conclusions do have machinery.** `MultiClassStopRule` blocks a parent conclusion and
  concludes nothing itself (`rules.py:543-566`) — a defeater in the technical sense. `−canPourTo`
  stays in the rule layer; it does not have to be folded into procedural grounding.
- **Simultaneous conclusions are native.** MCRDR accumulates across the top-rule chain
  (`rdr.py:1491-1533`), so one tree suffices; the one-tree-per-decision-family workaround is
  unnecessary.

What genuinely remains missing, and is what the chapter must argue: there is **no explicit
superiority relation over a flat rule set** — priority is tree-structural specificity, so the
paper's `25a > 18c` becomes a tree position, and encoding `r' > r` as a refinement of `r` **narrows
`r'` to `r`'s situations**. `25a`'s conditions are not a superset of `18c`'s (`almostGoalReached`
does not imply `slowFlowFrom`), so the encoding reinterprets the pair as "within `r`'s situations,
`r'` wins". That is what the paper's two pairs actually mean, but it is not Antoniou-equivalent in
general. Also absent: team defeat and ambiguity blocking. The defensible line is that RDR trades an
asserted superiority relation for one *derived from the order of correction*, with cornerstone-case
regression checking as the compensating guarantee — a different and arguably better engineering
contract, but a different one, and it must be argued rather than assumed.

Unlike revision 1, §8.3 **does** gate work: it determines WP1's structure.

### 8.4 Perception stays factored out — but the exit is designed, not hypothetical

Grounding reads the twin, assumed populated. For the pouring work this does not repair the paper's
fourth limitation, and is arguably further from real perception than the paper's geometric
grounding. Claim no hardware contribution for it.

Two things keep this from being a dead end, and both are already in the design rather than deferred
to a wish list. §2.6 fixes the seam a fact source plugs into (a state writer on the twin, not a
grounding variant), so nothing built here has to be revisited when perception arrives. And the
contextual safety theory (§2.5.4) is written to be the domain that *needs* it: its facts have no
analytic substitute, so it is where perceived semantic features become load-bearing rather than
merely more realistic. The honest statement for the chapter is that perception is factored out of
the pouring result by construction, and that the architecture's response to that limitation is
localized to one replaceable component — with a second theory already positioned to exercise it.

### 8.5 Thesis placement

CS1's theory on CS2's executive. A synthesis section needs Stage A plus a reduced B1; a chapter of
its own needs Stage C. Decide before Stage B, because arm A (WP3) is only worth building if the
comparison goes into the document.

---

## 9. Staging and surface

- **Stage A — demonstrator. Done, 2026-08-16** (§0). WP0's symbolic goal, the framework, both
  instantiations, the binding layer, the transcript, the Gantt chart and the demonstration, with the
  contextual safety theory pulled forward from the optional list because it is what makes the
  headline figure a context-driven regime change rather than a task progression.
  Deliverable: a semantic rule, not a topic publish, flips the constraint regime mid-pour at VJDF
  precision, with a decision transcript, retraction explanations and a Gantt figure. Measure
  reasoner latency here before committing to B.
- **Stage B1 — analytic ablation (~1.5 weeks).** WP3 + arms A/B/C + the model-mismatch sweep. The
  bridge claim, decided. There is no Stage B2 (§8.0).
- **Stage C — modularity (~2–3 weeks).** Draining (one decision, one chart, existing `PouringTask`)
  and scraping (one decision, one new task frame). This demonstrates **rule-set extension**, not
  theory pluggability (§1.4) — do not let the writing merge the two.
- **Recommended, any time after Stage A (~3 days) — contextual safety, phase 1** (§2.5.4). A second
  theory, its own situation type, no effect model of its own, coexisting in one chart. The only item
  that demonstrates **theory pluggability**, an order of magnitude cheaper than Stage C, and the
  seed of a later evaluation domain. If the schedule forces a choice between this and Stage C, this
  is the better buy.
- **Future work — contextual safety, phase 2.** The same theory on perceived facts, once a fact
  source is wired per §2.6. Not scheduled here; listed so phase 1 is built to survive it.

| Package | Source | Tests |
|---|---|---|
| WP0 library changes | ~250 edited | ~350 |
| WP1a framework interfaces (§2.5) | ~250–350 | ~250 |
| WP1b grounding + situation + theory + expectations (pouring) | ~700–1,000 | ~600 |
| *if retraction explanations are kept:* backward index over the RDR tree (§4.1.1) | ~100–150 | ~120 |
| WP2 binding nodes + policy | ~400–500 | ~400 |
| WP3 twist bridge (arm A) | ~250 | ~200 |
| WP4 analytic metrics + mismatch sweep (B1) | ~300 | — |
| **Total, Stage A + B1** | **~2,250–2,700** | **~1,800** |
| *optional:* contextual safety theory (§2.5.4) | ~250 | ~200 |

Roughly **5–6.5 weeks for Stage A + B1**, against revision 1's 4 weeks for A + B. The difference is
WP0's true size (§3) and the framework/instantiation split (§2.5); WP1's re-architecture cost is
largely refunded by the MCRDR decision (§4.1.1), and dropping B2 removes the schedule's least
predictable item entirely. Every remaining item runs on the analytic world with no simulator, no
ROS and no external assets, which is what makes this estimate more trustworthy than revision 1's.

### What earlier revisions got wrong

| Claim | Reality |
|---|---|
| "the pouring rule set transcribes close to literally" | EQL has no forward chaining, no derived-fact negation, no working memory (§4.1) |
| WP0 is "a few lines" | plus tick-time comparison crashes, ORM column loss, JSON corruption (§3.1) |
| terminal rows are per-task | one per compiled chart, enforced before gating (§3.2) |
| reasoner thread reads the twin | four unlocked read races; grounding must run on the control thread (§2.3) |
| `ThreadPayloadMonitor` is an established pattern to reuse | its observation never reaches life-cycle transitions (§5) |
| arm A uses `add_velocity_constraint` | that bounds; commanding needs the zero-caller `add_velocity_eq_constraint` (§6.1) |
| MuJoCo scenario ≈ 200 lines | no twin↔MuJoCo bridge exists at all (§6.2) |
| expectations are the 10 Hz bottleneck | grounding is microseconds; EQL evaluation is the cost (§4.5) |
| `Refinement`-vs-Antoniou "gates no work package" | it determines WP1's entire structure (§8.3) |
| "most of the theory is untouched" | ~41 % unchanged by rule count; the defensible claim is by decision pattern (§1.2) |
| activation takes effect next cycle | same cycle (§2.2) — the one correction in the plan's favour |
| `Near` exists in `predicates.py` | it does not (§3.3) |
| the design was pouring-shaped throughout | the paper's headline contribution is theory pluggability; framework and instantiation are now separated and the boundary is tested (§1.4, §2.5) |
| *(rev 2)* negative conclusions have no machinery, so defeaters fold into procedural grounding | `MultiClassStopRule` is a defeater; `−canPourTo` stays in the rule layer (§4.1.1, §8.3) |
| *(rev 2)* simultaneous decisions need one rule tree per decision family | MCRDR accumulates conclusions across the top-rule chain; one tree suffices (§4.1.1) |
| *(rev 2)* WP1 must be a hand-built stratified pipeline | MCRDR writes conclusions back into the case mid-pass, so stratification is top-rule ordering (§4.1.1) |

---

## 10. Relation to the other experiments

Report §1's runtime regime switch is the *mechanism* this work needs; this work supplies the
*decision layer* the regime switch currently takes from a hand-published ROS topic. Build §1 first
with a swappable trigger source and write it up as the mechanism characterization; write this up as
the closed semantic loop that uses it. WP0's `BoolTopicMonitor` is shared. Report §§2–4 are
independent — but note §4's verification predicates (`Causes`, `CanPerform`) are on the grounding
blacklist here (§2.3): they mutate world state and belong to that experiment, not to servoing.
