# Semantic Safety Filter in Giskardpy — Implementation Concept

Based on: Brunke et al. (2025), "Semantically Safe Robot Manipulation: From Semantic Scene Understanding to Motion Safeguards"

---

## Paper Summary

Brunke et al. (2025) propose a **Semantic Safety Filter** that certifies robot motions against semantically derived constraints — e.g. "don't move a cup of water above a laptop." Three constraint types are synthesized via LLM from a 3D semantic scene map:

1. **Spatial relationship constraints** — CBFs fitted as superquadrics to object point clouds, encoding unsafe zones (above, below, around, ...)
2. **Behavioral constraints** — slow down when approaching objects requiring caution, implemented by tuning the CBF class-K∞ function α
3. **Pose constraints** — penalize end-effector rotation when holding containers

These are stacked with geometric collision/self-collision CBFs and solved as a single online QP that minimally deviates from the commanded velocity.

---

## Architecture Fit

Giskardpy is already a QP-based motion controller. The paper's safety filter **is** a QP — so there is no separate filter layer needed. The semantic constraints become additional `Task` nodes added in parallel to the primary motion goal. The existing `add_velocity_constraint` API is exactly the right hook for CBF conditions because it constrains `ḣ = (∂h/∂q)u`, which is what `ḣ ≥ −α(h)` requires.

---

## Component Map

| Paper component | Giskardpy component |
|---|---|
| RGB-D + open-vocab segmentation | SemanticDigitalTwin (existing, extend with point clouds) |
| 3D environment map (object labels) | SDT Body entities with semantic labels (existing) |
| LLM semantic constraint synthesis | SemanticConstraintSynthesizer (new, KRROOD/RDR backend) |
| Superquadric g_i(x_ee; θ) | SuperquadricBarrierFunction (new, `giskardpy/qp/`) |
| CBF h_sem,i = g_i − 1 | symbolic Scalar via sm.*, evaluated at FK position |
| ḣ_sem ≥ −α_sem(h_sem) | `add_velocity_constraint(lower=−α(h), upper=∞, expr=h)` |
| Behavioral caution (α scaling) | `CautionLevel` enum → α_sem scalar parameter in constraint |
| Pose constraint (rotation penalty) | soft `add_velocity_constraint` on rotation expression |
| Safety filter QP | `SemanticSafetyGoal` (Parallel of tasks, added alongside primary goal) |

---

## New Classes and Files

### `giskardpy/qp/cbf.py` — Control Barrier Function primitives

```python
@dataclass
class SuperquadricParameters:
    epsilon_1: float          # shape exponent (outer)
    epsilon_2: float          # shape exponent (inner)
    scale: np.ndarray         # (a_x, a_y, a_z)
    transform: np.ndarray     # 4x4 pose of superquadric in world frame

@dataclass
class SuperquadricBarrierFunction:
    """
    Computes h(x_ee) = g(x_ee; θ) − 1 where g is a superquadric.
    h ≥ 0 ↔ x_ee is outside the unsafe region.
    """
    parameters: SuperquadricParameters

    def evaluate(self, end_effector_position: Point3) -> sm.Scalar:
        """Returns symbolic CBF value, differentiable w.r.t. joint angles via CasADi."""
        ...

    @classmethod
    def fit_to_point_cloud(cls, points: np.ndarray,
                           relation: SpatialRelation) -> SuperquadricBarrierFunction:
        """Offline: fit superquadric to object point cloud, extended for spatial relation."""
        ...
```

The `evaluate` method builds a symbolic CasADi expression. Because giskardpy automatically differentiates through forward kinematics, `(∂h/∂q) · u` is computed implicitly when the constraint is added via `add_velocity_constraint(task_expression=h_sym)`.

For the `above` relation, `fit_to_point_cloud` duplicates the point cloud in +z before fitting, exactly as described in the paper.

---

### Semantic Constraint Representation

New file: `giskardpy/motion_statechart/goals/semantic_safety_types.py`
(or extend `giskardpy/motion_statechart/data_types.py`)

```python
class SpatialRelation(str, Enum):
    ABOVE = "above"
    BELOW = "below"
    AROUND = "around"
    UNDER = "under"
    # ... 12 total from paper

class CautionLevel(str, Enum):
    NONE = "no_caution"
    CAUTION = "caution"

class PoseConstraintType(str, Enum):
    FREE_ROTATION = "free_rotation"
    CONSTRAINED_ROTATION = "constrained_rotation"

@dataclass
class ObjectSpatialConstraint:
    object_body: Body
    relation: SpatialRelation
    caution_level: CautionLevel
    barrier: SuperquadricBarrierFunction    # pre-fitted

@dataclass
class SemanticConstraintContext:
    spatial_constraints: list[ObjectSpatialConstraint]
    pose_constraint: PoseConstraintType
    manipulated_object: Body
```

---

### Semantic Constraint Synthesis

New file: `giskardpy/motion_statechart/goals/semantic_constraint_synthesizer.py`

```python
class SemanticConstraintSynthesizer(ABC):
    @abstractmethod
    def synthesize(self, manipulated_object: Body,
                   scene_objects: list[Body]) -> SemanticConstraintContext:
        """Query which spatial relations and behaviors are unsafe for this object."""

@dataclass
class RippleDownRulesSynthesizer(SemanticConstraintSynthesizer):
    """
    Uses KRROOD RippleDownRules for offline, deterministic semantic inference.
    Rules like: IF object.label == 'laptop' AND relation == 'above' THEN unsafe.
    Enables running without LLM at runtime.
    """
    rule_set: RippleDownRules

@dataclass
class LLMSynthesizer(SemanticConstraintSynthesizer):
    """
    Uses an LLM (multi-prompt strategy from paper) for open-vocabulary reasoning.
    Results are cached per (manipulated_object_class, scene_object_class) pair.
    """
    llm_client: ...
    cache: dict = field(default_factory=dict)
```

Both implementations produce the same `SemanticConstraintContext`. In this codebase, the **RDR backend is the natural fit** since KRROOD already has ripple-down rules, EQL for semantic queries, and ontology support. The LLM backend is an optional enhancement.

---

### Tasks

New file: `giskardpy/motion_statechart/tasks/semantic_safety_tasks.py`

```python
@dataclass(eq=False, repr=False)
class SpatialRelationshipSafetyTask(CartesianTask):
    """
    Enforces ḣ_sem,i ≥ −α_sem,i(h_sem,i) for one spatial relationship constraint.
    CBF condition is added as a velocity inequality constraint.
    """
    object_constraint: ObjectSpatialConstraint
    alpha_gain: float = 1.0  # scales how aggressively boundary is enforced

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = super().build(context)
        fk = context.world.compute_fk(self.root_link, self.tip_link)
        end_effector_position = fk.translation

        h = self.object_constraint.barrier.evaluate(end_effector_position)
        alpha = self._compute_alpha(h)  # class-K∞ function, tuned by caution level

        constraints = ConstraintCollection()
        constraints.add_velocity_constraint(
            lower_velocity_limit=-alpha,
            upper_velocity_limit=1e6,
            quadratic_weight=DefaultWeights.WEIGHT_ABOVE_CA,
            task_expression=h,
            velocity_limit=0.5,
            name=f"cbf_sem/{self.object_constraint.object_body.name}/{self.object_constraint.relation}",
            lower_slack_limit=0.0,   # hard: never violate the barrier
        )
        artifacts.add_constraints(constraints)
        return artifacts

    def _compute_alpha(self, h: sm.Scalar) -> sm.Scalar:
        if self.object_constraint.caution_level == CautionLevel.CAUTION:
            return (self.alpha_gain / 4) * h * h   # α_sem,c(h) = (1/4)h²
        return self.alpha_gain * h * h              # α_sem(h) = h²


@dataclass(eq=False, repr=False)
class EndEffectorPoseConstraintTask(CartesianTask):
    """
    Penalizes end effector rotation when pose_constraint == CONSTRAINED_ROTATION.
    Prevents spillage by both limiting angular velocity and penalizing deviation
    from the desired orientation captured at object pick-up.
    """
    desired_rotation: RotationMatrix    # captured at grasp time
    w_rotation: float = 1.0

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = super().build(context)
        fk = context.world.compute_fk(self.root_link, self.tip_link)
        # log(R_des R_cur^T)^∨ — rotation error in Lie algebra
        rotation_error = sm.log_rotation(self.desired_rotation @ fk.rotation.T)

        constraints = ConstraintCollection()
        constraints.add_velocity_constraint(
            lower_velocity_limit=-self.w_rotation * 0.1,
            upper_velocity_limit=self.w_rotation * 0.1,
            quadratic_weight=DefaultWeights.WEIGHT_BELOW_CA,
            task_expression=rotation_error.norm(),
            velocity_limit=0.5,
            name="cbf_pose/rotation_limit",
        )
        artifacts.add_constraints(constraints)
        return artifacts
```

---

### Top-level Goal

New file: `giskardpy/motion_statechart/goals/semantic_safety.py`

```python
@dataclass(eq=False, repr=False)
class SemanticSafetyGoal(Goal):
    """
    Adds semantic CBF safety constraints for the currently manipulated object.
    Runs in parallel with the primary motion goal (e.g. CartesianPose).
    Reads the semantic constraint context from SemanticConstraintSynthesizer and
    activates one SpatialRelationshipSafetyTask per unsafe spatial relationship.
    """
    manipulated_object: Body
    synthesizer: SemanticConstraintSynthesizer
    tip_link: KinematicStructureEntity
    root_link: KinematicStructureEntity

    def expand(self, context: MotionStatechartContext) -> None:
        scene_objects = context.world.get_all_bodies_except(self.manipulated_object)
        constraint_context = self.synthesizer.synthesize(
            self.manipulated_object, scene_objects
        )
        tasks = [
            SpatialRelationshipSafetyTask(
                object_constraint=c,
                root_link=self.root_link,
                tip_link=self.tip_link,
            )
            for c in constraint_context.spatial_constraints
        ]
        if constraint_context.pose_constraint == PoseConstraintType.CONSTRAINED_ROTATION:
            tasks.append(EndEffectorPoseConstraintTask(
                desired_rotation=self._capture_current_rotation(context),
                root_link=self.root_link,
                tip_link=self.tip_link,
            ))
        self.add_nodes(tasks)
```

**Usage alongside a primary goal:**

```python
Parallel(nodes=[
    CartesianPoseStraight(root_link=base, tip_link=gripper, goal_pose=target),
    SemanticSafetyGoal(
        manipulated_object=cup_of_water,
        synthesizer=RippleDownRulesSynthesizer(rule_set=kitchen_rules),
        root_link=base,
        tip_link=gripper,
    ),
])
```

---

## Superquadric Fitting Pipeline

The paper fits superquadrics offline per object per spatial relation. The fitting step is a preprocessing concern, not an online one:

```
SemanticDigitalTwin.get_object_point_cloud(body)
    → SuperquadricBarrierFunction.fit_to_point_cloud(points, relation)
    → stored in SDT as object metadata
    → loaded by SemanticConstraintSynthesizer at synthesis time
```

- For the `above` relation, point clouds are extended in +z before fitting.
- For `around`, the point cloud is used as-is.
- Non-convex objects use a union of superquadrics with `h_eff = max(h_i)` approximated smoothly.

---

## Mapping to Equation (3) of the Paper

| Paper constraint | Giskardpy mapping |
|---|---|
| `ḣ_sem ≥ −α_sem(h_sem; S_b)` | `SpatialRelationshipSafetyTask` with `lower_slack_limit=0` (hard) |
| `ḣ_env ≥ −α_env(h_env)` | Extend existing `_ExternalCollisionAvoidanceTask` with superquadric CBFs |
| `ḣ_self ≥ −α_self` | Already handled; extend with spherical CBFs along robot body |
| `ḣ_lim ≥ −α_lim` | Already handled by joint limit constraints |
| `\|\|u − u_cmd\|\|²` objective | Primary motion goal (CartesianPose) provides this implicitly |
| `w_rot L_rot(q,u)` cost term | `EndEffectorPoseConstraintTask` |
| Behavioral caution (α scaling) | `CautionLevel` → `_compute_alpha` in `SpatialRelationshipSafetyTask` |

---

## Key Design Decisions

1. **No separate safety filter QP** — the main giskardpy QP absorbs the safety constraints. This is architecturally cleaner and avoids a secondary solve. The tradeoff is that safety constraints are soft unless `lower_slack_limit=0` is explicitly set. Setting it to 0 makes CBF conditions hard, which matches the paper.

2. **RDR over LLM at runtime** — KRROOD's RippleDownRules are the right backend for deterministic, fast inference during execution. LLM synthesis runs offline (or once per new scene) and its output is compiled into RDR rules or cached as `SemanticConstraintContext` objects.

3. **Superquadric fitting is preprocessing** — fits happen when an object first enters the SDT, not during control. This keeps online CBF evaluation cheap (just symbolic evaluation + CasADi autodiff).

4. **α as symbolic expression** — because `h` is a CasADi symbolic Scalar, `α(h) = h²` is also symbolic and differentiable. The `lower_velocity_limit` parameter in `add_velocity_constraint` accepts `sm.ScalarData`, so `α` values can be full symbolic expressions evaluated at each QP step.

---

## Implementation Phases

| Phase | Scope | Files |
|---|---|---|
| 1 | `SuperquadricBarrierFunction` + unit tests (pure math, no giskardpy dependency) | `giskardpy/qp/cbf.py` |
| 2 | `SemanticConstraintContext` dataclasses + `RippleDownRulesSynthesizer` (KRROOD integration) | `giskardpy/motion_statechart/goals/semantic_safety_types.py`, `semantic_constraint_synthesizer.py` |
| 3 | `SpatialRelationshipSafetyTask` + integration test with a static superquadric | `giskardpy/motion_statechart/tasks/semantic_safety_tasks.py` |
| 4 | `EndEffectorPoseConstraintTask` + test with rotation deviation | same file |
| 5 | `SemanticSafetyGoal` composing the above + end-to-end test | `giskardpy/motion_statechart/goals/semantic_safety.py` |
| 6 | SDT extension for point cloud storage and superquadric fitting | `semantic_digital_twin/` |
| 7 | (Optional) `LLMSynthesizer` with prompt caching | `semantic_constraint_synthesizer.py` |
