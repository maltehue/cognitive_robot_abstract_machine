"""
Two theories reasoning over one motion.

The substance-transfer theory drives the pour; the contextual-safety theory, which knows
nothing about pouring and owns no effect model, restricts the motion because of what the
scene contains. Neither theory is aware of the other: they meet only at the binding
policy, which is what the framework's pluggability claim amounts to in practice.
"""

from __future__ import annotations

import pytest

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights, LifeCycleValues
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.knowledge_servoing.concluded_monitor import (
    ConcludedMonitor,
)
from giskardpy.motion_statechart.knowledge_servoing.decision_binding_policy import (
    DecisionBindingPolicy,
)
from giskardpy.motion_statechart.knowledge_servoing.decision_slot import DecisionSlot
from giskardpy.motion_statechart.knowledge_servoing.symbolic_theory_node import (
    SymbolicTheoryNode,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianVelocityLimit
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
)
from semantic_digital_twin.reasoning.contextual_safety import (
    CautionReason,
    EnforceCaution,
    SafetySituationGrounding,
    build_contextual_safety_theory,
)
from semantic_digital_twin.reasoning.substance_transfer import (
    AbandonTransfer,
    AlignSourceOverReceiver,
    ConcludeTransfer,
    PourIntoReceiver,
    RetargetFillLevel,
    TransferSituationGrounding,
    build_substance_transfer_theory,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from .test_pouring import (  # noqa: F401 - fixtures are used by name
    tracy_pouring_world,
    tracy_transfer_world,
)

REQUESTED_FILL_LEVEL = 0.4
"""
Fill level the transfer theory is asked to reach.
"""

CAUTIOUS_LINEAR_VELOCITY = 0.03
"""
Linear speed cap the safety theory's caution regime imposes, in metres per second.
"""


def _spawn_sensitive_body(world, receiving_cup) -> Body:
    """
    Places a body the twin marks as not-to-be-spilled-on beside the receiving cup.
    """
    sensitive_body = Body.from_shape_collection(
        shape_collection=ShapeCollection([Box(scale=Scale(0.2, 0.2, 0.02))]),
        name=PrefixedName("laptop"),
    )
    with world.modify_world():
        world.add_body(sensitive_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=receiving_cup.root,
                child=sensitive_body,
                name=PrefixedName("receiving_cup_T_laptop"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(),
            )
        )
    return sensitive_body


@pytest.fixture
def two_theory_statechart(
    tracy_transfer_world,
):  # noqa: F811 - pytest fixture injection
    """
    Builds one statechart driven by a transfer theory and a safety theory together.
    """
    world, source_cup, receiving_cup, _tool = tracy_transfer_world
    source_equation = source_cup.fill_equation
    receiving_cup.recouple_outflow_from(
        source=source_cup,
        world=world,
        fill_equation=ArticulatedPouringEquation(
            container_height=source_equation.container_height,
            container_width=source_equation.container_width,
            outflow_rate_constant=0.12,
            discharge_coefficient=source_equation.discharge_coefficient,
        ),
    )
    sensitive_body = _spawn_sensitive_body(world, receiving_cup)

    goal_fill_variable = sm.FloatVariable(name="transfer_goal_fill_level")
    aim = KeepProjectileInReceiver(
        receiver=receiving_cup, source=source_cup, weight=DefaultWeights.WEIGHT_MAXIMUM
    )
    clearance = KeepSourceRimAboveReceiverRim(
        receiver=receiving_cup, source=source_cup, minimum_clearance=0.08
    )
    transfer = FillByTransferTask(
        receiver=receiving_cup,
        goal_value=goal_fill_variable,
        fill_level_tolerance=0.05,
    )
    return_upright = AlignPlanes(
        root_link=world.root,
        tip_link=source_cup.root,
        goal_normal=Vector3.Z(reference_frame=world.root),
        tip_normal=Vector3.Z(reference_frame=source_cup.root),
    )
    # The safety theory's remedy: a constraint the controller already knows how to enforce, which
    # the transfer theory neither declares nor is aware of.
    speed_cap = CartesianVelocityLimit(
        root_link=world.root,
        tip_link=source_cup.root,
        max_linear_velocity=CAUTIOUS_LINEAR_VELOCITY,
    )

    transfer_slot = DecisionSlot()
    safety_slot = DecisionSlot()
    align_monitor = ConcludedMonitor(
        decision_type=AlignSourceOverReceiver, decision_slot=transfer_slot
    )
    pour_monitor = ConcludedMonitor(
        decision_type=PourIntoReceiver, decision_slot=transfer_slot
    )
    concluded_monitor = ConcludedMonitor(
        decision_type=ConcludeTransfer, decision_slot=transfer_slot
    )
    caution_monitor = ConcludedMonitor(
        decision_type=EnforceCaution, decision_slot=safety_slot
    )

    transfer_policy = DecisionBindingPolicy()
    transfer_policy.activate(AlignSourceOverReceiver, aim)
    transfer_policy.activate(PourIntoReceiver, transfer)
    transfer_policy.activate(ConcludeTransfer, return_upright)
    transfer_policy.activate(AbandonTransfer, concluded_monitor)
    transfer_policy.parameterize(
        RetargetFillLevel, lambda decision: decision.goal_fill_level, goal_fill_variable
    )
    safety_policy = DecisionBindingPolicy()
    safety_policy.activate(EnforceCaution, speed_cap)

    transfer_node = SymbolicTheoryNode(
        grounding=TransferSituationGrounding(
            source=source_cup,
            receiver=receiving_cup,
            requested_fill_level=REQUESTED_FILL_LEVEL,
        ),
        theory=build_substance_transfer_theory(),
        binding_policy=transfer_policy,
        decision_slot=transfer_slot,
    )
    safety_node = SymbolicTheoryNode(
        grounding=SafetySituationGrounding(
            carried_container=source_cup, sensitive_bodies=[sensitive_body]
        ),
        theory=build_contextual_safety_theory(),
        binding_policy=safety_policy,
        decision_slot=safety_slot,
    )

    aim.start_condition = align_monitor.observation_variable
    clearance.start_condition = align_monitor.observation_variable
    transfer.start_condition = pour_monitor.observation_variable
    return_upright.start_condition = concluded_monitor.observation_variable
    speed_cap.start_condition = caution_monitor.observation_variable

    statechart = MotionStatechart()
    for node in (
        transfer_node,
        safety_node,
        align_monitor,
        pour_monitor,
        concluded_monitor,
        caution_monitor,
        aim,
        clearance,
        transfer,
        return_upright,
        speed_cap,
    ):
        statechart.add_node(node)
    statechart.add_node(EndMotion.when_true(concluded_monitor))

    executor = Executor(
        MotionStatechartContext(world=world), pacer=SimulationPacer(real_time_factor=1)
    )
    executor.compile(motion_statechart=statechart)
    return executor, safety_slot, speed_cap, receiving_cup


class TestTwoTheoriesInOneStatechart:
    """
    Whether a second theory can restrict a motion the first theory is driving.
    """

    def test_the_safety_theory_activates_its_constraint(self, two_theory_statechart):
        executor, safety_slot, speed_cap, _cup = two_theory_statechart
        executor.tick()
        assert safety_slot.latest.contains_type(EnforceCaution)
        assert speed_cap.life_cycle_state == LifeCycleValues.RUNNING

    def test_the_caution_reason_changes_when_the_pour_starts(
        self, two_theory_statechart
    ):
        """
        The regime change is derived from the scene, not published into it: the same
        rule set reports a different reason once contents are in flight rather than
        merely carried.
        """
        executor, safety_slot, _speed_cap, _cup = two_theory_statechart
        observed_reasons = []
        for _ in range(2000):
            executor.tick()
            for decision in safety_slot.latest.of_type(EnforceCaution):
                if decision.reason not in observed_reasons:
                    observed_reasons.append(decision.reason)
            if executor.motion_statechart.is_end_motion():
                break

        assert observed_reasons == [
            CautionReason.CARRYING_CONTENTS_OVER_SENSITIVE_OBJECT,
            CautionReason.SPILL_WOULD_REACH_SENSITIVE_OBJECT,
        ]

    def test_the_transfer_still_completes_under_the_safety_constraint(
        self, two_theory_statechart
    ):
        executor, _safety_slot, _speed_cap, receiving_cup = two_theory_statechart
        executor.tick_until_end(timeout=4000)
        assert receiving_cup.fill_level >= REQUESTED_FILL_LEVEL
