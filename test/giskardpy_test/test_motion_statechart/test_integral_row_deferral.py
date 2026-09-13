"""
Reproduction of an inequality constraint failing to hold its bound during a motion.

Self-contained: it builds its own scene from the ``tracy_world`` fixture. Run it with
``--runxfail`` to see the failure itself.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pytest

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.feature_functions import HeightGoal
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp import constraint_collection
from giskardpy.qp.enforcement_strategy import IntegralStrategy
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy, TracyJoint
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% scene constants

_TABLE_SURFACE_Z = 0.9
"""
Height, in metres, of the table surface the reference cup stands on.
"""

_CUP_HEIGHT = 0.15
"""
Height, in metres, of the box-shaped cups.
"""

_CUP_WIDTH = 0.07
"""
Side length, in metres, of the box-shaped cups.
"""

_RECEIVER_RIM_XY = (1.0, 0.1)
"""
Planar position, in metres, of the reference cup standing on the table.
"""

_JOINT_VELOCITY_LIMIT = 1.0
"""
Velocity limit, in radians per second, given to every joint of the robot, matching the
URDF limits the pouring demo runs at.
"""

_CONTROL_FREQUENCY = 80
"""
Control frequency, in hertz.
"""

_PREDICTION_HORIZON = 180
"""
Number of velocity blocks in the controller's prediction horizon.
"""

# %% disturbance and band constants

_WRIST_ROTATION = 2.0
"""
Angle, in radians, the wrist rotates the held cup by while the height goal guards it.
"""

_WRIST_SPEED = 0.3
"""
Speed, in radians per second, of the wrist rotation; slow enough for the rest of the arm
to compensate within its joint velocity limits.
"""

_MINIMUM_RIM_HEIGHT = 0.03
"""
Floor, in metres, on the held cup's rim height above the reference cup's rim.
"""

_RIM_HEIGHT_BAND = 0.05
"""
Width, in metres, of the band above :data:`_MINIMUM_RIM_HEIGHT` the rim may settle
within.
"""

_BAND_TOLERANCE = 1e-4
"""
How far below the floor a sample may lie before it counts as having left the band,
absorbing the integrator's own step.
"""

# %% scene


def _box_cup_body(name: str) -> Body:
    body = Body(name=PrefixedName(name))
    cup_shape = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=_CUP_HEIGHT / 2, reference_frame=body
        ),
        scale=Scale(_CUP_WIDTH, _CUP_WIDTH, _CUP_HEIGHT),
    )
    body.visual = ShapeCollection(shapes=[cup_shape])
    body.collision = ShapeCollection(shapes=[cup_shape])
    body.collision.reference_frame = body
    return body


@dataclass
class HeldCupScene:
    """
    Tracy holding an upright cup in its left gripper above a table with a reference cup
    on it.
    """

    world: World
    """
    The world holding the robot and the cups.
    """

    cup: Body
    """
    The cup fixed in the left gripper.
    """

    receiver_rim: np.ndarray
    """
    Centre of the reference cup's rim, in the world root frame.
    """

    def rim_point(self) -> Point3:
        """
        Centre of the held cup's rim, in the cup's frame.
        """
        return Point3(0.0, 0.0, _CUP_HEIGHT, reference_frame=self.cup)

    def rim_height(self) -> float:
        """
        Height, in metres, of the held cup's rim centre above the reference rim.
        """
        rim = self.world.transform(
            target_frame=self.world.root, spatial_object=self.rim_point()
        ).to_np()[:3]
        return float(rim[2] - self.receiver_rim[2])


@pytest.fixture(scope="function")
def held_cup_scene(tracy_world) -> HeldCupScene:
    """
    Tracy with every joint limited to :data:`_JOINT_VELOCITY_LIMIT`, both arms parked,
    the left gripper moved to an upright pose above the table and a cup fixed in it,
    slightly tilted by a wrist offset.
    """
    world = deepcopy(tracy_world)
    [tracy] = world.get_semantic_annotations_by_type(Tracy)
    for connection in tracy.connections:
        if isinstance(connection, ActiveConnection1DOF):
            connection.raw_dof.limits.lower.velocity = -_JOINT_VELOCITY_LIMIT
            connection.raw_dof.limits.upper.velocity = _JOINT_VELOCITY_LIMIT
    for arm in (tracy.left_arm, tracy.right_arm):
        park = arm.get_joint_state_by_type(StaticJointState.PARK)
        JointState.from_mapping(dict(park.items())).apply_to(world)

    left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")
    upright_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=1,
        pos_y=0.2,
        pos_z=_TABLE_SURFACE_Z + 0.3,
        quat_z=0.5,
        quat_x=0.5,
        quat_y=0.5,
        quat_w=0.5,
        reference_frame=world.root,
    ).to_pose()
    approach = MotionStatechart()
    approach_task = CartesianPose(
        root_link=world.root, tip_link=left_tool_frame, goal_pose=upright_pose
    )
    approach.add_node(approach_task)
    approach.add_node(EndMotion.when_true(approach_task))
    approach_executor = Executor(
        MotionStatechartContext(world=world),
        pacer=SimulationPacer(real_time_factor=1),
    )
    approach_executor.compile(motion_statechart=approach)
    approach_executor.tick_until_end(timeout=1000)

    cup = _box_cup_body("held_cup")
    with world.modify_world():
        world.add_body(cup)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=left_tool_frame,
                child=cup,
                name=PrefixedName("l_gripper_T_held_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    roll=-math.pi / 2.0
                ),
            )
        )
    wrist = world.get_connection_by_name(TracyJoint.LEFT_WRIST_3)
    JointState.from_mapping({wrist: wrist.position + 0.1}).apply_to(world)

    receiver_rim = np.array([*_RECEIVER_RIM_XY, _TABLE_SURFACE_Z + _CUP_HEIGHT])
    return HeldCupScene(world=world, cup=cup, receiver_rim=receiver_rim)


# %% the reproduction


@dataclass
class RimHeightTrace:
    """
    The rim height sampled once per control tick, judged against its band.
    """

    lower: float
    """
    Floor of the band.
    """

    upper: float
    """
    Ceiling of the band.
    """

    samples: list[float] = field(default_factory=list)
    """
    The rim height per tick, in execution order.
    """

    def entry_index(self) -> int | None:
        """
        Index of the first sample inside the band, or ``None`` if never reached.
        """
        for index, sample in enumerate(self.samples):
            if self.lower <= sample <= self.upper:
                return index
        return None

    def worst_excursion_after_entry(self) -> float:
        """
        How far, in metres, the rim sank below the floor after the band was entered, or
        ``0.0`` if it never did.
        """
        entry = self.entry_index()
        if entry is None:
            return 0.0
        return max(
            0.0, *(self.lower - _BAND_TOLERANCE - s for s in self.samples[entry:])
        )


class TestHeightGoalWhileTheWristRotatesTheCup:
    """
    A height goal keeps the held cup's rim inside a band above the reference rim.

    Once the rim is inside, a joint goal rotates the wrist holding the cup by two
    radians, which alone would drop the rim far below the floor; the rest of the arm has
    to compensate. Once inside the band, the rim must never leave it again.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="An integral row over the whole prediction horizon is satisfied by a plan "
        "that defers the correction past the step it executes, so the rim sinks below "
        "its floor while the row reports no violation. PredictedValueStrategy replaces "
        "this row; the test keeps the original behaviour reproducible.",
    )
    def test_integral_row_lets_the_rim_sink_below_its_floor(
        self, held_cup_scene, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            constraint_collection, "INEQUALITY_ENFORCEMENT_STRATEGY", IntegralStrategy
        )
        scene = held_cup_scene
        guard = HeightGoal(
            root_link=scene.world.root,
            tip_link=scene.cup,
            tip_point=scene.rim_point(),
            reference_point=Point3(
                *scene.receiver_rim, reference_frame=scene.world.root
            ),
            lower_limit=_MINIMUM_RIM_HEIGHT,
            upper_limit=_MINIMUM_RIM_HEIGHT + _RIM_HEIGHT_BAND,
            weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
        )
        wrist = scene.world.get_connection_by_name(TracyJoint.LEFT_WRIST_3)
        wrist_goal = JointPositionList(
            goal_state=JointState.from_mapping(
                {wrist: wrist.position + _WRIST_ROTATION}
            ),
            max_velocity=_WRIST_SPEED,
        )
        wrist_goal.start_condition = guard.observation_variable

        trace = RimHeightTrace(lower=guard.lower_limit, upper=guard.upper_limit)
        original_on_tick = guard.on_tick

        def recording_on_tick(context):
            trace.samples.append(scene.rim_height())
            return original_on_tick(context)

        guard.on_tick = recording_on_tick

        motion = Parallel([wrist_goal, guard])
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(motion)
        motion_statechart.add_node(EndMotion.when_true(motion))
        executor = Executor(
            MotionStatechartContext(
                world=scene.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=_CONTROL_FREQUENCY,
                    prediction_horizon=_PREDICTION_HORIZON,
                ),
            ),
            pacer=SimulationPacer(real_time_factor=1),
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end(timeout=4000)

        assert wrist_goal.observation_state == ObservationStateValues.TRUE
        assert trace.entry_index() is not None, "the rim never reached its band"
        assert trace.worst_excursion_after_entry() == 0.0, (
            "the rim sank below its floor after entering the band, by "
            f"{trace.worst_excursion_after_entry() * 1000:.2f} mm"
        )
