from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

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
from giskardpy.motion_statechart.tasks.feature_functions import (
    AngleGoal,
    DistanceGoal,
    FeatureFunctionGoal,
    HeightGoal,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy, TracyJoint
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
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

_CONTROL_FREQUENCY = 80
"""
Control frequency, in hertz, the motions run at.
"""

_PREDICTION_HORIZON = 180
"""
Number of velocity blocks in the controller's prediction horizon.
"""

_JOINT_VELOCITY_LIMIT = 1.0
"""
Velocity limit, in radians per second, given to every joint of the robot.

Tracy tightens its joints to 0.2 rad/s by default; the limit used here matches the URDF
limits the pouring demo runs at, under which the deferred excursion is an order of
magnitude larger.
"""

_BAND_TOLERANCE = 1e-4
"""
How far outside a band a sample may lie before it counts as having left it, absorbing
the integrator's own step.
"""

# %% disturbance and band constants

_WRIST_ROTATION = 2.0
"""
Angle, in radians, the wrist rotates the held cup by while a feature goal guards it.
"""

_WRIST_SPEED = 0.3
"""
Speed, in radians per second, of the wrist rotation.

Slow enough that the rest of the arm can compensate for it within its joint velocity
limits, so a guarded quantity leaving its band is the optimizer's choice rather than a
saturation the guard could not have prevented.
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

_MINIMUM_RIM_DISTANCE = 0.15
"""
Floor, in metres, on the planar distance between the two cups' rim centres.
"""

_RIM_DISTANCE_BAND = 0.05
"""
Width, in metres, of the band above :data:`_MINIMUM_RIM_DISTANCE` the rims may settle
within.
"""

_MINIMUM_TILT = 0.3
"""
Floor, in radians, on the held cup's tilt away from upright.
"""

_TILT_BAND = 0.2
"""
Width, in radians, of the band above :data:`_MINIMUM_TILT` the tilt may settle within.
"""

# %% scene


def _box_cup_body(name: str) -> Body:
    """
    Create a box-shaped cup body whose bounding box spans ``[0, _CUP_HEIGHT]`` in z.
    """
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

    def cup_up(self) -> Vector3:
        """
        The held cup's up direction, in the cup's frame.
        """
        return Vector3.Z(reference_frame=self.cup)

    def root_point(self, point: Point3) -> np.ndarray:
        """
        The point's coordinates in the world root frame.
        """
        return self.world.transform(
            target_frame=self.world.root, spatial_object=point
        ).to_np()[:3]

    def rim_height(self) -> float:
        """
        Height, in metres, of the held cup's rim centre above the reference rim.
        """
        return float(self.root_point(self.rim_point())[2] - self.receiver_rim[2])

    def rim_distance(self) -> float:
        """
        Planar distance, in metres, between the held cup's rim centre and the reference
        rim.
        """
        return float(
            np.linalg.norm((self.root_point(self.rim_point()) - self.receiver_rim)[:2])
        )

    def tilt(self) -> float:
        """
        Angle, in radians, between the held cup's up direction and the world's.
        """
        root_up = self.world.transform(
            target_frame=self.world.root, spatial_object=self.cup_up()
        ).to_np()[:3]
        return float(
            math.acos(np.clip(root_up[2] / np.linalg.norm(root_up), -1.0, 1.0))
        )


def _controller_context(world: World) -> MotionStatechartContext:
    """
    Build a context whose controller runs at :data:`_CONTROL_FREQUENCY` over
    :data:`_PREDICTION_HORIZON` blocks.
    """
    return MotionStatechartContext(
        world=world,
        qp_controller_config=QPControllerConfig(
            target_frequency=_CONTROL_FREQUENCY,
            prediction_horizon=_PREDICTION_HORIZON,
        ),
    )


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


# %% band traces


@dataclass
class BandTrace:
    """
    Samples of a bounded quantity over a motion, judged against the band the motion was
    asked to keep it in.

    A motion may start outside the band and be brought into it; the safety contract is
    that once inside, the quantity never leaves again.
    """

    lower: float
    """
    Lower end of the band.
    """

    upper: float
    """
    Upper end of the band.
    """

    tolerance: float = _BAND_TOLERANCE
    """
    How far outside the band a sample may lie before it counts as having left it.
    """

    samples: list[float] = field(default_factory=list)
    """
    The quantity, sampled once per control tick in execution order.
    """

    def entry_index(self) -> int | None:
        """
        Index of the first sample inside the band, or ``None`` if the band was never
        reached.
        """
        for index, sample in enumerate(self.samples):
            if self.lower <= sample <= self.upper:
                return index
        return None

    def worst_excursion_after_entry(self) -> float:
        """
        The farthest, in the quantity's unit, any sample after the band was entered lies
        outside the band beyond the tolerance, or ``0.0`` if none does.
        """
        entry = self.entry_index()
        if entry is None:
            return 0.0
        excursions = (
            max(
                self.lower - self.tolerance - sample,
                sample - self.upper - self.tolerance,
            )
            for sample in self.samples[entry:]
        )
        return max(0.0, *excursions)


# %% feature goals guarding a held cup under a wrist rotation


@dataclass
class GuardedWristRotation:
    """
    A wrist rotation of a held cup run against one feature goal that bounds a quantity
    the rotation disturbs.

    The rotation is an equality goal on a single wrist joint; the guard is an inequality
    goal whose band the quantity is first brought into. The rotation starts only once
    the guard observes the quantity inside its band, so the disturbance acts entirely
    while the guard is meant to hold, and the rest of the arm must compensate for it.
    """

    scene: HeldCupScene
    """
    The scene the motion runs in.
    """

    guard: FeatureFunctionGoal
    """
    The feature goal bounding the disturbed quantity.
    """

    sample_quantity: Callable[[], float]
    """
    Reads the guarded quantity off the scene, in the unit the guard's limits use.
    """

    wrist_goal: JointPositionList = field(init=False)
    """
    The joint goal rotating the wrist that holds the cup.
    """

    def __post_init__(self) -> None:
        wrist = self.scene.world.get_connection_by_name(TracyJoint.LEFT_WRIST_3)
        self.wrist_goal = JointPositionList(
            goal_state=JointState.from_mapping(
                {wrist: wrist.position + _WRIST_ROTATION}
            ),
            max_velocity=_WRIST_SPEED,
        )

    def execute(self, trace: BandTrace) -> None:
        """
        Run the rotation to completion in simulation, sampling the guarded quantity on
        every tick of the guard.

        :param trace: The trace the samples are appended to.
        """
        original_on_tick = self.guard.on_tick

        def recording_on_tick(context):
            trace.samples.append(self.sample_quantity())
            return original_on_tick(context)

        self.guard.on_tick = recording_on_tick

        self.wrist_goal.start_condition = self.guard.observation_variable
        motion = Parallel([self.wrist_goal, self.guard])
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(motion)
        motion_statechart.add_node(EndMotion.when_true(motion))
        executor = Executor(
            _controller_context(self.scene.world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end(timeout=4000)


class TestFeatureGoalGuardsHeldCupWhileWristRotates:
    """
    A feature goal bounding a quantity of a held cup brings that quantity into its band
    and keeps it there while a joint goal, started once the band is reached, rotates the
    wrist holding the cup, which on its own would push the quantity out of the band.

    Only integral rows take part: the bounded quantity is guarded by the same constraint
    form every threshold in the system uses.
    """

    _DEFERRAL_REASON = (
        "The guard's row is an integral over the whole prediction horizon, so the "
        "optimizer satisfies it with a plan that defers the correction past the step "
        "it executes. The quantity leaves its band while the row reports no violation."
    )

    @pytest.mark.xfail(strict=True, reason=_DEFERRAL_REASON)
    def test_height_goal_keeps_the_rim_in_its_band(self, held_cup_scene) -> None:
        """
        The rim, once brought down from well above its band into it, never leaves the
        band while the wrist rotates the cup.
        """
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
        rotation = GuardedWristRotation(
            scene=scene, guard=guard, sample_quantity=scene.rim_height
        )
        trace = BandTrace(lower=guard.lower_limit, upper=guard.upper_limit)

        rotation.execute(trace)

        assert rotation.wrist_goal.observation_state == ObservationStateValues.TRUE
        assert trace.entry_index() is not None, "the rim never reached its band"
        assert trace.worst_excursion_after_entry() == 0.0, (
            "the rim left its height band after entering it, by "
            f"{trace.worst_excursion_after_entry() * 1000:.2f} mm"
        )

    @pytest.mark.xfail(
        strict=True,
        raises=TimeoutError,
        reason="Besides its band row, the distance goal adds a zero-target row per axis "
        "of the rim-to-rim vector, which resists any motion of the rim. The wrist goal "
        "is outweighed by them and never reaches its target.",
    )
    def test_distance_goal_keeps_the_rims_apart(self, held_cup_scene) -> None:
        """
        The planar rim-to-rim distance, once grown from below its band into it, never
        leaves the band while the wrist rotates the cup toward the reference cup.
        """
        scene = held_cup_scene
        guard = DistanceGoal(
            root_link=scene.world.root,
            tip_link=scene.cup,
            tip_point=scene.rim_point(),
            reference_point=Point3(
                *scene.receiver_rim, reference_frame=scene.world.root
            ),
            lower_limit=_MINIMUM_RIM_DISTANCE,
            upper_limit=_MINIMUM_RIM_DISTANCE + _RIM_DISTANCE_BAND,
            weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
        )
        rotation = GuardedWristRotation(
            scene=scene, guard=guard, sample_quantity=scene.rim_distance
        )
        trace = BandTrace(lower=guard.lower_limit, upper=guard.upper_limit)

        rotation.execute(trace)

        assert rotation.wrist_goal.observation_state == ObservationStateValues.TRUE
        assert trace.entry_index() is not None, "the rims never reached their band"
        assert trace.worst_excursion_after_entry() == 0.0, (
            "the rim distance left its band after entering it, by "
            f"{trace.worst_excursion_after_entry() * 1000:.2f} mm"
        )

    @pytest.mark.xfail(strict=True, reason=_DEFERRAL_REASON)
    def test_angle_goal_keeps_the_tilt_in_its_band(self, held_cup_scene) -> None:
        """
        The cup's tilt away from upright, once grown from below its band into it, never
        leaves the band while the wrist rotates the cup further over.
        """
        scene = held_cup_scene
        guard = AngleGoal(
            root_link=scene.world.root,
            tip_link=scene.cup,
            tip_vector=scene.cup_up(),
            reference_vector=Vector3.Z(reference_frame=scene.world.root),
            lower_angle=_MINIMUM_TILT,
            upper_angle=_MINIMUM_TILT + _TILT_BAND,
            weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
        )
        rotation = GuardedWristRotation(
            scene=scene, guard=guard, sample_quantity=scene.tilt
        )
        trace = BandTrace(lower=guard.lower_angle, upper=guard.upper_angle)

        rotation.execute(trace)

        assert rotation.wrist_goal.observation_state == ObservationStateValues.TRUE
        assert trace.entry_index() is not None, "the tilt never reached its band"
        assert trace.worst_excursion_after_entry() == 0.0, (
            "the tilt left its band after entering it, by "
            f"{trace.worst_excursion_after_entry():.4f} rad"
        )
