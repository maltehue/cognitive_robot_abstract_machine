"""
The closed-loop study: Tracy pours the source cup's grains into a receiver under the
pouring tasks, planning with a pouring model whose parameters may be wrong, and closing
the loop on what it measures to a varying degree.

The scene and the motion are those of ``demo_pouring_transfer_particles.py``, with the
model's parameters and the loop left open. Once the transfer motion ends the cup is
turned back upright and the grains are left to land, so what is counted is what the pour
really delivered rather than a snapshot of a cup still pouring.
"""

# %% imports

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from datetime import timedelta
from enum import StrEnum

import numpy
import pandas
from typing_extensions import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Self,
    Sequence,
    Tuple,
)

from experiments.pouring_model_verification.parameters import (
    ModelParameters,
    Perturbation,
)
from experiments.pouring_model_verification.inflow_model import (
    FirstOrderInflowModel,
)
from experiments.pouring_model_verification.concept_revision import (
    ConceptAwareRevision,
    Diagnosis,
    DischargeRevision,
    Launch,
    SourceWatch,
)
from experiments.pouring_model_verification.consistency import (
    ConsistencyCheck,
    Feature,
    RateResiduals,
)
from experiments.pouring_model_verification.hypothesis_bank import (
    Hypothesis,
    BankSettings,
    DrainGeometry,
    FreeParameter,
    HypothesisBank,
    DrainRateSpread,
    OnsetSpread,
    ModelPoint,
    Reading,
    StreamGeometry,
)
from experiments.pouring_model_verification.onset_rule import (
    ArrivalWatch,
    OnsetOverrideSwitch,
    OnsetResponder,
    OnsetResponse,
    OnsetRevision,
    OnsetRule,
    Revision,
    StallWatch,
    TiltOverride,
)
from experiments.pouring_model_verification.pid import (
    FillReached,
    NoCompensation,
    PidFillController,
    PidGains,
    PidVariant,
    RateProjection,
    ReadingCompensation,
    SmithPrediction,
    TiltTowards,
)
from experiments.pouring_model_verification.recovery import (
    NORMAL_TILT_RATE,
    BankCoupling,
    DrainCorrectionValues,
    CautionSettings,
    CautiousStart,
    hypotheses_for,
    sensor_knowledge,
)
from experiments.pouring_model_verification.poured_stream import PouredStream
from experiments.pouring_model_verification.second_lip import SecondLipPouringEquation
from experiments.pouring_model_verification.records import (
    Outcome,
    Phase,
    Sample,
    Study,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.scene import (
    PACKED_SHARE,
    PARTICLE_RADIUS,
    SETTLE_TIME,
    PourableContainer,
    Scene,
    SourceCup,
    SourceInWorld,
    settle_time_of,
    stepped_simulation,
)
from experiments.pouring_model_verification.sensor import (
    DELAY_TOLERANCE,
    CameraCondition,
    CrossingLog,
    SensorChannel,
    SensorCondition,
    SensorSet,
    WristSensor,
)
from giskardpy.executor import Executor, SteppedSimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import EndMotion, NodeArtifacts, Task
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
)
from giskardpy.qp.exceptions import InfeasibleException
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.symbolic_math.float_variable_data import FloatVariableData
import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.drain_calibration import CalibratedDrainScale
from semantic_digital_twin.physics.equations.pouring_equations import (
    STANDARD_GRAVITY,
)
from semantic_digital_twin.physics.particles import (
    MeasuredCommittedFillLevel,
    MeasuredFillLevel,
    ParticleFill,
)
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactParameters,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the scene

GOAL_GRAINS = 41.6
"""
How many grains the receiver is to hold, whichever receiver it is: what the demo's goal
fill level of 0.06 amounts to in its wide receiver.

Each receiver's goal fill level is this count's share of that receiver's capacity.
"""

FILL_TOLERANCE_GRAINS = 7.0
"""
How many grains either side of the goal count as reached: what the demo's fill tolerance
of 0.01 amounts to in its wide receiver.
"""

RECEIVER_STAND = (1.0, 0.1)
"""
Where the receiving cup stands on the table, as x and y in metres.
"""

CARRY_STAND = (1.0, 0.2, 0.3)
"""
Where the source cup is carried, as x and y in metres and a height above the table.
"""

CARRY_ORIENTATION = (0.5, 0.5, 0.5, 0.5)
"""
The orientation the hand carries the cup upright in, as a quaternion in x, y, z, w.
"""

POUR_START_TILT_MARGIN = 0.05
"""
How far past the model's onset the cup is carried before the transfer, in radians.
"""

WRIST_NUDGE = 0.1
"""
How far the wrist is turned out of the carry pose, in radians, so the pour does not
start from a singular wrist.
"""

GRASP_ROLL = -math.pi / 2
"""
How the source cup sits in the gripper, as a roll about the tool's own x axis.
"""

MINIMUM_RIM_CLEARANCE = 0.1
"""
How far the source's rim is kept above the receiver's, in metres.
"""

# %% the controller

CONTROL_FREQUENCY = 80
"""
How often the controller ticks, in hertz.
"""

PREDICTION_HORIZON = 20
"""
How many control cycles the quadratic program looks ahead.
"""

FILL_PREDICTION_WINDOW = 3.0
"""
How far ahead the fill row predicts, in seconds.

Held fixed across the study: finding 11 of the evaluation note shows the overshoot moves
with it, so varying it alongside a model parameter would confound the two.
"""

TRANSFER_REFERENCE_VELOCITY = 0.03
"""
How fast the fill task asks its fill level to change, per second.
"""

PERCEPTION_FREQUENCY = 10
"""
How often the fill levels are measured and reported, in hertz.
"""

CALIBRATION_SMOOTHING = 0.3
"""
How far towards each measured inflow the drain's factor moves.
"""

TICK_LIMIT = 4000
"""
How many control cycles a motion is given before it counts as timed out.
"""

SAMPLE_EVERY = 4
"""
How many control cycles pass between two samples.
"""

LANDING_TIME = timedelta(seconds=2)
"""
How long the grains are given to land once the cup is upright again.
"""

MAXIMUM_PID_TILT = 2.4
"""
The largest tilt the PID baseline commands, in radians: past the furthest the model-
predictive controller turned the cup in any trial.
"""


class FillController(StrEnum):
    """
    What decides how far the cup is tilted during the transfer.
    """

    MODEL_PREDICTIVE = "model_predictive"
    """
    The fill row: the pouring model's prediction of the receiver's fill at the end of a
    window is driven to the goal.
    """

    PID = "pid"
    """
    A PID controller on the grains the receiver is missing, with no model of the pour;
    the cup is aimed by the same tasks as under :attr:`MODEL_PREDICTIVE`.
    """

    PID_FIXED_POSE = "pid_fixed_pose"
    """
    As :attr:`PID`, with the cup held still over the receiver instead of aimed: no model
    takes part at all.
    """

    @property
    def uses_pouring_model(self) -> bool:
        """
        :return: Whether the tilt is decided by the pouring model.
        """
        return self is FillController.MODEL_PREDICTIVE

    @property
    def aims(self) -> bool:
        """
        :return: Whether the cup is moved to keep the stream landing in the receiver.
        """
        return self is not FillController.PID_FIXED_POSE


class MissingPidGainsError(Exception):
    """
    Raised when a trial is to be run under a PID controller without gains for it.
    """

    def __init__(self, controller: FillController) -> None:
        super().__init__(f"the {controller} controller needs gains to run with")


class KeptReposeAngleWithoutRevisionError(Exception):
    """
    Raised for a pour told to start from a kept repose angle without a revision whose
    correction could hold it.
    """

    def __init__(self) -> None:
        super().__init__("a kept repose angle needs the revision to hold it")


class ConceptAwareRevisionWithoutWristError(Exception):
    """
    Raised for a concept-aware revision without a wrist sensor weighing the source,
    which is what tells a stall the onset explains from one the aim explains.
    """

    def __init__(self) -> None:
        super().__init__("a concept-aware revision needs the source weighed")


class OnsetRuleWithoutModelError(Exception):
    """
    Raised when the onset rule is asked of a controller that has no pouring model to
    correct or override.
    """

    def __init__(self, controller: FillController) -> None:
        super().__init__(f"the onset rule needs the pouring model, not {controller}")


class MissingInflowModelForSmithError(Exception):
    """
    Raised when a Smith predictor is to be run without the inflow model it predicts
    with.
    """

    def __init__(self) -> None:
        super().__init__("a Smith predictor needs an identified inflow model")


class PidWithoutFillFeedbackError(Exception):
    """
    Raised when a trial is to be run under a PID controller in a loop mode that reports
    it nothing to act on.
    """

    def __init__(self, loop_mode: LoopMode) -> None:
        super().__init__(f"a PID controller reads no fill level under {loop_mode}")


class BankWithoutSensorSetError(Exception):
    """
    Raised when a pour weighs a hypothesis bank but reads no sensor set, or reads the
    receiver or the source with sensors other than the set's.
    """

    def __init__(self) -> None:
        super().__init__(
            "A pour that weighs a hypothesis bank reads the receiver and the source with "
            "the sensors of one sensor set."
        )


class CautionWithoutBankError(Exception):
    """
    Raised when a pour starts cautiously without a hypothesis bank, which is what
    confirms the aim and releases the caution.
    """

    def __init__(self) -> None:
        super().__init__(
            "A cautious start is released by the bank confirming the aim, so it needs a "
            "hypothesis bank."
        )


class UnweighedReceiverWithoutBankError(Exception):
    """
    Raised when nothing weighs the receiver and no hypothesis bank is there to estimate
    what it holds.
    """

    def __init__(self) -> None:
        super().__init__(
            "With nothing under the receiver, the controller learns what it holds only "
            "from a hypothesis bank's estimate, so it needs one."
        )


class OnsetSpreadMissingError(Exception):
    """
    Raised when a bank is to carry the twin's spread of the repose angle but the twin
    holds none for the cup.
    """

    def __init__(self) -> None:
        super().__init__(
            "A bank that adapts the onset needs the twin's repose angle while turning "
            "and its spread."
        )


class DrainRateSpreadMissingError(Exception):
    """
    Raised when a bank is to carry the twin's spread of the drain rate but the twin
    holds none for the cup.
    """

    def __init__(self) -> None:
        super().__init__(
            "A bank that adapts the drain rate needs the twin's drain rate while "
            "turning and its spread."
        )


UNCAPPED_TILT_RATE = 10.0
"""
The tilt speed limit while the cap is released, in radians per second: far above any
speed a pour turns at.
"""


class PourFault(StrEnum):
    """
    A fault in the pour that no concept of the pouring model covers.
    """

    ROUGH_GRAINS = "rough_grains"
    """
    Grains that resist rolling and slide stiffly, like angular grains, so they jam and
    flow in avalanches rather than in a stream that grows with the head.
    """

    LIP_BAFFLE = "lip_baffle"
    """
    A thin plate standing inside the cup at the lip it pours over, blocking the middle
    of the lip, so the grains leave only around its sides: the outflow is restricted
    while the lip, and so where the pour starts and where it lands, stay as they were.
    """

    @property
    def contact(self) -> Optional[ContactParameters]:
        """
        :return: What the faulty grains' surface does in a contact, or ``None`` for
            sound grains.
        """
        if self is not PourFault.ROUGH_GRAINS:
            return None
        settling = ParticleFill.settling_contact()
        return ContactParameters(
            friction=ContactFriction(sliding=1.2, torsional=0.05, rolling=0.02),
            stiffness=settling.stiffness,
        )

    def obstacle(self, source: SourceCup) -> Optional[Box]:
        """
        :param source: The cup poured from.
        :return: What stands in the grains' way, in the cup's frame, or ``None``.
        """
        if self is not PourFault.LIP_BAFFLE:
            return None
        geometry = source.geometry
        return Box(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=source.pour_direction
                * (geometry.inner_radius - BAFFLE_THICKNESS / 2),
                z=geometry.height + (BAFFLE_RISE - BAFFLE_DEPTH) / 2,
            ),
            scale=Scale(
                BAFFLE_THICKNESS,
                BAFFLE_SHARE_OF_LIP * 2 * geometry.inner_radius,
                BAFFLE_DEPTH + BAFFLE_RISE,
            ),
        )


BAFFLE_THICKNESS = 0.003
"""
How thick the baffle at the lip is, in metres.
"""

BAFFLE_SHARE_OF_LIP = 0.5
"""
How much of the cup's inner diameter the baffle spans across the lip.
"""

BAFFLE_DEPTH = 0.02
"""
How far below the rim the baffle reaches into the cup, in metres.
"""

BAFFLE_RISE = 0.01
"""
How far above the rim the baffle stands, in metres.
"""


class SequenceKey(StrEnum):
    """
    A key of a pour's place in a sequence written to JSON.
    """

    POUR = "pour"
    SEQUENCE_SEED = "sequence_seed"
    KEPT_REPOSE_ANGLE = "kept_repose_angle"
    KEPT_MODEL = "kept_model"
    KEPT_ONSET = "kept_onset"


@dataclass(frozen=True)
class SequencePosition:
    """
    Where a pour stands in a sequence of pours of one cup, each refilled and settled
    anew.
    """

    pour: int
    """
    Which pour of the sequence it is, counted from one.
    """

    sequence_seed: int
    """
    The microstate the sequence's first pour starts from, which names the sequence.
    """

    kept_repose_angle: Optional[float] = None
    """
    The repose angle the previous pour ended with, which this one starts from, in
    radians, or ``None`` to start from the model's own.
    """

    kept_model: Optional[ModelPoint] = None
    """
    The model a hypothesis bank wrote back after the previous pour, which this one
    starts from, or ``None`` to start from the model as given.
    """

    kept_onset: Optional[OnsetSpread] = None
    """
    The twin's spread of the repose angle as the pours before this one wrote it back, or
    ``None`` for the twin's own.
    """

    @property
    def label(self) -> str:
        """
        :return: A short identifier of the place, usable in a file name.
        """
        kept = (
            ""
            if self.kept_repose_angle is None
            and self.kept_model is None
            and self.kept_onset is None
            else "_kept"
        )
        return f"_pour{self.pour}{kept}"

    def to_json(self) -> Dict[str, Any]:
        return {
            SequenceKey.POUR: self.pour,
            SequenceKey.SEQUENCE_SEED: self.sequence_seed,
            SequenceKey.KEPT_REPOSE_ANGLE: self.kept_repose_angle,
            SequenceKey.KEPT_MODEL: (
                None if self.kept_model is None else self.kept_model.to_json()
            ),
            SequenceKey.KEPT_ONSET: (
                None if self.kept_onset is None else self.kept_onset.to_json()
            ),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> SequencePosition:
        kept_model = data.get(SequenceKey.KEPT_MODEL)
        kept_onset = data.get(SequenceKey.KEPT_ONSET)
        return cls(
            pour=data[SequenceKey.POUR],
            sequence_seed=data[SequenceKey.SEQUENCE_SEED],
            kept_repose_angle=data[SequenceKey.KEPT_REPOSE_ANGLE],
            kept_model=None if kept_model is None else ModelPoint.from_json(kept_model),
            kept_onset=(
                None if kept_onset is None else OnsetSpread.from_json(kept_onset)
            ),
        )


class LoopMode(StrEnum):
    """
    How much of what is measured the controller gets to act on.
    """

    OPEN_LOOP = "open_loop"
    """
    The fill levels are measured once, before the pour, and then integrated from the
    model alone.
    """

    FILL_FEEDBACK = "fill_feedback"
    """
    Both fill levels are measured and reported into the controller's model as the pour
    runs; the receiver's counting what is already falling towards it.
    """

    FILL_FEEDBACK_AND_CALIBRATION = "fill_feedback_and_calibration"
    """
    As :attr:`FILL_FEEDBACK`, and the measured inflow also rescales the source's drain.
    """

    @property
    def reports_fill_levels(self) -> bool:
        """
        :return: Whether the measured fill levels reach the controller during the pour.
        """
        return self is not LoopMode.OPEN_LOOP

    @property
    def calibrates_drain(self) -> bool:
        """
        :return: Whether the measured inflow rescales the drain.
        """
        return self is LoopMode.FILL_FEEDBACK_AND_CALIBRATION


@dataclass
class TransferScene:
    """
    Everything a run of the transfer needs to reach.
    """

    world: World
    """
    The world the robot and the cups stand in.
    """

    robot: Tracy
    """
    The robot holding the source cup.
    """

    cup: SourceInWorld
    """
    The cup the robot pours from.
    """

    receiver: PourableContainer
    """
    The cup on the table the pour has to reach.
    """

    tool_frame: Body
    """
    The frame of the hand holding the source.
    """

    drain_scale: FloatVariable
    """
    The factor the source's drain is scaled by.
    """

    repose_correction: Optional[FloatVariable]
    """
    The angle the source's repose angle is corrected by, or ``None`` for a drain whose
    repose angle stays as fitted.
    """

    discharge_scale: Optional[FloatVariable]
    """
    The factor the source's discharge coefficient is corrected by, or ``None`` for a
    stream that stays as fitted.
    """

    variables: FloatVariableData
    """
    The one place the factor's value lives, shared by every controller of the run.
    """

    slot_height: Optional[FloatVariable] = None
    """
    The height of the slot edge the source may also pour over, or ``None`` for a drain
    with one lip.
    """

    slot_open: Optional[FloatVariable] = None
    """
    One while the source's drain pours over the slot edge as well, or ``None`` for a
    drain with one lip.
    """

    @property
    def source(self) -> HasFillLevel:
        """
        :return: The source cup's annotation.
        """
        return self.cup.annotation

    @classmethod
    def build(
        cls,
        parameters: ModelParameters,
        scene: Scene,
        revisable_repose: bool = False,
        revisable_discharge: bool = False,
        second_lip: bool = False,
        obstacle: Optional[Box] = None,
    ) -> Self:
        """
        Build Tracy with both arms parked, a source cup in its left hand and a receiving
        cup on the table, coupled so what leaves one enters the other.

        :param parameters: What the source's drain is evaluated with.
        :param scene: Which source cup and which receiver.
        :param revisable_repose: Whether the source's repose angle can be corrected
            while the controller runs.
        :param revisable_discharge: Whether the source's discharge coefficient can be
            corrected while the controller runs.
        :param second_lip: Whether the source's drain can be switched to pour over the
            slot edge of its wall as well, at a height that can be moved while the
            controller runs.
        :param obstacle: What stands in the grains' way, fixed to the source cup in its
            frame, or ``None``.
        :return: The scene.
        """
        world = World()
        with world.modify_world():
            world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
        robot = RobotSpecification(Tracy).spawn(world)
        for arm in robot.get_arms():
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        world.notify_state_change()

        tool_frame = robot.left_arm.end_effector.tool_frame
        cup = SourceInWorld.build(
            world=world,
            source=scene.source,
            parent=tool_frame,
            parent_T_cup=HomogeneousTransformationMatrix.from_xyz_rpy(
                roll=GRASP_ROLL, reference_frame=tool_frame
            ),
        )
        if obstacle is not None:
            blocker = Body.from_shape_collection(
                name=PrefixedName("source_obstacle"),
                shape_collection=ShapeCollection([obstacle]),
            )
            with world.modify_world():
                world.add_kinematic_structure_entity(blocker)
                world.add_connection(
                    FixedConnection(parent=cup.annotation.root, child=blocker)
                )
        receiver_body = scene.receiver.geometry.body(PrefixedName("receiving_cup"))
        with world.modify_world():
            world.add_kinematic_structure_entity(receiver_body)
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=receiver_body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=RECEIVER_STAND[0],
                        y=RECEIVER_STAND[1],
                        z=robot.table.top_z,
                        reference_frame=world.root,
                    ),
                )
            )

        source = cup.annotation
        receiver = PourableContainer(name=PrefixedName("receiver"), root=receiver_body)
        with world.modify_world():
            world.add_semantic_annotation(receiver)
        source.initialize_fill_level(world=world, initial_fill=PACKED_SHARE)
        receiver.initialize_fill_level(world=world, initial_fill=0.0)
        drain_scale = FloatVariable("source_drain_scale")
        variables = FloatVariableData()
        variables.register_expression(drain_scale)
        variables.set_value(drain_scale, 1.0)
        repose_correction = None
        if revisable_repose:
            repose_correction = FloatVariable("source_repose_correction")
            variables.register_expression(repose_correction)
            variables.set_value(repose_correction, 0.0)
        discharge_scale = None
        if revisable_discharge:
            discharge_scale = FloatVariable("source_discharge_scale")
            variables.register_expression(discharge_scale)
            variables.set_value(discharge_scale, 1.0)
        slot_height = slot_open = None
        equation = replace(
            parameters.applied_to(source.fill_equation),
            outflow_scale=drain_scale,
            repose_correction=repose_correction,
            discharge_scale=discharge_scale,
        )
        if second_lip:
            slot_height = FloatVariable("source_slot_height")
            slot_open = FloatVariable("source_slot_open")
            for variable, value in (
                (slot_height, equation.container_height),
                (slot_open, 0.0),
            ):
                variables.register_expression(variable)
                variables.set_value(variable, value)
            equation = SecondLipPouringEquation(
                **vars(equation),
                slot_offset=equation.container_width / 2,
                slot_height=slot_height,
                slot_open=slot_open,
            )
        with world.modify_world():
            source.add_fill_equation(equation)
        receiver.receive_outflow_from(source=source, world=world)
        return cls(
            world=world,
            robot=robot,
            cup=cup,
            receiver=receiver,
            tool_frame=tool_frame,
            drain_scale=drain_scale,
            repose_correction=repose_correction,
            discharge_scale=discharge_scale,
            variables=variables,
            slot_height=slot_height,
            slot_open=slot_open,
        )

    def run_on(
        self,
        point: ModelPoint,
        built: ModelPoint,
        cautious: bool,
        slot_open: Optional[float] = None,
    ) -> None:
        """
        Switch the source's drain to the parameters of a hypothesis, through the drain
        corrections its compiled expressions read from the world state.

        :param point: The parameters to pour with.
        :param built: The parameters the drain was built with.
        :param cautious: Whether the pour is finished slowly, taking the grains to flow
            faster than the model says.
        :param slot_open: How far the slot edge is switched on, between zero and one, or
            ``None`` for fully where the point has a slot edge and not at all otherwise.
        """
        variables = self.variables
        DrainCorrectionValues.between(point, built, cautious).write_to(
            self.world, self.source.fill_connection.drain_correction
        )
        if self.slot_open is not None:
            if slot_open is None:
                slot_open = 0.0 if point.slot_height is None else 1.0
            variables.set_value(self.slot_open, slot_open)
            if point.slot_height is not None:
                variables.set_value(self.slot_height, point.slot_height)

    @property
    def cup_tilt(self) -> float:
        """
        :return: How far the source cup is turned from upright, in radians.
        """
        cup_up = self.world.compute_forward_kinematics_np(
            self.world.root, self.source.root
        )[:3, 2]
        return math.acos(min(1.0, max(-1.0, float(cup_up[2]))))

    @property
    def receiver_opening_height(self) -> float:
        """
        :return: How high the receiver's opening stands in the world, in metres.
        """
        base = self.world.compute_forward_kinematics_np(
            self.world.root, self.receiver.root
        )[2, 3]
        return float(base) + self.receiver.root.collision.height

    @property
    def fall_time(self) -> float:
        """
        :return: How long a grain takes to fall from the source cup to the receiver's
            opening, from rest, in seconds.
        """
        source = self.world.compute_forward_kinematics_np(
            self.world.root, self.source.root
        )[2, 3]
        drop = max(0.0, float(source) - self.receiver_opening_height)
        return math.sqrt(2.0 * drop / STANDARD_GRAVITY)

    def nudge_wrist(self) -> None:
        """
        Turn the wrist out of the carry pose, so the pour does not start from a wrist
        that can only tilt one way.
        """
        wrist = self.world.get_connection_by_name("left_wrist_3_joint")
        JointState.from_mapping({wrist: wrist.position + WRIST_NUDGE}).apply_to(
            self.world
        )

    def pour_start_tilt(
        self, contents: ParticleFill, margin: float = POUR_START_TILT_MARGIN
    ) -> float:
        """
        :param contents: The grains standing in the source.
        :param margin: How far past the onset the cup is carried, in radians.
        :return: The tilt at which the model, with its repose angle as currently
            corrected, says those grains reach the lip, plus a margin, so the drain has
            flow and gradient from the start.
        """
        equation = replace(self.source.fill_equation.ungated(), repose_correction=None)
        depth = contents.filled_height_in(self.source.root)
        return (
            float(equation.onset_tilt(depth).evaluate()[0])
            + self.repose_correction_value
            + margin
        )

    @property
    def repose_correction_value(self) -> float:
        """
        :return: The angle the source's repose angle is currently corrected by, in
            radians: by the drain correction a hypothesis bank switches, and by the
            onset rule's revision where the drain has one.
        """
        corrected = self.source.fill_connection.drain_correction.repose_connection_in(
            self.world
        ).position
        if self.repose_correction is None:
            return corrected
        return corrected + float(self.variables.get_value(self.repose_correction))

    @property
    def repose_angle(self) -> float:
        """
        :return: The repose angle the source's drain currently pours with, in radians.
        """
        return self.source.fill_equation.repose_angle + self.repose_correction_value

    @property
    def discharge_coefficient(self) -> float:
        """
        :return: The discharge coefficient the source's stream currently leaves with.
        """
        coefficient = self.source.fill_equation.discharge_coefficient * math.exp(
            self.source.fill_connection.drain_correction.discharge_scale_connection_in(
                self.world
            ).position
        )
        if self.discharge_scale is None:
            return coefficient
        return coefficient * float(self.variables.get_value(self.discharge_scale))


# %% the motions


def carry_pose(scene: TransferScene) -> HomogeneousTransformationMatrix:
    """
    :param scene: The scene the cup is carried in.
    :return: Where the hand carries the cup upright, over the receiver.
    """
    return HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=CARRY_STAND[0],
        pos_y=CARRY_STAND[1],
        pos_z=scene.robot.table.top_z + CARRY_STAND[2],
        quat_x=CARRY_ORIENTATION[0],
        quat_y=CARRY_ORIENTATION[1],
        quat_z=CARRY_ORIENTATION[2],
        quat_w=CARRY_ORIENTATION[3],
        reference_frame=scene.world.root,
    )


def carry_motion(scene: TransferScene, tilt: float = 0.0) -> MotionStatechart:
    """
    :param scene: The scene the motion runs in.
    :param tilt: How far the cup is turned about its own pouring axis on the way, in
        radians.
    :return: The motion bringing the hand to the carry pose, ending there.
    """
    world_T_carry = carry_pose(scene)
    carry = CartesianPose(
        root_link=scene.world.root,
        tip_link=scene.tool_frame,
        goal_pose=(
            world_T_carry @ HomogeneousTransformationMatrix.from_xyz_rpy(yaw=tilt)
        ).to_pose(),
    )
    statechart = MotionStatechart()
    statechart.add_node(carry)
    statechart.add_node(EndMotion.when_true(carry))
    return statechart


@dataclass(eq=False, repr=False)
class TiltRateLimit(Task):
    """
    Keeps the source cup from turning faster than a limit read anew every cycle, so a
    rule outside the motion can tighten or release it while the motion runs.
    """

    source: HasFillLevel = field(kw_only=True)
    """
    The cup whose tilt is limited.
    """

    maximum_rate: FloatVariable = field(kw_only=True)
    """
    The fastest the tilt may change, in radians per second.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    How expensive it is to turn faster.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        :param context: The build context.
        :return: One row bounding the tilt's speed, never holding the motion open.
        """
        artifacts = NodeArtifacts(observation=sm.Scalar.const_true())
        artifacts.constraints.add_velocity_constraint(
            lower_velocity_limit=-self.maximum_rate,
            upper_velocity_limit=self.maximum_rate,
            quadratic_weight=self.weight,
            task_expression=self.source.pour_tilt_expression,
            velocity_limit=NORMAL_TILT_RATE,
            name=f"{self.name}_tilt_rate",
        )
        return artifacts


def transfer_motion(
    scene: TransferScene, grain_volume: float, extra: Sequence[Task] = ()
) -> MotionStatechart:
    """
    :param scene: The scene the motion runs in.
    :param grain_volume: The volume one grain takes up once poured, in cubic metres.
    :param extra: Further tasks run alongside the pouring tasks.
    :return: The motion pouring the source into the receiver, ending when the
        receiver's fill level has reached the goal of :data:`GOAL_GRAINS`.
    """
    capacity = scene.receiver.capacity
    transfer = FillByTransferTask(
        receiver=scene.receiver,
        goal_value=GOAL_GRAINS * grain_volume / capacity,
        fill_level_tolerance=FILL_TOLERANCE_GRAINS * grain_volume / capacity,
        reference_velocity=TRANSFER_REFERENCE_VELOCITY,
        prediction_duration=FILL_PREDICTION_WINDOW,
    )
    motion = Parallel(
        [transfer] + aiming_tasks(scene) + [pour_axis_level(scene)] + list(extra)
    )
    statechart = MotionStatechart()
    statechart.add_node(motion)
    statechart.add_node(EndMotion.when_true(motion))
    return statechart


def aiming_tasks(scene: TransferScene) -> List[Task]:
    """
    :param scene: The scene the motion runs in.
    :return: The tasks that keep the stream landing in the receiver and the source's rim
        above the receiver's.
    """
    return [
        KeepProjectileInReceiver(receiver=scene.receiver, source=scene.source),
        KeepSourceRimAboveReceiverRim(
            receiver=scene.receiver,
            source=scene.source,
            minimum_clearance=MINIMUM_RIM_CLEARANCE,
            clearance_band=0.02,
            weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
        ),
    ]


def pour_axis_level(scene: TransferScene) -> AlignPlanes:
    """
    :param scene: The scene the motion runs in.
    :return: The task keeping the axis the cup is poured about horizontal.
    """
    return AlignPlanes(
        root_link=scene.world.root,
        tip_link=scene.tool_frame,
        goal_normal=Vector3.X(reference_frame=scene.world.root),
        tip_normal=Vector3.Z(reference_frame=scene.tool_frame),
    )


def pid_transfer_motion(
    scene: TransferScene,
    controller: PidFillController,
    setpoint: FloatVariable,
    aims: bool,
) -> MotionStatechart:
    """
    :param scene: The scene the motion runs in.
    :param controller: The PID controller whose command the tilt follows.
    :param setpoint: Where that command is written for the motion to read.
    :param aims: Whether the cup is aimed at the receiver or held still over it.
    :return: The motion pouring the source into the receiver under the PID controller,
        ending once a reading has reached the goal.
    """
    tilt = TiltTowards(
        root_link=scene.world.root,
        tip_link=scene.tool_frame,
        root_R_carry=carry_pose(scene).to_rotation_matrix(),
        goal=setpoint,
    )
    placement = (
        aiming_tasks(scene)
        if aims
        else [
            CartesianPosition(
                root_link=scene.world.root,
                tip_link=scene.tool_frame,
                goal_point=carry_pose(scene).to_position(),
            )
        ]
    )
    reached = FillReached(controller=controller)
    statechart = MotionStatechart()
    statechart.add_node(Parallel([tilt] + placement + [pour_axis_level(scene)]))
    statechart.add_node(reached)
    statechart.add_node(EndMotion.when_true(reached))
    return statechart


# %% what the controller is told


@dataclass
class Perception:
    """
    What measures the pour and what of it reaches the controller, for one loop mode.
    """

    scene: TransferScene
    """
    The scene being measured.
    """

    contents: ParticleFill
    """
    The grains being poured.
    """

    loop_mode: LoopMode
    """
    How much of what is measured the controller acts on.
    """

    sensor: Optional[SensorCondition]
    """
    What reports the fill levels to the controller, or ``None`` if nothing weighs the
    receiver and the controller is told an estimate of it instead.
    """

    seed: int
    """
    Where the sensor's noise is drawn from.
    """

    wrist: Optional[WristSensor] = None
    """
    The force-torque sensor weighing the source at the wrist, or ``None`` to read the
    source with the same sensor as the receiver, or not at all where nothing weighs the
    receiver either.
    """

    source_level: MeasuredFillLevel = field(init=False)
    """
    The source's fill level, read off the grains standing in it.
    """

    receiver_level: MeasuredFillLevel = field(init=False)
    """
    The receiver's fill level, read off the grains that have landed in it.
    """

    committed_receiver_level: MeasuredCommittedFillLevel = field(init=False)
    """
    The receiver's fill level counting the grains already falling towards it.
    """

    calibration: CalibratedDrainScale = field(init=False)
    """
    What rescales the drain to the measured inflow.
    """

    source_channel: Optional[SensorChannel] = field(init=False)
    """
    The source's fill level as the wrist reports it, or ``None`` if nothing weighs the
    source once the pour has started.
    """

    receiver_channel: Optional[SensorChannel] = field(init=False)
    """
    The receiver's fill level as the sensor reports it, or ``None`` if nothing weighs
    the receiver.
    """

    initial_receiver_fill: float = field(init=False, default=0.0)
    """
    The receiver's fill level when the pour started, which an estimate of what the pour
    delivers is added to.
    """

    reported_source_fill: float = field(init=False, default=0.0)
    """
    The source's fill level the sensor last reported.
    """

    reported_receiver_fill: float = field(init=False, default=0.0)
    """
    The receiver's fill level the sensor last reported.
    """

    _previous_report: Optional[Tuple[float, float]] = field(
        init=False, default=None, repr=False
    )
    """
    The receiver's previously reported fill level, with the time it was reported.
    """

    stop_short: float = field(init=False, default=0.0)
    """
    How many grains short of the goal the controller is made to stop: the receiver is
    reported to it as holding that many more.
    """

    def __post_init__(self) -> None:
        scene = self.scene
        grain_volume = self.contents.bulk_volume_per_particle
        source_sensor = self.wrist if self.wrist is not None else self.sensor
        self.source_channel = (
            None
            if source_sensor is None
            else SensorChannel(
                model=source_sensor.model,
                grain_share=grain_volume / scene.source.capacity,
                random=numpy.random.default_rng([self.seed, 0]),
            )
        )
        self.receiver_channel = (
            None
            if self.sensor is None
            else SensorChannel(
                model=self.sensor.model,
                grain_share=grain_volume / scene.receiver.capacity,
                random=numpy.random.default_rng([self.seed, 1]),
            )
        )
        self.source_level = MeasuredFillLevel(
            contents=self.contents,
            container=scene.source.root,
            capacity=scene.source.capacity,
            connection=scene.source.fill_connection,
            world=scene.world,
            spout=scene.cup.spout,
        )
        self.receiver_level = MeasuredFillLevel(
            contents=self.contents,
            container=scene.receiver.root,
            capacity=scene.receiver.capacity,
            connection=scene.receiver.fill_connection,
            world=scene.world,
        )
        self.committed_receiver_level = MeasuredCommittedFillLevel(
            contents=self.contents,
            container=scene.receiver.root,
            capacity=scene.receiver.capacity,
            connection=scene.receiver.fill_connection,
            world=scene.world,
            source=scene.source.root,
            source_spout=scene.cup.spout,
            opening_height=scene.receiver_opening_height,
        )
        self.calibration = CalibratedDrainScale(
            scale=scene.drain_scale,
            inflow=scene.receiver.fill_connection.inflow_equation.symbolic_velocity(
                scene.receiver.fill_connection
            ),
            variables=scene.variables,
            smoothing=CALIBRATION_SMOOTHING,
        )

    def report_initial_levels(self) -> None:
        """
        Start both fill levels from what is measured, whatever the loop mode: an open
        loop still knows what it starts from.
        """
        self.source_level.report()
        self.committed_receiver_level.report()
        self.initial_receiver_fill = self.committed_receiver_level.measure()
        self.reported_receiver_fill = self.initial_receiver_fill
        self.reported_source_fill = self.source_level.measure()

    @property
    def predicted_receiver_inflow(self) -> float:
        """
        :return: How many grains per second the model predicts are arriving now.
        """
        return (
            self.calibration.predicted()
            * self.scene.receiver.capacity
            / self.contents.bulk_volume_per_particle
        )

    @property
    def reported_source_grains(self) -> float:
        """
        :return: How many grains the sensor last reported the source to hold.
        """
        return (
            self.reported_source_fill
            * self.scene.source.capacity
            / self.contents.bulk_volume_per_particle
        )

    @property
    def reported_receiver_grains(self) -> float:
        """
        :return: How many grains the sensor last reported the receiver to hold.
        """
        return (
            self.reported_receiver_fill
            * self.scene.receiver.capacity
            / self.contents.bulk_volume_per_particle
        )

    @property
    def committed_receiver_grains(self) -> float:
        """
        :return: How many grains the receiver truly holds or has on their way to it.
        """
        return (
            self.committed_receiver_level.measure()
            * self.scene.receiver.capacity
            / self.contents.bulk_volume_per_particle
        )

    def update(self, at: float) -> None:
        """
        Measure the pour and hand the controller what its loop mode lets it act on, as
        the sensor reports it.

        :param at: The time of this measurement, in seconds.
        """
        if not self.loop_mode.reports_fill_levels:
            return
        if self.source_channel is not None:
            source_fill = self.source_channel.read(self.source_level.measure(), at)
            self.reported_source_fill = source_fill
            JointState.from_mapping(
                {self.scene.source.fill_connection: source_fill}
            ).apply_to(self.scene.world)
        if self.receiver_channel is None:
            return
        receiver_level = (
            self.committed_receiver_level
            if self.sensor.model.counts_falling_grains
            else self.receiver_level
        )
        receiver_fill = self.receiver_channel.read(receiver_level.measure(), at)
        JointState.from_mapping(
            {
                self.scene.receiver.fill_connection: min(
                    1.0, receiver_fill + self._shortfall
                ),
            }
        ).apply_to(self.scene.world)
        self.reported_receiver_fill = receiver_fill
        measured_inflow = self._reported_inflow(receiver_fill, at)
        if self.loop_mode.calibrates_drain:
            self.calibration.calibrate(measured_inflow=measured_inflow)

    @property
    def weighs_source(self) -> bool:
        """
        :return: Whether a sensor reports the source's fill level while pouring.
        """
        return self.source_channel is not None

    def report_source_estimate(self, grains: float) -> None:
        """
        Tell the controller what the source holds where nothing weighs it.

        :param grains: The grains the source is estimated to hold.
        """
        source_fill = (
            grains * self.contents.bulk_volume_per_particle / self.scene.source.capacity
        )
        JointState.from_mapping(
            {self.scene.source.fill_connection: source_fill}
        ).apply_to(self.scene.world)
        self.reported_source_fill = source_fill

    @property
    def weighs_receiver(self) -> bool:
        """
        :return: Whether a sensor reports the receiver's fill level.
        """
        return self.receiver_channel is not None

    @property
    def _shortfall(self) -> float:
        """
        :return: How far short of the goal the controller is made to stop, as a fill
            level of the receiver.
        """
        return (
            self.stop_short
            * self.contents.bulk_volume_per_particle
            / self.scene.receiver.capacity
        )

    def report_received_estimate(self, delivered_grains: float) -> None:
        """
        Tell the controller what the receiver holds where nothing weighs it: what it
        held at the start and what the pour is estimated to have delivered since.

        :param delivered_grains: The grains the pour is estimated to have delivered.
        """
        receiver_fill = (
            self.initial_receiver_fill
            + delivered_grains
            * self.contents.bulk_volume_per_particle
            / self.scene.receiver.capacity
        )
        JointState.from_mapping(
            {
                self.scene.receiver.fill_connection: min(
                    1.0, receiver_fill + self._shortfall
                )
            }
        ).apply_to(self.scene.world)
        self.reported_receiver_fill = receiver_fill

    def _reported_inflow(self, receiver_fill: float, at: float) -> float:
        """
        :param receiver_fill: The receiver's fill level reported now.
        :param at: The time now, in seconds.
        :return: How fast the reported fill level has risen since the previous report,
            per second; nothing for the first report.
        """
        previous, self._previous_report = self._previous_report, (receiver_fill, at)
        if previous is None or at <= previous[1]:
            return 0.0
        return (receiver_fill - previous[0]) / (at - previous[1])

    def sample(self, at: float, phase: Phase) -> Dict[Sample, Any]:
        """
        :param at: Simulated time since the grains came to rest, in seconds.
        :param phase: Which part of the trial is running.
        :return: One row of the trial's samples, taken now.
        """
        scene = self.scene
        return {
            Sample.TIME: at,
            Sample.PHASE: phase,
            Sample.TILT: scene.cup_tilt,
            Sample.MEASURED_SOURCE_FILL: self.source_level.measure(),
            Sample.MODEL_SOURCE_FILL: scene.source.fill_level,
            Sample.MEASURED_RECEIVER_FILL: self.receiver_level.measure(),
            Sample.COMMITTED_RECEIVER_FILL: self.committed_receiver_level.measure(),
            Sample.MODEL_RECEIVER_FILL: scene.receiver.fill_level,
            Sample.DRAIN_SCALE: float(scene.variables.get_value(scene.drain_scale)),
            Sample.GRAINS_IN_SOURCE: self.contents.count_inside(*scene.cup.parts),
            Sample.GRAINS_IN_RECEIVER: self.contents.count_inside(scene.receiver.root),
            Sample.GRAINS_IN_SPOUT: (
                0
                if scene.cup.spout is None
                else self.contents.count_inside(scene.cup.spout)
            ),
        }


# %% what acts on each measurement


@dataclass
class TransferSteering:
    """
    The transfer motion of one controller, and what acts on each measurement for it.
    """

    motion: MotionStatechart
    """
    The motion pouring the source into the receiver.
    """

    @property
    def stalls(self) -> List[Stall]:
        """
        :return: Every stall the steering noticed, in order.
        """
        return []

    @property
    def revisions(self) -> List[Revision]:
        """
        :return: Every revision of the model the steering made, in order.
        """
        return []

    @property
    def diagnoses(self) -> List[Diagnosis]:
        """
        :return: Every stall the steering blamed on a concept of the model, in order.
        """
        return []

    def observe_cycle(self, at: float) -> None:
        """
        Watch the pour at one control cycle, between measurements.

        :param at: The time now, in seconds.
        """

    def steer(self, at: float) -> None:
        """
        Act on the measurement just reported; the model-predictive controller reads it
        from the world itself.

        :param at: The time of the measurement, in seconds.
        """


@dataclass
class PidSteering(TransferSteering):
    """
    A PID controller turning each reported fill level into the tilt the motion follows.
    """

    controller: PidFillController
    """
    The controller computing the tilt.
    """

    setpoint: FloatVariable
    """
    Where the tilt is written for the motion to read.
    """

    variables: FloatVariableData
    """
    Where that variable's value lives.
    """

    perception: Perception
    """
    What reports the receiver's fill level.
    """

    scene: TransferScene
    """
    The scene, whose cup tilt a compensation may read.
    """

    def steer(self, at: float) -> None:
        """
        :param at: The time of the measurement, in seconds.
        """
        tilt = self.controller.update(
            self.perception.reported_receiver_grains, at, tilt=self.scene.cup_tilt
        )
        self.variables.set_value(self.setpoint, tilt)


@dataclass(frozen=True)
class Stall:
    """
    A moment the onset rule found nothing arriving.
    """

    time: float
    """
    When the stall began, in seconds.
    """

    tilt: float
    """
    The cup's tilt then, in radians.
    """

    predicted_inflow: float
    """
    The grains per second the model predicted were arriving then.
    """

    def to_json(self) -> Dict[str, float]:
        return {
            StallKey.TIME: self.time,
            StallKey.TILT: self.tilt,
            StallKey.PREDICTED_INFLOW: self.predicted_inflow,
        }


class StallKey(StrEnum):
    """
    A key of a logged stall written to JSON.
    """

    TIME = "time"
    TILT = "tilt"
    PREDICTED_INFLOW = "predicted_inflow"


@dataclass
class OnsetSteering(TransferSteering):
    """
    The model's own steering, with the onset rule watching each reading and responding
    to a stall.
    """

    watch: StallWatch
    """
    What notices a stall.
    """

    responder: OnsetResponder
    """
    What responds to one.
    """

    perception: Perception
    """
    What reports the fill levels.
    """

    scene: TransferScene
    """
    The scene, whose cup tilt the response reads.
    """

    _stalls: List[Stall] = field(init=False, default_factory=list, repr=False)
    """
    Every stall noticed, in order.
    """

    _stalled: bool = field(init=False, default=False, repr=False)
    """
    Whether the previous reading was stalled.
    """

    arrival: Optional[ArrivalWatch] = None
    """
    What notices the first grains arriving, for a rule that also lowers the onset, or
    ``None``.
    """

    lag: float = 0.0
    """
    How long after grains leave the cup the sensor reports them, in seconds.
    """

    _tilts: TiltHistory = field(init=False, default_factory=lambda: TiltHistory())
    """
    The cup's tilt at each reading.
    """

    @property
    def stalls(self) -> List[Stall]:
        return self._stalls

    @property
    def revisions(self) -> List[Revision]:
        return self.responder.revisions

    @property
    def diagnoses(self) -> List[Diagnosis]:
        return self.responder.diagnoses

    def steer(self, at: float) -> None:
        """
        :param at: The time of the measurement, in seconds.
        """
        reading = self.perception.reported_receiver_grains
        tilt = self.scene.cup_tilt
        self._tilts.record(at, tilt)
        self.responder.observe(at, self.perception.reported_source_grains)
        if self.arrival is not None:
            arrived_at = self.arrival.observe(reading, at)
            if arrived_at is not None and self.responder.lower(
                at,
                tilt=self._tilts.at(arrived_at - self.lag),
                source_fill=self.perception.reported_source_fill,
            ):
                self.watch.restart(at)
        stalled = self.watch.observe(reading, at)
        if stalled and not self._stalled:
            self._stalls.append(
                Stall(
                    time=at,
                    tilt=tilt,
                    predicted_inflow=self.perception.predicted_receiver_inflow,
                )
            )
        self._stalled = stalled
        changed = self.responder.respond(
            stalled, at, tilt=tilt, source_fill=self.perception.reported_source_fill
        )
        if changed:
            self.watch.restart(at)
            self._stalled = False


CROSSING_BAND = 0.02
"""
How far above or below the receiver's opening a grain may be and still count as crossing
it, in metres.
"""


@dataclass
class StreamCamera:
    """
    Sees where the grains that left the source cross the receiver's opening plane, as a
    camera watching the stream would, and evaluates where the model throws it.
    """

    scene: TransferScene
    """
    The scene the stream falls in.
    """

    contents: ParticleFill
    """
    The grains being poured.
    """

    sees_hits: bool = False
    """
    Whether grains crossing into the receiver's opening are seen too, rather than only
    those that miss it.
    """

    _log: CrossingLog = field(init=False, default_factory=CrossingLog, repr=False)
    """
    Every grain seen crossing the opening plane, with when it was seen.
    """

    def record(self, at: float) -> None:
        """
        Note every grain outside both cups near the receiver's opening plane.

        :param at: The time now, in seconds.
        """
        scene = self.scene
        self._log.look(at)
        positions = self.contents.world_positions()
        if not len(positions):
            return
        outside = ~self.contents.is_inside(*scene.cup.parts)
        if not self.sees_hits:
            outside &= ~self.contents.is_inside(scene.receiver.root)
        near_plane = (
            numpy.abs(positions[:, 2] - scene.receiver_opening_height) <= CROSSING_BAND
        )
        for crossing in positions[outside & near_plane, :2]:
            self._log.record(at, crossing)

    def observed_distance(
        self, since: float, until: Optional[float] = None
    ) -> Optional[float]:
        """
        :param since: The earliest time to count, in seconds.
        :param until: The latest time to count, or ``None`` for up to now.
        :return: How far from the lip, along the stream's horizontal direction, the
            grains seen in that time crossed the opening plane on average, in metres,
            or ``None`` if none were seen.
        """
        seen = self._log.between(since, until)
        if not seen:
            return None
        return self.stream.distance_from_lip(seen)

    def observed_rate(self, since: float, until: Optional[float] = None) -> float:
        """
        :param since: The earliest time to count, in seconds.
        :param until: The latest time to count, or ``None`` for up to now.
        :return: How many grains per second crossed the opening plane in that time: the
            grains seen near it per look, over the time a grain falling at the stream's
            speed there takes to pass the band they are seen in.
        """
        launch = self.launch()
        vertical = launch.speed * launch.vertical_share
        speed = math.sqrt(vertical**2 + 2 * STANDARD_GRAVITY * launch.drop)
        return self._log.grains_per_look(since, until) * speed / (2 * CROSSING_BAND)

    @property
    def stream(self) -> PouredStream:
        """
        :return: The stream the model throws from the source's lip into the receiver.
        """
        return PouredStream(
            source=self.scene.source,
            receiver=self.scene.receiver,
            world=self.scene.world,
        )

    def launch(self) -> Launch:
        """
        :return: The stream as the model predicts it leaves the lip now.
        """
        return self.stream.launch()

    def stream_geometry(self, opening_radius: float) -> StreamGeometry:
        """
        :param opening_radius: The receiver opening's radius, in metres.
        :return: How the model says the stream leaves the lip now, and where the
            receiver's centre lies from there.
        """
        return self.stream.geometry(opening_radius)


SWITCH_TIME_CONSTANT = 0.5
"""
How fast the controller's model follows a switch of hypothesis, in seconds: fast enough
to act within the pour, slow enough that the optimisation stays feasible across it.
"""


@dataclass
class ModelBlend:
    """
    The model a controller runs on, following the one called for with a first-order lag
    rather than jumping to it.
    """

    point: ModelPoint
    """
    The model now.
    """

    time: float
    """
    When it was last moved, in seconds.
    """

    slot_open: float = 0.0
    """
    How far the slot edge is switched on now, between zero and one.
    """

    def towards(self, target: ModelPoint, at: float) -> ModelPoint:
        """
        :param target: The model called for.
        :param at: The time now, in seconds.
        :return: The model moved part of the way towards it: the repose angle and the
            slot edge's height in step, the coefficient and the rate by their ratio.
        """
        share = 1.0 - math.exp(-max(0.0, at - self.time) / SWITCH_TIME_CONSTANT)
        self.time = at
        current = self.point

        def linear(old: float, new: float) -> float:
            return old + share * (new - old)

        def ratio(old: float, new: float) -> float:
            return old * (new / old) ** share

        slot = target.slot_height
        if slot is not None and current.slot_height is not None:
            slot = linear(current.slot_height, slot)
        if slot is None:
            slot = current.slot_height
        self.slot_open = linear(
            self.slot_open, 0.0 if target.slot_height is None else 1.0
        )
        self.point = ModelPoint(
            repose_angle=linear(current.repose_angle, target.repose_angle),
            discharge_coefficient=ratio(
                current.discharge_coefficient, target.discharge_coefficient
            ),
            outflow_rate_constant=ratio(
                current.outflow_rate_constant, target.outflow_rate_constant
            ),
            slot_height=slot,
        )
        return self.point


class BankReadingKey(StrEnum):
    """
    A key of one reading a hypothesis bank was handed, written to JSON.
    """

    TIME = "time"
    TILT = "tilt"
    RECEIVER = "receiver"
    SOURCE = "source"
    CROSSING = "crossing"
    TRUE_CROSSING = "true_crossing"
    STREAM = "stream"
    STREAM_SEEN = "stream_seen"
    COUNTED_GRAINS = "counted_grains"


@dataclass
class BankSteering(TransferSteering):
    """
    The model's own steering, with a bank of hypotheses weighing each reading and the
    controller switched to whichever hypothesis the readings call for.
    """

    coupling: BankCoupling
    """
    The bank, and the model the controller runs on as a result.
    """

    perception: Perception
    """
    What reports the fill levels.
    """

    scene: TransferScene
    """
    The scene, whose drain the switch acts on.
    """

    camera: StreamCamera
    """
    What sees where the stream crosses the receiver's opening plane.
    """

    camera_condition: CameraCondition
    """
    How the camera departs from where the stream really crosses.
    """

    built: ModelPoint
    """
    The parameters the source's drain was built with.
    """

    opening_radius: float
    """
    The receiver opening's radius, in metres.
    """

    random: numpy.random.Generator
    """
    Where the camera's noise and dropouts are drawn from.
    """

    caution: Optional[CautiousStart] = None
    """
    Whether the tilt speed is capped while the aim is unconfirmed, or ``None`` for no
    cap.
    """

    tilt_rate_cap: Optional[FloatVariable] = None
    """
    Where the tilt speed limit is written for the motion to read.
    """

    readings: List[Dict[str, Any]] = field(init=False, default_factory=list)
    """
    Every reading the bank was handed, with where the stream truly crossed.
    """

    handed: List[Reading] = field(init=False, default_factory=list)
    """
    Every reading the bank was handed, as it was handed.
    """

    blend: Optional[ModelBlend] = field(init=False, default=None)
    """
    The model the controller runs on, moving towards the one the bank calls for.
    """

    _crossed: float = field(init=False, default=0.0)
    """
    How many grains truly crossed the opening plane while the camera saw the stream.
    """

    _count_scale: float = field(init=False, default=1.0)
    """
    How far off the camera's count is over this pour: its calibration error, drawn once.
    """

    def steer(self, at: float) -> None:
        """
        :param at: The time of the measurement, in seconds.
        """
        scene, perception = self.scene, self.perception
        tilt = scene.cup_tilt
        model = self.camera_condition.model
        delay = 0.0 if model is None else model.delay
        since = at - delay - 1.0 / PERCEPTION_FREQUENCY + DELAY_TOLERANCE
        true_crossing = self.camera.observed_distance(since=since, until=at - delay)
        stream = self.camera.stream_geometry(self.opening_radius)
        crossing = self._seen(true_crossing)
        counts = model is not None and model.count_noise is not None
        if counts and not self.handed:
            self._count_scale = max(
                0.0, 1.0 + self.random.normal(0.0, model.count_noise)
            )
        if counts and crossing is not None:
            self._crossed += self.camera.observed_rate(since, until=at - delay) * (
                1.0 / PERCEPTION_FREQUENCY
            )
        reading = Reading(
            time=at,
            tilt=tilt,
            receiver_grains=(
                perception.reported_receiver_grains
                if perception.weighs_receiver
                else None
            ),
            source_grains=(
                perception.reported_source_grains
                if perception.weighs_source or not self.handed
                else None
            ),
            crossing_distance=crossing,
            stream=stream,
            stream_seen=crossing is not None if counts else None,
            counted_grains=self._crossed * self._count_scale if counts else None,
        )
        self.handed.append(reading)
        self.coupling.observe(
            reading,
            perception.committed_receiver_grains / GOAL_GRAINS,
            trickle=min(
                self.coupling.bank.settings.trickle,
                (
                    GOAL_GRAINS
                    - perception.reported_receiver_grains
                    - self.coupling.margin
                )
                / (2 * FILL_PREDICTION_WINDOW),
            ),
        )
        if not perception.weighs_receiver:
            perception.report_received_estimate(self.coupling.received_grains)
        if not perception.weighs_source:
            perception.report_source_estimate(
                self.handed[0].source_grains
                - model.grains_crossed(reading.counted_grains)
            )
        if self.caution is not None:
            capped = self.caution.observe(
                at,
                reading.crossing_distance,
                aim_confirmed=FreeParameter.DISCHARGE_COEFFICIENT
                in self.coupling.active.concepts,
            )
            scene.variables.set_value(
                self.tilt_rate_cap,
                (self.caution.settings.tilt_rate_cap if capped else UNCAPPED_TILT_RATE),
            )
        applied = self.model_at(at)
        scene.run_on(
            applied,
            self.built,
            self.coupling.cautious,
            slot_open=self.blend.slot_open,
        )
        perception.stop_short = self.coupling.margin
        self.readings.append(
            {
                BankReadingKey.TIME: at,
                BankReadingKey.TILT: tilt,
                BankReadingKey.RECEIVER: reading.receiver_grains,
                BankReadingKey.SOURCE: reading.source_grains,
                BankReadingKey.CROSSING: reading.crossing_distance,
                BankReadingKey.TRUE_CROSSING: true_crossing,
                BankReadingKey.STREAM: stream.to_json(),
                BankReadingKey.STREAM_SEEN: reading.stream_seen,
                BankReadingKey.COUNTED_GRAINS: reading.counted_grains,
            }
        )

    def model_at(self, at: float) -> ModelPoint:
        """
        :param at: The time of the reading, in seconds.
        :return: The model the controller runs on now, moving towards the one the bank
            calls for; the first reading starts that move from the model the controller
            ran on before it, so the bank's first point is blended in, not jumped to.
        """
        if self.blend is None:
            self.blend = ModelBlend(point=self.coupling.bank.base, time=at)
        return self.blend.towards(self.coupling.point, at)

    def observe_cycle(self, at: float) -> None:
        """
        Let the camera look at the stream at every control cycle, as a camera running at
        the control rate would.

        :param at: The time now, in seconds.
        """
        self.camera.record(at)

    def _seen(self, crossing: Optional[float]) -> Optional[float]:
        """
        :param crossing: Where the stream truly crossed, or ``None`` if it did not.
        :return: Where the camera reports it crossed, or ``None`` if it saw nothing.
        """
        model = self.camera_condition.model
        if crossing is None or model is None:
            return None
        if self.random.random() < model.dropout:
            return None
        return crossing + self.random.normal(0.0, model.noise)


@dataclass
class TiltHistory:
    """
    The cup's tilt at each reading of a pour.
    """

    times: List[float] = field(default_factory=list)
    """
    When each tilt was read, in seconds, oldest first.
    """

    tilts: List[float] = field(default_factory=list)
    """
    The tilt read then, in radians.
    """

    def record(self, at: float, tilt: float) -> None:
        """
        :param at: The time now, in seconds.
        :param tilt: The cup's tilt now, in radians.
        """
        self.times.append(at)
        self.tilts.append(tilt)

    def at(self, time: float) -> float:
        """
        :param time: A time of the pour, in seconds.
        :return: The tilt last read at or before that time, or the first one read.
        """
        index = (
            int(numpy.searchsorted(self.times, time + DELAY_TOLERANCE, side="right"))
            - 1
        )
        return self.tilts[max(0, index)]


# %% running one trial


class MotionEnding(StrEnum):
    """
    How a motion of a trial came to an end.
    """

    REACHED_GOAL = "reached_goal"
    """
    Its end condition held.
    """

    TIMED_OUT = "timed_out"
    """
    It ran out of control cycles first.
    """

    INFEASIBLE = "infeasible"
    """
    The controller found no command that satisfies its hard constraints.
    """

    @property
    def reached_goal(self) -> bool:
        """
        :return: Whether the motion ended by reaching its goal.
        """
        return self is MotionEnding.REACHED_GOAL


class RunsMotion(Protocol):
    """
    Something that ticks a compiled motion to its end.
    """

    control_cycles: float
    """
    How many control cycles have run.
    """

    def tick_until_end(self, timeout: int) -> None:
        """
        :param timeout: How many control cycles the motion is given.
        """


@dataclass
class MotionResult:
    """
    How one motion of a trial ran.
    """

    cycles: int
    """
    How many control cycles it ran for.
    """

    ending: MotionEnding
    """
    How it came to an end.
    """

    @classmethod
    def of(cls, executor: RunsMotion, timeout: int) -> MotionResult:
        """
        Tick a motion to its end and record how it ended.

        A motion that times out or that the controller cannot solve is an outcome of the
        trial, not a failure of it, so both are recorded rather than raised.

        :param executor: What runs the motion.
        :param timeout: How many control cycles the motion is given.
        :return: How the motion ran.
        """
        try:
            executor.tick_until_end(timeout=timeout)
        except TimeoutError:
            return cls(
                cycles=int(executor.control_cycles), ending=MotionEnding.TIMED_OUT
            )
        except InfeasibleException:
            return cls(
                cycles=int(executor.control_cycles), ending=MotionEnding.INFEASIBLE
            )
        return cls(
            cycles=int(executor.control_cycles), ending=MotionEnding.REACHED_GOAL
        )


@dataclass(frozen=True)
class TransferTrial:
    """
    One pour, in one scene, with one set of model parameters, one loop mode and one
    microstate of the grains.
    """

    scene: Scene
    """
    Which source cup pours into which receiver.
    """

    baseline: ModelParameters
    """
    The parameters the perturbation is taken from: those fitted to the source cup.
    """

    perturbation: Perturbation
    """
    Which parameters the controller's model is evaluated with.
    """

    loop_mode: LoopMode
    """
    How much of what is measured the controller acts on.
    """

    seed: int
    """
    Which microstate the grains start from; see :func:`settle_time_of`, and which noise
    the sensor draws.
    """

    controller: FillController = FillController.MODEL_PREDICTIVE
    """
    What decides how far the cup is tilted.
    """

    sensor: Optional[SensorCondition] = SensorCondition.IDEAL
    """
    What reports the fill levels to the controller, or ``None`` if nothing weighs the
    receiver and a hypothesis bank estimates it from the wrist and the camera.
    """

    gains: Optional[PidGains] = None
    """
    The gains a PID controller runs with, or ``None`` under the model-predictive one.

    A PID controller acts on the reported fill levels alone, so it needs a loop mode
    that reports them, and the drain's calibration has no effect on it.
    """

    pid_variant: PidVariant = PidVariant.PLAIN
    """
    How a PID controller makes up for a late reading.
    """

    inflow_model: Optional[FirstOrderInflowModel] = None
    """
    The inflow model a Smith predictor predicts with, identified from readings.
    """

    onset_rule: Optional[OnsetRule] = None
    """
    The rule that tilts a stalled pour further, or ``None`` to leave the model alone.
    """

    start_margin: float = POUR_START_TILT_MARGIN
    """
    How far past the model's onset the cup is carried before the transfer, in radians.
    """

    sequence: Optional[SequencePosition] = None
    """
    Where the pour stands in a sequence of pours, or ``None`` for a pour on its own.
    """

    wrist_sensor: Optional[WristSensor] = None
    """
    The force-torque sensor weighing the source at the wrist, or ``None`` to read the
    source with the same sensor as the receiver.
    """

    bank: Optional[BankSettings] = None
    """
    The settings of the hypothesis bank weighing each reading and switching the
    controller's model, or ``None`` to run the model as given.
    """

    sensor_set: Optional[SensorSet] = None
    """
    Every sensor a pour with a hypothesis bank reads, the camera included.
    """

    forces_second_lip: bool = False
    """
    Whether the bank weighs the second lip even for a cup without a slot.
    """

    fault: Optional[PourFault] = None
    """
    A fault in the grains no concept of the model covers, or ``None`` for sound grains.
    """

    fixed_model: Optional[ModelPoint] = None
    """
    A model the controller runs on throughout instead of the perturbed one, such as one
    fitted offline, or ``None``.
    """

    fixed_label: str = "oracle"
    """
    A short name of the fixed model, usable in a file name.
    """

    caution: Optional[CautionSettings] = None
    """
    How the pour starts cautiously while the bank has not confirmed the aim, or ``None``
    for a pour at full speed throughout.
    """

    check_residuals: Optional[Dict[Feature, RateResiduals]] = None
    """
    How far each feature the consistency check reads landed from its prediction on the
    cup's sound pours, by tilt rate, or ``None`` for an uncalibrated check.
    """

    @property
    def onset(self) -> Optional[OnsetSpread]:
        """
        :return: The twin's spread of the repose angle this pour's bank carries: the one
            the sequence wrote back, or the twin's own with the error set into its
            centre; ``None`` for a bank that does not adapt the onset.
        :raises OnsetSpreadMissingError: If the bank adapts the onset and the twin has
            no spread for the cup.
        """
        if self.bank is None or not self.bank.adapts_onset:
            return None
        if self.sequence is not None and self.sequence.kept_onset is not None:
            return self.sequence.kept_onset
        parameters = self.perturbation.around(self.baseline)
        if parameters.dynamic_repose_spread is None:
            raise OnsetSpreadMissingError()
        return OnsetSpread(
            centre=parameters.while_turning().repose_angle,
            spread=parameters.dynamic_repose_spread,
        )

    @property
    def drain_rate(self) -> Optional[DrainRateSpread]:
        """
        :return: The twin's spread of the drain rate this pour's bank carries, with the
            error set into its centre; ``None`` for a bank that does not adapt the drain
            rate.
        :raises DrainRateSpreadMissingError: If the bank adapts the drain rate and the
            twin has no spread for the cup.
        """
        if self.bank is None or not self.bank.adapts_drain_rate:
            return None
        parameters = self.perturbation.around(self.baseline)
        if parameters.dynamic_outflow_rate_spread is None:
            raise DrainRateSpreadMissingError()
        return DrainRateSpread(
            centre=parameters.while_turning().outflow_rate_constant,
            spread=parameters.dynamic_outflow_rate_spread,
        )

    @property
    def start_model_values(self) -> Dict[FreeParameter, float]:
        """
        :return: The centres of the twin's spreads the bank carries, which the
            controller's model starts on.
        """
        values = {}
        if self.onset is not None:
            values[FreeParameter.REPOSE_ANGLE] = self.onset.centre
        if self.drain_rate is not None:
            values[FreeParameter.OUTFLOW_RATE_CONSTANT] = self.drain_rate.centre
        return values

    def __post_init__(self) -> None:
        if not self.controller.uses_pouring_model and self.gains is None:
            raise MissingPidGainsError(self.controller)
        if (
            not self.controller.uses_pouring_model
            and not self.loop_mode.reports_fill_levels
        ):
            raise PidWithoutFillFeedbackError(self.loop_mode)
        if self.pid_variant is PidVariant.SMITH and self.inflow_model is None:
            raise MissingInflowModelForSmithError()
        if self.onset_rule is not None and not self.controller.uses_pouring_model:
            raise OnsetRuleWithoutModelError(self.controller)
        if self.kept_repose_angle is not None and not self.revises_repose_angle:
            raise KeptReposeAngleWithoutRevisionError()
        if self.revises_discharge_coefficient and self.wrist_sensor is None:
            raise ConceptAwareRevisionWithoutWristError()
        if self.bank is not None and (
            self.sensor_set is None
            or self.sensor is not self.sensor_set.receiver
            or self.wrist_sensor is not self.sensor_set.wrist
        ):
            raise BankWithoutSensorSetError()
        if self.caution is not None and self.bank is None:
            raise CautionWithoutBankError()
        if self.sensor is None and self.bank is None:
            raise UnweighedReceiverWithoutBankError()

    @property
    def revises_repose_angle(self) -> bool:
        """
        :return: Whether the onset rule revises the model's repose angle.
        """
        return (
            self.onset_rule is not None
            and self.onset_rule.response.revises_repose_angle
        )

    @property
    def revises_discharge_coefficient(self) -> bool:
        """
        :return: Whether the onset rule may revise the discharge coefficient.
        """
        return (
            self.onset_rule is not None
            and self.onset_rule.response is OnsetResponse.CONCEPT_AWARE_REVISION
        )

    @property
    def kept_repose_angle(self) -> Optional[float]:
        """
        :return: The repose angle kept from the previous pour of a sequence, in
            radians, or ``None`` to start from the model's own.
        """
        if self.sequence is None:
            return None
        return self.sequence.kept_repose_angle

    @property
    def kept_model(self) -> Optional[ModelPoint]:
        """
        :return: The model a hypothesis bank wrote back after the previous pour of a
            sequence, which this one starts from, or ``None`` to start from the model
            as given.
        """
        if self.sequence is None:
            return None
        return self.sequence.kept_model

    @property
    def _weighs_second_lip(self) -> bool:
        """
        :return: Whether the bank weighs a second lip at the slot edge.
        """
        return self.scene.source is SourceCup.SPOUTED_CUP or self.forces_second_lip

    @property
    def name(self) -> str:
        """
        :return: The directory name this trial writes its record into; a
            model-predictive trial with an ideal sensor keeps the name it was first
            recorded under.
        """
        sensor = (
            "" if self.sensor in (SensorCondition.IDEAL, None) else f"_{self.sensor}"
        )
        if not self.controller.uses_pouring_model:
            variant = (
                "" if self.pid_variant is PidVariant.PLAIN else f"_{self.pid_variant}"
            )
            return (
                f"{Study.TRANSFER}_{self.scene.label}_{self.controller}{variant}"
                f"_{self.gains.label}_{self.sensor}_seed{self.seed}"
            )
        rule = "" if self.onset_rule is None else f"_{self.onset_rule.label}"
        margin = (
            ""
            if self.start_margin == POUR_START_TILT_MARGIN
            else f"_margin{self.start_margin:g}"
        )
        sequence = "" if self.sequence is None else self.sequence.label
        wrist = "" if self.wrist_sensor is None else f"_wrist_{self.wrist_sensor}"
        if self.bank is not None:
            lip = "_lip" if self.forces_second_lip else ""
            wrist = f"_{self.bank.label}_{self.sensor_set}{lip}"
        fault = "" if self.fault is None else f"_{self.fault}"
        if self.fixed_model is not None:
            fault = f"{fault}_{self.fixed_label}"
        return (
            f"{Study.TRANSFER}_{self.scene.label}_{self.perturbation.label}"
            f"_{self.loop_mode}{sensor}{rule}{margin}{wrist}{fault}{sequence}"
            f"{'' if self.caution is None else f'_{self.caution.label}'}"
            f"_seed{self.seed}"
        )

    def run(self) -> TrialRecord:
        """
        Carry the cup, fill it, pre-tilt it, pour under the pouring tasks, turn it back
        upright and let the grains land.

        :return: The trial's record.
        """
        started = time.monotonic()
        parameters = self.perturbation.around(self.baseline)
        weighs = self.bank is not None
        revisable = weighs or self.fixed_model is not None
        scene = TransferScene.build(
            parameters,
            self.scene,
            revisable_repose=self.revises_repose_angle or revisable,
            revisable_discharge=self.revises_discharge_coefficient or revisable,
            second_lip=(weighs and self._weighs_second_lip)
            or (
                self.fixed_model is not None
                and self.fixed_model.slot_height is not None
            ),
            obstacle=(
                None if self.fault is None else self.fault.obstacle(self.scene.source)
            ),
        )
        built = ModelPoint(
            repose_angle=parameters.repose_angle,
            discharge_coefficient=parameters.discharge_coefficient,
            outflow_rate_constant=parameters.outflow_rate_constant,
        )
        start_model = (self.kept_model or self.fixed_model or built).with_values(
            self.start_model_values
        )
        scene.run_on(start_model, built, cautious=False)
        if self.kept_repose_angle is not None:
            scene.variables.set_value(
                scene.repose_correction,
                self.kept_repose_angle - parameters.repose_angle,
            )
        repose_angle_at_start = scene.repose_angle
        geometry = self.scene.source.geometry
        ParticleFill.settling_contact().apply_to(scene.world.bodies)
        rows: List[Dict[Sample, Any]] = []
        with stepped_simulation(scene.world) as simulation:
            simulation.step_simulation(timedelta(milliseconds=20))
            self._run_motion(carry_motion(scene), scene, simulation)
            scene.nudge_wrist()
            contents = ParticleFill.spawn_in(
                simulator=simulation.simulator,
                container=scene.source.root,
                world_T_container=scene.world.compute_forward_kinematics_np(
                    scene.world.root, scene.source.root
                ),
                positions=geometry.particle_positions(
                    PARTICLE_RADIUS,
                    geometry.particle_capacity(
                        PARTICLE_RADIUS, fill_fraction=PACKED_SHARE
                    ),
                ),
                particle_radius=PARTICLE_RADIUS,
                contact=None if self.fault is None else self.fault.contact,
            )
            simulation.step_simulation(settle_time_of(self.seed, SETTLE_TIME))
            contents.bulk_volume_per_particle = contents.bulk_volume_per_particle_in(
                scene.source.root, scene.source.capacity
            )
            perception = Perception(
                scene=scene,
                contents=contents,
                loop_mode=self.loop_mode,
                sensor=self.sensor,
                seed=self.seed,
                wrist=self.wrist_sensor,
            )
            perception.report_initial_levels()
            total_grains = contents.count_inside(*scene.cup.parts)
            clock = _Clock()
            rows.append(perception.sample(clock.now, Phase.PRE_TILT))

            pre_tilt = (
                scene.pour_start_tilt(contents, margin=self.start_margin)
                if self.controller.uses_pouring_model
                else self.gains.start_tilt
            )
            if start_model.slot_height is not None:
                pre_tilt = (
                    self._drain_geometry(scene, contents).onset_tilt(
                        total_grains, start_model
                    )
                    + self.start_margin
                )
            self._run_motion(
                carry_motion(scene, tilt=pre_tilt),
                scene,
                simulation,
                perception=perception,
                phase=Phase.PRE_TILT,
                clock=clock,
                rows=rows,
                measures=False,
            )
            transfer_start = clock.now
            steering = self._steering(scene, perception, contents)
            transfer = self._run_motion(
                steering.motion,
                scene,
                simulation,
                perception=perception,
                phase=Phase.TRANSFER,
                clock=clock,
                rows=rows,
                after_measuring=steering.steer,
                every_cycle=steering.observe_cycle,
            )
            transfer_duration = clock.now - transfer_start
            bank_record = self._bank_record(steering)
            peak_tilt = max(
                row[Sample.TILT] for row in rows if row[Sample.PHASE] == Phase.TRANSFER
            )
            delivered_at_end = contents.count_inside(scene.receiver.root)
            self._run_motion(
                carry_motion(scene),
                scene,
                simulation,
                perception=perception,
                phase=Phase.RETURN,
                clock=clock,
                rows=rows,
                measures=False,
            )
            self._land(simulation, perception, clock, rows)
            retained = contents.count_inside(*scene.cup.parts)
            delivered = contents.count_inside(scene.receiver.root)
        return TrialRecord(
            description={
                TrialKey.STUDY: Study.TRANSFER,
                TrialKey.NAME: self.name,
                TrialKey.SOURCE: self.scene.source,
                TrialKey.RECEIVER: self.scene.receiver,
                TrialKey.SEED: self.seed,
                TrialKey.PERTURBATION: self.perturbation.to_json(),
                TrialKey.PARAMETERS: parameters.to_json(),
                TrialKey.LOOP_MODE: self.loop_mode,
                TrialKey.CONTROLLER: self.controller,
                TrialKey.SENSOR: self.sensor,
                TrialKey.GAINS: None if self.gains is None else self.gains.to_json(),
                TrialKey.PID_VARIANT: self.pid_variant,
                TrialKey.INFLOW_MODEL: (
                    None if self.inflow_model is None else self.inflow_model.to_json()
                ),
                TrialKey.ONSET_RULE: (
                    None if self.onset_rule is None else self.onset_rule.to_json()
                ),
                TrialKey.START_MARGIN: self.start_margin,
                TrialKey.SEQUENCE: (
                    None if self.sequence is None else self.sequence.to_json()
                ),
                TrialKey.REPOSE_ANGLE_AT_START: repose_angle_at_start,
                TrialKey.REPOSE_ANGLE_AT_END: scene.repose_angle,
                TrialKey.WRIST_SENSOR: self.wrist_sensor,
                TrialKey.DIAGNOSES: [
                    diagnosis.to_json() for diagnosis in steering.diagnoses
                ],
                TrialKey.DISCHARGE_COEFFICIENT_AT_END: scene.discharge_coefficient,
                TrialKey.STALLS: [stall.to_json() for stall in steering.stalls],
                TrialKey.REVISIONS: [
                    revision.to_json() for revision in steering.revisions
                ],
                TrialKey.EQUATION: scene.source.fill_equation.ungated().to_json(),
                TrialKey.FAULT: self.fault,
                **bank_record,
                TrialKey.OUTCOME: {
                    Outcome.GOAL_GRAINS: GOAL_GRAINS,
                    Outcome.TOTAL_GRAINS: total_grains,
                    Outcome.DELIVERED_AT_END: delivered_at_end,
                    Outcome.DELIVERED: delivered,
                    Outcome.RETAINED: retained,
                    Outcome.SPILLED: total_grains - delivered - retained,
                    Outcome.PEAK_TILT: peak_tilt,
                    Outcome.TRANSFER_DURATION: transfer_duration,
                    Outcome.REACHED_GOAL: transfer.ending.reached_goal,
                    Outcome.ENDING: transfer.ending,
                    Outcome.PRE_TILT: pre_tilt,
                },
                TrialKey.WALL_TIME: time.monotonic() - started,
            },
            samples=pandas.DataFrame(rows),
        )

    @staticmethod
    def _run_motion(
        statechart: MotionStatechart,
        scene: TransferScene,
        simulation: MujocoSim,
        perception: Optional[Perception] = None,
        phase: Optional[Phase] = None,
        clock: Optional[_Clock] = None,
        rows: Optional[List[Dict[Sample, Any]]] = None,
        measures: bool = True,
        after_measuring: Optional[Callable[[float], None]] = None,
        every_cycle: Optional[Callable[[float], None]] = None,
    ) -> MotionResult:
        """
        Tick one motion to its end in lockstep with the physics.

        :param statechart: The motion to run.
        :param scene: The scene the motion runs in.
        :param simulation: The physics it runs in lockstep with.
        :param perception: What measures the pour, once there are grains to measure.
        :param phase: Which part of the trial the motion is, for the samples.
        :param clock: The trial's clock, advanced once per cycle.
        :param rows: Where the samples are appended.
        :param measures: Whether the controller is told what is measured during this
            motion; only the transfer is driven by the fill levels.
        :param every_cycle: What watches the pour at every control cycle, given the time
            in seconds.
        :param after_measuring: What acts on each measurement once it has been reported,
            given its time in seconds.
        :return: How the motion ran.
        """
        executor = Executor(
            context=MotionStatechartContext(
                world=scene.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=CONTROL_FREQUENCY,
                    prediction_horizon=PREDICTION_HORIZON,
                ),
                float_variable_data=scene.variables,
            ),
            pacer=SteppedSimulationPacer(simulation),
        )
        executor.compile(motion_statechart=statechart)
        if perception is not None:
            tick = executor.tick
            cycles_between_measurements = max(
                1, round(CONTROL_FREQUENCY / PERCEPTION_FREQUENCY)
            )

            def tick_and_measure() -> None:
                tick()
                clock.advance(1 / CONTROL_FREQUENCY)
                cycle = int(executor.control_cycles)
                if every_cycle is not None:
                    every_cycle(clock.now)
                if measures and cycle % cycles_between_measurements == 0:
                    perception.update(at=clock.now)
                    if after_measuring is not None:
                        after_measuring(clock.now)
                if cycle % SAMPLE_EVERY == 0:
                    rows.append(perception.sample(clock.now, phase))

            executor.tick = tick_and_measure
        return MotionResult.of(executor, timeout=TICK_LIMIT)

    def _steering(
        self, scene: TransferScene, perception: Perception, contents: ParticleFill
    ) -> TransferSteering:
        """
        :param scene: The scene the transfer runs in.
        :param perception: What reports the fill levels.
        :param contents: The grains being poured.
        :return: The transfer motion of this trial's controller, and what acts on each
            measurement for it.
        """
        if self.controller.uses_pouring_model and self.bank is not None:
            return self._bank_steering(scene, perception, contents)
        if self.controller.uses_pouring_model and self.onset_rule is not None:
            return self._onset_steering(scene, perception, contents)
        if self.controller.uses_pouring_model:
            return TransferSteering(
                motion=transfer_motion(scene, contents.bulk_volume_per_particle)
            )
        controller = PidFillController(
            gains=self.gains,
            goal=GOAL_GRAINS,
            tolerance=FILL_TOLERANCE_GRAINS,
            maximum_tilt=MAXIMUM_PID_TILT,
            compensation=self._compensation(scene),
        )
        setpoint = FloatVariable("pid_tilt_setpoint")
        scene.variables.register_expression(setpoint)
        scene.variables.set_value(setpoint, controller.setpoint)
        return PidSteering(
            motion=pid_transfer_motion(
                scene, controller, setpoint, aims=self.controller.aims
            ),
            controller=controller,
            setpoint=setpoint,
            variables=scene.variables,
            perception=perception,
            scene=scene,
        )

    def _onset_steering(
        self, scene: TransferScene, perception: Perception, contents: ParticleFill
    ) -> OnsetSteering:
        """
        :param scene: The scene the transfer runs in.
        :param perception: What reports the fill levels.
        :param contents: The grains being poured.
        :return: The model's transfer motion, with the onset rule watching each reading
            and responding to a stall as the rule says.
        """
        extra: List[Task] = []
        match self.onset_rule.response:
            case OnsetResponse.OVERRIDE:
                active = FloatVariable("onset_override_active")
                scene.variables.register_expression(active)
                scene.variables.set_value(active, 0.0)
                extra.append(
                    TiltOverride(
                        root_link=scene.world.root,
                        tip_link=scene.tool_frame,
                        root_R_carry=carry_pose(scene).to_rotation_matrix(),
                        active=active,
                        velocity=self.onset_rule.override_velocity,
                        weight=self.onset_rule.override_weight,
                    )
                )
                responder = OnsetOverrideSwitch(
                    active=active, variables=scene.variables
                )
            case OnsetResponse.REVISION:
                responder = OnsetRevision(
                    equation=scene.source.fill_equation,
                    correction=scene.repose_correction,
                    variables=scene.variables,
                )
            case OnsetResponse.CONCEPT_AWARE_REVISION:
                responder = ConceptAwareRevision(
                    onset=OnsetRevision(
                        equation=scene.source.fill_equation,
                        correction=scene.repose_correction,
                        variables=scene.variables,
                    ),
                    discharge=DischargeRevision(
                        coefficient=scene.source.fill_equation.discharge_coefficient,
                        scale=scene.discharge_scale,
                        variables=scene.variables,
                    ),
                    source=SourceWatch(
                        noise=self.wrist_sensor.model.noise,
                        patience=self.onset_rule.patience,
                    ),
                    stream=StreamCamera(scene=scene, contents=contents),
                    patience=self.onset_rule.patience,
                )
        return OnsetSteering(
            motion=transfer_motion(scene, contents.bulk_volume_per_particle, extra),
            watch=StallWatch(
                rule=self.onset_rule, goal=GOAL_GRAINS, tolerance=FILL_TOLERANCE_GRAINS
            ),
            responder=responder,
            perception=perception,
            scene=scene,
            arrival=(
                ArrivalWatch(
                    noise=self.sensor.model.noise,
                    reference=perception.reported_receiver_grains,
                )
                if self.onset_rule.lowers
                else None
            ),
            lag=self._sensor_lag(scene),
        )

    @staticmethod
    def _drain_geometry(scene: TransferScene, contents: ParticleFill) -> DrainGeometry:
        """
        :return: The source cup as the model sees it, counted in grains.
        """
        equation = scene.source.fill_equation
        return DrainGeometry(
            container_height=equation.container_height,
            lip_offset=equation.lip_offset,
            grains_per_fill=scene.source.capacity / contents.bulk_volume_per_particle,
            slot_offset=equation.container_width / 2,
        )

    def _bank_steering(
        self, scene: TransferScene, perception: Perception, contents: ParticleFill
    ) -> BankSteering:
        """
        :param scene: The scene the transfer runs in.
        :param perception: What reports the fill levels.
        :param contents: The grains being poured.
        :return: The model's transfer motion, with the hypothesis bank weighing each
            reading and switching the model the controller runs on.
        """
        built = ModelPoint(
            repose_angle=scene.source.fill_equation.repose_angle,
            discharge_coefficient=scene.source.fill_equation.discharge_coefficient,
            outflow_rate_constant=scene.source.fill_equation.outflow_rate_constant,
        )
        base = (self.kept_model or built).with_values(self.start_model_values)
        bank = HypothesisBank(
            hypotheses=hypotheses_for(
                self.scene.source,
                self.forces_second_lip,
                outflow_capacity=self.bank.outflow_capacity,
            ),
            geometry=self._drain_geometry(scene, contents),
            base=base,
            sensors=sensor_knowledge(self.sensor_set, fall_time=scene.fall_time),
            settings=self.bank,
            seed=self.seed,
            onset=self.onset,
            drain_rate=self.drain_rate,
        )
        caution = tilt_rate_cap = None
        extra: List[Task] = []
        if self.caution is not None:
            caution = CautiousStart(
                settings=self.caution,
                opening_radius=self.scene.receiver.geometry.inner_radius,
            )
            tilt_rate_cap = FloatVariable("source_tilt_rate_cap")
            scene.variables.register_expression(tilt_rate_cap)
            scene.variables.set_value(tilt_rate_cap, self.caution.tilt_rate_cap)
            extra.append(TiltRateLimit(source=scene.source, maximum_rate=tilt_rate_cap))
        return BankSteering(
            motion=transfer_motion(scene, contents.bulk_volume_per_particle, extra),
            coupling=BankCoupling(bank=bank),
            caution=caution,
            tilt_rate_cap=tilt_rate_cap,
            perception=perception,
            scene=scene,
            camera=StreamCamera(scene=scene, contents=contents, sees_hits=True),
            camera_condition=self.sensor_set.camera,
            built=built,
            opening_radius=self.scene.receiver.geometry.inner_radius,
            random=numpy.random.default_rng([self.seed, 2]),
        )

    def _onset_record(self, steering: BankSteering) -> Dict[str, Any]:
        """
        :param steering: The bank's steering the transfer ran with.
        :return: The twin's spreads of the repose angle and the drain rate the bank
            carried, the pour's repose angle as the onset hypothesis estimates it, how
            much of each posterior lay within its spread, and how the explanation the pour ended on held up
            against what it showed.
        """
        bank = steering.coupling.bank
        onset_filter = bank.filters.get(Hypothesis.ONSET)
        consistency = None
        if (
            self.bank.checks_consistency
            and bank.started
            and self.sensor_set.wrist is not None
        ):
            explanation = bank.filters[steering.coupling.active]
            consistency = ConsistencyCheck(
                geometry=bank.geometry,
                sensors=bank.sensors,
                settings=bank.settings,
                seed=self.seed,
                residuals=self.check_residuals or {},
            ).verdict(explanation.parameters, explanation.log_weights, steering.handed)
        return {
            TrialKey.TWIN_ONSET: None if bank.onset is None else bank.onset.to_json(),
            TrialKey.ONSET_ESTIMATE: (
                None
                if onset_filter is None or not bank.started
                else onset_filter.mean().repose_angle
            ),
            TrialKey.ONSET_SHARE_INSIDE: bank.share_inside(FreeParameter.REPOSE_ANGLE),
            TrialKey.TWIN_DRAIN_RATE: (
                None if bank.drain_rate is None else bank.drain_rate.to_json()
            ),
            TrialKey.DRAIN_RATE_SHARE_INSIDE: bank.share_inside(
                FreeParameter.OUTFLOW_RATE_CONSTANT
            ),
            TrialKey.CONSISTENCY: (
                None if consistency is None else consistency.to_json()
            ),
        }

    def _bank_record(self, steering: TransferSteering) -> Dict[str, Any]:
        """
        :param steering: The steering the transfer ran with.
        :return: What a pour with a hypothesis bank adds to its record: the settings and
            sensors, every reading, the probabilities after each, every switch, what the
            bank concluded, and the model it writes back.
        """
        if not isinstance(steering, BankSteering):
            return {}
        coupling = steering.coupling
        switch = coupling.switch
        chosen, support = switch.target(coupling.bank.probabilities)
        final = coupling.point
        writes_back = coupling.active.revises and support >= switch.threshold
        return {
            TrialKey.BANK: self.bank.to_json(),
            TrialKey.SENSOR_SET: self.sensor_set,
            TrialKey.FORCES_SECOND_LIP: self.forces_second_lip,
            TrialKey.CAUTION: (
                None if self.caution is None else self.caution.to_json()
            ),
            TrialKey.CAUTION_RELEASED_AT: (
                None if steering.caution is None else steering.caution.released_at
            ),
            **self._onset_record(steering),
            TrialKey.BANK_READINGS: steering.readings,
            TrialKey.BANK_TRACE: coupling.trace,
            TrialKey.COMMITS: [commit.to_json() for commit in coupling.commits],
            TrialKey.BANK_SUMMARY: coupling.summary(),
            TrialKey.FINAL_MODEL: final.to_json(),
            TrialKey.WRITES_BACK: writes_back,
        }

    def _compensation(self, scene: TransferScene) -> ReadingCompensation:
        """
        :param scene: The scene the transfer runs in.
        :return: What the PID controller makes of each reading, over the lag the sensor
            is known to have: its delay, and the stream's fall for a sensor that sees
            only landed grains.
        """
        horizon = self._sensor_lag(scene)
        match self.pid_variant:
            case PidVariant.PLAIN:
                return NoCompensation()
            case PidVariant.STOP_EARLY:
                return RateProjection(horizon=horizon)
            case PidVariant.SMITH:
                return SmithPrediction(model=self.inflow_model, horizon=horizon)

    def _sensor_lag(self, scene: TransferScene) -> float:
        """
        :param scene: The scene the transfer runs in.
        :return: How long after grains leave the cup the sensor is known to report
            them, in seconds: its delay, and the stream's fall for a sensor that sees
            only landed grains.
        """
        model = self.sensor.model
        return model.delay + (0.0 if model.counts_falling_grains else scene.fall_time)

    @staticmethod
    def _land(
        simulation: MujocoSim,
        perception: Perception,
        clock: _Clock,
        rows: List[Dict[Sample, Any]],
    ) -> None:
        """
        Step the physics with nothing moving until the last grains have landed.
        """
        period = SAMPLE_EVERY / CONTROL_FREQUENCY
        for _ in range(math.ceil(LANDING_TIME.total_seconds() / period)):
            simulation.step_simulation(timedelta(seconds=period))
            clock.advance(period)
            rows.append(perception.sample(clock.now, Phase.SETTLE))


@dataclass
class _Clock:
    """
    Simulated time since the grains came to rest, in seconds.
    """

    now: float = 0.0
    """
    The current time, in seconds.
    """

    def advance(self, seconds: float) -> None:
        """
        :param seconds: How much simulated time has passed.
        """
        self.now += seconds
