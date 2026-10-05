"""
A plan as the Plan Builder writes it: the ordered steps one robot performs.

The builder hands its steps over as data rather than as the Python it also generates, so
what a plan can ask for stays bounded, and a step naming an arm the robot has not got is
refused when it is read instead of failing inside a motion. Each step reads itself from
the builder's form of it and writes itself back, and turns into the coraplex action
carrying it out.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.factories import a, variable
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasSupportingSurface,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    CounterTop,
    Cupboard,
    Dishwasher,
    Drawer,
    Dresser,
    Floor,
    Fridge,
    ShelfLayer,
    Sofa,
    Table,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from cramera.live.placement_surface import PlacementSurface
from cramera.model_catalog import BuilderStep
from typing_extensions import Any, ClassVar, Dict, Iterator, List, Optional, Type, Union

# %% the builder's form of a step


class StepField(StrEnum):
    """
    The keys a step is written with in the builder's form of it.
    """

    TYPE = "type"
    PARAMETERS = "params"


class StepParameter(StrEnum):
    """
    The parameters a step's builder form names.
    """

    ARM = "arm"
    TORSO = "torso"
    X = "x"
    Y = "y"
    Z = "z"
    YAW = "yaw"
    OBJECT = "object"
    TARGET_MODE = "targetMode"
    SURFACE_TYPE = "surfaceType"
    SURFACE_NAME = "surfaceName"


class TargetMode(StrEnum):
    """
    How a step that puts an object down says where: at an exact pose, or on a surface
    found in the world when the step's turn comes.
    """

    POSE = "pose"
    SEMANTIC = "semantic"


SURFACE_TYPES: Dict[str, Type[HasSupportingSurface]] = {
    kind.__name__: kind
    for kind in (
        CounterTop,
        Table,
        ShelfLayer,
        Floor,
        Sofa,
        Drawer,
        Fridge,
        Cabinet,
        Cupboard,
        Dresser,
        Dishwasher,
    )
}
"""
The surfaces an object may be put down on, by the name the builder offers them under:
the same ones the live bridge lists to it, and what the placement backend samples poses
on.
"""


class MalformedPlanError(Exception):
    """
    Raised when the builder's form of a plan cannot be read.
    """


def _number(parameters: Dict[str, Any], key: StepParameter) -> float:
    """
    One finite coordinate of a step.

    :param parameters: The step's parameters.
    :param key: The parameter to read.
    :raises MalformedPlanError: If the value is missing or not a finite number.
    """
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MalformedPlanError(f"{key!r} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise MalformedPlanError(f"{key!r} must be finite, got {value!r}")
    return float(value)


def _member(enumeration: Type, parameters: Dict[str, Any], key: StepParameter) -> Any:
    """
    The enum member a step names.

    :param enumeration: The enumeration the name has to belong to.
    :param parameters: The step's parameters.
    :param key: The parameter holding the member's name.
    :raises MalformedPlanError: If the name is not one of the enumeration's members.
    """
    name = parameters.get(key)
    if name not in enumeration.__members__:
        raise MalformedPlanError(
            f"{key!r} must be one of {list(enumeration.__members__)}, got {name!r}"
        )
    return enumeration[name]


def _name(parameters: Dict[str, Any], key: StepParameter) -> str:
    """
    The name a step gives something.

    :param parameters: The step's parameters.
    :param key: The parameter holding the name.
    :raises MalformedPlanError: If the name is missing or not a string.
    """
    value = parameters.get(key)
    if not isinstance(value, str) or not value:
        raise MalformedPlanError(f"{key!r} must name something, got {value!r}")
    return value


# %% places a step names


@dataclass(frozen=True)
class Point:
    """
    A point in the world's frame, as the builder's scene lets one be set.
    """

    x: float
    """
    Position along the world's x axis, in metres.
    """

    y: float
    """
    Position along the world's y axis, in metres.
    """

    z: float
    """
    Height above the world's origin, in metres.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> Point:
        """
        :param parameters: A step's parameters.
        :return: The point they name.
        :raises MalformedPlanError: If a coordinate is missing or unusable.
        """
        return cls(
            x=_number(parameters, StepParameter.X),
            y=_number(parameters, StepParameter.Y),
            z=_number(parameters, StepParameter.Z),
        )

    def to_parameters(self) -> Dict[str, float]:
        """
        :return: The point in a step's parameters.
        """
        return {
            StepParameter.X: self.x,
            StepParameter.Y: self.y,
            StepParameter.Z: self.z,
        }

    def pose(self, world: World) -> Pose:
        """
        :param world: The world the point is expressed in.
        :return: A pose at the point, with no turn.
        """
        return Pose.from_xyz_rpy(self.x, self.y, self.z, reference_frame=world.root)


@dataclass(frozen=True)
class LevelPose(Point):
    """
    A pose given as a place and a heading, with no roll or pitch.
    """

    yaw: float = 0.0
    """
    Heading about the vertical axis, in radians.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> LevelPose:
        point = Point.from_parameters(parameters)
        return cls(
            x=point.x,
            y=point.y,
            z=point.z,
            yaw=_number(parameters, StepParameter.YAW),
        )

    def to_parameters(self) -> Dict[str, float]:
        return {**super().to_parameters(), StepParameter.YAW: self.yaw}

    def pose(self, world: World) -> Pose:
        return Pose.from_xyz_rpy(
            self.x, self.y, self.z, yaw=self.yaw, reference_frame=world.root
        )


@dataclass(frozen=True)
class SurfaceTarget:
    """
    A place to put an object down given as a kind of surface: any surface of that kind
    in the world, the one nearest the object first, or one named among them.

    The poses themselves are only found when the step's turn comes, on the world as it
    is then, so a surface another step has cleared or filled is read as it stands.
    """

    surface_type: str
    """
    The kind of surface, as :data:`SURFACE_TYPES` names it.
    """

    surface_name: Optional[str] = None
    """
    The name of the one surface to use, or ``None`` for whichever of the kind has room.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> SurfaceTarget:
        """
        :param parameters: A step's parameters.
        :return: The surface they name.
        :raises MalformedPlanError: If the kind is not one objects are put down on.
        """
        kind = parameters.get(StepParameter.SURFACE_TYPE)
        if kind not in SURFACE_TYPES:
            raise MalformedPlanError(
                f"{StepParameter.SURFACE_TYPE!r} must be one of {list(SURFACE_TYPES)},"
                f" got {kind!r}"
            )
        name = parameters.get(StepParameter.SURFACE_NAME) or None
        if name is not None and not isinstance(name, str):
            raise MalformedPlanError(
                f"{StepParameter.SURFACE_NAME!r} must be a name, got {name!r}"
            )
        return cls(surface_type=kind, surface_name=name)

    def to_parameters(self) -> Dict[str, Any]:
        """
        :return: The surface in a step's parameters.
        """
        return {
            StepParameter.SURFACE_TYPE: self.surface_type,
            StepParameter.SURFACE_NAME: self.surface_name or "",
        }

    def poses(self, world: World, body: Body) -> PlacementSurface:
        """
        :param world: The world the surface is looked for in.
        :param body: The object to put down, whose footprint has to fit.
        :return: The free, supported poses for the object on the surface, nearest first,
            found when first iterated.
        """
        return PlacementSurface(
            world=world,
            body=body,
            surface_type=SURFACE_TYPES[self.surface_type],
            surface_name=self.surface_name,
        )


DropOff = Union[LevelPose, SurfaceTarget]
"""
Where a step puts an object down: at a pose, or on a surface.
"""


def _drop_off(parameters: Dict[str, Any]) -> DropOff:
    """
    :param parameters: The parameters of a step that puts an object down.
    :return: Where it puts it, as the step's target mode says; at a pose when no mode is
        given, which is how a plan written before there were surfaces says it.
    :raises MalformedPlanError: If the mode is neither, or its target cannot be read.
    """
    mode = parameters.get(StepParameter.TARGET_MODE, TargetMode.POSE)
    if mode not in TargetMode.__members__.values():
        raise MalformedPlanError(
            f"{StepParameter.TARGET_MODE!r} must be one of"
            f" {[member.value for member in TargetMode]}, got {mode!r}"
        )
    if TargetMode(mode) is TargetMode.SEMANTIC:
        return SurfaceTarget.from_parameters(parameters)
    return LevelPose.from_parameters(parameters)


def _drop_off_parameters(target: DropOff) -> Dict[str, Any]:
    """
    :param target: Where a step puts an object down.
    :return: The target in the step's parameters, its mode among them.
    """
    mode = (
        TargetMode.SEMANTIC if isinstance(target, SurfaceTarget) else TargetMode.POSE
    )
    return {StepParameter.TARGET_MODE: mode.value, **target.to_parameters()}


# %% the objects a step acts on


def body_named(world: World, name: str) -> Body:
    """
    :param world: The world a plan is performed in.
    :param name: An object's name, as the builder's scene lists it: its body's own name,
        which for a carried object is the mesh name the viewer streams it under.
    :return: The body.
    :raises MalformedPlanError: If no body of the world is called that.
    """
    for body in world.bodies:
        if body.name.name == name or str(body.name) == name:
            return body
    raise MalformedPlanError(f"the world holds no object named {name!r}")


def graspable(world: World, body: Body) -> HasRootBody:
    """
    :param world: The world the body lies in.
    :param body: An object's body.
    :return: The annotation the manipulation actions take for the object: the one the
        world holds for it, or a bare one for a body annotated as nothing.
    """
    for annotation in world.get_semantic_annotations_by_type(HasRootBody):
        if annotation.root is body:
            return annotation
    return HasRootBody(root=body)


@dataclass
class DefaultGraspWhenGrounded:
    """
    The grasp the planner would choose for an object, chosen when the pick's turn comes
    rather than when the plan is written.

    The default grasp approaches from the side the robot's reach favours, so it depends
    on where the robot stands; in a plan that first navigates to the object, that is
    only known once the navigation is done. A plan's actions are expanded as soon as the
    plan is built - the viewer publishes its tree then - so the choice is kept out of
    the actions and made the domain of a query variable, which is only iterated when
    the pick is grounded.

    .. warning::
        :meth:`__iter__` must stay a generator, as
        :class:`~coraplex.locations.base.DeferredLocation` explains.
    """

    body: Body
    """
    The object to pick up.
    """

    arm: Arms
    """
    The arm that takes it.
    """

    robot: AbstractRobot
    """
    The robot it belongs to.
    """

    def __iter__(self) -> Iterator[GraspDescription]:
        yield GraspDescription.robot_relative_default(
            ViewManager.get_end_effector_view(self.arm, self.robot),
            self.body.global_pose,
            self.body,
        )


# %% the steps themselves


@dataclass(frozen=True)
class PlanStep(ABC):
    """
    One step of a plan the builder wrote.
    """

    STEP_TYPE: ClassVar[BuilderStep]
    """
    What the step is called in the builder's form of it.
    """

    @classmethod
    @abstractmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> PlanStep:
        """
        Read the step off its parameters.

        :param parameters: The step's parameters.
        :raises MalformedPlanError: If a parameter is missing or unusable.
        """

    @abstractmethod
    def to_parameters(self) -> Dict[str, Any]:
        """
        :return: The step's parameters, as :meth:`from_parameters` reads them.
        """

    @abstractmethod
    def action(self, context: Context) -> Union[ActionDescription, Match]:
        """
        The coraplex action carrying this step out, or the query that grounds it when
        its turn comes.

        :param context: The running scene, whose world the step's poses and objects are
            resolved in.
        """

    def to_payload(self) -> Dict[str, Any]:
        """
        :return: The builder's form of the step.
        """
        return {
            StepField.TYPE: self.STEP_TYPE.value,
            StepField.PARAMETERS: self.to_parameters(),
        }


@dataclass(frozen=True)
class ParkArms(PlanStep):
    """
    Bring an arm back to its parked pose.
    """

    STEP_TYPE: ClassVar[BuilderStep] = BuilderStep.PARK_ARMS

    arm: Arms
    """
    The arm to park.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> ParkArms:
        return cls(arm=_member(Arms, parameters, StepParameter.ARM))

    def to_parameters(self) -> Dict[str, Any]:
        return {StepParameter.ARM: self.arm.name}

    def action(self, context: Context) -> ActionDescription:
        return ParkArmsAction(self.arm)


@dataclass(frozen=True)
class MoveTorso(PlanStep):
    """
    Raise or lower the torso.
    """

    STEP_TYPE: ClassVar[BuilderStep] = BuilderStep.MOVE_TORSO

    torso_state: TorsoState
    """
    The height the torso is moved to.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> MoveTorso:
        return cls(torso_state=_member(TorsoState, parameters, StepParameter.TORSO))

    def to_parameters(self) -> Dict[str, Any]:
        return {StepParameter.TORSO: self.torso_state.name}

    def action(self, context: Context) -> ActionDescription:
        return MoveTorsoAction(self.torso_state)


@dataclass(frozen=True)
class Navigate(PlanStep):
    """
    Drive the robot's base somewhere.
    """

    STEP_TYPE: ClassVar[BuilderStep] = BuilderStep.NAVIGATE

    target: LevelPose
    """
    Where the robot drives to, and which way it ends up facing.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> Navigate:
        return cls(target=LevelPose.from_parameters(parameters))

    def to_parameters(self) -> Dict[str, Any]:
        return self.target.to_parameters()

    def action(self, context: Context) -> ActionDescription:
        return NavigateAction(self.target.pose(context.world))


@dataclass(frozen=True)
class LookAt(PlanStep):
    """
    Point the robot's default camera at a point.
    """

    STEP_TYPE: ClassVar[BuilderStep] = BuilderStep.LOOK_AT

    target: Point
    """
    The point looked at.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> LookAt:
        return cls(target=Point.from_parameters(parameters))

    def to_parameters(self) -> Dict[str, Any]:
        return self.target.to_parameters()

    def action(self, context: Context) -> ActionDescription:
        return LookAtAction(self.target.pose(context.world))


@dataclass(frozen=True)
class Pick(PlanStep):
    """
    Pick an object up, with the grasp the planner chooses for where the robot stands
    when the pick's turn comes.
    """

    STEP_TYPE: ClassVar[BuilderStep] = BuilderStep.PICK

    object_name: str
    """
    The object's name in the world.
    """

    arm: Arms
    """
    The arm that takes it.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> Pick:
        return cls(
            object_name=_name(parameters, StepParameter.OBJECT),
            arm=_member(Arms, parameters, StepParameter.ARM),
        )

    def to_parameters(self) -> Dict[str, Any]:
        return {StepParameter.OBJECT: self.object_name, StepParameter.ARM: self.arm.name}

    def action(self, context: Context) -> Match:
        body = body_named(context.world, self.object_name)
        return a(PickUpAction)(
            object_designator=graspable(context.world, body),
            arm=self.arm,
            grasp_description=variable(
                GraspDescription,
                domain=DefaultGraspWhenGrounded(body, self.arm, context.robot),
            ),
        )


@dataclass(frozen=True)
class Place(PlanStep):
    """
    Put a held object down, at a pose or on a surface.
    """

    STEP_TYPE: ClassVar[BuilderStep] = BuilderStep.PLACE

    object_name: str
    """
    The object's name in the world.
    """

    arm: Arms
    """
    The arm holding it.
    """

    target: DropOff
    """
    Where it is put down.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> Place:
        return cls(
            object_name=_name(parameters, StepParameter.OBJECT),
            arm=_member(Arms, parameters, StepParameter.ARM),
            target=_drop_off(parameters),
        )

    def to_parameters(self) -> Dict[str, Any]:
        return {
            StepParameter.OBJECT: self.object_name,
            StepParameter.ARM: self.arm.name,
            **_drop_off_parameters(self.target),
        }

    def action(self, context: Context) -> Match:
        # Grounded when its turn comes, as a query: a place action built outright reads
        # the grasp off the hand, which holds nothing until the pick before it is done.
        body = body_named(context.world, self.object_name)
        target = (
            self.target.pose(context.world)
            if isinstance(self.target, LevelPose)
            else variable(Pose, domain=self.target.poses(context.world, body))
        )
        return a(PlaceAction)(object_designator=body, target_location=target, arm=self.arm)


@dataclass(frozen=True)
class Transport(PlanStep):
    """
    Carry an object somewhere: drive to it, pick it up, drive to where it goes and put
    it down, with the robot choosing where to stand for each.
    """

    STEP_TYPE: ClassVar[BuilderStep] = BuilderStep.TRANSPORT

    object_name: str
    """
    The object's name in the world.
    """

    arm: Arms
    """
    The arm that carries it.
    """

    target: DropOff
    """
    Where it is put down.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> Transport:
        return cls(
            object_name=_name(parameters, StepParameter.OBJECT),
            arm=_member(Arms, parameters, StepParameter.ARM),
            target=_drop_off(parameters),
        )

    def to_parameters(self) -> Dict[str, Any]:
        return {
            StepParameter.OBJECT: self.object_name,
            StepParameter.ARM: self.arm.name,
            **_drop_off_parameters(self.target),
        }

    def action(self, context: Context) -> ActionDescription:
        body = body_named(context.world, self.object_name)
        target = (
            self.target.pose(context.world)
            if isinstance(self.target, LevelPose)
            else self.target.poses(context.world, body)
        )
        return TransportAction(
            object_designator=graspable(context.world, body),
            target_location=target,
            arm=self.arm,
        )


# %% a whole plan


@dataclass(frozen=True)
class ObjectStepInPlanError(MalformedPlanError):
    """
    Raised for a step that looks for an object through perception, which a plan read
    on its own has no perception for.
    """

    step_type: BuilderStep
    """
    The step that was asked for.
    """

    def __str__(self) -> str:
        return (
            f"a {self.step_type} step looks for an object, and this plan has no"
            f" perception; readable are {[kind.STEP_TYPE.value for kind in STEP_KINDS]}"
        )


STEP_KINDS: List[Type[PlanStep]] = [
    ParkArms,
    MoveTorso,
    Navigate,
    LookAt,
    Pick,
    Place,
    Transport,
]
"""
The steps a plan read on its own can hold. The ones acting on an object name it as the
world the plan is performed in names its body.
"""


@dataclass(frozen=True)
class BuilderPlan:
    """
    The steps one robot performs, in order.
    """

    steps: List[PlanStep] = field(default_factory=list)
    """
    The steps to perform, in order.
    """

    @classmethod
    def from_payload(cls, payload: List[Any]) -> BuilderPlan:
        """
        :param payload: The builder's form of the steps.
        :return: The plan they describe.
        :raises MalformedPlanError: If the plan or one of its steps is unusable.
        """
        if not isinstance(payload, list):
            raise MalformedPlanError("a plan is a list of steps")
        return cls(steps=[cls._step(entry) for entry in payload])

    @staticmethod
    def _step(entry: Any) -> PlanStep:
        """
        :param entry: One step in the builder's form.
        :return: The step it describes.
        :raises MalformedPlanError: If the entry names no step a plan can hold.
        """
        if not isinstance(entry, dict):
            raise MalformedPlanError("every step must be an object")
        named = entry.get(StepField.TYPE)
        if named not in BuilderStep.__members__.values():
            raise MalformedPlanError(
                f"{StepField.TYPE!r} must be one of {[kind.value for kind in BuilderStep]},"
                f" got {named!r}"
            )
        by_type = {kind.STEP_TYPE: kind for kind in STEP_KINDS}
        if BuilderStep(named) not in by_type:
            raise ObjectStepInPlanError(BuilderStep(named))
        parameters = entry.get(StepField.PARAMETERS) or {}
        if not isinstance(parameters, dict):
            raise MalformedPlanError(f"{StepField.PARAMETERS!r} must be an object")
        return by_type[BuilderStep(named)].from_parameters(parameters)

    def to_payload(self) -> List[Dict[str, Any]]:
        """
        :return: The builder's form of the steps.
        """
        return [step.to_payload() for step in self.steps]

    def plan(self, context: Context) -> PlanNode:
        """
        :param context: The running scene's context, whose world the steps resolve in.
        :return: The coraplex plan performing these steps.
        """
        actions = [step.action(context) for step in self.steps]
        return sequential(actions, context=context).plan
