"""
A demo setup: the environment a demo runs in, the robots standing in it and what each of
them does.

A setup is kept as a file of its own, so the Plan Builder can open one, move its robots
elsewhere and save the result as another, and a scene can be brought up from one without
being told again where anything stands. Each robot either performs a plan of its own or
follows the joint states of the real robot it stands for.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.adapters.usd.stage_parser import RootPlacement
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from typing_extensions import Any, ClassVar, Dict, List, Mapping, Optional, Self

from cramera.body_geometry import DrawnGeometry
from cramera.environment_file import EnvironmentFile, USDSceneEnvironmentFile
from cramera.model_catalog import EnvironmentKind
from cramera.multi_robot import (
    RobotInstance,
    RobotScene,
    UnknownEnvironmentJointError,
    stand_joints_at,
    yaw_of,
)
from cramera.paths import SCENE_NAME_PATTERN, setups_directory
from cramera.plan_steps import BuilderPlan

# %% the builder's form of a setup


class SetupField(StrEnum):
    """
    The keys a setup is written with in the Plan Builder's form of it.
    """

    ENVIRONMENT = "environment"
    PATH = "path"
    KIND = "kind"
    CLASS = "cls"
    ROOT_PLACEMENT = "rootPlacement"
    ROBOTS = "robots"
    IDENTIFIER = "identifier"
    LABEL = "label"
    MODEL = "model"
    X = "x"
    Y = "y"
    YAW = "yaw"
    JOINT_STATE_TOPIC = "jointStateTopic"
    LOCALIZATION_TOPIC = "localizationTopic"
    REPEATS_PLAN = "repeatsPlan"
    STEPS = "steps"
    ENVIRONMENT_JOINT_POSITIONS = "environmentJointPositions"
    ENVIRONMENT_GEOMETRY = "environmentGeometry"
    OBJECTS = "objects"
    NAME = "name"
    Z = "z"
    SIZE = "size"


class MalformedSetupError(Exception):
    """
    Raised when the Plan Builder's form of a setup cannot be read.
    """


@dataclass
class UnknownRobotModelError(MalformedSetupError):
    """
    Raised for a robot model the installation offers no annotation for.
    """

    model: str
    """
    The model that was named.
    """

    def __str__(self) -> str:
        return f"no robot model called {self.model!r} is installed"


@dataclass
@dataclass
class MapEnvironmentInSetupError(MalformedSetupError):
    """
    Raised for an environment a class builds rather than a file describes: a setup names
    the environment's file, so it cannot stand its robots in a map.
    """

    map: str
    """
    The map class that was named.
    """

    def __str__(self) -> str:
        return (
            f"{self.map} is a map built by a class, not an environment file;"
            f" a setup names a file, so it cannot stand its robots in {self.map}"
        )


# %% one robot of a setup


@dataclass
class RobotSetup:
    """
    One robot of a demo setup: where it stands and what moves it.
    """

    instance: RobotInstance
    """
    The robot, its name and where it starts.
    """

    joint_state_topic: Optional[str] = None
    """
    The ``sensor_msgs/JointState`` topic the robot's joints follow, for a robot standing
    in for a real one, or ``None`` for a robot moved by its plan alone.
    """

    localization_topic: Optional[str] = None
    """
    The ``nav_msgs/Odometry`` topic the robot's base follows, for a robot standing in
    for a real one that localizes itself, or ``None`` for a base that stays where it is
    put.

    A localized robot's pose in :attr:`instance` is where the origin of the map it
    localizes in lies in the environment, so the reported pose is taken from there.
    """

    plan: BuilderPlan = field(default_factory=BuilderPlan)
    """
    What the robot does once the scene is up; or, for a robot that follows a real one,
    the plan the plan builder opens with for it, ready to be run on it: a scene leaves
    such a robot to the real one, since what the real robot reports would take every
    position a plan moved to straight back.
    """

    repeats_plan: bool = False
    """
    Whether the robot starts its plan again each time it has finished it, until the
    scene is stopped.
    """

    @property
    def follows_a_real_robot(self) -> bool:
        """
        :return: Whether the robot is moved by what a real robot reports rather than by
            a plan of its own.
        """
        return bool(self.joint_state_topic or self.localization_topic)

    @property
    def yaw(self) -> float:
        """
        :return: Which way the robot faces where it starts, in radians.
        """
        return yaw_of(self.instance.pose.to_np())

    @classmethod
    def from_payload(
        cls,
        payload: Dict[str, Any],
        robot_types: Mapping[str, type[AbstractRobot]],
    ) -> Self:
        """
        :param payload: The Plan Builder's form of one robot.
        :param robot_types: The installed robot annotations, by the model name the
            builder lists them under.
        :return: The robot it describes.
        :raises MalformedSetupError: If the robot or its plan cannot be read.
        """
        model = payload.get(SetupField.MODEL)
        if model not in robot_types:
            raise UnknownRobotModelError(str(model))
        coordinates = [
            payload.get(key) for key in (SetupField.X, SetupField.Y, SetupField.YAW)
        ]
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in coordinates
        ):
            raise MalformedSetupError(f"a robot's x, y and yaw are numbers: {payload}")
        x, y, yaw = coordinates
        topic = payload.get(SetupField.JOINT_STATE_TOPIC) or None
        return cls(
            instance=RobotInstance(
                identifier=str(payload.get(SetupField.IDENTIFIER, "")),
                label=str(payload.get(SetupField.LABEL, "")),
                robot_type=robot_types[model],
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=x, y=y, yaw=yaw),
            ),
            joint_state_topic=topic,
            localization_topic=payload.get(SetupField.LOCALIZATION_TOPIC) or None,
            plan=BuilderPlan.from_payload(payload.get(SetupField.STEPS, [])),
            repeats_plan=bool(payload.get(SetupField.REPEATS_PLAN, False)),
        )

    def to_payload(self) -> Dict[str, Any]:
        """
        :return: The Plan Builder's form of this robot.
        """
        pose = self.instance.pose.to_np()
        return {
            SetupField.IDENTIFIER: self.instance.identifier,
            SetupField.LABEL: self.instance.label,
            SetupField.MODEL: self.instance.robot_type.__name__,
            SetupField.X: float(pose[0, 3]),
            SetupField.Y: float(pose[1, 3]),
            SetupField.YAW: self.yaw,
            SetupField.JOINT_STATE_TOPIC: self.joint_state_topic or "",
            SetupField.LOCALIZATION_TOPIC: self.localization_topic or "",
            SetupField.REPEATS_PLAN: self.repeats_plan,
            SetupField.STEPS: self.plan.to_payload(),
        }


# %% the objects lying about


@dataclass
class ObjectSetup:
    """
    One box of a demo setup, lying somewhere in the environment to be carried.
    """

    name: str
    """
    What the box is called in the world; what a plan's pick, place and transport steps
    name it by. A name ending in a mesh file's suffix is one the viewer streams pose by
    pose, so the box can change hands without the viewer loading the scene again.
    """

    pose: HomogeneousTransformationMatrix
    """
    Where the middle of the box starts out, in the environment's frame.
    """

    size: Scale = field(default_factory=lambda: Scale(0.06, 0.06, 0.1))
    """
    How big the box is, in metres.
    """

    @property
    def yaw(self) -> float:
        """
        :return: Which way the box is turned, in radians.
        """
        return yaw_of(self.pose.to_np())

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> Self:
        """
        :param payload: The Plan Builder's form of one box.
        :return: The box it describes.
        :raises MalformedSetupError: If the box has no name, no place or no size.
        """
        name = payload.get(SetupField.NAME)
        if not isinstance(name, str) or not name:
            raise MalformedSetupError(f"an object needs a name, not {name!r}")
        try:
            coordinates = [
                float(payload[key]) for key in (SetupField.X, SetupField.Y, SetupField.Z)
            ]
            yaw = float(payload.get(SetupField.YAW, 0.0))
            size = [float(side) for side in payload[SetupField.SIZE]]
        except (KeyError, TypeError, ValueError) as error:
            raise MalformedSetupError(
                f"an object lies at x, y, z with a yaw and a size of three sides: {name}"
            ) from error
        if len(size) != 3 or not all(math.isfinite(v) for v in coordinates + size):
            raise MalformedSetupError(f"{name} has no finite place or size")
        return cls(
            name=name,
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(*coordinates, yaw=yaw),
            size=Scale(*size),
        )

    def to_payload(self) -> Dict[str, Any]:
        """
        :return: The Plan Builder's form of this box.
        """
        pose = self.pose.to_np()
        return {
            SetupField.NAME: self.name,
            SetupField.X: float(pose[0, 3]),
            SetupField.Y: float(pose[1, 3]),
            SetupField.Z: float(pose[2, 3]),
            SetupField.YAW: self.yaw,
            SetupField.SIZE: [self.size.x, self.size.y, self.size.z],
        }


# %% a whole setup


@dataclass
class DemoSetup:
    """
    The environment of a demo, the robots standing in it and the boxes lying about in
    it.
    """

    environment: Optional[EnvironmentFile]
    """
    What the robots stand in, or ``None`` for an empty world.
    """

    robots: List[RobotSetup]
    """
    The robots, in the order they are listed.
    """

    objects: List[ObjectSetup] = field(default_factory=list)
    """
    The boxes lying about to be carried, each where it starts out.
    """

    environment_joint_positions: Dict[str, float] = field(default_factory=dict)
    """
    Where the environment's own joints stand, by connection name - how far each door
    of a building stands open, say. A joint left out stands where its description puts
    it.
    """

    environment_geometry: DrawnGeometry = DrawnGeometry.VISUAL
    """
    Which of its geometries the environment is drawn with - as the boxes it collides as,
    say, for a scanned building too heavy to draw as it looks.
    """

    def pose_environment(self, world: World) -> None:
        """
        Stand the environment's joints of a world built from this setup where the setup
        says.

        :param world: The world built from this setup.
        :raises UnknownEnvironmentJointError: If the world has no joint of a name the
            setup gives a position for.
        """
        stand_joints_at(world, self.environment_joint_positions)

    @property
    def robot_scene(self) -> RobotScene:
        """
        :return: The robots as a scene of independently named instances, the first one
            selected.
        """
        return RobotScene(
            instances=[robot.instance for robot in self.robots],
            active_identifier=self.robots[0].instance.identifier,
            environment_joint_positions=dict(self.environment_joint_positions),
        )

    # %% the file a setup is kept in

    def save(self, path: Path) -> None:
        """
        Write the setup to a file.

        An environment beside the file is named relative to it, so the two can be moved
        together, as :meth:`load` reads it.

        :param path: The file to write.
        """
        written = replace(self, environment=self._environment_seen_from(path.parent))
        path.write_text(json.dumps(to_json(written), indent=2))

    def _environment_seen_from(self, directory: Path) -> Optional[EnvironmentFile]:
        """
        :param directory: Where the setup is written.
        :return: The environment, named relative to that directory if it lies there.
        """
        if self.environment is None or _is_relative(self.environment.path):
            return self.environment
        file = Path(self.environment.path)
        if not file.is_relative_to(directory.resolve()):
            return self.environment
        return replace(
            self.environment, path=str(file.relative_to(directory.resolve()))
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Read a setup from a file.

        An environment named by a relative path is found beside the file, so a setup can
        travel with the environment it names.

        :param path: The file to read.
        :return: The setup it holds.
        """
        setup: DemoSetup = from_json(json.loads(path.read_text()))
        if setup.environment is not None and _is_relative(setup.environment.path):
            setup.environment.path = str(
                (path.parent / setup.environment.path).resolve()
            )
        return setup

    # %% the builder's form

    @classmethod
    def from_payload(
        cls,
        payload: Dict[str, Any],
        robot_types: Mapping[str, type[AbstractRobot]],
    ) -> Self:
        """
        :param payload: The Plan Builder's form of a setup.
        :param robot_types: The installed robot annotations, by the model name the
            builder lists them under.
        :return: The setup it describes.
        :raises MalformedSetupError: If the setup cannot be read.
        """
        robots = payload.get(SetupField.ROBOTS)
        if not isinstance(robots, list) or not robots:
            raise MalformedSetupError("a setup holds at least one robot")
        return cls(
            environment=cls._environment_from_payload(
                payload.get(SetupField.ENVIRONMENT)
            ),
            robots=[RobotSetup.from_payload(robot, robot_types) for robot in robots],
            objects=cls._objects_from_payload(payload.get(SetupField.OBJECTS) or []),
            environment_joint_positions=cls._joint_positions_from_payload(
                payload.get(SetupField.ENVIRONMENT_JOINT_POSITIONS) or {}
            ),
            environment_geometry=cls._drawn_geometry_from_payload(
                payload.get(SetupField.ENVIRONMENT_GEOMETRY) or DrawnGeometry.VISUAL
            ),
        )

    @staticmethod
    def _objects_from_payload(payload: Any) -> List[ObjectSetup]:
        """
        :param payload: The builder's form of the boxes lying about.
        :return: The boxes it describes.
        :raises MalformedSetupError: If it is not a list of boxes, or two share a name.
        """
        if not isinstance(payload, list):
            raise MalformedSetupError(f"objects are a list, not {payload!r}")
        objects = [ObjectSetup.from_payload(entry) for entry in payload]
        names = [box.name for box in objects]
        if len(set(names)) != len(names):
            raise MalformedSetupError(f"an object is placed only once, got {names}")
        return objects

    @staticmethod
    def _drawn_geometry_from_payload(payload: Any) -> DrawnGeometry:
        """
        :param payload: The builder's form of how the environment is drawn.
        :return: The geometry it names.
        :raises MalformedSetupError: If it names no geometry a body is drawn with.
        """
        if payload not in DrawnGeometry.__members__.values():
            raise MalformedSetupError(
                f"the environment is drawn as one of {[kind.value for kind in DrawnGeometry]},"
                f" not {payload!r}"
            )
        return DrawnGeometry(payload)

    @staticmethod
    def _joint_positions_from_payload(payload: Any) -> Dict[str, float]:
        """
        :param payload: The builder's form of the environment's joint positions.
        :return: The positions it names, by connection name.
        :raises MalformedSetupError: If it is not a mapping of names to finite numbers.
        """
        if not isinstance(payload, dict) or not all(
            isinstance(position, (int, float))
            and not isinstance(position, bool)
            and math.isfinite(position)
            for position in payload.values()
        ):
            raise MalformedSetupError(
                f"joint positions are names mapped to numbers, not {payload!r}"
            )
        return {str(name): float(position) for name, position in payload.items()}

    @staticmethod
    def _environment_from_payload(
        payload: Optional[Dict[str, Any]],
    ) -> Optional[EnvironmentFile]:
        """
        :param payload: The builder's form of an environment, or ``None`` for none.
        :return: The environment file it names.
        :raises MapEnvironmentInSetupError: If it names a map instead of a file.
        """
        if not payload:
            return None
        if payload.get(SetupField.KIND) == EnvironmentKind.MAP:
            raise MapEnvironmentInSetupError(str(payload.get(SetupField.CLASS, "")))
        environment = EnvironmentFile.from_path(str(payload[SetupField.PATH]))
        placement = payload.get(SetupField.ROOT_PLACEMENT)
        if isinstance(environment, USDSceneEnvironmentFile) and placement:
            environment.root_placement = RootPlacement(placement)
        return environment

    def to_payload(self) -> Dict[str, Any]:
        """
        :return: The Plan Builder's form of this setup.
        """
        environment = None
        if self.environment is not None:
            environment = {SetupField.PATH: self.environment.path}
            if isinstance(self.environment, USDSceneEnvironmentFile):
                environment[SetupField.ROOT_PLACEMENT] = (
                    self.environment.root_placement.value
                )
        return {
            SetupField.ENVIRONMENT: environment,
            SetupField.ROBOTS: [robot.to_payload() for robot in self.robots],
            SetupField.OBJECTS: [box.to_payload() for box in self.objects],
            SetupField.ENVIRONMENT_JOINT_POSITIONS: dict(
                self.environment_joint_positions
            ),
            SetupField.ENVIRONMENT_GEOMETRY: self.environment_geometry.value,
        }


def _is_relative(path: str) -> bool:
    """
    :param path: A path or a ``package://`` URL.
    :return: Whether it is a path relative to somewhere else.
    """
    return "://" not in path and not Path(path).is_absolute()


# %% the setups the Plan Builder saves


@dataclass
class SetupNameTakenError(Exception):
    """
    Raised when a setup is saved under the name of one already saved.
    """

    name: str
    """
    The name asked for.
    """

    def __str__(self) -> str:
        return f"a setup called {self.name!r} is saved already; choose another name"


@dataclass
class InvalidSetupNameError(ValueError):
    """
    Raised for a setup name that cannot name a file.
    """

    name: str
    """
    The name asked for.
    """

    def __str__(self) -> str:
        return (
            f"{self.name!r} is no setup name: letters, digits, '_' and '-',"
            f" up to 64 of them"
        )


@dataclass
class SetupLibrary:
    """
    The directory the Plan Builder saves setups into and opens them from.

    A setup saved there is never overwritten: saving under a name already taken is
    refused, so a setup that has been moved about is kept beside the one it came from.
    """

    directory: Path = field(default_factory=setups_directory)
    """
    Where the setups are kept, one file each.
    """

    SUFFIX: ClassVar[str] = ".json"
    """
    The ending of a setup's file.
    """

    def names(self) -> List[str]:
        """
        :return: The name of every saved setup, alphabetically.
        """
        if not self.directory.is_dir():
            return []
        return sorted(path.stem for path in self.directory.glob(f"*{self.SUFFIX}"))

    def path_of(self, name: str) -> Path:
        """
        :param name: A setup's name.
        :return: The file it is kept in.
        :raises InvalidSetupNameError: If the name cannot name a file.
        """
        if not SCENE_NAME_PATTERN.fullmatch(name):
            raise InvalidSetupNameError(name)
        return self.directory / f"{name}{self.SUFFIX}"

    def open(self, name: str) -> DemoSetup:
        """
        :param name: A saved setup's name.
        :return: The setup.
        """
        return DemoSetup.load(self.path_of(name))

    def save(self, name: str, setup: DemoSetup) -> Path:
        """
        Save a setup under a name nothing is saved under yet.

        :param name: The name to save it under.
        :param setup: The setup.
        :return: The file it was written to.
        :raises SetupNameTakenError: If a setup is saved under that name already.
        """
        path = self.path_of(name)
        if path.exists():
            raise SetupNameTakenError(name)
        self.directory.mkdir(parents=True, exist_ok=True)
        setup.save(path)
        return path
