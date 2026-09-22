"""Plan-builder models and capabilities from CRAM's semantic annotations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path

from typing_extensions import Any, ClassVar, get_args

from cramera.payload import CrameraPayload
from krrood.class_diagrams.attribute_introspector import DataclassOnlyIntrospector
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.icub3 import ICub3
from semantic_digital_twin.robots.justin import Justin
from semantic_digital_twin.robots.mmp_dresden import MMPDresden
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_part_mixins import (
    HasLeftRightArm,
    HasMobileBase,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    AbstractRobotPart,
    Arm,
    Torso,
)
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1

# %% frontend vocabulary


class BuilderStep(StrEnum):
    """Operations offered by the plan builder."""

    PARK_ARMS = "park_arms"
    """Move an arm to its park state."""
    MOVE_TORSO = "move_torso"
    """Move a torso to a named height."""
    NAVIGATE = "navigate"
    """Drive a mobile base to a pose."""
    TRANSPORT = "transport"
    """Pick up and place an object."""
    PICK = "pick"
    """Pick up an object."""
    PLACE = "place"
    """Place a held object."""


class CatalogField(StrEnum):
    """Keys of the model catalog's browser payload."""

    OK = "ok"
    """Whether the catalog request succeeded."""
    NAME = "name"
    """Human-readable model name."""
    CLASS = "cls"
    """Concrete robot annotation class name."""
    IMPORT = "import"
    """Import statement for the annotation."""
    STEPS = "steps"
    """Supported builder operations."""
    ARMS = "arms"
    """Supported coraplex arm selections."""
    PATH = "path"
    """Local environment description path."""
    ROBOTS = "robots"
    """Registered robot choices."""
    ENVIRONMENTS = "environments"
    """Installed environment choices."""


class BuilderArm(StrEnum):
    """Arm selections understood by coraplex actions."""

    LEFT = "LEFT"
    """The robot's left arm."""
    RIGHT = "RIGHT"
    """The robot's right arm."""
    BOTH = "BOTH"
    """Both arms, or the sole available arm."""


# %% robot capabilities


@dataclass
class RobotModel:
    """An annotated robot and the operations its part structure supports."""

    annotation: type[AbstractRobot]
    """The CRAM annotation that owns the robot's description and parts."""

    def part_types(self) -> set[type[AbstractRobot | AbstractRobotPart]]:
        """Collect nested part annotations through CRAM's dataclass introspector.

        :return: Robot and part classes declared by the annotation.
        """
        pending = [self.annotation]
        result = set()
        introspector = DataclassOnlyIntrospector()
        while pending:
            candidate = pending.pop()
            arguments = get_args(candidate)
            if arguments:
                pending.extend(arguments)
                continue
            if (
                not isinstance(candidate, type)
                or not issubclass(candidate, (AbstractRobot, AbstractRobotPart))
                or candidate in result
            ):
                continue
            result.add(candidate)
            pending.extend(
                attribute.field.type for attribute in introspector.discover(candidate)
            )
        return result

    @property
    def steps(self) -> list[BuilderStep]:
        """Return operations supported by the robot's declared semantic parts."""
        parts = self.part_types()
        supported = []
        if any(issubclass(part, Arm) for part in parts):
            supported.extend(
                [
                    BuilderStep.PARK_ARMS,
                    BuilderStep.TRANSPORT,
                    BuilderStep.PICK,
                    BuilderStep.PLACE,
                ]
            )
        if any(issubclass(part, Torso) for part in parts):
            supported.append(BuilderStep.MOVE_TORSO)
        if issubclass(self.annotation, HasMobileBase):
            supported.append(BuilderStep.NAVIGATE)
        return [step for step in BuilderStep if step in supported]

    @property
    def arms(self) -> list[BuilderArm]:
        """Return named arms, or coraplex's default selection for a single arm."""
        if any(issubclass(part, HasLeftRightArm) for part in self.part_types()):
            return list(BuilderArm)
        return [BuilderArm.BOTH] if BuilderStep.PICK in self.steps else []

    def to_payload(self) -> dict[str, Any]:
        """Return the browser's import and capability information."""
        return {
            CatalogField.NAME: self.annotation.__name__,
            CatalogField.CLASS: self.annotation.__name__,
            CatalogField.IMPORT: f"from {self.annotation.__module__} import {self.annotation.__name__}",
            CatalogField.STEPS: self.steps,
            CatalogField.ARMS: self.arms,
        }


# %% installed environments


@dataclass
class EnvironmentModel:
    """An installed world description selectable as a builder environment."""

    path: str
    """Absolute path passed to CRAM's world specification."""

    def to_payload(self) -> dict[str, str]:
        """Return a readable environment name and its description path."""
        return {
            CatalogField.NAME: Path(self.path).stem.replace("_", " "),
            CatalogField.PATH: self.path,
        }


@dataclass
class ModelCatalog(CrameraPayload):
    """CRAM robot annotations and world descriptions available for authoring."""

    robots: list[RobotModel]
    """Registered robot annotations with derived capabilities."""

    worlds_directory: Path
    """Directory containing installed coraplex environment descriptions."""

    ANNOTATIONS: ClassVar[tuple[type[AbstractRobot], ...]] = (
        PR2,
        Garmi,
        HSRB,
        Tiago,
        Stretch,
        Armar7,
        Justin,
        ICub3,
        MMPDresden,
        Tracy,
        DAiSy,
        UnitreeG1,
    )
    """Annotated robots with a concrete model-description provider."""

    @classmethod
    def installed(cls) -> ModelCatalog:
        """Read the installed CRAM packages' model inventory.

        :return: The authoring catalog for this installation.
        """
        return cls(
            robots=[RobotModel(annotation) for annotation in cls.ANNOTATIONS],
            worlds_directory=Path(str(files("coraplex"))).parent.parent
            / "resources"
            / "worlds",
        )

    @property
    def environments(self) -> list[EnvironmentModel]:
        """Return every URDF world shipped in the environment directory."""
        return [
            EnvironmentModel(str(path))
            for path in sorted(self.worlds_directory.glob("*.urdf"))
        ]

    def to_payload(self) -> dict[str, Any]:
        """Return robot capabilities and environment choices for the browser."""
        return {
            CatalogField.OK: self.ok,
            CatalogField.ROBOTS: [robot.to_payload() for robot in self.robots],
            CatalogField.ENVIRONMENTS: [
                environment.to_payload() for environment in self.environments
            ],
        }
