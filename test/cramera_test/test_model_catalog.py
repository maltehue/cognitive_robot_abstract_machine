"""
The model catalog derives choices from CRAM's semantic robot annotations.
"""

from pathlib import Path

import pytest

from cramera.model_catalog import ModelCatalog, RobotModel, BuilderStep
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.minimal_robot import MinimalRobot

# %% registered models


def test_catalog_includes_supported_annotations(supported_abstract_robots) -> None:
    """
    Every robot supported by the shared CRAM fixture is offered.

    :param supported_abstract_robots: CRAM's existing supported annotation inventory.
    """
    assert set(supported_abstract_robots) <= {
        model.annotation for model in ModelCatalog.installed().robots
    }


@pytest.mark.parametrize(
    "annotation, excluded",
    [
        (Tracy, {BuilderStep.NAVIGATE, BuilderStep.MOVE_TORSO}),
        (MinimalRobot, set(BuilderStep)),
    ],
)
def test_capabilities_follow_robot_parts(annotation, excluded) -> None:
    """
    Unavailable operations are omitted from the annotated robot's palette.

    :param annotation: The robot annotation to inspect.
    :param excluded: Operations the annotation cannot support.
    """
    assert not (set(RobotModel(annotation).steps) & excluded)


def test_mobile_manipulator_has_every_builder_step() -> None:
    """
    PR2's nested base, torso and arms expose the complete palette.
    """
    assert set(RobotModel(PR2).steps) == set(BuilderStep)


def test_one_arm_robot_uses_coraplex_default_arm() -> None:
    """
    A single arm is selected through CRAM's BOTH/default-arm convention.
    """
    assert RobotModel(HSRB).arms == ["BOTH"]


def test_environment_catalog_matches_installed_worlds() -> None:
    """
    Every installed URDF environment is offered without another HTML list.
    """
    catalog = ModelCatalog.installed()
    assert {Path(model.path) for model in catalog.environments} == set(
        catalog.worlds_directory.glob("*.urdf")
    )


def test_payload_contains_imports_and_capabilities() -> None:
    """
    The frontend receives the concrete import and supported operation names.
    """
    payload = RobotModel(Tracy).to_payload()
    assert payload["import"] == f"from {Tracy.__module__} import {Tracy.__name__}"
    assert payload["steps"] == [step.value for step in RobotModel(Tracy).steps]
    assert payload["arms"] == ["LEFT", "RIGHT", "BOTH"]


def test_catalog_payload_matches_installed_models() -> None:
    """
    Every domain entry is represented in the browser's catalog.
    """
    catalog = ModelCatalog.installed()
    payload = catalog.to_payload()
    assert payload["ok"] is True
    assert payload["robots"] == [robot.to_payload() for robot in catalog.robots]
    assert payload["environments"] == [
        environment.to_payload() for environment in catalog.environments
    ]
