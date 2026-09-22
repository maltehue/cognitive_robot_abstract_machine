"""
Recorded-scene questions address every robot instance without ambiguity.
"""

from pathlib import Path

from cramera.knowledge.eql_session import EqlSession
from cramera.knowledge.presets import Preset
from cramera.live.bridge import Bridge
from semantic_digital_twin.robots.robot_parts import AbstractRobot

from .test_multi_robot_knowledge import repeated_robot_scene
from .test_multi_robot_live import two_robot_bridge


# %% shared scene questions
def test_robot_preset_returns_each_recorded_instance(
    repeated_robot_scene: Path,
) -> None:
    """
    The offered robot question remains executable with repeated models.

    :param repeated_robot_scene: Recording with two copies of the same robot model.
    """
    session = EqlSession.of_scene("fixture")
    preset = Preset.of_scene("fixture")[0]
    result = session.run(preset.code)
    assert result.ok
    assert {row["__entity__"] for row in result.rows} == {
        robot.name for robot in session.knowledge_base.robots
    }
    assert "robots" in preset.text


def test_arm_preset_names_and_returns_parts_of_each_robot(
    repeated_robot_scene: Path,
) -> None:
    """
    The arm question identifies the owners of all returned robot parts.

    :param repeated_robot_scene: Recording with two copies of the same robot model.
    """
    session = EqlSession.of_scene("fixture")
    preset = Preset.of_scene("fixture")[1]
    result = session.run(preset.code)
    assert result.ok
    assert {(row["__entity__"], row["robot"]) for row in result.rows} == {
        (arm.name, arm.robot) for arm in session.knowledge_base.arms
    }
    assert "arms" in preset.text


def test_live_robot_preset_returns_every_native_robot(two_robot_bridge: Bridge) -> None:
    """
    Live questions include every annotated instance in the shared world.

    :param two_robot_bridge: Attached world with two articulated robot instances.
    """
    preset = two_robot_bridge.query_source.presets()[0]
    result = two_robot_bridge.run_query(preset.code)
    robots = two_robot_bridge.world.get_semantic_annotations_by_type(AbstractRobot)
    assert result.ok
    assert {row["__entity__"] for row in result.rows} == {
        str(robot.name) for robot in robots
    }
