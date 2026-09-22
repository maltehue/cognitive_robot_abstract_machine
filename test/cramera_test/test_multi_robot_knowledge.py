"""
Recorded robot queries preserve every instance and its native parts.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from cramera.knowledge.eql_session import EqlSession
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.knowledge.scene_bundle import ParsedUrdf
from cramera.knowledge.enums import JointRegion
from cramera.knowledge.views.dispatcher import GraphPanelViews
from cramera.robot_parts import RobotPartAnnotation, RobotPartRole, ArmSide


# %% repeated recorded robot model
@pytest.fixture()
def repeated_robot_scene(fixture_scene: Path) -> Path:
    """
    Expand the existing scene fixture to two namespaced copies.

    :param fixture_scene: Complete recorded scene and architecture fixture.
    :return: Directory containing the expanded scene bundle.
    """
    directory = fixture_scene / "scenes" / "fixture"
    scene = json.loads((directory / "scene.json").read_text())
    source = deepcopy(scene["robot"])
    scene["robots"] = []
    scene["models"] = []
    for identifier in ("first", "second"):
        robot = {**deepcopy(source), "identifier": identifier, "prefix": identifier}
        scene["robots"].append(robot)
        scene["models"].append(
            {
                "name": identifier,
                "identifier": identifier,
                "prefix": identifier,
                "robot": True,
                "urdf": identifier + ".urdf",
            }
        )
        urdf = (directory / "robot.urdf").read_text()
        urdf = urdf.replace('name="', 'name="' + identifier + "/").replace(
            'link="', 'link="' + identifier + "/"
        )
        (directory / (identifier + ".urdf")).write_text(urdf)
    scene["robot"] = scene["robots"][1]
    scene["activeRobot"] = scene["robot"]["identifier"]
    (directory / "scene.json").write_text(json.dumps(scene))
    trajectory = json.loads((directory / "trajectory.json").read_text())
    trajectory["frames"] = [
        {
            identifier + "/" + key.partition("/")[2]: value
            for identifier in ("first", "second")
            for key, value in frame.items()
        }
        for frame in trajectory["frames"]
    ]
    (directory / "trajectory.json").write_text(json.dumps(trajectory))
    EpisodeKnowledgeBase.reset()
    return directory


def test_recorded_robot_queries_include_both_instances(
    repeated_robot_scene: Path,
) -> None:
    """
    The ordinary robot variable and plural namespace contain both instances.
    """
    session = EqlSession.of_scene("fixture")
    domain = next(domain for domain in session.domains() if domain.name == "robot")
    assert {robot.name for robot in domain.objects} == {"first", "second"}
    assert {robot.name for robot in session.namespace()["robots"]} == {
        "first",
        "second",
    }
    assert session.knowledge_base.robot.name == "second"


def test_recorded_arms_keep_owners_and_unique_names(repeated_robot_scene: Path) -> None:
    """
    Same-class arms and grippers remain distinguishable in query results.
    """
    knowledge = EpisodeKnowledgeBase.of_scene("fixture")
    assert {(arm.name, arm.robot, arm.gripper.name) for arm in knowledge.arms} == {
        (identifier + "/left_arm", identifier, identifier + "/left_gripper")
        for identifier in ("first", "second")
    }
    transport = next(episode for episode in knowledge.episodes if episode.picks)
    assert transport.performed_by.robot == knowledge.robot.name


def test_recorded_joint_queries_keep_instance_identity(
    repeated_robot_scene: Path,
) -> None:
    """
    A second robot's joints retain their namespace and are not environment joints.
    """
    knowledge = EpisodeKnowledgeBase.of_scene("fixture")
    expected = {
        name
        for frame in json.loads((repeated_robot_scene / "trajectory.json").read_text())[
            "frames"
        ]
        for name in frame
    }
    assert {joint.name for joint in knowledge.joints} == expected
    assert all(
        joint.region is not JointRegion.ENVIRONMENT for joint in knowledge.joints
    )


def test_recorded_kinematic_tree_contains_each_robot(
    repeated_robot_scene: Path,
) -> None:
    """
    The ordinary URDF view can inspect the complete shared scene.
    """
    parsed = ParsedUrdf.of_scene("fixture")
    assert {name.partition("/")[0] for name in parsed.links} == {"first", "second"}
    assert {joint.name.partition("/")[0] for joint in parsed.joints} == {
        "first",
        "second",
    }


def test_each_robot_opens_the_recorded_kinematic_view(
    repeated_robot_scene: Path,
) -> None:
    """
    Both recorded robot identities offer the existing graph inspection action.
    """
    views = GraphPanelViews.of_scene("fixture")
    for identifier in ("first", "second"):
        payload = views.for_node(identifier)
        assert payload is not None
        assert "urdf:" + identifier + "/base_link" in payload.details


def test_native_part_annotations_keep_each_gripper_attachment(
    repeated_robot_scene: Path,
) -> None:
    """
    Structured annotations retain the same instance-qualified arm ownership.
    """
    scene_path = repeated_robot_scene / "scene.json"
    scene = json.loads(scene_path.read_text())
    for robot in scene["robots"]:
        robot["partAnnotations"] = [
            RobotPartAnnotation(
                name="left_arm",
                role=RobotPartRole.ARM,
                side=ArmSide.LEFT,
                links=robot["parts"]["left_arm"],
            ).to_payload(),
            RobotPartAnnotation(
                name="left_gripper",
                role=RobotPartRole.END_EFFECTOR,
                side=ArmSide.LEFT,
                links=robot["parts"]["left_gripper"],
                attached_to="left_arm",
            ).to_payload(),
        ]
    scene_path.write_text(json.dumps(scene))
    knowledge = EpisodeKnowledgeBase.of_scene("fixture")
    assert {(arm.name, arm.robot, arm.gripper.name) for arm in knowledge.arms} == {
        (identifier + "/left_arm", identifier, identifier + "/left_gripper")
        for identifier in ("first", "second")
    }
