"""
A demo setup is kept as a file, read back as it was written, and handed to and from the
Plan Builder in the builder's own form.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cramera.demo_setup import (
    DemoSetup,
    InvalidSetupNameError,
    MalformedSetupError,
    MapEnvironmentInSetupError,
    ObjectSetup,
    RobotSetup,
    SetupField,
    SetupLibrary,
    SetupNameTakenError,
    UnknownEnvironmentJointError,
    UnknownRobotModelError,
)
from cramera.body_geometry import DrawnGeometry
from cramera.live.bridge import Bridge
from cramera.environment_file import URDFEnvironmentFile, USDSceneEnvironmentFile
from cramera.model_catalog import EnvironmentKind
from semantic_digital_twin.predetermined_maps.apartment_environment import (
    ApartmentEnvironment,
)
from cramera.multi_robot import RobotInstance
from cramera.plan_steps import BuilderPlan, LookAt, Point
from semantic_digital_twin.adapters.usd.stage_parser import RootPlacement
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale

from .dataset.standing_robot import StandingRobot

ROBOT_TYPES = {StandingRobot.__name__: StandingRobot}
"""
The installed robot models a builder payload may name, in these tests.
"""


def looking_robot(identifier: str = "camera_arm") -> RobotSetup:
    return RobotSetup(
        instance=RobotInstance(
            identifier=identifier,
            label="Camera arm",
            robot_type=StandingRobot,
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.5, y=-0.5, yaw=0.3),
        ),
        plan=BuilderPlan(steps=[LookAt(Point(1.0, 2.0, 1.2))]),
        repeats_plan=True,
    )


def mirrored_robot() -> RobotSetup:
    return RobotSetup(
        instance=RobotInstance(
            identifier="humanoid",
            label="Humanoid",
            robot_type=StandingRobot,
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.0, y=2.0),
        ),
        joint_state_topic="/humanoid/joint_states",
        localization_topic="/humanoid/odom",
    )


def block_on_a_table() -> ObjectSetup:
    return ObjectSetup(
        name="block.stl",
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=13.27, y=-2.02, z=1.02, yaw=0.4),
        size=Scale(0.06, 0.06, 0.2),
    )


def setup_in(environment_path: str) -> DemoSetup:
    return DemoSetup(
        environment=USDSceneEnvironmentFile(
            path=environment_path, root_placement=RootPlacement.STAGE_ORIGIN
        ),
        robots=[looking_robot(), mirrored_robot()],
        objects=[block_on_a_table()],
    )


def payload_of(setup: DemoSetup) -> dict:
    return setup.to_payload()


# %% the file a setup is kept in


def test_a_saved_setup_reads_back_as_it_was(tmp_path: Path) -> None:
    setup = setup_in("/somewhere/world.usda")
    setup.save(tmp_path / "demo.json")

    loaded = DemoSetup.load(tmp_path / "demo.json")

    assert loaded.environment == setup.environment
    for read, written in zip(loaded.robots, setup.robots, strict=True):
        assert read.instance.identifier == written.instance.identifier
        assert read.instance.robot_type is written.instance.robot_type
        np.testing.assert_allclose(
            read.instance.pose.to_np(), written.instance.pose.to_np(), atol=1e-6
        )
        assert read.joint_state_topic == written.joint_state_topic
        assert read.localization_topic == written.localization_topic
        assert read.plan == written.plan
        assert read.repeats_plan == written.repeats_plan


def test_an_environment_named_relative_to_its_setup_is_found_beside_it(
    tmp_path: Path,
) -> None:
    setup_in("lab/world.usda").save(tmp_path / "demo.json")

    loaded = DemoSetup.load(tmp_path / "demo.json")

    assert loaded.environment.path == str(tmp_path / "lab" / "world.usda")


def test_a_setup_saved_again_keeps_naming_its_environment_beside_it(
    tmp_path: Path,
) -> None:
    setup_in("lab/world.usda").save(tmp_path / "demo.json")

    DemoSetup.load(tmp_path / "demo.json").save(tmp_path / "demo.json")

    written = json.loads((tmp_path / "demo.json").read_text())
    assert written["environment"]["path"] == "lab/world.usda"


def test_an_environment_elsewhere_is_named_by_its_whole_path(tmp_path: Path) -> None:
    setup_in("/somewhere/world.usda").save(tmp_path / "demo.json")

    written = json.loads((tmp_path / "demo.json").read_text())
    assert written["environment"]["path"] == "/somewhere/world.usda"


def test_a_package_url_is_left_as_it_is(tmp_path: Path) -> None:
    url = "package://some_world/worlds/hall.world"
    DemoSetup(environment=URDFEnvironmentFile(path=url), robots=[looking_robot()]).save(
        tmp_path / "demo.json"
    )

    assert DemoSetup.load(tmp_path / "demo.json").environment.path == url


# %% the builder's form


def test_a_setup_reads_back_from_the_builders_form() -> None:
    setup = setup_in("/somewhere/world.usda")

    read = DemoSetup.from_payload(payload_of(setup), ROBOT_TYPES)

    assert read.to_payload() == payload_of(setup)
    assert read.environment == setup.environment


def test_the_builders_form_states_where_a_robot_faces() -> None:
    [looking, _] = payload_of(setup_in("/w.usda"))["robots"]

    assert looking["yaw"] == pytest.approx(0.3)
    assert looking["x"] == pytest.approx(1.5)


def test_a_robot_model_that_is_not_installed_is_refused() -> None:
    payload = payload_of(setup_in("/w.usda"))
    payload["robots"][0]["model"] = "Unheard"

    with pytest.raises(UnknownRobotModelError):
        DemoSetup.from_payload(payload, ROBOT_TYPES)


def test_a_setup_of_no_robots_is_refused() -> None:
    payload = payload_of(setup_in("/w.usda"))
    payload["robots"] = []

    with pytest.raises(MalformedSetupError):
        DemoSetup.from_payload(payload, ROBOT_TYPES)


def test_a_robot_standing_nowhere_is_refused() -> None:
    payload = payload_of(setup_in("/w.usda"))
    payload["robots"][0]["x"] = "by the door"

    with pytest.raises(MalformedSetupError):
        DemoSetup.from_payload(payload, ROBOT_TYPES)


def test_a_robot_following_a_real_one_keeps_its_plan_for_the_builder() -> None:
    # The scene leaves such a robot to the real one; the plan is what the builder
    # opens with for it, so it travels with the setup like any other.
    payload = payload_of(setup_in("/w.usda"))
    payload["robots"][0]["jointStateTopic"] = "/camera_arm/joint_states"
    payload["robots"][0]["localizationTopic"] = "/camera_arm/odom"

    [robot, _] = DemoSetup.from_payload(payload, ROBOT_TYPES).robots

    assert robot.follows_a_real_robot
    assert robot.plan == looking_robot().plan


def test_a_localization_topic_travels_in_the_builders_form() -> None:
    setup = setup_in("/w.usda")

    read = DemoSetup.from_payload(payload_of(setup), ROBOT_TYPES)

    assert read.robots[1].localization_topic == setup.robots[1].localization_topic


def test_the_scene_of_a_setup_starts_with_its_first_robot_selected() -> None:
    setup = setup_in("/w.usda")

    assert setup.robot_scene.active_identifier == setup.robots[0].instance.identifier


# %% the environment's own joints


def hinged_world():
    """
    A world of a wall and a door leaf on a hinge, the way a scanned door is read.
    """
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.spatial_types import Vector3
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.connections import RevoluteConnection
    from semantic_digital_twin.world_description.degree_of_freedom import (
        DegreeOfFreedomLimits,
    )
    from semantic_digital_twin.world_description.world_entity import Body

    world = World()
    wall = Body(name=PrefixedName("wall", prefix="lab"))
    leaf = Body(name=PrefixedName("door_0", prefix="lab"))
    with world.modify_world():
        world.add_body(wall)
        world.add_body(leaf)
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=wall,
                child=leaf,
                axis=Vector3.Z(reference_frame=wall),
            )
        )
    [hinge] = world.connections
    return world, hinge


def test_a_setup_stands_the_environments_joints_where_it_says(tmp_path: Path) -> None:
    world, hinge = hinged_world()
    setup = setup_in("/w.usda")
    setup.environment_joint_positions = {str(hinge.name): 1.2}
    setup.save(tmp_path / "demo.json")

    DemoSetup.load(tmp_path / "demo.json").pose_environment(world)

    assert world.state[hinge.dof.id].position == pytest.approx(1.2)


def test_a_joint_the_environment_does_not_have_is_refused() -> None:
    world, _ = hinged_world()
    setup = setup_in("/w.usda")
    setup.environment_joint_positions = {"lab/no_such_hinge": 1.2}

    with pytest.raises(UnknownEnvironmentJointError):
        setup.pose_environment(world)


def test_the_environments_joints_travel_in_the_builders_form() -> None:
    setup = setup_in("/w.usda")
    setup.environment_joint_positions = {"lab/wall_T_lab/door_0": 1.2}

    read = DemoSetup.from_payload(payload_of(setup), ROBOT_TYPES)

    assert read.environment_joint_positions == setup.environment_joint_positions


# %% how a setup's environment is drawn


def test_a_setup_keeps_how_its_environment_is_drawn(tmp_path: Path) -> None:
    setup = setup_in("/somewhere/world.usda")
    setup.environment_geometry = DrawnGeometry.COLLISION
    setup.save(tmp_path / "demo.json")

    assert DemoSetup.load(tmp_path / "demo.json").environment_geometry is (
        DrawnGeometry.COLLISION
    )


def test_a_setup_saved_before_it_said_how_to_draw_its_environment_draws_it_as_it_looks(
    tmp_path: Path,
) -> None:
    setup = setup_in("/somewhere/world.usda")
    setup.save(tmp_path / "demo.json")
    written = json.loads((tmp_path / "demo.json").read_text())
    del written["environment_geometry"]
    (tmp_path / "demo.json").write_text(json.dumps(written))

    assert DemoSetup.load(tmp_path / "demo.json").environment_geometry is (
        DrawnGeometry.VISUAL
    )


def test_how_the_environment_is_drawn_travels_in_the_builders_form() -> None:
    setup = setup_in("/w.usda")
    setup.environment_geometry = DrawnGeometry.COLLISION

    read = DemoSetup.from_payload(payload_of(setup), ROBOT_TYPES)

    assert read.environment_geometry is DrawnGeometry.COLLISION


def test_registering_a_setup_tells_the_bridge_how_its_environment_is_drawn() -> None:
    setup = setup_in("/w.usda")
    bridge = Bridge()

    bridge.register_setup(setup)

    assert bridge.presentation == setup.environment.presentation()


# %% the setups the builder saves


def test_a_saved_setup_is_listed_and_opens(tmp_path: Path) -> None:
    library = SetupLibrary(directory=tmp_path / "setups")
    library.save("moved_about", setup_in("/w.usda"))

    assert library.names() == ["moved_about"]
    assert library.open("moved_about").environment == setup_in("/w.usda").environment


def test_a_setup_is_never_saved_over_another(tmp_path: Path) -> None:
    library = SetupLibrary(directory=tmp_path)
    library.save("demo", setup_in("/first.usda"))

    with pytest.raises(SetupNameTakenError):
        library.save("demo", setup_in("/second.usda"))
    assert library.open("demo").environment.path == "/first.usda"


def test_a_name_that_cannot_name_a_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InvalidSetupNameError):
        SetupLibrary(directory=tmp_path).save("../elsewhere", setup_in("/w.usda"))


# %% a map environment, which no setup file can name


def test_a_setup_in_a_map_environment_is_refused() -> None:
    """
    A setup names the environment's file; the real lab is built by a class and has none.
    """
    payload = payload_of(setup_in("/w.usda"))
    payload[SetupField.ENVIRONMENT] = {
        SetupField.KIND: EnvironmentKind.MAP,
        SetupField.CLASS: ApartmentEnvironment.__name__,
    }

    with pytest.raises(MapEnvironmentInSetupError) as refused:
        DemoSetup.from_payload(payload, ROBOT_TYPES)

    assert refused.value.map == ApartmentEnvironment.__name__
    assert isinstance(refused.value, MalformedSetupError)


def test_a_setup_keeps_how_big_the_textures_of_its_scan_are_drawn(
    tmp_path: Path,
) -> None:
    setup = DemoSetup(
        environment=USDSceneEnvironmentFile(
            path="/somewhere/world.usda", maximum_texture_size=2048
        ),
        robots=[looking_robot()],
    )
    setup.save(tmp_path / "demo.json")

    assert DemoSetup.load(tmp_path / "demo.json").environment.maximum_texture_size == (
        2048
    )


# %% the boxes lying about


def test_a_saved_setup_keeps_its_boxes(tmp_path: Path) -> None:
    path = tmp_path / "setup.json"
    setup_in("/scans/lab/world.usda").save(path)

    [box] = DemoSetup.load(path).objects

    assert box.name == "block.stl"
    assert box.pose.to_np() == pytest.approx(block_on_a_table().pose.to_np())
    assert box.size == Scale(0.06, 0.06, 0.2)


def test_a_setup_saved_before_there_were_boxes_has_none(tmp_path: Path) -> None:
    path = tmp_path / "setup.json"
    setup_in("/scans/lab/world.usda").save(path)
    written = json.loads(path.read_text())
    del written["objects"]
    path.write_text(json.dumps(written))

    assert DemoSetup.load(path).objects == []


def test_the_boxes_travel_in_the_builders_form() -> None:
    setup = setup_in("/scans/lab/world.usda")

    read = DemoSetup.from_payload(setup.to_payload(), ROBOT_TYPES)

    [box] = read.objects
    assert box.name == "block.stl"
    assert box.pose.to_np() == pytest.approx(block_on_a_table().pose.to_np())
    assert box.yaw == pytest.approx(0.4)
    assert box.size == Scale(0.06, 0.06, 0.2)
    assert setup.to_payload()[SetupField.OBJECTS] == [
        {
            SetupField.NAME: "block.stl",
            SetupField.X: pytest.approx(13.27),
            SetupField.Y: pytest.approx(-2.02),
            SetupField.Z: pytest.approx(1.02),
            SetupField.YAW: pytest.approx(0.4),
            SetupField.SIZE: [0.06, 0.06, 0.2],
        }
    ]


def test_a_builders_form_without_boxes_reads_as_a_setup_of_none() -> None:
    payload = setup_in("/scans/lab/world.usda").to_payload()
    del payload[SetupField.OBJECTS]

    assert DemoSetup.from_payload(payload, ROBOT_TYPES).objects == []


@pytest.mark.parametrize(
    "box",
    [
        {"x": 1.0, "y": 2.0, "z": 0.9, "yaw": 0.0, "size": [0.1, 0.1, 0.1]},
        {"name": "block.stl", "x": 1.0, "y": 2.0, "yaw": 0.0, "size": [0.1, 0.1, 0.1]},
        {"name": "block.stl", "x": 1.0, "y": 2.0, "z": 0.9, "size": [0.1, 0.1]},
        {"name": "block.stl", "x": "there", "y": 2.0, "z": 0.9, "size": [0.1, 0.1, 0.1]},
    ],
)
def test_a_box_without_a_name_a_place_or_a_size_is_refused(box: dict) -> None:
    payload = setup_in("/scans/lab/world.usda").to_payload()
    payload[SetupField.OBJECTS] = [box]

    with pytest.raises(MalformedSetupError):
        DemoSetup.from_payload(payload, ROBOT_TYPES)


def test_a_box_is_placed_only_once() -> None:
    payload = setup_in("/scans/lab/world.usda").to_payload()
    payload[SetupField.OBJECTS] = payload[SetupField.OBJECTS] * 2

    with pytest.raises(MalformedSetupError):
        DemoSetup.from_payload(payload, ROBOT_TYPES)
