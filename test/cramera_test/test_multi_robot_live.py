"""Every robot instance retains its own live and recorded model."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import pytest

from cramera.body_geometry import rounded_pose
from cramera.live.bridge import Bridge
from cramera.live.live_bundle import bundle_world_models
from cramera.live.recording import Recording
from cramera.live.recording_bundle import write_recording_bundle
from cramera.live.http import serve
from cramera.live.teleop import TeleopController, TeleopRequest
from cramera.live import teleop as teleop_module
from cramera.live.teleop import TeleopUnavailable
from cramera.live.recording_bundle import finalize_recording
from cramera.live.recording_storage import trim_recording_bundle
from cramera.live.frame_range import FrameRange
from cramera import paths
from coraplex.datastructures.dataclasses import Context
from cramera.live.robot_models import RobotSelectionBusy, UnknownRobot
from cramera.live.bridge import TaskStatusName
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    RevoluteConnection,
)

from .test_live_bundle import shaped
from .test_live_bridge import make_plan_node, PlanWithRoot
from .test_live_http import post
from .test_server import get_json


@dataclass
class FinishingTeleop(TeleopController):
    """Hold a real driver's last servo tick until the test lets it finish."""

    world: World
    """Native world used by the driver."""
    robot: AbstractRobot
    """Robot whose final servo tick is in progress."""
    entered: threading.Event = field(default_factory=threading.Event)
    """Signals that the worker started its final world write."""
    release: threading.Event = field(default_factory=threading.Event)
    """Allows that pending world write to complete."""
    completed: threading.Event = field(default_factory=threading.Event)
    """Signals that the final servo tick has returned."""

    def __post_init__(self) -> None:
        """Initialize the production worker with the fixture's world and robot."""
        TeleopController.__init__(self, self.world, self.robot)

    def _servo(self, targets: dict[str, TeleopRequest]) -> None:
        """Simulate an in-flight native motion until released.

        :param targets: Last hand command copied by the actual driver loop.
        """
        self.entered.set()
        self.release.wait(timeout=3)
        self.completed.set()


@dataclass
class PlanWithSelectedRobot(PlanWithRoot):
    """A published plan with the native execution context selecting its robot."""

    context: Context
    """Native execution context selecting one annotation in the world."""


# %% shared articulated world
@pytest.fixture()
def two_robot_bridge(world_with_two_bodies: tuple[World, Body, Body]) -> Bridge:
    """Populate the existing empty world fixture with two articulated instances.

    :param world_with_two_bodies: Native world fixture receiving both robot branches.
    :return: A bound bridge with its initial state published.
    """
    world, root, _ = world_with_two_bodies
    with world.modify_world():
        world.add_body(root)
        for index, identifier in enumerate(("first", "second")):
            base = shaped(identifier, "base")
            arm = shaped(identifier, "arm")
            root_connection = Connection6DoF.create_with_dofs(
                parent=world.root,
                child=base,
                world=world,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=index * 3,
                ),
            )
            world.add_connection(root_connection)
            world.add_connection(
                RevoluteConnection.create_with_dofs(
                    parent=base,
                    child=arm,
                    world=world,
                    axis=Vector3.Z(),
                )
            )
            robot = MinimalRobot.from_branch_in_world(base)
            robot.update_name(PrefixedName(identifier, prefix=identifier))
    bridge = Bridge()
    bridge.attach(world)
    bridge.snapshot()
    return bridge


# %% independent live instances
def test_all_robot_roots_are_streamed_without_loose_robot_objects(
    two_robot_bridge: Bridge,
) -> None:
    """Free robot roots remain articulated models and receive independent poses."""
    bridge = two_robot_bridge
    robots = bridge.world.get_semantic_annotations_by_type(AbstractRobot)
    assert bridge.get_state()["modelBases"] == {
        robot.root.name.prefix: rounded_pose(robot.root) for robot in robots
    }
    assert bridge.object_keys() == []


def test_active_robot_selection_survives_rebinding(two_robot_bridge: Bridge) -> None:
    """Selecting another annotation persists through periodic model discovery."""
    bridge = two_robot_bridge
    bridge.select_robot("second")
    bridge.bind()
    bridge.snapshot()
    assert bridge.get_robots()["active_identifier"] == "second"
    assert bridge.robot.root.name.prefix == "second"
    assert bridge.get_state()["base"] == rounded_pose(bridge.robot.root)


def test_robot_catalog_preserves_individual_joint_positions(
    two_robot_bridge: Bridge,
) -> None:
    """A later Builder run can restore inactive arms as well as their root poses."""
    bridge = two_robot_bridge
    for index, connection in enumerate(bridge._connections):
        connection.position = 0.2 * (index + 1)
    bridge.snapshot()
    for robot in bridge.get_robots()["robots"]:
        expected = {
            str(connection.name): connection.position
            for connection in bridge._connections
            if connection.child.name.prefix == robot["identifier"]
        }
        assert robot["joint_positions"] == expected


@pytest.mark.parametrize("status", [TaskStatusName.RUNNING, TaskStatusName.PAUSE])
def test_selection_rejects_an_unfinished_plan(
    two_robot_bridge: Bridge, status: TaskStatusName
) -> None:
    """Robot identity cannot change underneath a running or paused plan."""
    bridge = two_robot_bridge
    bridge.begin_plan(PlanWithRoot(make_plan_node("SequentialNode", status=status)))
    selected = bridge.robot
    with pytest.raises(RobotSelectionBusy):
        bridge.select_robot("second")
    assert bridge.robot is selected


def test_unknown_robot_leaves_current_selection_unchanged(
    two_robot_bridge: Bridge,
) -> None:
    """A nonexistent identifier cannot silently fall back to another robot."""
    selected = two_robot_bridge.robot
    with pytest.raises(UnknownRobot):
        two_robot_bridge.select_robot("missing")
    assert two_robot_bridge.robot is selected


def test_reselecting_the_active_robot_keeps_the_snapshot(
    two_robot_bridge: Bridge,
) -> None:
    """An unchanged selection does not reset the robot or restart its live state."""
    bridge = two_robot_bridge
    before = bridge.get_state()
    bridge.select_robot(bridge.robot.root.name.prefix)
    assert bridge.get_state() == before


def test_live_robot_selection_endpoint(two_robot_bridge: Bridge) -> None:
    """The actual bridge endpoint selects a shared-world instance and reports failures."""
    bridge = two_robot_bridge
    server = serve(bridge, 0)
    address = f"http://localhost:{server.server_address[1]}"
    try:
        assert get_json(address + "/robots") == bridge.get_robots()
        status, response = post(address + "/robot", {"identifier": "second"})
        assert status == 200
        assert response == {"ok": True, **bridge.get_robots()}
        assert response["active_identifier"] == "second"
        assert post(address + "/robot", {"identifier": "absent"})[0] == 400
        assert post(address + "/robot", ["second"])[0] == 400
        assert post(address + "/robot", {"identifier": 12})[0] == 400
        bridge.begin_plan(
            PlanWithRoot(
                make_plan_node("SequentialNode", status=TaskStatusName.RUNNING)
            )
        )
        assert post(address + "/robot", {"identifier": "first"})[0] == 409
    finally:
        server.shutdown()


def test_robot_selection_waits_for_previous_teleop_tick(
    two_robot_bridge: Bridge,
) -> None:
    """No previous robot can continue moving after selection returns."""
    bridge = two_robot_bridge
    driver = FinishingTeleop(bridge.world, bridge.robot)
    bridge._teleop = driver
    driver.submit(TeleopRequest(arm="left", position=[0, 0, 0]))
    assert driver.entered.wait(timeout=2)
    with ThreadPoolExecutor(max_workers=1) as executor:
        changed = executor.submit(bridge.select_robot, "second")
        try:
            assert not changed.done() or driver.completed.is_set()
            driver.release.set()
            changed.result(timeout=3)
            assert driver.completed.is_set()
            assert bridge._teleop is None
            assert bridge.robot.root.name.prefix == "second"
        finally:
            driver.release.set()
            driver.stop()


def test_timed_out_teleop_keeps_previous_robot_selected(
    two_robot_bridge: Bridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A driver that still owns a world write prevents selection from succeeding."""
    bridge = two_robot_bridge
    selected = bridge.robot
    driver = FinishingTeleop(bridge.world, selected)
    bridge._teleop = driver
    driver.submit(TeleopRequest(arm="left", position=[0, 0, 0]))
    assert driver.entered.wait(timeout=2)
    monkeypatch.setattr(teleop_module, "STOP_TIMEOUT_SECONDS", 0)
    try:
        with pytest.raises(TeleopUnavailable):
            bridge.select_robot("second")
        assert bridge.robot is selected
    finally:
        driver.release.set()
        monkeypatch.undo()
        driver.stop()


def test_plan_context_selects_the_executing_robot(two_robot_bridge: Bridge) -> None:
    """A generated native context chooses the intended instance before execution."""
    bridge = two_robot_bridge
    robot = next(
        robot
        for robot in bridge.world.get_semantic_annotations_by_type(AbstractRobot)
        if robot.root.name.prefix == "second"
    )
    context = Context(world=bridge.world, robot=robot)
    bridge.begin_plan(PlanWithSelectedRobot(make_plan_node("SequentialNode"), context))
    assert bridge.robot is context.robot
    assert bridge.get_robots()["active_identifier"] == robot.root.name.prefix


def test_trimming_preserves_all_root_track_alignment(
    two_robot_bridge: Bridge,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clipped fallback still assigns every robot the pose of the kept frame."""
    bridge = two_robot_bridge
    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
    recording = Recording()
    recording.start()
    expected = []
    for position in (1, 2, 3):
        bridge.robot.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=position, reference_frame=bridge.world.root
            )
        )
        bridge.snapshot()
        recording.append(bridge.state)
        expected.append(bridge.get_state()["modelBases"])
    finalize_recording(bridge, recording)
    trim_recording_bundle(FrameRange(first=1, last=2))
    trajectory = json.loads(
        (
            paths.local_scenes_directory()
            / paths.RECORDING_SCENE_NAME
            / "trajectory.json"
        ).read_text()
    )
    assert trajectory["modelBases"] == expected[1:]
    assert len(trajectory["frames"]) == len(trajectory["modelBases"])


def test_every_robot_is_a_separate_articulated_model(
    two_robot_bridge: Bridge,
    tmp_path: Path,
) -> None:
    """Same-class instances receive distinct files with disjoint native links."""
    bridge = two_robot_bridge
    bundle = bundle_world_models(bridge.world, bridge.robot, tmp_path, "recording")
    robots = bridge.world.get_semantic_annotations_by_type(AbstractRobot)
    models = [model for model in bundle.models if model["robot"]]
    assert {model["prefix"] for model in models} == {
        robot.root.name.prefix for robot in robots
    }
    assert len({model["urdf"] for model in models}) == len(robots)
    for model in models:
        document = ElementTree.parse(tmp_path / model["urdf"])
        links = {link.attrib["name"] for link in document.iter("link")}
        assert links == {
            "world_root",
            model["prefix"] + "/base",
            model["prefix"] + "/arm",
        }
        assert len(list(document.iter("joint"))) == 2


def test_recording_keeps_all_model_root_tracks(
    two_robot_bridge: Bridge,
    tmp_path: Path,
) -> None:
    """Recorded robot bases survive later active-instance changes."""
    bridge = two_robot_bridge
    recording = Recording()
    recording.start()
    recording.append(bridge.state)
    first = bridge.get_state()["modelBases"]
    bridge.select_robot("second")
    bridge.robot.root.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=4, reference_frame=bridge.world.root
        )
    )
    bridge.snapshot()
    recording.append(bridge.state)
    final = bridge.get_state()["modelBases"]
    frames = recording.stop()
    scene = write_recording_bundle(bridge, frames, 30, tmp_path / "bundle", "robots")
    trajectory = json.loads((tmp_path / "bundle" / "trajectory.json").read_text())
    assert trajectory["modelBases"] == [first, final]
    assert scene["activeRobot"] == "second"
    assert {robot["identifier"] for robot in scene["robots"]} == set(first)
