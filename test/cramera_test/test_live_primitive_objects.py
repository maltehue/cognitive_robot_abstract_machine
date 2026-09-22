"""
Movable primitive objects stay visible and recorded through attachment changes.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import pytest

from cramera.body_geometry import rounded_pose
from cramera.live.bridge import Bridge
from cramera.live.live_bundle import build_live_scene
from cramera.live.recording import Recording
from cramera.live.recording_bundle import write_recording_bundle
from semantic_digital_twin.api import BodySpecification, Connection6DoFSpecification
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

from .test_live_bundle import use_scratch_scenes_directory

# %% movable geometry without mesh filenames


class PrimitiveName(StrEnum):
    """
    Object identity shared by the world, bridge and recording.
    """

    CUBE = "transport_cube"
    """
    A primitive body with no mesh suffix in its name.
    """


def spawn_cube(world: World) -> Body:
    """
    Add a shaped free object using the native world specification.

    :param world: Existing robot world receiving the transport object.
    :return: The newly spawned movable cube.
    """
    return BodySpecification.box(
        PrimitiveName.CUBE,
        Scale(0.06, 0.06, 0.14),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(2, 0, 0.4),
        connection_specification=Connection6DoFSpecification(),
    ).spawn(world)


@pytest.fixture()
def primitive_bridge(cylinder_bot_world: World) -> Bridge:
    """
    Bind the live bridge to an annotated robot and a free primitive object.

    :param cylinder_bot_world: Existing mobile robot fixture.
    :return: Attached bridge with the initial object pose published.
    """
    spawn_cube(cylinder_bot_world)
    bridge = Bridge()
    bridge.attach(cylinder_bot_world)
    bridge.snapshot()
    return bridge


def test_primitive_is_published_without_a_mesh_filename(
    primitive_bridge: Bridge,
) -> None:
    """
    World-declared movable primitives receive catalog entries and live poses.

    :param primitive_bridge: Attached bridge containing the movable cube.
    """
    bridge = primitive_bridge
    body = bridge.world.get_body_by_name(PrimitiveName.CUBE)
    assert bridge.object_keys() == [PrimitiveName.CUBE]
    assert bridge.get_state()["objects"] == {PrimitiveName.CUBE: rounded_pose(body)}
    assert bridge.object_catalog()[0]["key"] == PrimitiveName.CUBE


def test_object_tracking_survives_pick_carry_and_release(
    primitive_bridge: Bridge,
) -> None:
    """
    The same object remains streamed when its free connection becomes fixed.

    :param primitive_bridge: Attached bridge containing the movable cube.
    """
    bridge = primitive_bridge
    world = bridge.world
    body = world.get_body_by_name(PrimitiveName.CUBE)
    signature = bridge.bundle_signature()
    initial = rounded_pose(body)
    with world.modify_world():
        world.move_branch_with_fixed_connection(body, bridge.robot.root)
    bridge.observe_model_change()
    bridge.robot.drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(1)
    bridge.snapshot()
    assert bridge.object_keys() == [PrimitiveName.CUBE]
    assert bridge.get_state()["objects"][PrimitiveName.CUBE] == rounded_pose(body)
    assert rounded_pose(body)[0] == initial[0] + 1
    assert bridge.bundle_signature() == signature
    with world.modify_world():
        world.move_branch_with_fixed_connection(body, world.root)
    bridge.observe_model_change()
    bridge.snapshot()
    assert bridge.object_keys() == [PrimitiveName.CUBE]
    assert bridge.get_state()["objects"][PrimitiveName.CUBE] == rounded_pose(body)
    assert bridge.bundle_signature() == signature


@pytest.mark.parametrize("held", [False, True])
def test_live_bundle_excludes_streamed_primitive_from_all_models(
    primitive_bridge: Bridge,
    held: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Neither environment nor robot URDF duplicates the separately moving object.

    :param primitive_bridge: Attached bridge containing the movable cube.
    :param held: Whether the cube has become part of the robot's kinematic subtree.
    :param monkeypatch: Redirects generated bundles to the test directory.
    :param tmp_path: Scratch directory for serialized models.
    """
    bridge = primitive_bridge
    if held:
        with bridge.world.modify_world():
            bridge.world.move_branch_with_fixed_connection(
                bridge.world.get_body_by_name(PrimitiveName.CUBE), bridge.robot.root
            )
        bridge.observe_model_change()
    scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
    scene_name = build_live_scene(bridge)
    scene_directory = scenes / scene_name
    scene = json.loads((scene_directory / "scene.json").read_text())
    for model in scene["models"]:
        assert PrimitiveName.CUBE not in (scene_directory / model["urdf"]).read_text()


def test_removed_primitive_is_no_longer_published(primitive_bridge: Bridge) -> None:
    """
    Remembering an attachment does not retain a body removed from the world.

    :param primitive_bridge: Attached bridge containing the movable cube.
    """
    bridge = primitive_bridge
    body = bridge.world.get_body_by_name(PrimitiveName.CUBE)
    with bridge.world.modify_world():
        bridge.world.remove_connection(body.parent_connection)
        bridge.world.remove_kinematic_structure_entity(body)
    bridge.observe_model_change()
    bridge.snapshot()
    assert bridge.object_keys() == []
    assert bridge.get_state()["objects"] == {}


# %% scene population after recording startup


def test_recording_keeps_a_primitive_first_seen_after_initial_frame(
    cylinder_bot_world: World, tmp_path: Path
) -> None:
    """
    Objects spawned during demo setup retain their first pose and carried motion.

    :param cylinder_bot_world: Existing annotated robot world before scene population.
    :param tmp_path: Scratch directory for the finalized recording.
    """
    bridge = Bridge()
    bridge.attach(cylinder_bot_world)
    recording = Recording()
    recording.start()
    bridge.snapshot()
    recording.append(bridge.state)
    body = spawn_cube(cylinder_bot_world)
    bridge.observe_model_change()
    bridge.snapshot()
    recording.append(bridge.state)
    first_pose = rounded_pose(body)
    with cylinder_bot_world.modify_world():
        cylinder_bot_world.move_branch_with_fixed_connection(body, bridge.robot.root)
    bridge.observe_model_change()
    bridge.robot.drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(1)
    bridge.snapshot()
    recording.append(bridge.state)
    frames = recording.stop()
    scene = write_recording_bundle(bridge, frames, 30, tmp_path / "recording", "carry")
    assert len(scene["objects"]) == 1
    assert scene["objects"][0]["key"] == PrimitiveName.CUBE
    assert scene["objects"][0]["spawn"] == first_pose
    assert scene["objects"][0]["box"] == [0.06, 0.06, 0.14]
    trajectory = json.loads((tmp_path / "recording" / "trajectory.json").read_text())
    assert trajectory["objects"] == [
        {},
        {PrimitiveName.CUBE: first_pose},
        {PrimitiveName.CUBE: rounded_pose(body)},
    ]
    for model in scene["models"]:
        assert (
            PrimitiveName.CUBE
            not in (tmp_path / "recording" / model["urdf"]).read_text()
        )
