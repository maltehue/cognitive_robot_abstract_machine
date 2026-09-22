"""
Recorded objects preserve the body-local visual geometry shown by the live view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import trimesh
from typing_extensions import Any

from cramera.body_geometry import rounded_pose
from cramera.live.bridge import Bridge
from cramera.live.recording import RecordedFrame
from cramera.live.recording_bundle import write_recording_bundle
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale, Shape
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% native object recording fixture


@dataclass
class ObjectShapeRecording:
    """
    Serialize one shaped object with a distinct collision approximation.
    """

    world: World
    """Existing annotated robot world receiving the free object."""

    directory: Path
    """
    Directory receiving the finalized recording and mesh assets.
    """

    body: Body = field(init=False)
    """
    Recorded body whose visual geometry supplies the expected mesh.
    """

    def write(self, shape: Shape) -> dict[str, Any]:
        """
        Record a movable body using the given visual shape.

        :param shape: Visual geometry expressed relative to the new body's origin.
        :return: The recorded object's scene entry.
        """
        self.body = Body(
            name=PrefixedName("shaped_object"),
            visual=ShapeCollection([shape]),
            collision=ShapeCollection([Box(scale=Scale(0.01, 0.01, 0.01))]),
        )
        with self.world.modify_world():
            connection = Connection6DoF.create_with_dofs(
                parent=self.world.root, child=self.body, world=self.world
            )
            self.world.add_connection(connection)
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            2, 3, 0.8, reference_frame=self.world.root
        )
        bridge = Bridge()
        bridge.attach(self.world)
        bridge.snapshot()
        frame = RecordedFrame(
            frames=bridge.state.frames,
            base=bridge.state.base,
            objects=bridge.state.objects,
        )
        scene = write_recording_bundle(bridge, [frame], 30, self.directory, "geometry")
        return scene["objects"][0]


@pytest.fixture()
def shape_recording(cylinder_bot_world: World, tmp_path: Path) -> ObjectShapeRecording:
    """
    Prepare a native world and isolated recording destination.

    :param cylinder_bot_world: Existing mobile robot fixture.
    :param tmp_path: Scratch directory containing the recording.
    :return: Recorder for a visual shape with independent body and collision poses.
    """
    return ObjectShapeRecording(cylinder_bot_world, tmp_path / "recording")


# %% shape-local transforms and scaling


def test_transformed_box_retains_visual_origin(
    shape_recording: ObjectShapeRecording,
) -> None:
    """
    A translated, rotated box replays at its body-local visual position.

    :param shape_recording: Native recording fixture with a separate collision shape.
    """
    shape = Box(
        scale=Scale(0.2, 0.1, 0.3),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(0.3, -0.2, 0.1, yaw=0.7),
    )
    entry = shape_recording.write(shape)
    assert "mesh" in entry
    mesh = trimesh.load_mesh(shape_recording.directory / entry["mesh"])
    np.testing.assert_allclose(
        mesh.bounds, shape_recording.body.visual.combined_mesh.bounds, atol=1e-7
    )
    assert entry["spawn"] == rounded_pose(shape_recording.body)


@pytest.mark.parametrize(
    "transformed, scaled", [(True, False), (False, True), (True, True)]
)
def test_mesh_retains_shape_transform_and_scale(
    shape_recording: ObjectShapeRecording,
    tmp_path: Path,
    transformed: bool,
    scaled: bool,
) -> None:
    """
    Mesh files preserve local origin and nonuniform scale when recorded.

    :param shape_recording: Native recording fixture with distinct visual geometry.
    :param tmp_path: Directory for the untransformed source mesh.
    :param transformed: Whether the visual shape has a nonidentity local pose.
    :param scaled: Whether the visual shape rescales its mesh coordinates.
    """
    source = tmp_path / "source.stl"
    trimesh.creation.box(extents=[0.2, 0.1, 0.3]).export(source)
    shape = Mesh(
        filename=str(source),
        scale=Scale(2, 0.5, 1.5) if scaled else Scale(),
        origin=(
            HomogeneousTransformationMatrix.from_xyz_rpy(0.3, -0.2, 0.1, yaw=0.7)
            if transformed
            else HomogeneousTransformationMatrix()
        ),
    )
    entry = shape_recording.write(shape)
    mesh = trimesh.load_mesh(shape_recording.directory / entry["mesh"])
    np.testing.assert_allclose(
        mesh.bounds, shape_recording.body.visual.combined_mesh.bounds, atol=1e-7
    )
    assert entry["spawn"] == rounded_pose(shape_recording.body)
