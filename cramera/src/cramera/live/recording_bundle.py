"""
Writing a finalized live recording to disk as a replayable scene bundle.

Produces exactly the ``scene.json``/``trajectory.json``/``meshes/`` shape an onboarded
scene is recorded into (see :meth:`cramera.onboard.demo.SceneBuilder.build`), so the
frontend's existing trajectory playback needs no changes to replay a recording — only
the scene name differs. Always writes under
:func:`cramera.paths.local_scenes_directory`, never a shared scenes root: a recording is
a local capture until the user explicitly saves it, and saving must never touch a git-
tracked scenes checkout.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

import numpy as np
from typing_extensions import Any, Dict, List, Optional

from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale, Shape
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cramera import paths
from cramera.body_geometry import measure_body, POSE_PRECISION, rounded_scale
from cramera.generated_json import write_json_atomically
from cramera.knowledge.detected_events import SceneField
from cramera.knowledge.recorded_statecharts import (
    RecordedStatecharts,
    STATECHART_FILE,
)
from cramera.live.bridge import Bridge, ObjectCatalogEntry, ObjectKind
from cramera.live.live_bundle import bundle_world_models
from cramera.live.recording import Recording, RecordedFrame, RecordingState
from cramera.live.recording_segments import derive_segments
from cramera.live.robot_models import RobotModels
from cramera.robot_fields import RobotField
from cramera.mesh_format import MeshFormat
from cramera.onboard.bundle_urdf import BundledAssets
from cramera.spatial_annotations import SpatialAnnotations

RECORDING_BUILD_LOCK = threading.Lock()
"""
Guards a recording bundle's output directory, mirroring
:data:`cramera.live.live_bundle.BUILD_LOCK` — the viewer may poll ``/recording`` while a
save is in flight.
"""

MESH_SUBDIRECTORY = "recording"
"""
Directory the recording's models' meshes nest under, inside the bundle.
"""


class NothingToBundle(Exception):
    """
    Raised by :func:`write_recording_bundle` when the recording has no frames.
    """


def write_recording_bundle(
    bridge: Bridge,
    frames: List[RecordedFrame],
    frames_per_second: float,
    output_directory: Path,
    scene_name: str,
) -> Dict[str, Any]:
    """
    Write a finalized recording's geometry and trajectory to disk.

    Clears ``output_directory`` first, mirroring
    :func:`cramera.live.live_bundle.build_live_scene`'s own throwaway-bundle behaviour.

    :param bridge: The live bridge whose world and object catalog are bundled.
    :param frames: The recording's buffered ticks, in order.
    :param frames_per_second: The recording's estimated frame rate.
    :param output_directory: Directory the bundle is written into.
    :param scene_name: Name the bundle's ``scene.json`` carries.
    :raises NothingToBundle: If no ticks were recorded.
    """
    if not frames:
        raise NothingToBundle("the recording has no frames")
    with RECORDING_BUILD_LOCK:
        if output_directory.exists():
            shutil.rmtree(output_directory)
        output_directory.mkdir(parents=True)
        geometry = bundle_world_models(
            bridge.world,
            bridge.robot,
            output_directory,
            MESH_SUBDIRECTORY,
            overlay_bodies=bridge.query_objects(),
        )
        objects = _loose_object_entries(bridge, frames, output_directory)
        scene = {
            "name": scene_name,
            "framesPerSecond": frames_per_second,
            "trajectory": "trajectory.json",
            "models": geometry.models,
            "robot": geometry.robot,
            RobotField.ROBOTS: geometry.robots,
            RobotField.ACTIVE_ROBOT: (
                RobotModels.identifier(bridge.robot)
                if bridge.robot is not None
                else None
            ),
            "objects": objects,
            "segments": [segment.to_payload() for segment in derive_segments(frames)],
            "missingAssets": geometry.missing_assets,
            "worldBound": True,
            "bundleSignature": bridge.bundle_signature(),
            SceneField.PLAN_TREES: bridge.plan_state.recorded_trees(),
            **SpatialAnnotations.of_world(
                bridge.world, bridge.query_objects()
            ).scene_fields(),
        }
        statecharts = RecordedStatecharts.of_snapshots(
            frame.statechart for frame in frames
        )
        if not statecharts.is_empty():
            scene["statecharts"] = STATECHART_FILE
            write_json_atomically(
                output_directory / STATECHART_FILE, statecharts.to_payload()
            )
        write_json_atomically(output_directory / "scene.json", scene, indent=1)
        write_json_atomically(
            output_directory / "trajectory.json",
            {
                "framesPerSecond": frames_per_second,
                "frames": [frame.frames for frame in frames],
                "base": [frame.base for frame in frames],
                "objects": [frame.objects for frame in frames],
                RobotField.MODEL_BASES: [frame.model_bases for frame in frames],
            },
        )
        return scene


def finalize_recording(bridge: Bridge, recording: Recording) -> Optional[str]:
    """
    Ensure a recording's buffered frames have been written to disk, bundling them on
    first call.

    Idempotent and safe to call more than once — e.g. once from the viewer's explicit
    ``/recording/stop`` request, and again as a safety net if the demo process exits
    before that request ever arrives (see :mod:`cramera.live.visualization`): a
    recording already bundled, or one with nothing to bundle, is left untouched.

    :param bridge: The live bridge whose world and object catalog are bundled.
    :param recording: The recording to finalize.
    :return: The scene name the recording was bundled under, or None if there is nothing
        to bundle (capture never started, or no ticks were recorded).
    """
    if recording.state is RecordingState.IDLE:
        return None
    frames = recording.stop()
    if recording.scene_name is not None:
        return recording.scene_name
    if not frames:
        return None
    output_directory = paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME
    write_recording_bundle(
        bridge,
        frames,
        recording.frames_per_second(),
        output_directory,
        paths.RECORDING_SCENE_NAME,
    )
    recording.scene_name = paths.RECORDING_SCENE_NAME
    return recording.scene_name


def _loose_object_entries(
    bridge: Bridge, frames: List[RecordedFrame], output_directory: Path
) -> List[Dict[str, Any]]:
    """
    Declare each recorded catalog object at its first observed pose.

    :param bridge: The live bridge whose object catalog and bodies are read.
    :param frames: Recorded ticks supplying the first appearance of each object.
    :param output_directory: Directory a mesh-backed object's file is copied into.
    """
    first_poses: Dict[str, List[float]] = {}
    for frame in frames:
        for key, pose in frame.objects.items():
            first_poses.setdefault(key, pose)
    entries = []
    for entry in bridge.object_metadata:
        spawn = first_poses.get(entry.key)
        body = bridge.object_body(entry.key)
        if spawn is None or body is None:
            continue
        entries.append(_object_entry(entry, body, spawn, output_directory))
    return entries


def _object_entry(
    entry: ObjectCatalogEntry, body: Body, spawn: List[float], output_directory: Path
) -> Dict[str, Any]:
    """
    One loose object's ``scene.json`` entry: an inline box, or a copied/exported mesh.

    A body with no shapes at all (:attr:`ObjectCatalogEntry.kind` is
    :attr:`~cramera.live.bridge.ObjectKind.BOX`) reuses the catalog's own placeholder
    size rather than touching the body's geometry, which does not exist to measure or
    export.

    :param entry: The object's geometry-catalog entry, for its id, colour and — for a
        shapeless body — its placeholder size.
    :param body: The world body the object is published from.
    :param spawn: The object's first recorded pose.
    :param output_directory: Directory a mesh file is written into.
    """
    payload: Dict[str, Any] = {
        "id": entry.id,
        "key": entry.key,
        "spawn": spawn,
        "color": entry.color,
    }
    if entry.kind is ObjectKind.BOX:
        payload["box"] = list(entry.size)
        payload["height"] = entry.size[2]
        return payload
    extent = measure_body(body)
    if extent is not None:
        payload["height"] = round(extent.z, POSE_PRECISION)
    shapes = _body_shapes(body)
    if (
        len(shapes) == 1
        and isinstance(shapes[0], Box)
        and np.array_equal(shapes[0].origin.to_np(), np.eye(4))
    ):
        payload["box"] = rounded_scale(shapes[0].scale, POSE_PRECISION)
        return payload
    payload["mesh"] = _write_object_mesh(body, entry.key, shapes, output_directory)
    return payload


def _body_shapes(body: Body) -> List[Shape]:
    """
    The body's shapes to render from: its visual ones, else its collision ones.

    :param body: The body whose shapes are read.
    """
    for collection in (body.visual, body.collision):
        if collection.shapes:
            return list(collection.shapes)
    return []


def _write_object_mesh(
    body: Body, key: str, shapes: List[Shape], output_directory: Path
) -> str:
    """
    Write a loose object's geometry into the bundle and answer the path it is served at.

    An untransformed, unit-scale mesh is copied with its side assets. Other geometry is
    combined in the body's frame, preserving each selected shape's origin and scale.

    :param body: The body whose geometry is written.
    :param key: The object's catalog key, used as the written file's basename.
    :param shapes: The body's shapes, as :func:`_body_shapes` selected them.
    :param output_directory: Directory the mesh is written into.
    """
    objects_directory = output_directory / "meshes" / "objects"
    objects_directory.mkdir(parents=True, exist_ok=True)
    if (
        len(shapes) == 1
        and isinstance(shapes[0], Mesh)
        and np.array_equal(shapes[0].origin.to_np(), np.eye(4))
        and shapes[0].scale == Scale()
    ):
        source = shapes[0].filename
        if source and Path(source).is_file():
            destination = objects_directory / (key + Path(source).suffix)
            assets = BundledAssets(bundle_root=str(output_directory))
            if assets.copy(source, str(destination)):
                assets.copy_side_assets(source, str(destination))
                return "meshes/objects/" + destination.name
    destination = objects_directory / (key + MeshFormat.OBJ.value)
    ShapeCollection(shapes=shapes, reference_frame=body).combined_mesh.export(
        str(destination)
    )
    return "meshes/objects/" + destination.name
