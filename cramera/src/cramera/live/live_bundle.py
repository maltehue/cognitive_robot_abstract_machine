"""
Bundles a live world's *current* model into a throwaway scene.

Serializes the world object the bridge is attached to — the robot subtree as one
model, everything else as the environment model — through the same
:class:`~cramera.onboard.world_to_urdf.UrdfDocument` serializer every recorded scene
goes through. Whatever a demo built its world from, and however it built it, is what
the viewer shows; no model source tracking or manual onboarding is involved.

A build is idempotent: while the world model is unchanged, the existing bundle is left
untouched — a viewer may be downloading its files at any moment, and deleting them
mid-flight breaks the page that requested the rebuild in the first place.
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Any, Dict, Iterable, List, Optional

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from cramera import paths
from cramera.generated_json import GeneratedJson
from cramera.live.bridge import Bridge
from cramera.live.robot_models import RobotModels
from cramera.robot_fields import RobotField
from cramera.onboard.bundle_urdf import BundleReport
from cramera.onboard.world_to_urdf import UrdfDocument
from cramera.robot_parts import RobotPartAnnotation
from cramera.world_objects import WorldObjects

BUILD_LOCK = threading.Lock()
"""
Serializes bundling across this process.

The output directory is one fixed path, and a build clears it before writing it, so two
overlapping builds delete and overwrite each other's files. The bridge answers on a
threading HTTP server and the viewer keeps asking for the live scene while a demo starts
up, which makes that overlap routine rather than rare.
"""

ENVIRONMENT_MODEL_NAME = "environment"
"""
Name of the model holding every world body outside the robot and the object overlay.
"""

MESH_SUBDIRECTORY = "live"
"""
Directory bundled meshes nest under inside the scene's ``meshes/`` tree.
"""


@dataclass(frozen=True)
class WorldModelsBundle:
    """
    A live world's geometry, serialized into a scene bundle's ``models``/``robot``
    fields.
    """

    models: List[Dict[str, Any]]
    """
    The scene's ``models`` entries: the environment (if any bodies remain outside the
    robot and the object overlay) followed by the robot, when one is bound.
    """

    robot: Optional[Dict[str, Any]]
    """
    The scene's ``robot`` field, or None without a bound robot.
    """

    missing_assets: List[str]
    """
    Every asset a model referenced but could not resolve, across all models.
    """

    robots: List[Dict[str, Any]] = field(default_factory=list)
    """Metadata for every articulated robot in the shared world."""


def bundle_world_models(
    world: World,
    robot: Optional[AbstractRobot],
    output_directory: Path,
    mesh_subdirectory: str,
    *,
    overlay_bodies: Iterable[Body] | None = None,
) -> WorldModelsBundle:
    """
    Serialize a world's robot and environment bodies into URDF models on disk.

    Shared between the throwaway ``__live__`` bundle and a finalized live recording
    (:mod:`cramera.live.recording_bundle`) — both bundle the same live world through the
    same :class:`~cramera.onboard.world_to_urdf.UrdfDocument` serializer, and would
    otherwise duplicate the environment/robot split and the ``UrdfDocument.of_bodies``
    wiring.

    :param world: The world whose bodies are serialized.
    :param robot: The world's robot annotation, or None to bundle everything as
        environment.
    :param output_directory: Directory the URDF and mesh files are written into.
    :param mesh_subdirectory: Directory the meshes nest under inside
        ``output_directory``.
    :param overlay_bodies: Bodies rendered independently, including previously loose
        objects now attached to the robot; inferred from the world when omitted.
    """
    excluded_bodies = set(
        WorldObjects(world, robot).overlay_bodies()
        if overlay_bodies is None
        else overlay_bodies
    )
    robot_models = RobotModels(world, robot)
    bodies_by_robot = [
        (
            annotation,
            [
                body
                for body in _robot_bodies(world, annotation)
                if body not in excluded_bodies
            ],
        )
        for annotation in robot_models.robots()
    ]
    robot_body_set = {body for _, bodies in bodies_by_robot for body in bodies}
    models: List[Dict[str, Any]] = []
    environment_bodies = [
        body
        for body in world.bodies_topologically_sorted
        if body not in robot_body_set and body not in excluded_bodies
    ]
    if environment_bodies:
        report = UrdfDocument.of_bodies(
            environment_bodies,
            ENVIRONMENT_MODEL_NAME,
            str(output_directory),
            mesh_subdirectory,
        )
        models.append(_model_payload(report, is_robot=False))
    for annotation, robot_bodies in bodies_by_robot:
        if not robot_bodies:
            continue
        report = UrdfDocument.of_bodies(
            robot_bodies,
            robot_models.model_name(annotation),
            str(output_directory),
            mesh_subdirectory,
            identity_root=annotation.root,
        )
        model = _model_payload(report, is_robot=True)
        model["prefix"] = RobotModels.identifier(annotation)
        model[RobotField.IDENTIFIER] = RobotModels.identifier(annotation)
        model[RobotField.POSE] = robot_models.root_poses()[
            RobotModels.identifier(annotation)
        ]
        models.append(model)
    missing_assets = sorted(
        {missing for model in models for missing in model.pop("missing")}
    )
    return WorldModelsBundle(
        models=models,
        robot=_robot_payload(robot),
        missing_assets=missing_assets,
        robots=[_robot_payload(annotation) for annotation in robot_models.robots()],
    )


def build_live_scene(bridge: Bridge) -> Optional[str]:
    """
    Bundle the live world's current model into a throwaway scene.

    :param bridge: The live bridge whose current world is bundled.
    :return: :data:`cramera.paths.LIVE_SCENE_NAME`, the scene name to navigate the
        viewer to, or None while no world is attached yet.
    """
    if bridge.world is None:
        return None
    output_directory = paths.local_scenes_directory() / paths.LIVE_SCENE_NAME
    with BUILD_LOCK:
        signature = bridge.bundle_signature()
        if _existing_signature(output_directory) == signature:
            return paths.LIVE_SCENE_NAME
        return _write_bundle(bridge, output_directory, signature)


def _existing_signature(output_directory: Path) -> Optional[str]:
    """
    The signature the existing bundle was built from, or None without a readable one.

    :param output_directory: Directory the previous bundle was written to.
    """
    scene = GeneratedJson(output_directory / "scene.json").read()
    if not isinstance(scene, dict):
        return None
    return scene.get("bundleSignature")


def _write_bundle(bridge: Bridge, output_directory: Path, signature: str) -> str:
    """
    Clear the throwaway scene's directory and serialize the world into it.

    Only ever called while :data:`BUILD_LOCK` is held.

    :param bridge: The live bridge whose current world is bundled.
    :param output_directory: Directory the bundle is written to, cleared first.
    :param signature: The digest of the world model this bundle is built from, recorded
        so the next build can tell whether anything changed.
    """
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
    scene = {
        "name": paths.LIVE_SCENE_NAME,
        "models": geometry.models,
        "robot": geometry.robot,
        RobotField.ROBOTS: geometry.robots,
        RobotField.ACTIVE_ROBOT: (
            RobotModels.identifier(bridge.robot) if bridge.robot is not None else None
        ),
        "objects": [],
        "segments": [],
        "missingAssets": geometry.missing_assets,
        # the world was attached before this bundle was built; kept for the viewer,
        # which rebuilds a bundle that reports False
        "worldBound": True,
        # what this bundle was built from; while it is unchanged, /live_scene leaves
        # the bundle untouched instead of deleting files a viewer may be downloading
        "bundleSignature": signature,
    }
    (output_directory / "scene.json").write_text(json.dumps(scene, indent=1))
    return paths.LIVE_SCENE_NAME


def _robot_bodies(world: World, robot: Optional[AbstractRobot]) -> List[Body]:
    """
    The robot's subtree in serialization order, or an empty list without a robot.

    :param world: The world the robot lives in.
    :param robot: The robot annotation, or None.
    """
    if robot is None:
        return []
    subtree = set(world.get_kinematic_structure_entities_of_branch(robot.root))
    return [body for body in world.bodies_topologically_sorted if body in subtree]


def _model_payload(report: BundleReport, is_robot: bool) -> Dict[str, Any]:
    """
    One entry of the scene's ``models`` list.

    :param report: What the serializer wrote for this model.
    :param is_robot: Whether the entry is the robot rather than the environment.
    """
    return {
        "name": report.name,
        "urdf": "%s.urdf" % report.name,
        "prefix": "",
        "robot": is_robot,
        "links": len(report.links),
        "movableJoints": report.movable_joints,
        "posedJoints": report.posed_joints,
        "missing": report.missing,
    }


def _robot_payload(robot: Optional[AbstractRobot]) -> Optional[Dict[str, Any]]:
    """
    The scene's ``robot`` field, or None if no robot is bound.

    :param robot: The robot's semantic annotation, or None if no robot is bound.
    """
    if robot is None:
        return None
    root_name = str(robot.root.name)
    part_annotations = RobotPartAnnotation.of_robot(robot)
    return {
        "name": type(robot).__name__.lower(),
        RobotField.IDENTIFIER: RobotModels.identifier(robot),
        RobotField.LABEL: (
            robot.name.name
            if isinstance(robot, AbstractRobot)
            else type(robot).__name__
        ),
        "prefix": root_name.split("/", 1)[0] if "/" in root_name else "",
        "baseBody": root_name.split("/", 1)[-1],
        "parts": {annotation.name: annotation.links for annotation in part_annotations},
        "partAnnotations": [annotation.to_payload() for annotation in part_annotations],
    }
