"""
Turn a coraplex demo into a self-contained web-viewer scene.

Runs the demo file *unmodified* under instrumentation and emits a scene bundle
into :func:`cramera.paths.scenes_directory`::

    <scenes dir>/<name>/
        scene.json         models, robot parts, objects, segments, targets
        trajectory.json    per-tick joints + robot base + object world poses
        <model>.urdf       package:// resolved & rewritten
        meshes/...         all meshes + textures the scene needs

What the hooks capture while the demo runs:
  - every package:// asset resolution and every URDF/STL the world loads
  - per-tick positions of all movable connections (giskardpy Executor.tick)
  - world pose of the robot base and of every loose (STL) object
  - one segment per executed plan ActionNode, with nesting depth
  - the robot's semantic annotation: base body, arms, end-effector link sets

Usage (the interpreter needs the CRAM stack on it)::

    cramera-onboard path/to/demo.py --name pr2_kitchen

A demo file that parses its own CLI arguments (its own ``argparse.ArgumentParser``
or ``sys.argv`` reads) receives everything after a ``--`` separator unchanged::

    cramera-onboard path/to/demo.py --name pr2_kitchen -- --robot pr2 --seed 3
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import os
import runpy
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from semantic_digital_twin.adapters.gazebo import GazeboParser
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.adapters.package_resolver import PackageUriResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
)
from semantic_digital_twin.world_description.geometry import Box, Mesh
from typing_extensions import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    TYPE_CHECKING,
)

from cramera import paths
from cramera.logging_setup import get_logger
from cramera.body_geometry import (
    measure_body,
    POSE_PRECISION,
    rounded_pose,
    rounded_scale,
)
from cramera.knowledge.detected_events import (
    DetectedEventRecord,
    SceneField,
    TrajectoryField,
)
from cramera.knowledge.recorded_statecharts import STATECHART_FILE, RecordedStatecharts
from cramera.live.chart_observer import ChartObserver
from cramera.live.chart_structure import ChartSnapshot
from cramera.onboard.detection_recorder import DetectionRecorder
from cramera.generated_json import write_json_atomically
from cramera.live.bridge import ROBOT_BASE_KEY
from cramera.monkey_patch import MethodPatch
from cramera.robot_parts import RobotPartAnnotation
from cramera.mesh_format import MeshFormat
from cramera.onboard.bundle_urdf import (
    BundledAssets,
    BundledModel,
    BundleReport,
    bundle_model,
    companion_material_library,
)
from cramera.onboard.bundle_world import BundledWorld
from cramera.onboard.scene_index import write_scene_index
from cramera.onboard.world_to_urdf import UrdfDocument
from cramera.palette import ObjectPalette
from cramera.spatial_annotations import SpatialAnnotations
from cramera.world_objects import WorldObjects

if TYPE_CHECKING:
    from coraplex.plans.executables import Executable
    from coraplex.plans.plan_node import ActionNode
    from giskardpy.executor import Executor
    from semantic_digital_twin.world_description.world_entity import Body

logger = get_logger(__name__)

_STARTED_AT = time.time()
"""
When this process started, so progress lines can show elapsed recording time.
"""

TARGET_BUNDLE_FRAMES = 1500
"""
Frame count a bundle is downsampled towards when no explicit step is given.
"""

MISSING_ASSETS_LOGGED = 10
"""
How many unresolved assets the summary lists before truncating.
"""


@runtime_checkable
class NamesAWorldEntity(Protocol):
    """
    Anything carrying a world-entity name, such as a body a designator refers to.
    """

    name: Any


@runtime_checkable
class DescribesAnAction(Protocol):
    """
    A plan node carrying the designator that describes what it does.
    """

    designator: Any


def log(*parts: object) -> None:
    """
    Emit a progress line prefixed with the elapsed recording time.

    :param parts: Values joined by a space to form the log message.
    """
    logger.info(
        "[%6.1fs] %s",
        time.time() - _STARTED_AT,
        " ".join(str(part) for part in parts),
    )


# %% recorder
@dataclass
class SpawnedBox:
    """
    One primitive box body spawned from a specification while the demo ran.

    Loose objects usually enter the world as mesh files, which the mesh-parser hook
    captures; a box body has no file, so its geometry is remembered here instead.
    """

    name: str
    """
    The body's world name, the key its poses are recorded under.
    """

    scale: List[float]
    """
    The box extents in metres, as ``[x, y, z]``.
    """

    color: str
    """
    The authored colour, as a css hex string.
    """

    @classmethod
    def of_specification(
        cls, specification: BodySpecification, name: Optional[str] = None
    ) -> Optional[SpawnedBox]:
        """
        The recordable box a specification describes, or None when it describes anything
        other than a single box shape.

        :param specification: The specification a body was materialized from.
        :param name: The spawn-time name override, if one was given.
        """
        shapes = specification.shapes.shapes
        if len(shapes) != 1 or not isinstance(shapes[0], Box):
            return None
        [shape] = shapes
        return cls(
            name=str(name or specification.name),
            scale=[
                round(float(value), POSE_PRECISION) for value in shape.scale.to_np()[:3]
            ],
            color=ObjectPalette.css_color(shape.color),
        )


@dataclass
class Recorder:
    """
    Records one demo run: assets, per-tick motion and the executed plan.

    .. note:: ``giskardpy`` and ``coraplex`` are only imported inside the
       ``install_*`` hook methods that need them. Unlike ``semantic_digital_twin``,
       which this module already imports at the top, they are not required to parse
       a finished recording into a scene bundle, and this module is imported by the
       ``cramera-onboard`` console script, which has to stay importable without
       them. This is one of the documented exceptions to the imports-at-top rule.
    """

    FRAME_LOG_INTERVAL: ClassVar[int] = 2000
    """
    How many recorded frames pass between progress lines.
    """

    MAX_SERIALIZED_PLAN_NODES: ClassVar[int] = 400
    """
    Upper bound on the plan nodes written into a bundle.
    """

    DEFAULT_DETECTION_INTERVAL: ClassVar[int] = 50
    """
    How many run ticks pass between two detector ticks by default.

    One tick of segmind's detectors compares every tracked object against every body
    with collision geometry in the world, which on a furnished world costs seconds -
    times the tens of thousands of ticks a run takes. Looking at every 50th tick keeps
    the moments it recognizes within a third of a second of when they happened, which is
    finer than the events themselves.
    """

    resolutions: Dict[str, str] = field(default_factory=dict)
    """
    ``package://`` URI to the path it resolved to while the demo ran.
    """

    urdf_sources: List[str] = field(default_factory=list)
    """
    URDF/xacro files the world was built from, in load order.
    """

    task: Optional[str] = None
    """
    What the run being recorded does, as the person recording it named it, or None to
    take what its plan did.
    """

    chart_frames: List[Optional[ChartSnapshot]] = field(default_factory=list)
    """
    What the motion statechart looked like at each recorded frame, None for a frame with
    nothing executing.

    A statechart exists only while it is being ticked, so replaying one afterwards is
    only possible from what was kept of it while it ran.
    """

    _chart_observer: ChartObserver = field(default_factory=ChartObserver, init=False)
    """
    Reads what the executing statechart looks like, remembering what it last saw.
    """

    frame_times: List[float] = field(default_factory=list)
    """
    When each recorded frame was taken, in seconds since the epoch, one per entry of
    :attr:`frames`.

    An answer names a moment in wall-clock time, so a replay needs to know which frames
    were taken around it.
    """

    detections: List[DetectedEventRecord] = field(default_factory=list)
    """
    What segmind's detectors saw while the run was recorded, oldest first, or nothing
    for a run recorded without them.
    """

    detect_events: bool = False
    """
    Whether to tick segmind's detectors along the run, which is what gives the bundle
    its detections.
    """

    detection_interval: int = 50
    """
    How many run ticks pass between two detector ticks.
    """

    _ticks_since_detection: int = field(default=0, init=False)
    """
    Run ticks since the detectors last looked, counted up to
    :attr:`detection_interval`.
    """

    _detection_recorder: Optional[DetectionRecorder] = field(default=None, init=False)
    """
    The detectors watching the run, once the world they watch exists.
    """

    gazebo_sources: List[str] = field(default_factory=list)
    """
    Gazebo/SDF worlds and models the demo loaded, in load order.
    """

    mjcf_sources: List[str] = field(default_factory=list)
    """
    MJCF robots and scenes the demo loaded, in load order.
    """

    mesh_sources: List[str] = field(default_factory=list)
    """
    Mesh files of the loose objects, in load order.
    """

    spawned_boxes: List[SpawnedBox] = field(default_factory=list)
    """
    Loose objects spawned as a primitive box rather than loaded from a mesh file.
    """

    frames: List[Dict[str, float]] = field(default_factory=list)
    """
    Per-tick joint positions, keyed by prefixed connection name.
    """

    base_frames: List[Optional[List[float]]] = field(default_factory=list)
    """
    Per-tick robot base pose as ``[x, y, z, qx, qy, qz, qw]``.
    """

    object_frames: List[Dict[str, List[float]]] = field(default_factory=list)
    """
    Per-tick world pose of every tracked object, keyed by mesh basename.
    """

    actions: List[Dict[str, Any]] = field(default_factory=list)
    """
    One entry per parsed action: its class, arm and target object.
    """

    plan_nodes: List[Any] = field(default_factory=list)
    """
    The plan nodes the demo parsed, used to serialize the executed plan tree.
    """

    world: Optional[Any] = None
    """
    The executing world, captured on the first tick.
    """

    robot: Optional[AbstractRobot] = None
    """
    The robot annotation of :attr:`world`.
    """

    control_timestep: Optional[float] = None
    """
    The controller's timestep, from which the recording's frame rate follows.
    """

    _connections: Optional[List[Any]] = field(default=None)
    """
    Connections whose position is recorded; None until the first tick binds.
    """

    _asset_hook_uninstallers: List[Callable[[], None]] = field(default_factory=list)
    """
    Restores the methods :meth:`install_asset_hooks` last replaced.
    """

    _bodies: Optional[Dict[str, Any]] = field(default=None)
    """
    Recorded bodies by mesh basename, plus :data:`ROBOT_BASE_KEY`.
    """

    # %% asset hooks
    def install_asset_hooks(self) -> None:
        """
        Record every asset resolution so the bundler can copy the files.
        """
        self._asset_hook_uninstallers = [
            MethodPatch(PackageUriResolver, "resolve").install(
                self._remember_resolution
            ),
            MethodPatch(URDFParser, "from_file").install(self._remember_urdf_source),
            MethodPatch(GazeboParser, "from_file").install(
                self._remember_gazebo_source
            ),
            MethodPatch(MJCFParser, "__init__").install(self._remember_mjcf_source),
            MethodPatch(STLParser, "__init__").install(self._remember_mesh_source),
            MethodPatch(BodySpecification, "to_domain_object").install(
                self._remember_spawned_box
            ),
        ]

    def uninstall_asset_hooks(self) -> None:
        """
        Restore the methods :meth:`install_asset_hooks` replaced.

        Bundling re-parses a recorded Gazebo or MJCF source to build a clean URDF for
        it, which the hooks would otherwise mistake for another source to record.
        """
        for uninstall in self._asset_hook_uninstallers:
            uninstall()
        self._asset_hook_uninstallers = []

    def _remember_resolution(
        self,
        original: Callable[[PackageUriResolver, str], str],
        resolver: PackageUriResolver,
        uri: str,
    ) -> str:
        """
        Resolve as usual, but remember the uri -> path mapping.

        :param original: The real, unpatched ``PackageUriResolver.resolve`` bound
            method.
        :param resolver: The resolver resolving the URI.
        :param uri: The ``package://`` URI being resolved.
        """
        resolved = original(resolver, uri)
        self.resolutions[uri] = resolved
        return resolved

    def _remember_urdf_source(
        self,
        original: Callable[..., URDFParser],
        cls: type,
        file_path: str,
        **kwargs: Any,
    ) -> URDFParser:
        """
        Parse as usual, but remember this URDF/xacro source file.

        :param original: The real, unpatched ``URDFParser.from_file`` classmethod.
        :param cls: The ``URDFParser`` class the method is bound to.
        :param file_path: Path of the URDF/xacro source file being parsed.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        if file_path not in self.urdf_sources:
            self.urdf_sources.append(file_path)
        return original(cls, file_path, **kwargs)

    def _remember_gazebo_source(
        self,
        original: Callable[..., GazeboParser],
        cls: type,
        file_path: str,
        **kwargs: Any,
    ) -> GazeboParser:
        """
        Parse as usual, but remember this Gazebo/SDF world or model source file.

        :param original: The real, unpatched ``GazeboParser.from_file``.
        :param cls: The parser class the classmethod was called on.
        :param file_path: Path of the source file.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        if file_path not in self.gazebo_sources:
            self.gazebo_sources.append(file_path)
        return original(cls, file_path, **kwargs)

    def _remember_mjcf_source(
        self,
        original: Callable[..., None],
        mjcf_parser: MJCFParser,
        file_path: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize as usual, but remember this MJCF source file.

        :param original: The real, unpatched ``MJCFParser.__init__`` bound method.
        :param mjcf_parser: The parser being initialized.
        :param file_path: Path of the source file.
        :param args: Positional arguments forwarded to the wrapped call.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        if file_path not in self.mjcf_sources:
            self.mjcf_sources.append(file_path)
        return original(mjcf_parser, file_path, *args, **kwargs)

    def _remember_mesh_source(
        self,
        original: Callable[..., None],
        stl_parser: STLParser,
        file_path: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize as usual, but remember this loose object's mesh file.

        :param original: The real, unpatched ``STLParser.__init__`` bound method.
        :param stl_parser: The parser being initialized.
        :param file_path: Path of the object's mesh file.
        :param args: Positional arguments forwarded to the wrapped call.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        if file_path not in self.mesh_sources:
            self.mesh_sources.append(file_path)
        return original(stl_parser, file_path, *args, **kwargs)

    def _remember_spawned_box(
        self,
        original: Callable[..., Any],
        specification: BodySpecification,
        name: Optional[str] = None,
    ) -> Any:
        """
        Materialize as usual, but remember box bodies so their poses get recorded.

        :param original: The real, unpatched ``BodySpecification.to_domain_object``.
        :param specification: The specification a body is being materialized from.
        :param name: The spawn-time name override, if one was given.
        """
        spawned = SpawnedBox.of_specification(specification, name)
        if spawned is not None and all(
            recorded.name != spawned.name for recorded in self.spawned_boxes
        ):
            self.spawned_boxes.append(spawned)
        return original(specification, name)

    # %% trajectory hook
    def install_tick_hook(self) -> None:
        """
        Wrap Executor.tick so every simulation step is snapshotted.
        """
        from giskardpy.executor import Executor

        MethodPatch(Executor, "tick").install(self._record_tick)

    def _record_tick(
        self,
        original: Callable[..., None],
        executor: Executor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Run the real tick, then record its resulting world state.

        The detectors are ticked by an executor of their own, and this patch is
        installed on every executor there is: its tick is not a step of the run, so it
        is run and nothing else.

        :param original: The real, unpatched ``Executor.tick`` bound method.
        :param executor: The executor whose tick is being run.
        :param args: Positional arguments forwarded to the wrapped call.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        if (
            self._detection_recorder is not None
            and self._detection_recorder.owns_executor(executor)
        ):
            return original(executor, *args, **kwargs)
        result = original(executor, *args, **kwargs)
        self.record_frame(executor)
        self.record_detections()
        return result

    @staticmethod
    def object_key(body: Body) -> str:
        """
        The key a body's recorded poses are filed under: its local name, without the world
        prefix a composed world gives it.

        :param body: The body to key.
        """
        return str(body.name).split("/")[-1]

    def recorded_objects(self) -> Dict[str, Body]:
        """
        The objects this recording followed, by the key it followed them under: what it
        bound itself to when the run started, the robot's own base aside.

        Asked of the recording rather than of the world at the end: a run that carries
        something re-parents it to the gripper, and a body under a gripper is no longer
        one the world lets move freely -- so asking the world again would leave out
        exactly the objects the run was about.
        """
        return {
            key: body for key, body in self._bodies.items() if key != ROBOT_BASE_KEY
        }

    def free_floating_bodies(self) -> List[Body]:
        """
        Every body its world lets move freely, which is how a world states that a body is
        loose rather than part of the furniture.

        The robot's own bodies are left out: a mobile base is free-floating too, and it is
        recorded as the robot rather than as an object. So are bodies without geometry: a
        world lets frames move as well as things -- a mobile robot's ``odom`` is free and
        has no shape -- and a frame gives a viewer nothing to draw.
        """
        return WorldObjects(self.world, self.robot).free_floating()

    def bind_to_executor(self, executor: Executor) -> None:
        """
        Locate the world, robot and recordable objects of a running executor.

        Deferred until the first tick, because the world does not exist any earlier.

        :param executor: The executor whose world is bound to.
        """
        self.world = executor.context.world
        if self.detect_events:
            self._detection_recorder = DetectionRecorder(world=self.world)
            self._detection_recorder.start()
        self.control_timestep = executor.context.qp_controller_config.control_dt
        robots = self.world.get_semantic_annotations_by_type(AbstractRobot)
        self.robot = robots[0] if robots else None
        self._bodies = {}
        if self.robot is not None:
            self._bodies[ROBOT_BASE_KEY] = self.robot.root
        for mesh_source in self.mesh_sources:
            name = os.path.basename(mesh_source)
            body = self.world.get_body_by_name(name)
            if body is not None:
                self._bodies[name] = body
        for spawned in self.spawned_boxes:
            body = self.world.get_body_by_name(spawned.name)
            if body is not None:
                self._bodies[spawned.name] = body
        # a world built in code loads no mesh file and spawns no box, so its loose objects
        # can only be recognized by the world letting them move freely
        for body in self.free_floating_bodies():
            self._bodies.setdefault(self.object_key(body), body)
        self._connections = [
            connection
            for connection in self.world.connections
            if isinstance(connection, ActiveConnection1DOF)
        ]
        log(
            "bound: robot=%s, %d movable connections, objects=%s"
            % (
                type(self.robot).__name__ if self.robot else None,
                len(self._connections),
                [key for key in self._bodies if key != ROBOT_BASE_KEY],
            )
        )

    def record_detections(self) -> None:
        """
        Let the detectors look at the world the tick just left behind, and keep what they
        saw.

        Ticked inline here rather than on a thread of its own: segmind builds CasADi
        expressions, which are reference counted without atomics, so detecting while the
        planner builds expressions of its own corrupts them.
        """
        if self._detection_recorder is None:
            return
        if self._ticks_since_detection % self.detection_interval:
            self._ticks_since_detection += 1
            return
        self._ticks_since_detection += 1
        self._detection_recorder.tick()
        self.detections = self._detection_recorder.records()

    def record_frame(self, executor: Executor) -> None:
        """
        Append one frame: every movable connection's position, the robot base pose and
        every tracked object's pose.

        :param executor: The executor whose current tick is recorded.
        """
        if self._connections is None:
            self.bind_to_executor(executor)
        self.frame_times.append(time.time())
        self.chart_frames.append(
            self._chart_observer.snapshot(executor.motion_statechart)
        )
        self.frames.append(
            {
                str(connection.name): round(float(connection.position), POSE_PRECISION)
                for connection in self._connections
            }
        )
        self.base_frames.append(
            rounded_pose(self._bodies[ROBOT_BASE_KEY])
            if ROBOT_BASE_KEY in self._bodies
            else None
        )
        self.object_frames.append(
            {
                name: rounded_pose(body)
                for name, body in self._bodies.items()
                if name != ROBOT_BASE_KEY
            }
        )
        if len(self.frames) % self.FRAME_LOG_INTERVAL == 0:
            log("... %d frames" % len(self.frames))

    # %% action metadata hook
    # Plans compile all actions into merged motion statecharts, so there is no
    # per-action call boundary at execution time. ``ActionNode.parse`` does fire
    # once per action (in plan order) — the action class, its arm and its target
    # object are recorded there, and the segment *timing* is derived afterwards
    # from the recorded data (object attach/detach + first base motion).
    def _target_of(self, designator: Any) -> Optional[str]:
        """
        The recorded object a designator refers to, matched by its recorded key: the
        mesh basename for mesh objects, the body name for spawned boxes.

        :param designator: The designator to search for a world-entity reference.
        """
        keys = {os.path.basename(path) for path in self.mesh_sources}
        keys |= {spawned.name for spawned in self.spawned_boxes}
        keys |= set(self._bodies or {})
        for value in vars(designator).values():
            if not isinstance(value, NamesAWorldEntity):
                continue
            key = str(value.name).split("/")[-1]
            if key in keys:
                return key
        return None

    @staticmethod
    def _arm_of(designator: Any) -> Optional[str]:
        """
        The arm a designator names, whether it calls the field ``arm`` or ``arms``.

        :param designator: The designator to read the arm field from.
        """
        fields = vars(designator)
        arm = fields.get("arm") or fields.get("arms")
        return str(arm) if arm is not None else None

    def install_segment_hook(self) -> None:
        """
        Wrap ActionNode.parse to record each action's class, arm and target as it is
        parsed (in plan order).
        """
        from coraplex.plans.plan_node import ActionNode

        MethodPatch(ActionNode, "parse").install(self._record_segment)

    def _record_segment(
        self,
        original: Callable[..., Executable],
        node: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Executable:
        """
        Record this action's designator before letting it parse normally.

        :param original: The real, unpatched ``ActionNode.parse`` bound method.
        :param node: The action node being parsed.
        :param args: Positional arguments forwarded to the wrapped call.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        designator = node.designator
        self.actions.append(
            {
                "action": type(designator).__name__,
                "arm": self._arm_of(designator),
                "target": self._target_of(designator),
            }
        )
        self.plan_nodes.append(node)
        log(
            "action parsed:",
            self.actions[-1]["action"],
            "->",
            self.actions[-1]["target"] or "-",
        )
        return original(node, *args, **kwargs)

    def resolve_action_targets(self) -> None:
        """
        Fill in the target of every action that had none when it was parsed.

        An action parsed before the first tick was recorded while no world was bound yet,
        so an object that is neither a mesh file nor a spawned box could not be recognized
        as its target. The designators are still on the recorded plan nodes, so they are
        matched again against the objects that ended up tracked.
        """
        for action, node in zip(self.actions, self.plan_nodes):
            if action["target"] is None:
                action["target"] = self._target_of(node.designator)

    # %% the executed plan tree, serialized from the real PlanNode graph
    def serialize_plans(self, max_nodes: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        The executed plan trees, deduplicated by root, as nested dicts.

        :param max_nodes: Upper bound on the total node count across all trees; the
            recording stops descending once it is reached. Defaults to
            :attr:`MAX_SERIALIZED_PLAN_NODES`.
        """
        if max_nodes is None:
            max_nodes = self.MAX_SERIALIZED_PLAN_NODES
        roots: List[Any] = []
        seen_roots = set()
        for node in self.plan_nodes:
            root = node
            while root.parent is not None:
                root = root.parent
            if id(root) not in seen_roots:
                seen_roots.add(id(root))
                roots.append(root)
        serialized_count = itertools.count()
        return [
            tree
            for tree in (
                self._serialized_plan_node(root, serialized_count, max_nodes)
                for root in roots
            )
            if tree
        ]

    def _serialized_plan_node(
        self, node: Any, serialized_count: Iterator[int], max_nodes: int
    ) -> Optional[Dict[str, Any]]:
        """
        One plan node and its children as a dict, or None past ``max_nodes``.

        :param node: The plan node to serialize.
        :param serialized_count: Counter shared across the whole tree walk, advanced
            once per serialized node.
        :param max_nodes: Node count at which the walk stops descending.
        """
        if next(serialized_count) >= max_nodes:
            return None
        designator = node.designator if isinstance(node, DescribesAnAction) else None
        entry = {
            "kind": type(node).__name__,
            "label": (
                type(designator).__name__
                if designator is not None
                else type(node).__name__
            ),
            "status": node.status.name,
        }
        if designator is not None:
            target = self._target_of(designator)
            if target:
                entry["target"] = target
            arm = self._arm_of(designator)
            if arm is not None:
                entry["arm"] = arm
        children = node.children
        entry["children"] = [
            child
            for child in (
                self._serialized_plan_node(child, serialized_count, max_nodes)
                for child in children or ()
            )
            if child
        ]
        return entry


# %% post-processing the recording
@dataclass
class RecordingAnalysis:
    """
    Derives the played-back story of a recording: which objects were transported, when
    the robot drove, and the plan segments the viewer scrubs through.
    """

    MOVEMENT_TOLERANCE: ClassVar[float] = 0.02
    """
    How far a pose must travel to count as moved at all, in metres.
    """

    TRANSPORT_TOLERANCE: ClassVar[float] = 0.03
    """
    How far an object must travel over the whole run to count as transported, in metres.
    """

    BASE_MOTION_TOLERANCE: ClassVar[float] = 0.05
    """
    How far the robot base must travel to count as having driven off, in metres.
    """

    recorder: Recorder
    """
    The finished recording being analysed.
    """

    @classmethod
    def has_moved(
        cls,
        first: Sequence[float],
        second: Sequence[float],
        tolerance: Optional[float] = None,
    ) -> bool:
        """
        Whether two poses are more than ``tolerance`` apart (planar distance plus
        height).

        :param first: The pose to compare against ``second``.
        :param second: The pose to compare against ``first``.
        :param tolerance: Tolerance in metres, so sensor jitter does not read as
            movement. Defaults to :attr:`MOVEMENT_TOLERANCE`.
        """
        if tolerance is None:
            tolerance = cls.MOVEMENT_TOLERANCE
        return (
            math.hypot(first[0] - second[0], first[1] - second[1])
            + abs(first[2] - second[2])
            > tolerance
        )

    def object_windows(self) -> List[Dict[str, Any]]:
        """
        The attach..detach frame window of every object that travelled overall.

        An object whose first and last pose differ was transported; the window spans
        from the first frame that differs from where it started to just past the last
        frame that differs from where it ended up.
        """
        object_frames = self.recorder.object_frames
        frame_count = len(object_frames)
        windows = []
        for name in object_frames[0]:
            spawn = object_frames[0].get(name)
            final = object_frames[frame_count - 1].get(name)
            if (
                not spawn
                or not final
                or not self.has_moved(spawn, final, self.TRANSPORT_TOLERANCE)
            ):
                continue
            attach = next(
                (
                    index
                    for index in range(frame_count)
                    if name in object_frames[index]
                    and self.has_moved(object_frames[index][name], spawn)
                ),
                frame_count - 1,
            )
            detach = (
                next(
                    (
                        index
                        for index in range(frame_count - 1, -1, -1)
                        if name in object_frames[index]
                        and self.has_moved(object_frames[index][name], final)
                    ),
                    0,
                )
                + 1
            )
            if attach < detach:
                windows.append(
                    {
                        "object": name,
                        "attach": attach,
                        "detach": detach,
                        "place": final[:3],
                    }
                )
        windows.sort(key=lambda window: window["attach"])
        return windows

    def first_base_motion(self, before: int) -> int:
        """
        The first frame before ``before`` at which the robot base left its spawn.

        :param before: Frame index to stop searching before.
        :return: That frame's index, or ``before`` if the base never moved.
        """
        spawn = self.recorder.base_frames[0]
        for index in range(min(before, len(self.recorder.base_frames))):
            pose = self.recorder.base_frames[index]
            if not pose or not spawn:
                continue
            if (
                math.hypot(pose[0] - spawn[0], pose[1] - spawn[1])
                > self.BASE_MOTION_TOLERANCE
            ):
                return index
        return before

    def derive_segments(self) -> List[Dict[str, Any]]:
        """
        Segments = data-driven windows, labelled from the parsed action list.

        """
        frame_count = len(self.recorder.frames)
        transport_windows = self.object_windows()
        manipulation_actions = [
            action for action in self.recorder.actions if action.get("target")
        ]
        leading_actions = []
        for action in self.recorder.actions:
            if action.get("target"):
                break
            leading_actions.append(action)

        segments = []
        previous_end = 0
        if transport_windows:
            lead_end = self.first_base_motion(transport_windows[0]["attach"])
            if lead_end > 10:
                label = (
                    leading_actions[0]["action"].replace("Action", "").lower()
                    if len(leading_actions) == 1
                    else "prepare"
                )
                segments.append(
                    {
                        "step": label,
                        "action": ",".join(
                            leading_action["action"]
                            for leading_action in leading_actions
                        )
                        or None,
                        "arm": None,
                        "start": 0,
                        "end": lead_end,
                    }
                )
                previous_end = lead_end
        unmatched_actions = list(manipulation_actions)
        for window_index, window in enumerate(transport_windows):
            matching_action = next(
                (
                    action
                    for action in unmatched_actions
                    if action["target"] == window["object"]
                ),
                None,
            )
            if matching_action:
                unmatched_actions.remove(matching_action)
            object_id = os.path.splitext(window["object"])[0]
            verb = (
                matching_action["action"].replace("Action", "").lower()
                if matching_action
                else "move"
            )
            has_next_window = window_index + 1 < len(transport_windows)
            next_attach = (
                transport_windows[window_index + 1]["attach"]
                if has_next_window
                else frame_count - 1
            )
            end = (
                min(
                    next_attach,
                    window["detach"] + max(10, (next_attach - window["detach"]) // 2),
                )
                if has_next_window
                else frame_count - 1
            )
            segments.append(
                {
                    "step": "%s_%s" % (verb, object_id),
                    "action": matching_action["action"] if matching_action else None,
                    "arm": matching_action["arm"] if matching_action else None,
                    "start": previous_end,
                    "end": end,
                    "picks": object_id,
                    "attach": window["attach"],
                    "detach": window["detach"],
                    "place": window["place"],
                }
            )
            previous_end = end
        if not segments:
            label = (
                self.recorder.actions[0]["action"].replace("Action", "").lower()
                if len(self.recorder.actions) == 1
                else "plan"
            )
            segments.append(
                {
                    "step": label,
                    "action": None,
                    "arm": None,
                    "start": 0,
                    "end": frame_count - 1,
                }
            )
        return segments


@dataclass
class SceneBuilder:
    """
    Assembles one finished recording into a scene bundle on disk.
    """

    FALLBACK_FRAMES_PER_SECOND: ClassVar[float] = 50.0
    """
    Frame rate assumed when the controller does not report its timestep.
    """

    MINIMUM_FRAMES_PER_SECOND: ClassVar[int] = 10
    """
    The slowest playback the viewer is given, however hard the recording is downsampled.
    """

    ENVIRONMENT_MODEL_NAME: ClassVar[str] = "environment"
    """
    Name of the model synthesized from the bodies no recorded source describes, which is
    also the environment such a scene is listed under since a world built in code names
    none of its own.
    """

    PLACE_TARGET_DROP: ClassVar[float] = 0.02
    """
    How far below the lowest recorded place pose the place marker is drawn, in metres.
    """

    PLACE_BOUNDS_MARGIN: ClassVar[float] = 0.55
    """
    Half-extent of the draggable place area around the recorded place poses, in metres.
    """

    PLACE_BOUNDS_FRONT_MARGIN: ClassVar[float] = 0.65
    """
    Extra room in front of the place area, so a target can be dragged towards the robot.
    """

    DRAG_BOUNDS_MARGIN_X: ClassVar[float] = 0.35
    """
    How far beyond the objects' spawn poses they may be dragged, in metres.
    """
    DRAG_BOUNDS_MARGIN_Y: ClassVar[float] = 0.6

    PREFIX_PROBE_LINKS: ClassVar[int] = 12
    """
    How many of a model's links are probed to find its prefix in the composed world.
    """

    recorder: Recorder
    """
    The finished recording a scene is built from.
    """

    scene_name: str
    """
    Scene name, used as the output folder and in the scene metadata.
    """

    output_directory: str
    """
    Directory the scene bundle is written into.
    """

    step: int
    """
    Downsampling step; every ``step``-th frame is kept.
    """

    @staticmethod
    def _nearest_kept_frame(downsampled_index: Dict[int, int], raw_index: int) -> int:
        """
        The downsampled index closest to a raw frame index.

        :param downsampled_index: Raw frame indices mapped to their downsampled
            position.
        :param raw_index: Frame index in the original, un-downsampled recording.
        """
        return downsampled_index.get(
            raw_index,
            downsampled_index[
                min(downsampled_index, key=lambda kept: abs(kept - raw_index))
            ],
        )

    def _bundle_model(
        self,
        source: str,
        bundler: Callable[..., BundleReport],
        world_body_names: List[str],
        base_body: str,
    ) -> BundledModel:
        """
        Bundle one model source and turn its report into a ``models`` scene entry.

        :param source: Path or URI of the model's source file.
        :param bundler: Bundles the source into :attr:`output_directory`.
        :param world_body_names: Every body name in the composed world, used to find the
            model's prefix.
        :param base_body: The robot's base link name, used to tell a robot model apart
            from an environment model.
        :return: The model's ``models`` scene entry, and the bundler's report.
        """
        return bundle_model(
            source,
            bundler,
            world_body_names,
            base_body,
            self.output_directory,
            self.PREFIX_PROBE_LINKS,
            hints=self.recorder.resolutions,
        )

    @staticmethod
    def _body_name_of(model: BundledModel, link: str) -> str:
        """
        What the composed world calls a link of one bundled model.

        A model's own file names its links plainly; a world composed of several loaded
        worlds gives each of them a prefix, and the bodies are named under it.

        :param model: The model whose file the link is named in.
        :param link: The link name as that file spells it.
        """
        return "%s/%s" % (model.prefix, link) if model.prefix else link

    def _bundle_unclaimed_bodies(
        self, bundled_models: List[BundledModel], objects: List[Dict[str, Any]]
    ) -> Optional[BundledModel]:
        """
        Bundle the world's remaining bodies as one model, or answer None when a parsed
        source already describes every one of them.

        A world built in code -- bodies constructed directly rather than parsed out of a
        file -- otherwise leaves the viewer nothing to draw, because a model is only ever
        made from a recorded source. The bodies the recording tracks as objects are left
        out: those are drawn from their own geometry and moved per frame, so a copy of them
        in the environment would be a second, motionless one.

        :param bundled_models: The models built from the recorded sources.
        :param objects: The ``objects`` entries built for this scene.
        """
        described = {
            self._body_name_of(model, link)
            for model in bundled_models
            for link in model.report.links
        }
        tracked = {entry["key"] for entry in objects}
        unclaimed = [
            body
            for body in self.recorder.world.bodies_topologically_sorted
            if str(body.name) not in described
            and self.recorder.object_key(body) not in tracked
        ]
        if not unclaimed:
            return None
        report = UrdfDocument.of_bodies(
            bodies=unclaimed,
            name=self.ENVIRONMENT_MODEL_NAME,
            output_directory=self.output_directory,
            mesh_subdirectory=self.ENVIRONMENT_MODEL_NAME,
        )
        log(
            "bundled %-28s prefix=%-12s robot=%s meshes=%d missing=%d (%d bodies no source described)"
            % (
                self.ENVIRONMENT_MODEL_NAME,
                "-",
                False,
                report.meshes_copied,
                len(report.missing),
                len(unclaimed),
            )
        )
        return BundledModel(
            name=self.ENVIRONMENT_MODEL_NAME,
            prefix="",
            is_robot=False,
            report=report,
        )

    def _object_of_body(
        self, body: Body, key: str, index: int, palette: ObjectPalette
    ) -> Dict[str, Any]:
        """
        The ``objects`` entry describing a body the world holds only in memory.

        A single box is described as one, since the viewer can draw that without a file.
        Anything else needs geometry on disk: a mesh shape's own file is copied, so its
        materials come along, and a body of any other shape is written out as the single
        mesh it adds up to.

        :param body: The body to describe.
        :param key: The key the body's recorded poses are filed under.
        :param index: Position in the object list, which picks the entry's colour.
        :param palette: Supplies a distinguishable colour per object.
        """
        entry = {
            "id": key,
            "key": key,
            "spawn": self.recorder.object_frames[0][key],
            "color": palette.color_for(index),
        }
        extent = measure_body(body)
        if extent is not None:
            entry["height"] = round(extent.z, POSE_PRECISION)
        shapes = body.visual.shapes
        if len(shapes) == 1 and isinstance(shapes[0], Box):
            entry["box"] = rounded_scale(shapes[0].scale, POSE_PRECISION)
            return entry
        entry["mesh"] = self._write_object_mesh(body, key, shapes)
        # an OBJ brings its own materials and textures; without naming them the viewer
        # can only paint the mesh in the scene's fallback colour
        material_library = companion_material_library(
            Path(self.output_directory), entry["mesh"]
        )
        if material_library is not None:
            entry["mtl"] = material_library
        return entry

    def _write_object_mesh(self, body: Body, key: str, shapes: Sequence[Any]) -> str:
        """
        Write a body's geometry into the bundle and answer the path the scene records.

        :param body: The body whose geometry is written.
        :param key: The key the body's recorded poses are filed under.
        :param shapes: The body's visual shapes.
        """
        objects_directory = os.path.join(self.output_directory, "meshes", "objects")
        os.makedirs(objects_directory, exist_ok=True)
        if len(shapes) == 1 and isinstance(shapes[0], Mesh):
            source = shapes[0].filename
            if source and os.path.isfile(source):
                destination = os.path.join(
                    objects_directory, key + os.path.splitext(source)[1]
                )
                assets = BundledAssets(bundle_root=self.output_directory)
                if assets.copy(source, destination):
                    assets.copy_side_assets(source, destination)
                return "meshes/objects/" + os.path.basename(destination)
        destination = os.path.join(objects_directory, key + MeshFormat.OBJ.value)
        body.combined_mesh.export(destination)
        return "meshes/objects/" + os.path.basename(destination)

    def _kept_indices(self) -> List[int]:
        """
        The recorded frames the bundle keeps: every step-th one, and always the last.
        """
        frame_count = len(self.recorder.frames)
        kept = list(range(0, frame_count, self.step))
        if kept and kept[-1] != frame_count - 1:
            kept.append(frame_count - 1)
        return kept

    def task(self, steps: List[str]) -> Optional[str]:
        """
        What the recording says it is doing: the task it was named with, or the steps its
        plan performed, or None for a run that performed none.

        :param steps: The plan steps of this recording, in the order they ran.
        """
        if self.recorder.task:
            return self.recorder.task
        named: List[str] = []
        for step in steps:
            if step not in named:
                named.append(step)
        return ", ".join(named) or None

    def write_statecharts(self, scene: Dict[str, Any]) -> None:
        """
        Write what the run's statecharts looked like beside its trajectory, and name the
        file in the scene, or leave both alone for a run that executed none.

        The same file a live recording writes, so the viewer replays either the same way.

        :param scene: The scene being assembled, named into when there is something to
            name.
        """
        statecharts = RecordedStatecharts.of_snapshots(self.chart_frames())
        if statecharts.is_empty():
            return
        scene["statecharts"] = STATECHART_FILE
        write_json_atomically(
            Path(self.output_directory) / STATECHART_FILE, statecharts.to_payload()
        )

    def chart_frames(self) -> List[Optional[ChartSnapshot]]:
        """
        The statechart snapshot of each frame the bundle keeps.
        """
        snapshots = self.recorder.chart_frames
        return [
            snapshots[index] for index in self._kept_indices() if index < len(snapshots)
        ]

    def frame_times(self) -> List[float]:
        """
        When each frame the bundle keeps was taken, in seconds since the epoch, or
        nothing for a recording that stamped none.
        """
        stamps = self.recorder.frame_times
        return [stamps[index] for index in self._kept_indices() if index < len(stamps)]

    def detected_events(self) -> List[Dict[str, Any]]:
        """
        What the run's detectors saw, as the bundle carries it, or nothing for a run
        recorded without them.
        """
        return [record.to_payload() for record in self.recorder.detections]

    def build(self) -> Dict[str, Any]:
        """
        Downsample the recording to every step-th frame (always keeping the last) and
        assemble scene.json + trajectory.json from it.
        """
        kept_indices = self._kept_indices()
        downsampled_index = {
            raw_index: kept for kept, raw_index in enumerate(kept_indices)
        }

        frames = [self.recorder.frames[index] for index in kept_indices]
        base = [self.recorder.base_frames[index] for index in kept_indices]
        object_poses = [self.recorder.object_frames[index] for index in kept_indices]

        raw_frames_per_second = (
            1.0 / self.recorder.control_timestep
            if self.recorder.control_timestep
            else self.FALLBACK_FRAMES_PER_SECOND
        )
        frames_per_second = max(
            self.MINIMUM_FRAMES_PER_SECOND, round(raw_frames_per_second / self.step)
        )

        # %% robot description
        robot = self.recorder.robot
        root_name = str(robot.root.name)
        prefix = root_name.split("/", 1)[0] if "/" in root_name else ""
        base_body = root_name.split("/", 1)[1] if "/" in root_name else root_name
        part_annotations = RobotPartAnnotation.of_robot(robot)
        parts = {annotation.name: annotation.links for annotation in part_annotations}

        # %% segments: data-derived windows, labelled from the parsed actions
        segments = []
        for raw_segment in RecordingAnalysis(self.recorder).derive_segments():
            segment = dict(raw_segment)
            segment["start"] = self._nearest_kept_frame(
                downsampled_index, raw_segment["start"]
            )
            segment["end"] = self._nearest_kept_frame(
                downsampled_index, raw_segment["end"]
            )
            if "attach" in segment:
                segment["attach"] = self._nearest_kept_frame(
                    downsampled_index, raw_segment["attach"]
                )
                segment["detach"] = self._nearest_kept_frame(
                    downsampled_index, raw_segment["detach"]
                )
            segments.append(segment)
        # a scene with two transports of the same object would otherwise name both steps
        # identically, and the viewer keys its playback captions on the step name
        step_counts: Dict[str, int] = {}
        for segment in segments:
            step_counts[segment["step"]] = step_counts.get(segment["step"], 0) + 1
            if step_counts[segment["step"]] > 1:
                segment["step"] = "%s_%d" % (
                    segment["step"],
                    step_counts[segment["step"]],
                )

        # %% objects
        objects = []
        palette = ObjectPalette()
        for index, source in enumerate(self.recorder.mesh_sources):
            mesh = os.path.basename(source)
            if mesh not in self.recorder.object_frames[0]:
                continue
            destination = os.path.join(self.output_directory, "meshes", "objects", mesh)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(source, destination)
            entry = {
                "id": os.path.splitext(mesh)[0],
                "key": mesh,
                "mesh": "meshes/objects/" + mesh,
                "spawn": self.recorder.object_frames[0][mesh],
                "color": palette.color_for(index),
            }
            # recorded from the world, so the knowledge base does not have to guess it;
            # omitted when the object's shapes report no measurable size
            body = (self.recorder._bodies or {}).get(mesh)
            extent = measure_body(body) if body is not None else None
            if extent is not None:
                entry["height"] = round(extent.z, POSE_PRECISION)
            material_library = companion_material_library(
                Path(self.output_directory), entry["mesh"]
            )
            if material_library is not None:
                entry["mtl"] = material_library
            objects.append(entry)
        for spawned in self.recorder.spawned_boxes:
            if spawned.name not in self.recorder.object_frames[0]:
                continue
            objects.append(
                {
                    "id": spawned.name,
                    "key": spawned.name,
                    "box": spawned.scale,
                    "spawn": self.recorder.object_frames[0][spawned.name],
                    "color": spawned.color,
                    "height": spawned.scale[2],
                }
            )
        # the loose objects of a world built in code: no source file named them, so their
        # geometry is written out of the world itself
        emitted = {entry["key"] for entry in objects}
        for key, body in self.recorder.recorded_objects().items():
            if key in emitted or key not in self.recorder.object_frames[0]:
                continue
            objects.append(self._object_of_body(body, key, len(objects), palette))

        # %% place target + drag bounds
        places = [segment["place"] for segment in segments if segment.get("place")]
        place_target = None
        if places:
            center_x = sum(place[0] for place in places) / len(places)
            center_y = sum(place[1] for place in places) / len(places)
            lowest_z = min(place[2] for place in places)
            place_target = {
                "position": [round(center_x, 3), round(center_y, 3)],
                "z": round(lowest_z - self.PLACE_TARGET_DROP, 3),
                "bounds": {
                    "minX": round(center_x - self.PLACE_BOUNDS_MARGIN, 2),
                    "maxX": round(center_x + self.PLACE_BOUNDS_MARGIN, 2),
                    "minY": round(center_y - self.PLACE_BOUNDS_MARGIN, 2),
                    "maxY": round(center_y + self.PLACE_BOUNDS_FRONT_MARGIN, 2),
                },
            }
        drag_bounds = None
        if objects:
            spawn_x = [entry["spawn"][0] for entry in objects]
            spawn_y = [entry["spawn"][1] for entry in objects]
            drag_bounds = {
                "minX": round(min(spawn_x) - self.DRAG_BOUNDS_MARGIN_X, 2),
                "maxX": round(max(spawn_x) + self.DRAG_BOUNDS_MARGIN_X, 2),
                "minY": round(min(spawn_y) - self.DRAG_BOUNDS_MARGIN_Y, 2),
                "maxY": round(max(spawn_y) + self.DRAG_BOUNDS_MARGIN_Y, 2),
            }

        # %% bundle every model source the demo loaded
        world_body_names = [str(body.name) for body in self.recorder.world.bodies]
        models = []
        missing: List[str] = []
        bundled_sources = (
            [(source, BundleReport.of_source) for source in self.recorder.urdf_sources]
            + [
                (source, BundledWorld.of_gazebo_source)
                for source in self.recorder.gazebo_sources
            ]
            + [
                (source, BundledWorld.of_mjcf_source)
                for source in self.recorder.mjcf_sources
            ]
        )
        bundled_models = [
            self._bundle_model(source, bundler, world_body_names, base_body)
            for source, bundler in bundled_sources
        ]
        environment = self._bundle_unclaimed_bodies(bundled_models, objects)
        if environment is not None:
            bundled_models.append(environment)
        for bundled in bundled_models:
            models.append(bundled.to_payload())
            missing += bundled.report.missing

        scene = {
            "name": self.scene_name,
            "framesPerSecond": frames_per_second,
            "trajectory": "trajectory.json",
            "models": models,
            "robot": {
                "name": type(robot).__name__.lower(),
                "prefix": prefix,
                "baseBody": base_body,
                "parts": parts,
                "partAnnotations": [
                    annotation.to_payload() for annotation in part_annotations
                ],
            },
            "objects": objects,
            "segments": segments,
            "actions": self.recorder.actions,
            "planTrees": self.recorder.serialize_plans(),
            "placeTarget": place_target,
            "dragBounds": drag_bounds,
            "missingAssets": sorted(set(missing)),
            SceneField.TASK.value: self.task([entry["step"] for entry in segments]),
            SceneField.DETECTED_EVENTS.value: self.detected_events(),
            **SpatialAnnotations.of_world(
                self.recorder.world, list(self.recorder.recorded_objects().values())
            ).scene_fields(),
        }
        self.write_statecharts(scene)
        write_json_atomically(
            Path(self.output_directory) / "scene.json", scene, indent=1
        )
        write_json_atomically(
            Path(self.output_directory) / "trajectory.json",
            {
                "framesPerSecond": frames_per_second,
                "frames": frames,
                TrajectoryField.FRAME_TIMES.value: self.frame_times(),
                "base": base,
                "objects": object_poses,
            },
        )
        return scene


@dataclass
class ArgumentSplit:
    """
    Command-line arguments for ``cramera-onboard``, split at a ``--`` separator.
    """

    own: List[str]
    """Arguments that configure ``cramera-onboard`` itself."""

    passthrough: List[str]
    """Arguments left untouched for the demo file's own argument parsing."""


def split_passthrough_arguments(arguments: List[str]) -> ArgumentSplit:
    """
    Splits CLI arguments at a ``--`` separator.

    Everything before ``--`` is parsed by ``cramera-onboard``'s own
    :class:`argparse.ArgumentParser`. Everything after it is left untouched so a
    demo file that parses its own ``sys.argv`` (for example with its own
    ``argparse.ArgumentParser``) sees exactly the arguments meant for it.
    """
    if "--" not in arguments:
        return ArgumentSplit(own=list(arguments), passthrough=[])
    index = arguments.index("--")
    return ArgumentSplit(own=arguments[:index], passthrough=arguments[index + 1 :])


# %% the cramera-onboard entry point
def onboarding_parser() -> argparse.ArgumentParser:
    """
    What ``cramera-onboard`` accepts on its command line.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("demo", help="path to the coraplex demo .py file")
    parser.add_argument("--name", required=True, help="scene name (output folder)")
    parser.add_argument(
        "--out",
        default=str(paths.scenes_directory()),
        help="scenes directory (default: CRAMERA_SCENES or ~/.cramera/scenes)",
    )
    parser.add_argument(
        "--step", type=int, default=0, help="downsample step (0 = auto)"
    )
    parser.add_argument(
        "--task",
        help="what the recorded run does, e.g. 'make breakfast' (default: the steps its plan performed)",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="tick segmind's detectors along the run and record what they saw",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=Recorder.DEFAULT_DETECTION_INTERVAL,
        metavar="TICKS",
        help="run ticks between two detector ticks (default: %(default)s)",
    )
    return parser


def parse_onboarding_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    One command line, read.

    :param arguments: The command line, without the program name.
    """
    return onboarding_parser().parse_args(arguments)


def recorder_for(arguments: argparse.Namespace) -> Recorder:
    """
    The recorder one command line asks for.

    Every switch the command line offers has to reach the recorder, which is what this
    one place is for.

    :param arguments: The command line, as read.
    """
    return Recorder(
        detect_events=arguments.detect,
        detection_interval=arguments.detect_every,
        task=arguments.task,
    )


def main() -> None:
    """
    ``cramera-onboard`` -- record one demo run into a scene bundle.
    """
    # force: the demo's own imports configure the root logger before we get here
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    argument_split = split_passthrough_arguments(sys.argv[1:])
    args = parse_onboarding_arguments(argument_split.own)

    try:
        import coraplex  # noqa: F401
    except ModuleNotFoundError:
        sys.exit(
            "The CRAM stack is not importable — run under the action-cram venv:\n"
            "  the workspace venv (uv sync), then: cramera-onboard ..."
        )

    recorder = recorder_for(args)
    recorder.install_asset_hooks()
    recorder.install_tick_hook()
    recorder.install_segment_hook()

    demo = os.path.abspath(args.demo)
    log("running demo:", demo)
    sys.path.insert(0, os.path.dirname(demo))
    # make repo-level helper packages (e.g. test.conftest) importable — the
    # demos rely on pytest's rootdir behaviour for that
    candidate = os.path.dirname(demo)
    while candidate != os.path.dirname(candidate):
        if os.path.isdir(os.path.join(candidate, "coraplex")) and os.path.isdir(
            os.path.join(candidate, "test")
        ):
            sys.path.insert(0, candidate)
            log("repo root on sys.path:", candidate)
            break
        candidate = os.path.dirname(candidate)
    sys.argv = [demo, *argument_split.passthrough]
    runpy.run_path(demo, run_name="__main__")
    log(
        "demo finished: %d raw frames, %d actions"
        % (len(recorder.frames), len(recorder.actions))
    )

    if not recorder.frames:
        sys.exit("No frames captured — did the demo perform a plan?")
    if recorder.robot is None:
        sys.exit("No AbstractRobot semantic annotation found in the world.")

    step = args.step or max(1, len(recorder.frames) // TARGET_BUNDLE_FRAMES)
    output_directory = os.path.join(args.out, args.name)
    os.makedirs(output_directory, exist_ok=True)
    # bundling re-parses the recorded Gazebo/MJCF sources, which the still-installed
    # hooks would record as further sources
    recorder.uninstall_asset_hooks()
    recorder.resolve_action_targets()
    scene = SceneBuilder(recorder, args.name, output_directory, step).build()
    write_scene_index(Path(args.out) / "index.json", args.name)

    log("scene '%s' written to %s" % (args.name, output_directory))
    log(
        "  models:  %s"
        % ", ".join(
            "%s%s" % (model["name"], " (robot)" if model["robot"] else "")
            for model in scene["models"]
        )
    )
    log("  objects: %s" % ", ".join(entry["id"] for entry in scene["objects"]))
    log("  segments: %s" % " → ".join(entry["step"] for entry in scene["segments"]))
    if scene["missingAssets"]:
        log("  warning — %d missing assets:" % len(scene["missingAssets"]))
        for asset in scene["missingAssets"][:MISSING_ASSETS_LOGGED]:
            log("   ", asset)
    sys.stdout.flush()
    os._exit(0)  # don't hang on non-daemon ROS/viz threads the demo started


if __name__ == "__main__":
    main()
