"""
The live-viz bridge state: what a running demo publishes to the viewer.

This module is free of HTTP — it holds the :class:`Bridge` singleton whose snapshot
methods run on the *simulation* thread (driven by the world's own callbacks and the
plan callbacks of :mod:`cramera.live.visualization`) and whose ``get_*`` accessors
hand finished, plain-dict snapshots to the HTTP layer.

Node status is where the plan and the statechart differ: coraplex only performs the plan
root (``Plan.perform`` → ``root.perform``); ``ActionNode.notify`` expands its children
but never performs them, so every inner ``PlanNode`` keeps status ``CREATED`` for the
whole run. The real per-step progress arrives through the plan callbacks, which report
each motion node's start and end; those statuses are propagated up the plan tree and
flagged ``derived``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import urllib.parse
import math
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from http.server import ThreadingHTTPServer
from pathlib import Path

from typing_extensions import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Tuple,
    TYPE_CHECKING,
)

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Quaternion,
    RotationMatrix,
)
from giskardpy.motion_statechart.data_types import LifeCycleValues
from cramera.logging_setup import get_logger
from cramera.body_geometry import POSE_PRECISION, rounded_pose
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
)

from cramera.knowledge.enums import PlanNodeGroup
from cramera.live.chart_observer import ChartObserver
from cramera.live.chart_structure import (
    ChartEdgeEntry,
    ChartNodeEntry,
    ChartSnapshot,
    ObservationName,
)
from cramera.knowledge.presets import Preset
from cramera.world_objects import WorldObjects
from cramera.knowledge.query_runner import EqlQueryRunner, RenderResult
from cramera.knowledge.query_vocabulary import QueryVocabulary
from cramera.knowledge.queryable_knowledge import (
    QueryableKnowledge,
    QueryScope,
    UnknownQueryScope,
)
from cramera.knowledge.question_matching import QuestionMatcher, QuestionMatchResult
from cramera.knowledge.workspace_classes import WorkspaceClassIndex
from cramera.live.query import LiveQuerySource, NoQuerySourceRegistered
from cramera.live.world_query import WorldQuerySource
from cramera.live.markers import MarkerEntry, MarkerStore
from cramera.live.shape_catalog import ShapeEntry, served_mesh_file, shape_entry
from cramera.live.transforms import TransformGraph, TransformSnapshot
from cramera.live.robot_models import RobotModels, RobotSelectionBusy
from cramera.robot_fields import RobotField
from cramera.palette import ObjectPalette
from cramera.robot_parts import RobotPartAnnotation

if TYPE_CHECKING:
    from coraplex.datastructures.dataclasses import Context
    from coraplex.plans.plan import Plan
    from coraplex.plans.plan_node import MotionNode, PlanNode
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body, Connection

    from cramera.live.recording import Recording


@runtime_checkable
class PlanExecutionContext(Protocol):
    """A plan exposing the context that decides whether conditions execute."""

    context: Optional[Context]
    """Execution settings of the observed plan."""


logger = get_logger(__name__)


class TaskStatusName(StrEnum):
    """
    The status vocabulary the viewer styles plan and statechart nodes with.

    Mirrors coraplex's ``TaskStatus`` names. A :class:`StrEnum`, because the values
    travel to the frontend as JSON and are compared against the names coraplex itself
    reports.
    """

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    PAUSE = "PAUSE"

    @classmethod
    def of_native_name(cls, name: str) -> TaskStatusName:
        """Translate a native lifecycle name into the recorded plan vocabulary.

        :param name: A native lifecycle or recorded status name.
        :return: Its viewer status.
        """
        if name == LifeCycleValues.NOT_STARTED.name:
            return cls.CREATED
        if name == LifeCycleValues.PAUSED.name:
            return cls.PAUSE
        return cls(name)

    @classmethod
    def _precedence(cls) -> Tuple[TaskStatusName, ...]:
        """
        The statuses from lowest to highest precedence.
        """
        return (
            cls.CREATED,
            cls.SUCCEEDED,
            cls.PAUSE,
            cls.RUNNING,
            cls.INTERRUPTED,
            cls.FAILED,
        )

    @property
    def rank(self) -> int:
        """
        Precedence when a plan node's status is aggregated from its children: the higher
        rank wins.
        """
        return self._precedence().index(self)

    @classmethod
    def rank_of(cls, status: str) -> int:
        """
        The rank of a status name, or the lowest rank for one this enum does not know.

        :param status: A status name as reported by coraplex or the statechart.
        """
        if status not in cls._value2member_map_:
            return 0
        return cls(status).rank


ROBOT_BASE_KEY = "__base__"
"""
Key under which the robot's root body is published, instead of as a loose object.
"""

IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)
"""
``[qx, qy, qz, qw]`` of an unrotated pose, used where no orientation is known.
"""


@runtime_checkable
class DescribesAnAction(Protocol):
    """
    A plan node carrying the designator that describes what it does.

    Structural, because only some coraplex node types have a designator at all.
    """

    designator: Any


@runtime_checkable
class NamesAWorldEntity(Protocol):
    """
    Anything carrying a world-entity name, such as a body a designator refers to.
    """

    name: Any


ALLOWED_CONSTRAINT_GOALS = (
    "VectorsAligned",
    "PointingAt",
    "JointPositionReached",
    "HeightMonitor",
    "DistanceMonitor",
)
"""giskardpy goal/monitor class names the plan view may request (verified in
``giskardpy/motion_statechart/monitors``)."""


class MalformedConstraintRequest(Exception):
    """
    Raised when the plan view's constraint payload cannot be read.
    """


@dataclass(frozen=True)
class AttachConstraintRequest:
    """
    A natural-language constraint the plan view attached to a plan node, already
    compiled (viewer-side) to a giskardpy motion-statechart goal/monitor. The bridge
    validates it here and, once live injection lands, adds the corresponding node to
    the running :class:`MotionStatechart` at :attr:`target_node_id`.
    """

    target_node_id: str
    """Stable id of the plan node the constraint is attached to (viewer-assigned)."""

    text: str
    """The original natural-language constraint, kept for provenance and logging."""

    goal_type: str
    """giskardpy goal/monitor class name, e.g. ``"VectorsAligned"`` or ``"PointingAt"``."""

    params: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for the giskard goal, as compiled by the plan view."""

    node_kind: Optional[str] = None
    """Plan-node kind (e.g. ``"ActionNode"``), for logging and targeting."""

    node_label: Optional[str] = None
    """Plan-node label, for logging."""

    apply: str = "next_activation"
    """When to apply: ``"next_activation"`` (default) or ``"immediate"``."""

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "AttachConstraintRequest":
        """
        Build a request from a decoded ``POST /constraint`` body.

        :param payload: The decoded JSON body of a ``POST /constraint`` request.
        :raises MalformedConstraintRequest: If a required field is missing or unusable.
            Validating here keeps bad input off the simulation tick.
        """
        node = payload.get("target_plan_node") or {}
        if not isinstance(node, dict):
            raise MalformedConstraintRequest("'target_plan_node' must be an object")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise MalformedConstraintRequest(
                "'target_plan_node.id' must be a non-empty string"
            )
        giskard = payload.get("giskard_node") or {}
        if not isinstance(giskard, dict):
            raise MalformedConstraintRequest("'giskard_node' must be an object")
        goal_type = giskard.get("type")
        if goal_type not in ALLOWED_CONSTRAINT_GOALS:
            raise MalformedConstraintRequest(
                "'giskard_node.type' must be one of %s" % (ALLOWED_CONSTRAINT_GOALS,)
            )
        params = giskard.get("params") or {}
        if not isinstance(params, dict):
            raise MalformedConstraintRequest("'giskard_node.params' must be an object")
        text = payload.get("text")
        if not isinstance(text, str):
            text = ""
        apply_when = payload.get("apply") or "next_activation"
        if apply_when not in ("next_activation", "immediate"):
            raise MalformedConstraintRequest(
                "'apply' must be 'next_activation' or 'immediate'"
            )
        return cls(
            target_node_id=node_id,
            text=text,
            goal_type=goal_type,
            params=dict(params),
            node_kind=node.get("kind"),
            node_label=node.get("label"),
            apply=apply_when,
        )


class MalformedMoveRequest(Exception):
    """
    Raised when the viewer's move payload cannot be read as a pose.
    """


@dataclass(frozen=True)
class MoveRequest:
    """
    A drag the viewer performed on one object, to be written into the world.
    """

    object_key: str
    """
    Mesh key of the dragged object, as published in the geometry catalog.
    """

    position: List[float]
    """
    Target position ``[x, y, z]`` in world coordinates.
    """

    quaternion: Optional[List[float]] = None
    """
    Target orientation ``[qx, qy, qz, qw]``, or None to keep the current one.
    """

    is_final: bool = False
    """
    Whether this is the drag's last update, as opposed to an intermediate one.
    """

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> MoveRequest:
        """
        Build a request from a decoded ``POST /move`` body.

        :param payload: The decoded JSON body of a ``POST /move`` request.
        :raises MalformedMoveRequest: If the object key or the position is unusable.
            Validating here keeps bad input from raising inside the simulation tick,
            where the only recovery is to drop the whole snapshot.
        """
        object_key = payload.get("object")
        if not isinstance(object_key, str) or not object_key:
            raise MalformedMoveRequest("'object' must be a non-empty string")
        position = cls._coordinates(payload.get("position"), "position", 3)
        quaternion = (
            cls._coordinates(payload.get("quaternion"), "quaternion", 4)
            if payload.get("quaternion")
            else None
        )
        return cls(
            object_key=object_key,
            position=position,
            quaternion=quaternion,
            is_final=bool(payload.get("final")),
        )

    @staticmethod
    def _coordinates(value: Any, name: str, length: int) -> List[float]:
        """
        Read a fixed-length list of finite coordinates out of a payload field.

        :param value: The raw payload field to validate and convert.
        :param name: The field's name, used in the error message if it is invalid.
        :param length: The number of coordinates the field must contain.
        """
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise MalformedMoveRequest(
                "'%s' must be a list of %d numbers" % (name, length)
            )
        coordinates = []
        for entry in value:
            if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                raise MalformedMoveRequest("'%s' must contain only numbers" % name)
            if not math.isfinite(entry):
                raise MalformedMoveRequest("'%s' must be finite" % name)
            coordinates.append(float(entry))
        return coordinates


class MalformedJointMoveRequest(Exception):
    """
    Raised when the viewer's joint payload cannot be read as a joint position.
    """


@dataclass(frozen=True)
class JointMoveRequest:
    """
    A joint position the viewer set by hand, to be written into the world.
    """

    connection_name: str
    """
    Name of the connection, as the world state publishes it.
    """

    position: float
    """
    Target position in the connection's own unit: radians for a hinge, metres for a slide.
    """

    is_final: bool = False
    """
    Whether this is the slider's last update, as opposed to an intermediate one.
    """

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> JointMoveRequest:
        """
        Build a request from a decoded ``POST /joint`` body.

        :param payload: The decoded JSON body of a ``POST /joint`` request.
        :raises MalformedJointMoveRequest: If the joint name or the position is unusable.
        """
        connection_name = payload.get("joint")
        if not isinstance(connection_name, str) or not connection_name:
            raise MalformedJointMoveRequest("'joint' must be a non-empty string")
        position = payload.get("position")
        if (
            isinstance(position, bool)
            or not isinstance(position, (int, float))
            or not math.isfinite(position)
        ):
            raise MalformedJointMoveRequest("'position' must be a finite number")
        return cls(
            connection_name=connection_name,
            position=float(position),
            is_final=bool(payload.get("final")),
        )


@dataclass
class MotionNodeProgress:
    """
    What the bridge knows about one plan node's execution.

    Holds the node itself so the identity key derived from it stays unique for as long
    as the entry lives.
    """

    node: PlanNode
    """
    The plan node this progress belongs to.
    """

    status: Optional[TaskStatusName] = None
    """
    The node's last observed execution status, else None.
    """


# %% viewer payload shapes
class ObjectKind(StrEnum):
    """
    How a loose object's geometry is served to the viewer.
    """

    MESH = "mesh"
    BOX = "box"
    SHAPES = "shapes"


@dataclass(frozen=True)
class ObjectCatalogEntry:
    """
    One loose object's geometry-catalog entry, as the viewer spawns it.
    """

    key: str
    """
    Mesh basename this object is published under.
    """

    id: str
    """
    Stem of :attr:`key`, used as the object's display id.
    """

    kind: ObjectKind
    """
    Whether the viewer renders a served mesh or a placeholder box.
    """

    color: str
    """
    Colour assigned to this object from the shared palette.
    """

    mesh: Optional[str] = None
    """
    URL the mesh is served from, set only when :attr:`kind` is ``MESH``.
    """

    format: Optional[str] = None
    """
    Mesh file extension, set only when :attr:`kind` is ``MESH``.
    """

    size: Optional[List[float]] = None
    """
    Box extent in metres, set only when :attr:`kind` is ``BOX``.
    """

    shapes: Optional[List[ShapeEntry]] = None
    """
    The body's shapes, set only when :attr:`kind` is ``SHAPES``.
    """


@dataclass
class PlanNodeEntry:
    """
    One plan node's serialized state, mutated in place as its status resolves.
    """

    id: str
    """
    Identity-based id of this node (``p`` + ``id(node)``).
    """

    parent: Optional[str]
    """
    Id of this node's parent entry, or None for the root.
    """

    kind: str
    """
    The plan node's own class name.
    """

    group: PlanNodeGroup
    """
    Colour group the viewer draws this node in, from :attr:`kind`.
    """

    label: str
    """
    Designator class name if this node describes an action, else :attr:`kind`.
    """

    status: str
    """
    This node's status: its own if it reports one, else a derived one.

    Not typed as :class:`TaskStatusName`, because it mirrors whatever coraplex's own
    ``TaskStatus`` reports, which this bridge does not validate.
    """

    derived: bool
    """
    Whether :attr:`status` was derived (from the statechart or children) rather than
    the node's own reported status.
    """

    arm: Optional[str] = None
    """
    Arm the node's designator names, if any.
    """

    target: Optional[str] = None
    """
    Published object the node's designator refers to, if any.
    """


class PlanTreeField(StrEnum):
    """Fields joining the nodes of a recorded plan hierarchy."""

    CHILDREN = "children"
    """Nested plan steps in their execution order."""


@dataclass(frozen=True)
class PlanSnapshot:
    """
    The plan tree in the shape the viewer walks.
    """

    signature: str = ""
    """
    Node-id signature of the tree's shape, stable across status-only changes.
    """

    nodes: List[PlanNodeEntry] = field(default_factory=list)
    """
    Every node in the tree, flattened with parent references.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The snapshot plus the legend its groups are drawn with, so the viewer does not
        keep its own copy of the plan-node colour table.
        """
        payload = asdict(self)
        payload["legend"] = [
            {"group": group.value, "label": group.label}
            for group in PlanNodeGroup.legend()
        ]
        return payload

    def recorded_trees(self) -> List[Dict[str, Any]]:
        """Reconstruct the recorded hierarchy from the published parent references.

        :return: Plan roots containing their nested children and action metadata.
        """
        entries = {
            node.id: {**asdict(node), PlanTreeField.CHILDREN: []} for node in self.nodes
        }
        roots = []
        for node in self.nodes:
            entry = entries[node.id]
            if node.parent in entries:
                entries[node.parent][PlanTreeField.CHILDREN].append(entry)
            else:
                roots.append(entry)
        return roots


@dataclass(frozen=True)
class WorldStateSnapshot:
    """
    The world's joints, base pose and object poses at one simulation tick.
    """

    sequence_number: int = 0
    """
    Monotonic snapshot counter so the viewer can skip unchanged states.
    """

    frames: Dict[str, float] = field(default_factory=dict)
    """
    Movable connection position by prefixed name.
    """

    base: Optional[List[float]] = None
    """
    Robot base pose as ``[x, y, z, qx, qy, qz, qw]``, or None without a robot.
    """

    objects: Dict[str, List[float]] = field(default_factory=dict)
    """
    Loose-object pose by mesh key, in the same 7-element form as :attr:`base`.
    """

    ORIENTATION_START: ClassVar[int] = 3
    """
    Index the quaternion begins at in a ``[x, y, z, qx, qy, qz, qw]`` pose.
    """

    markers_version: int = 0
    """
    Version of the debug-marker overlay; the viewer refetches ``/markers`` on change.
    """

    def orientation_of(self, object_key: str) -> Optional[List[float]]:
        """
        The orientation one object stands at, or None if this snapshot has no pose for it.

        :param object_key: Mesh key of the object whose orientation is read.
        """
        pose = self.objects.get(object_key)
        if pose is None:
            return None
        return list(pose[self.ORIENTATION_START :])

    model_bases: Dict[str, List[float]] = field(default_factory=dict)
    """
    Every bundled model's root pose by world-instance prefix, in the same 7-element
    form as :attr:`base`, so a second robot or a moved environment model animates.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The snapshot in the camel-cased JSON shape the viewer reads.
        """
        payload = asdict(self)
        payload["sequenceNumber"] = payload.pop("sequence_number")
        payload["modelBases"] = payload.pop("model_bases")
        payload["markersVersion"] = payload.pop("markers_version")
        return payload


@dataclass(frozen=True)
class BridgeStatus:
    """
    What the viewer polls to decide whether a live demo is reachable.
    """

    running: bool
    robot: Optional[str]
    objects: List[str]
    movable: bool
    plan: bool
    chart: bool
    query: bool
    """
    Whether a running demo offered its state to be questioned (see
    :meth:`Bridge.register_query_source`).
    """

    sequence_number: int
    model_version: int = 0
    """
    How many model sources the demo has parsed so far; the viewer reloads the live
    scene when this grows, so a model loaded mid-run appears.
    """

    bundle_signature: str = ""
    """
    Digest of what a live bundle built right now would contain (see
    :meth:`ModelBundleContext.signature`); the viewer reloads the live scene when it
    no longer matches the one its loaded bundle carries.
    """

    robot_parts: List[RobotPartAnnotation] = field(default_factory=list)
    """
    The arms and end effectors of the live robot, as sem_dt annotates them.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The status in the JSON shape the viewer polls, with the robot parts in the same
        ``partAnnotations`` shape a recorded scene bundle carries.
        """
        payload = asdict(self)
        payload["sequenceNumber"] = payload.pop("sequence_number")
        payload["modelVersion"] = payload.pop("model_version")
        payload["bundleSignature"] = payload.pop("bundle_signature")
        payload.pop("robot_parts")
        payload["partAnnotations"] = [
            annotation.to_payload() for annotation in self.robot_parts
        ]
        return payload


@dataclass
class Bridge:
    """
    Shared state between the running demo and the viewer.

    All world reads and writes happen on the simulation thread (the tick hook); the HTTP
    handlers only ever read the finished snapshot dicts under :attr:`_lock`.
    """

    REBIND_INTERVAL_SECONDS: ClassVar[float] = 3.0
    """
    How long a world binding stays fresh before bodies are re-discovered.
    """

    DEFAULT_OBJECT_SIZE: ClassVar[Tuple[float, float, float]] = (0.06, 0.06, 0.12)
    """
    Fallback size for an object whose shapes carry no scale, in metres.
    """

    world: Optional[World] = None
    """
    The executing world, captured by the tick hook on its first call.
    """

    robot: Optional[AbstractRobot] = None
    """
    The robot annotation of :attr:`world`, re-discovered on every bind.
    """

    _robot_catalog: list[dict] = field(default_factory=list)
    """Finished robot-instance metadata read by HTTP handlers."""

    sequence_number: int = 0
    """
    Monotonic snapshot counter so the viewer can skip unchanged states.
    """

    state: WorldStateSnapshot = field(default_factory=WorldStateSnapshot)
    """
    The newest world snapshot in the trajectory-frame format.
    """

    object_metadata: List[ObjectCatalogEntry] = field(default_factory=list)
    """
    Geometry catalog for the viewer: one entry per loose object.
    """

    plan_state: PlanSnapshot = field(default_factory=PlanSnapshot)
    """
    The newest plan-tree snapshot (see :meth:`snapshot_plan`).
    """

    chart_state: ChartSnapshot = field(default_factory=ChartSnapshot)
    """
    The newest motion-statechart snapshot (see :meth:`observe_chart`).
    """

    _connections: List[ActiveConnection1DOF] = field(default_factory=list)
    """
    Actuated world connections whose positions are published as frames.
    """

    transform_state: TransformSnapshot = field(default_factory=TransformSnapshot)
    """
    The newest transform-graph snapshot (see :mod:`cramera.live.transforms`).
    """

    query_source: Optional[LiveQuerySource] = None
    """
    What the running demo offers to be queried about, once it registers itself.
    """

    _query_lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Serializes queries: EQL evaluation is not written to run twice at once, and the
    bridge answers several viewers from its own thread pool.

    ..note:: This does not keep a query apart from the demo thread, which evaluates EQL
        of its own; what they share is the ``SymbolGraph`` singleton, which serializes
        itself.
    """

    _transforms: TransformGraph = field(default_factory=TransformGraph)
    """
    Tracks when each world connection last changed, across ticks.
    """

    _kinematic_connections: List[Connection] = field(default_factory=list)
    """
    Every world connection, of any kind, as the last bind discovered them.
    """

    _bodies: Dict[str, Body] = field(default_factory=dict)
    """
    Published bodies by mesh key; :data:`ROBOT_BASE_KEY` is the robot root.
    """

    _last_bind_time: float = 0.0
    """
    Timestamp of the last world discovery (see :attr:`REBIND_INTERVAL_SECONDS`).
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards every snapshot dict that the HTTP layer reads.
    """

    _moves: List[MoveRequest] = field(default_factory=list)
    """
    Object moves queued by the viewer, applied on the simulation thread.
    """

    _joint_moves: List[JointMoveRequest] = field(default_factory=list)
    """
    Joint positions set in the viewer, applied on the simulation thread.
    """

    _last_moves: Dict[str, List[float]] = field(default_factory=dict)
    """
    Latest drag target per object mesh key (``[x, y, z, qx, qy, qz, qw]``), recorded on
    every move whether or not the sim applied it, for the Plan Builder's pose capture.
    """

    _constraints: List["AttachConstraintRequest"] = field(default_factory=list)
    """Constraints attached from the plan view, drained on the simulation thread."""

    _constraints_lock: threading.Lock = field(default_factory=threading.Lock)
    """Guards :attr:`_constraints`."""

    pending_constraints: List["AttachConstraintRequest"] = field(default_factory=list)
    """Validated constraints, kept for inspection until live statechart injection lands."""

    _resolved_constraints: List[Any] = field(default_factory=list)
    """(constraint, built giskard node) pairs, resolved against the live world and ready
    to add to the next motion statechart before it compiles."""

    _moves_lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards :attr:`_moves` (written by HTTP threads).
    """

    _teleop: Optional[Any] = None
    """
    The live teleop driver (a :class:`cramera.live.teleop.TeleopController`), created on
    the first ``POST /teleop`` once a world is bound. None until then.
    """

    _mesh_serve: Dict[str, str] = field(default_factory=dict)
    """
    Object key → absolute mesh path served via the ``/mesh`` endpoint.
    """

    _plan: Optional[Plan] = None
    """
    The coraplex plan captured by the ``Plan.perform`` hook.
    """

    _chart_observer: ChartObserver = field(default_factory=ChartObserver)
    """
    Reads what the executing statechart looks like, remembering what it last saw.
    """

    _chart_title: str = ""
    """
    Name of the action whose motion group is executing.
    """

    _ever_running: set = field(default_factory=set)
    """ids of plan nodes that have been RUNNING at least once, so a node that has run
    and gone idle (and not failed) reads as SUCCEEDED — a monotonic done-progression
    the raw coraplex status does not keep (expanded nodes revert to CREATED)."""

    _motion_nodes: Dict[int, MotionNodeProgress] = field(default_factory=dict)
    """
    Execution progress per plan node, keyed by the node's :func:`id`.

    Identity, not equality: coraplex's ``DesignatorNode`` compares by field value, so
    two structurally identical steps of one plan would otherwise share a status. The
    :class:`MotionNodeProgress` entry pins the node itself, which keeps CPython from
    handing its ``id`` to a later object.

    Reset whenever a new plan starts performing, which bounds it to one plan's nodes.
    """

    _tick_count: int = 0
    _last_tick_time: float = 0.0

    _model_revision: int = 0
    """
    Counts world attachments and model changes, reported as the status's model version.
    """

    _marker_stores: Dict[str, MarkerStore] = field(default_factory=dict)
    """
    The ROS debug markers per subscribed topic (see :mod:`cramera.live.ros_markers`).
    """

    _marker_lock: threading.Lock = field(default_factory=threading.Lock)
    _marker_revision: int = 0

    marker_listener: Optional[Any] = None
    """
    The ROS subscription feeding the marker overlay, while one runs — the viewer's
    marker settings manage its topics through the bridge.
    """

    marker_state: Dict[str, Any] = field(
        default_factory=lambda: {"version": 0, "markers": []}
    )
    """
    The newest marker-overlay snapshot the HTTP layer serves.
    """

    _published_marker_revision: int = -1
    """
    The aggregate store revision :attr:`marker_state` was built from.
    """

    _marker_state_version: int = 0
    """
    Monotonic version of :attr:`marker_state`; the sum of store revisions can revisit
    an old value after a topic is dropped, this never does.
    """

    _bundle_signature: str = ""
    """
    Cached digest of the bundled scene content, recomputed on attach and model change.
    """

    live_server: Optional[ThreadingHTTPServer] = None
    """
    The bridge's HTTP server once it is listening, so a second start reuses it.
    """

    recording: Optional[Recording] = None
    """
    The current live run's capture buffer, started alongside :meth:`attach` (see
    :mod:`cramera.live.visualization`); None before anything has ever attached.
    """

    # %% what the visualization drives
    def attach(self, world: World) -> None:
        """
        Bind to the world a demo is executing and publish its geometry catalog.

        :param world: The world the demo is executing in.
        """
        if self.world is not world:
            self._bodies = {}
            self.robot = None
        self.world = world
        if self.query_source is None or isinstance(self.query_source, WorldQuerySource):
            self.query_source = WorldQuerySource(world, self.query_objects)
        self._model_revision += 1
        self.bind()
        self._refresh_bundle_signature()
        logger.info(
            "attached to world (robot=%s, %d joints)",
            type(self.robot).__name__ if self.robot else "?",
            len(self._connections),
        )

    def observe_motion_tick(self, chart: MotionStatechart) -> None:
        """
        Apply queued viewer controls on the motion executor's thread.

        Applies queued viewer moves first, because the executor tick runs on the only
        thread allowed to write to the world; the world snapshot itself follows from
        the state change the tick causes.

        :param chart: The motion statechart the executor is ticking.
        """
        self.apply_moves()
        self.apply_constraints(chart)
        self._last_tick_time = time.monotonic()
        self._tick_count += 1

    def observe_motion_started(self, node: MotionNode) -> None:
        """
        Record that a plan node's motion started running.

        :param node: The node whose motion started.
        """
        self._motion_nodes[id(node)] = MotionNodeProgress(
            node=node, status=TaskStatusName.RUNNING
        )
        action_node = node.parent_action_node
        if action_node is not None and action_node.designator is not None:
            self._chart_title = type(action_node.designator).__name__

    def observe_motion_ended(self, node: MotionNode) -> None:
        """
        Pin the final status of a finished motion node and republish the plan.

        :param node: The node whose motion ended.
        """
        self._motion_nodes[id(node)] = MotionNodeProgress(
            node=node, status=TaskStatusName.of_native_name(node.status.name)
        )
        self.snapshot_plan()

    def begin_plan(self, plan: Plan) -> None:
        """
        Record the plan that started performing and publish its tree.

        Drops the previous plan's per-node progress, so a long-running process does not
        accumulate entries for nodes that no longer exist.

        :param plan: The plan that started performing.
        """
        if isinstance(plan, PlanExecutionContext) and plan.context is not None:
            robot = plan.context.robot
            if robot is not None and robot in RobotModels(self.world).robots():
                self._select_robot(robot)
        self._plan = plan
        self._motion_nodes.clear()
        self._ever_running.clear()
        self.snapshot_plan()

    def observe_model_change(self) -> None:
        """
        Refresh the catalogs and the bundle signature after a world model change.
        """
        self._model_revision += 1
        self.bind()
        self._refresh_bundle_signature()

    def observe_ros_markers(self, topic: str, markers: List[Any]) -> None:
        """
        Apply one received ``MarkerArray`` (called on the ROS subscriber thread).

        Only the store is touched here; the publishable payload is rebuilt on the
        simulation thread, which may read the world for frame resolution.

        :param topic: The topic the array arrived on.
        :param markers: The array's markers.
        """
        with self._marker_lock:
            store = self._marker_stores.setdefault(topic, MarkerStore())
            if store.observe(markers):
                self._marker_revision += 1

    def _refresh_marker_state(self) -> None:
        """
        Rebuild the marker overlay payload if any store changed since the last build.

        Runs on the simulation thread: excluding the world-model markers and
        resolving marker frames both read the world. Markers whose namespace names a
        world entity are the robot/environment geometry the scene already renders,
        and stay out of the overlay.
        """
        with self._marker_lock:
            revision = self._marker_revision
            if revision == self._published_marker_revision:
                return
            marker_entries = {
                topic: tuple(store.entries.values())
                for topic, store in self._marker_stores.items()
            }
        world_entity_names = set()
        if self.world is not None:
            world_entity_names = {str(body.name) for body in self.world.bodies} | {
                str(region.name) for region in self.world.regions
            }
        markers = []
        for topic in sorted(marker_entries):
            for entry in marker_entries[topic]:
                if entry.ns in world_entity_names:
                    continue
                markers.append(self._marker_payload(topic, entry))
        self._published_marker_revision = revision
        self._marker_state_version += 1
        with self._lock:
            self.marker_state = {
                "version": self._marker_state_version,
                "markers": markers,
            }

    def _marker_payload(self, topic: str, entry: MarkerEntry) -> Dict[str, Any]:
        """
        One marker as the viewer renders it, with its pose resolved into the world.

        :param topic: The topic the marker arrived on.
        :param entry: The marker to publish.
        """
        return {
            "topic": topic,
            "ns": entry.ns,
            "id": entry.id,
            "kind": entry.kind,
            "pose": self._marker_world_pose(entry),
            "scale": entry.scale,
            "color": entry.color,
            "opacity": entry.opacity,
            "points": entry.points,
            "text": entry.text,
        }

    def _marker_world_pose(self, entry: MarkerEntry) -> List[float]:
        """
        A marker's pose in world coordinates, as ``[x, y, z, qx, qy, qz, qw]``.

        A frame naming a world body anchors the marker to that body's current pose;
        the world root (under any of its usual names) and unknown frames read as the
        world itself.

        :param entry: The marker whose pose is resolved.
        """
        local = entry.position + entry.quaternion
        world = self.world
        if world is None:
            return local
        frame_body = self._marker_frame_body(entry.frame)
        if frame_body is None:
            return local
        frame_T_marker = HomogeneousTransformationMatrix.from_xyz_quaternion(*local)
        world_T_marker = frame_body.global_pose.to_homogeneous_matrix() @ frame_T_marker
        return [
            round(value, POSE_PRECISION)
            for value in world_T_marker.to_position_quaternion_list()
        ]

    def _marker_frame_body(self, frame: str) -> Optional[Body]:
        """
        The world body a marker frame names, or None for the world root and frames
        the world does not know.

        :param frame: The marker's ``frame_id``.
        """
        root_name = str(self.world.root.name)
        if frame in ("", "map", "world", root_name, root_name.split("/")[-1]):
            return None
        for body in self.world.bodies:
            name = str(body.name)
            if frame == name or frame == name.split("/")[-1]:
                return body
        return None

    def get_markers(self) -> Dict[str, Any]:
        """
        The debug-marker overlay the viewer renders.
        """
        with self._lock:
            return self.marker_state

    def marker_topics_payload(self) -> Dict[str, Any]:
        """
        The marker settings the viewer offers: what is watched and what the ROS graph
        advertises.
        """
        if self.marker_listener is None:
            return {"ok": True, "ros": False, "subscribed": [], "available": []}
        subscribed = self.marker_listener.subscribed_topics()
        return {
            "ok": True,
            "ros": True,
            "subscribed": subscribed,
            "available": sorted(
                set(self.marker_listener.available_marker_topics()) | set(subscribed)
            ),
        }

    def set_marker_topic(self, topic: str, subscribed: bool) -> Dict[str, Any]:
        """
        Start or stop watching one marker topic, as the viewer's settings ask.

        Stopping also drops the topic's markers, the way removing an RViz display
        clears what it showed.

        :param topic: The topic to watch or drop.
        :param subscribed: Whether the topic should be watched.
        """
        if self.marker_listener is None:
            return {"ok": False, "error": "no ROS in the demo process"}
        if not topic.startswith("/"):
            return {"ok": False, "error": "a topic starts with '/'"}
        if subscribed:
            self.marker_listener.subscribe(topic)
        else:
            self.marker_listener.unsubscribe(topic)
            with self._marker_lock:
                store = self._marker_stores.pop(topic, None)
                if store is not None and store.entries:
                    self._marker_revision += 1
        return self.marker_topics_payload()

    def publish_bodies(self, bodies: Dict[str, Body]) -> None:
        """
        Replace the published bodies and rebuild the viewer's geometry catalog.

        :param bodies: The current published bodies, keyed by mesh key.
        """
        self._bodies = bodies
        self._build_object_metadata(bodies)

    # %% what the HTTP layer reads
    def object_catalog(self) -> List[Dict[str, Any]]:
        """
        The geometry catalog the viewer spawns live objects from.
        """
        with self._lock:
            return [asdict(entry) for entry in self.object_metadata]

    def object_keys(self) -> List[str]:
        """
        Keys of independently published objects, excluding the robot root.
        """
        with self._lock:
            return [key for key in self._bodies if key != ROBOT_BASE_KEY]

    def query_objects(self) -> List[Body]:
        """Published loose bodies used for object-specific semantic questions."""
        with self._lock:
            published = [
                body for key, body in self._bodies.items() if key != ROBOT_BASE_KEY
            ]
        if self.world is None:
            return published
        known_names = {str(body.name) for body in published}
        return published + [
            body
            for body in WorldObjects(self.world, self.robot).free_floating()
            if str(body.name) not in known_names
        ]

    def mesh_path(self, key: str) -> Optional[str]:
        """
        Absolute path of an object's mesh file, or None if it is not served.

        :param key: Mesh key of the object, as published in the geometry catalog.
        """
        with self._lock:
            return self._mesh_serve.get(key)

    def object_body(self, key: str) -> Optional[Body]:
        """
        The published body behind an object-catalog key, or None if it is not published.

        :param key: Mesh key of the object, as published in the geometry catalog.
        """
        with self._lock:
            return self._bodies.get(key)

    def bundle_signature(self) -> str:
        """
        A digest of the bundled scene's content: the identity, parentage and connection
        type of every body the live bundle serializes, plus the robot's identity.

        Deliberately excludes the overlay's tracked objects — a demo re-parenting a
        grasped object changes the world model but not the bundled scene, and must not
        make the viewer reload it. State changes never touch it either.
        """
        return self._bundle_signature

    def _refresh_bundle_signature(self) -> None:
        """
        Recompute the cached bundle signature from the current world model.
        """
        if self.world is None:
            self._bundle_signature = ""
            return
        robot_name = RobotModels.identifier(self.robot) if self.robot else None
        overlay_bodies = {
            body for key, body in self._bodies.items() if key != ROBOT_BASE_KEY
        }
        entries: List[str] = []
        try:
            for body in self.world.bodies:
                name = str(body.name)
                if body in overlay_bodies:
                    continue
                connection = body.parent_connection
                entries.append(
                    "%s<-%s:%s"
                    % (
                        name,
                        str(connection.parent.name) if connection else "",
                        type(connection).__name__ if connection else "root",
                    )
                )
        except Exception as error:
            # boundary guard: the world is mid-modification and iterating it is not
            # safe; keep the previous signature rather than flapping the viewer.
            logger.debug("signature refresh skipped: %s", error)
            return
        digest = hashlib.sha1("|".join(sorted(entries)).encode()).hexdigest()[:16]
        self._bundle_signature = "world-%s-robot-%s" % (digest, robot_name)

    def status(self) -> Dict[str, Any]:
        """
        What the viewer polls to decide whether a live demo is reachable.
        """
        bundle_signature = self.bundle_signature()
        with self._lock:
            return BridgeStatus(
                running=self.world is not None,
                robot=type(self.robot).__name__ if self.robot else None,
                objects=[key for key in self._bodies if key != ROBOT_BASE_KEY],
                movable=True,
                plan=bool(self.plan_state.nodes),
                chart=bool(self.chart_state.nodes),
                query=self.query_source is not None,
                sequence_number=self.sequence_number,
                model_version=self._model_revision,
                bundle_signature=bundle_signature,
                robot_parts=(
                    RobotPartAnnotation.of_robot(self.robot)
                    if self.robot is not None
                    else []
                ),
            ).to_payload()

    # %% viewer -> questions about the running demo
    def register_query_source(self, source: LiveQuerySource) -> None:
        """
        Offer the running demo's state to the viewer's queries.

        :param source: What the demo declares as queryable.
        """
        self.query_source = source
        logger.info("live queries answered by '%s'", source.title())

    def _registered_query_source(self) -> LiveQuerySource:
        """
        The registered query source.

        :raises NoQuerySourceRegistered: When no demo offered one.
        """
        if self.query_source is None:
            raise NoQuerySourceRegistered()
        return self.query_source

    def query_title(self) -> str:
        """
        Short name of what queries are answered from.

        :raises NoQuerySourceRegistered: When no demo offered one.
        """
        return self._registered_query_source().title()

    def query_presets(self) -> List[Preset]:
        """
        The ready-made queries the panel offers as buttons, each with its question read
        back as English by the scope it declares.

        :raises NoQuerySourceRegistered: When no demo offered one.
        """
        presets = self._registered_query_source().presets()
        with self._query_lock:
            return [
                preset.worded(self._scope_runner(preset.scope)) for preset in presets
            ]

    def match_question(self, text: str) -> QuestionMatchResult:
        """
        Recognize which of the running demo's ready-made queries a natural-language
        question is asking, if any.

        The questions the panel shows are matched against their English wording as well
        as their label; the ones it does not show are matched against their label alone,
        which is already the words they are asked in, and wording each of them would
        mean building that many queries per asked question.

        :param text: The question as asked, in natural language.
        :raises NoQuerySourceRegistered: When no demo offered one.
        """
        unlisted = self._registered_query_source().unlisted_presets()
        return QuestionMatcher(self.query_presets() + unlisted).match(text)

    def query_scopes(self) -> List[QueryScope]:
        """
        The bodies of knowledge the running demo offers, in the order it offers them.

        :raises NoQuerySourceRegistered: When no demo offered one.
        """
        return [
            knowledge.scope for knowledge in self._registered_query_source().knowledge()
        ]

    def query_variables(
        self, scope: QueryScope = QueryScope.CURRENT_STATE
    ) -> List[str]:
        """
        Names a query of one scope may range over, for the panel to advertise.

        :param scope: The body of knowledge the names belong to.
        :raises NoQuerySourceRegistered: When no demo offered one.
        :raises UnknownQueryScope: When the demo offers no such body of knowledge.
        """
        return [domain.name for domain in self._queryable_knowledge(scope).domains]

    def query_vocabulary(
        self, scope: QueryScope = QueryScope.CURRENT_STATE
    ) -> QueryVocabulary:
        """
        Everything a query of one scope may name, for the query box to offer.

        :param scope: The body of knowledge the names belong to.
        :raises NoQuerySourceRegistered: When no demo offered one.
        :raises UnknownQueryScope: When the demo offers no such body of knowledge.
        """
        knowledge = self._queryable_knowledge(scope)
        return QueryVocabulary(
            domains=knowledge.domains,
            extra_names=knowledge.extra_names,
            class_index=WorkspaceClassIndex.of_repository(),
        )

    def _queryable_knowledge(self, scope: QueryScope) -> QueryableKnowledge:
        """
        What answers questions of one scope.

        :param scope: The body of knowledge being asked.
        :raises NoQuerySourceRegistered: When no demo offered one.
        :raises UnknownQueryScope: When the demo offers no such body of knowledge.
        """
        for knowledge in self._registered_query_source().knowledge():
            if knowledge.scope is scope:
                return knowledge
        raise UnknownQueryScope(name=scope.value)

    def highlightable_ids(self) -> FrozenSet[str]:
        """
        Ids the viewer can light up: every published object, by key and by display id.

        An answer value naming one of these glows in the scene, whatever the query
        asked for (see :attr:`~cramera.knowledge.query_runner.RowRenderer.highlightable_ids`).
        """
        keys = self.object_keys()
        return frozenset(keys) | frozenset(Path(key).stem for key in keys)

    def run_query(
        self, code: str, scope: QueryScope = QueryScope.CURRENT_STATE
    ) -> RenderResult:
        """
        Answer one EQL query about the running demo.

        :param code: The EQL query source.
        :param scope: Which of the demo's bodies of knowledge to ask.
        :raises NoQuerySourceRegistered: When no demo offered one.
        :raises UnknownQueryScope: When the demo offers no such body of knowledge.
        """
        with self._query_lock:
            return self._scope_runner(scope).run(code)

    def _scope_runner(self, scope: QueryScope) -> EqlQueryRunner:
        """
        The runner answering questions of one scope, over the demo's current state.

        Krrood's SymbolGraph singleton is not threadsafe, so callers hold
        :attr:`_query_lock` around whatever they do with the runner.

        :param scope: The body of knowledge being asked.
        :raises NoQuerySourceRegistered: When no demo offered one.
        :raises UnknownQueryScope: When the demo offers no such body of knowledge.
        """
        knowledge = self._queryable_knowledge(scope)
        return EqlQueryRunner(
            domains=knowledge.domains,
            extra_names=knowledge.extra_names,
            evaluation=knowledge.evaluation,
            highlightable_ids=self.highlightable_ids(),
        )

    # %% viewer -> world (teleoperation)
    def get_robots(self) -> dict[str, Any]:
        """Return stable instance identities and the currently selected robot."""
        with self._lock:
            return {
                RobotField.ROBOTS: list(self._robot_catalog),
                RobotField.ACTIVE_IDENTIFIER: (
                    RobotModels.identifier(self.robot)
                    if self.robot is not None
                    else None
                ),
            }

    def select_robot(self, identifier: str) -> None:
        """Select an idle world's robot without resetting any world state.

        :param identifier: Native root namespace of the requested instance.
        :raises RobotSelectionBusy: If a plan is running or paused.
        :raises UnknownRobot: If the requested instance is absent.
        """
        if any(
            node.parent is None
            and node.status in (TaskStatusName.RUNNING, TaskStatusName.PAUSE)
            for node in self.plan_state.nodes
        ):
            raise RobotSelectionBusy(
                "Wait for the current plan to finish before selecting another robot"
            )
        self._select_robot(RobotModels(self.world, self.robot).named(identifier))

    def _select_robot(self, robot: AbstractRobot) -> None:
        """Publish a selected annotation and release any previous teleop driver.

        :param robot: Native annotation owned by the attached world.
        """
        if robot is self.robot:
            return
        self.stop_teleop()
        self._teleop = None
        self.robot = robot
        self._bodies[ROBOT_BASE_KEY] = robot.root
        self._refresh_bundle_signature()
        self.snapshot()

    def queue_teleop(self, request: "TeleopRequest") -> None:
        """
        Feed one streamed hand target to the teleop driver, starting it on first call.

        Called on an HTTP thread; the driver it hands off to owns the world writes, on
        its own thread, exactly as the motion tick does for queued moves.

        :param request: The latest hand target for one arm.
        :raises TeleopUnavailable: If no world/robot is bound yet.
        """
        from cramera.live.teleop import TeleopController, TeleopUnavailable

        # the robot annotation is added to the world by the demo (PR2.from_world) after
        # the bridge first binds, so resolve it live rather than trusting the early bind
        robot = self.robot
        if robot is None and self.world is not None:
            robots = self.world.get_semantic_annotations_by_type(AbstractRobot)
            robot = robots[0] if robots else None
        if self.world is None or robot is None:
            raise TeleopUnavailable("no live robot bound yet -- start a scene first")
        if self._teleop is None:
            self._teleop = TeleopController(
                self.world,
                robot,
                is_busy=lambda: time.monotonic() - self._last_tick_time < 0.5,
            )
        self._teleop.submit(request)

    def stop_teleop(self) -> None:
        """Stop the teleop driver, if one is running; the arm holds its last pose."""
        if self._teleop is not None:
            self._teleop.stop()

    # %% viewer -> world
    def queue_move(self, request: MoveRequest) -> None:
        """
        Queue an object move from the viewer (called on an HTTP thread).

        :param request: The move to apply on the next simulation tick.
        """
        with self._lock:
            snapshot_orientation = self.state.orientation_of(request.object_key)
        with self._moves_lock:
            self._moves.append(request)
            # remember the drag target regardless of whether the sim applies it (it only
            # applies on a motion tick); the Plan Builder reads these via /captured_objects
            # to snap an object's pose into a start point or an action target.
            orientation = self._orientation_after(
                request,
                previous_target=self._last_moves.get(request.object_key),
                snapshot_orientation=snapshot_orientation,
            )
            self._last_moves[request.object_key] = list(request.position) + orientation

    def queue_joint_move(self, request: JointMoveRequest) -> None:
        """
        Queue a joint position from the viewer (called on an HTTP thread).

        :param request: The joint position to apply on the next simulation tick.
        """
        with self._moves_lock:
            self._joint_moves.append(request)

    def pending_joint_moves(self) -> List[JointMoveRequest]:
        """
        The joint positions queued and not yet applied by the simulation thread.
        """
        with self._moves_lock:
            return list(self._joint_moves)

    @staticmethod
    def _orientation_after(
        request: MoveRequest,
        previous_target: Optional[List[float]],
        snapshot_orientation: Optional[List[float]],
    ) -> List[float]:
        """
        The orientation an object stands at once ``request`` is applied.

        A drag naming no orientation keeps the object's current one, which is what
        :meth:`_apply_move` writes into the world; the recorded target has to say the same,
        or capturing after such a drag reports the object as unrotated. While the sim is
        idle it has applied no earlier drag, so the newest of those is what the object
        stands at, and the snapshot only speaks for an object never dragged.

        :param request: The move whose resulting orientation is asked for.
        :param previous_target: The drag target recorded for this object before, if any.
        :param snapshot_orientation: The orientation of the newest world snapshot, if it
            has a pose for this object.
        """
        if request.quaternion is not None:
            return list(request.quaternion)
        if previous_target is not None:
            return list(previous_target[WorldStateSnapshot.ORIENTATION_START :])
        if snapshot_orientation is not None:
            return snapshot_orientation
        return list(IDENTITY_QUATERNION)

    def get_captured_objects(self) -> Dict[str, Any]:
        """
        Object poses for the Plan Builder: the live world snapshot, overlaid with the
        latest drag target per object (so a drag that the idle sim has not applied is
        still capturable). Each pose is ``[x, y, z, qx, qy, qz, qw]``.
        """
        with self._lock:
            objects = dict(self.state.objects)
        with self._moves_lock:
            objects.update(self._last_moves)
        return {"objects": objects}

    def get_surfaces(self) -> Dict[str, Any]:
        """
        Placement surfaces in the live world, for the Plan Builder's "place on a surface"
        target mode. Returns supporting-surface annotations (counters, tables, shelves,
        floors) as ``{"type": <class>, "name": <root body name>}`` so the UI can offer a
        concrete instance to sample a place pose on.
        """
        out: List[Dict[str, str]] = []
        if self.world is None:
            return {"surfaces": out}
        try:
            from semantic_digital_twin.semantic_annotations.semantic_annotations import (
                Cabinet,
                CounterTop,
                Cupboard,
                Dishwasher,
                Drawer,
                Dresser,
                Floor,
                Fridge,
                ShelfLayer,
                Sofa,
                Table,
            )
        except Exception:  # semantic annotations unavailable — return nothing
            return {"surfaces": out}
        # supporting surfaces ("on") + case containers ("in"): both expose the same
        # sample_points_from_surface via HasSupportingSurface / HasCaseAsRootBody.
        for surface_type in (
            CounterTop,
            Table,
            ShelfLayer,
            Floor,
            Sofa,
            Drawer,
            Fridge,
            Cabinet,
            Cupboard,
            Dresser,
            Dishwasher,
        ):
            try:
                annotations = self.world.get_semantic_annotations_by_type(surface_type)
            except Exception:
                continue
            for annotation in annotations:
                root = getattr(annotation, "root", None)
                name = str(root.name) if root is not None else surface_type.__name__
                out.append({"type": surface_type.__name__, "name": name})
        return {"surfaces": out}

    def queue_constraint(self, request: "AttachConstraintRequest") -> None:
        """
        Queue a constraint attached from the plan view (called on an HTTP thread).

        Prints an immediate receipt to the demo terminal so inserting a constraint is
        visible even between motions (the full resolve-against-the-world confirmation
        follows on the next simulation tick, in :meth:`_inject_constraint`).

        :param request: The constraint to apply on the next simulation tick.
        """
        params = ", ".join("%s=%s" % (k, v) for k, v in request.params.items())
        line = "=" * 68
        print(
            "\n%s\n[CONSTRAINT] received from plan view\n"
            "  text        : %r\n"
            "  giskard node: %s(%s)\n"
            "  plan node   : %s (%s)\n"
            "  status      : queued — the motion statechart applies it on the next tick\n%s\n"
            % (
                line,
                request.text,
                request.goal_type,
                params,
                request.node_label,
                request.target_node_id,
                line,
            ),
            file=sys.stderr,
            flush=True,
        )
        with self._constraints_lock:
            self._constraints.append(request)

    def apply_constraints(self, chart: Optional["MotionStatechart"] = None) -> None:
        """
        Drain queued plan-view constraints on the simulation thread.

        Runs from the tick hook, next to :meth:`apply_moves`, because the simulation
        thread is the only one allowed to touch the running motion statechart.

        :param chart: The motion statechart the executor is currently ticking, if any.
        """
        with self._constraints_lock:
            constraints, self._constraints = self._constraints, []
        for constraint in constraints:
            try:
                self._inject_constraint(constraint, chart)
            except Exception:  # a bad constraint must never kill the tick
                logger.exception("failed to inject constraint %r", constraint.text)

    def _resolve_body(self, name: Optional[str]) -> Optional[Any]:
        """
        Look up a world :class:`Body` by name, tolerating the viewer's short names.

        :param name: The body/link name from the compiled constraint (e.g. ``"milk"``,
            ``"head_camera"``, ``"map"``), or None.
        :return: The matching body, the world root for ``"map"``/None, or None if unknown.
        """
        if self.world is None:
            return None
        if not name or name == "map":
            return self.world.root
        # objects are published to the viewer under mesh keys ("milk.stl"); that is the
        # same dict MoveRequest resolves against, so try it first for object constraints.
        for key in (name, name + ".stl", name + ".obj", name + ".dae"):
            body = self._bodies.get(key)
            if body is not None:
                return body
        try:
            return self.world.get_body_by_name(name)
        except Exception:
            pass
        # last resort: suffix match against every body name (names are prefixed/typed)
        for body in self.world.bodies:
            leaf = str(body.name).split("/")[-1]
            if leaf == name or leaf == name + ".stl" or leaf.rsplit(".", 1)[0] == name:
                return body
        logger.warning(
            "constraint link %r not found; known objects=%s",
            name,
            sorted(self._bodies)[:12],
        )
        return None

    def _build_giskard_node(
        self, constraint: "AttachConstraintRequest"
    ) -> Optional[Any]:
        """
        Turn a compiled constraint into a real giskardpy motion-statechart node,
        resolving its link names to live :class:`Body` objects.

        :param constraint: The validated constraint from the plan view.
        :return: The constructed monitor node, or None if it cannot be built (unknown
            body, placeholder target, or an unsupported goal type).
        """
        from semantic_digital_twin.spatial_types import Point3, Vector3
        from giskardpy.motion_statechart.monitors.cartesian_monitors import (
            VectorsAligned,
            PointingAt,
        )
        from giskardpy.motion_statechart.monitors.feature_monitors import (
            HeightMonitor,
            DistanceMonitor,
        )

        params = constraint.params
        root = self._resolve_body(params.get("root_link"))
        tip = self._resolve_body(params.get("tip_link"))
        if root is None or tip is None:
            logger.warning(
                "constraint %r: could not resolve links root=%r tip=%r in the live world",
                constraint.text,
                params.get("root_link"),
                params.get("tip_link"),
            )
            return None

        def vec(value):
            return Vector3(*value)

        if constraint.goal_type == "VectorsAligned":
            return VectorsAligned(
                root_link=root,
                tip_link=tip,
                goal_normal=vec(params.get("goal_normal", [0, 0, 1])),
                tip_normal=vec(params.get("tip_normal", [0, 0, 1])),
                threshold=float(params.get("threshold", 0.1)),
            )
        if constraint.goal_type == "PointingAt":
            goal = params.get("goal_point")
            if isinstance(goal, (list, tuple)):
                goal_point = Point3(*goal, reference_frame=root)
            else:
                # a placeholder like "@operation_target": point at the object being
                # handled, if we can resolve it, else give up gracefully
                target = self._resolve_body(params.get("goal_point_body"))
                if target is None:
                    logger.warning(
                        "constraint %r: PointingAt has no concrete target to aim at",
                        constraint.text,
                    )
                    return None
                goal_point = Point3(0, 0, 0, reference_frame=target)
            return PointingAt(
                root_link=root,
                tip_link=tip,
                goal_point=goal_point,
                pointing_axis=vec(params.get("pointing_axis", [0, 0, 1])),
                threshold=float(params.get("threshold", 0.05)),
            )
        if constraint.goal_type in ("HeightMonitor", "DistanceMonitor"):
            # reference is the world root; the monitored point is the object's origin.
            cls = (
                HeightMonitor
                if constraint.goal_type == "HeightMonitor"
                else DistanceMonitor
            )
            return cls(
                root_link=root,
                tip_link=tip,
                reference_point=Point3(0, 0, 0, reference_frame=root),
                tip_point=Point3(0, 0, 0, reference_frame=tip),
                lower_limit=float(params.get("lower_limit", 0.0)),
                upper_limit=float(params.get("upper_limit", 2.0)),
            )
        logger.info(
            "constraint %r: goal type %s not yet wired for live injection",
            constraint.text,
            constraint.goal_type,
        )
        return None

    def _inject_constraint(
        self,
        constraint: "AttachConstraintRequest",
        chart: Optional["MotionStatechart"] = None,
    ) -> None:
        """
        Resolve a plan-view constraint to a real giskardpy node and attach it to the
        live motion statechart.

        Runs on the simulation thread (from :meth:`apply_constraints`), so touching the
        chart is race-free. The node is built with live :class:`Body` objects and added
        via :meth:`MotionStatechart.add_node`, which grows the chart's state vectors.

        .. note:: ``add_node`` registers the node and grows the state, but the currently
           executing QP was already compiled; the added constraint is picked up when the
           chart next compiles (``apply == "next_activation"``). Forcing a recompile of a
           mid-tick QP is deliberately avoided here, as it would race the executor.

        :param constraint: The validated constraint from the plan view.
        :param chart: The motion statechart the executor is ticking, if any.
        """
        self.pending_constraints.append(constraint)
        node = self._build_giskard_node(constraint)
        if node is None:
            self._announce_constraint(constraint, node, chart, ok=False)
            return
        # The node is built against the LIVE world (bodies resolved, giskard goal
        # constructed) — this is the constraint giskardpy will enforce. It is NOT added
        # to the currently-ticking chart: a compiled MotionStatechart binds several fixed
        # -size updaters (observation state, life-cycle state, the QP), so add_node mid-run
        # raises a shape mismatch and kills the executor (verified). Instead the resolved
        # node is queued for the next motion activation, where the chart is (re)compiled.
        self._resolved_constraints.append((constraint, node))
        self._announce_constraint(constraint, node, chart, ok=True)

    def _announce_constraint(
        self,
        constraint: "AttachConstraintRequest",
        node: Optional[Any],
        chart: Optional["MotionStatechart"],
        ok: bool,
    ) -> None:
        """
        Print a clear, terminal-visible confirmation that a plan-view constraint reached
        the running demo and became a real giskardpy motion-statechart node.

        Printed to stderr (not the logger) so it shows up in the demo terminal regardless
        of the log level.

        :param constraint: The constraint from the plan view.
        :param node: The built giskardpy node, or None if it could not be built.
        :param chart: The motion statechart currently ticking, if any.
        :param ok: Whether a node was successfully built.
        """
        line = "=" * 68
        if not ok or node is None:
            msg = (
                "\n%s\n[CONSTRAINT] received from plan view but NOT applicable\n"
                "  text : %r\n  goal : %s\n  note : could not resolve links / goal against the live world\n%s\n"
                % (line, constraint.text, constraint.goal_type, line)
            )
            print(msg, file=sys.stderr, flush=True)
            return
        root = getattr(node, "root_link", None)
        tip = getattr(node, "tip_link", None)
        params = ", ".join("%s=%s" % (k, v) for k, v in constraint.params.items())
        chart_title = self._chart_title if chart is not None else None
        msg = (
            "\n%s\n[CONSTRAINT] plan view -> live giskardpy motion statechart\n"
            "  text        : %r\n"
            "  giskard node: %s(%s)\n"
            "  resolved    : root_link=%s  tip_link=%s\n"
            "  plan node   : %s (%s)\n"
            "  current MS  : %s\n"
            "  status      : compiled against the live world; the motion statechart will\n"
            "                enforce it at the next motion activation (%d pending)\n%s\n"
            % (
                line,
                constraint.text,
                constraint.goal_type,
                params,
                root is not None and str(root.name),
                tip is not None and str(tip.name),
                constraint.node_label,
                constraint.target_node_id,
                chart_title or "(no motion running right now)",
                len(self._resolved_constraints),
                line,
            )
        )
        print(msg, file=sys.stderr, flush=True)

    def apply_moves(self) -> None:
        """
        Apply queued object moves to the world.

        Called from the tick hook — the simulation thread is the only place that may
        write to the world.
        """
        with self._moves_lock:
            moves, self._moves = self._moves, []
            joint_moves, self._joint_moves = self._joint_moves, []
        if self.world is None:
            return
        for move in moves:
            body = self._bodies.get(move.object_key)
            if body is None:
                logger.debug("no published body for %s — move skipped", move.object_key)
                continue
            self._apply_move(move, body)
        for joint_move in joint_moves:
            self._apply_joint_move(joint_move)

    def _apply_joint_move(self, move: JointMoveRequest) -> None:
        """
        Write one viewer joint position into the world, held within the joint's limits.

        :param move: The queued joint position to apply.
        """
        connection = next(
            (
                candidate
                for candidate in self._connections
                if str(candidate.name) == move.connection_name
            ),
            None,
        )
        if connection is None:
            logger.debug(
                "no actuated connection %s — joint move skipped", move.connection_name
            )
            return
        limits = connection.dof.limits
        position = move.position
        if limits.lower.position is not None:
            position = max(limits.lower.position, position)
        if limits.upper.position is not None:
            position = min(limits.upper.position, position)
        connection.position = position
        self._transforms.note_viewer_write(str(connection.name))
        logger.info(
            "set %s -> %.3f [final=%s]", move.connection_name, position, move.is_final
        )

    def _apply_move(self, move: MoveRequest, body: Body) -> None:
        """
        Write one viewer move into the world.

        Only free-floating (:class:`Connection6DoF`) objects are draggable.
        Objects rigidly fixed to furniture — e.g. a spoon on a drawer that
        must ride along when the drawer opens — keep their ``FixedConnection``
        and are left untouched (a fixed connection has no settable origin).

        .. note:: The object is not re-parented. That is a structural change of the
           kinematic tree (``modify_world`` + forward kinematics recompile), and
           running it inside the tick hook while a giskard goal is live hangs the
           executor. The plain pose write already makes ``body.global_pose`` correct,
           which is what the plan's navigate/pick reachability reads.

        :param move: The queued move to apply.
        :param body: The body the move targets.
        """
        connection = body.parent_connection
        if not isinstance(connection, Connection6DoF):
            logger.info(
                "%s is fixed (%s) — not draggable, skipping",
                move.object_key,
                type(connection).__name__,
            )
            return
        position = move.position
        rotation_matrix = (
            RotationMatrix.from_quaternion(
                Quaternion(
                    x=move.quaternion[0],
                    y=move.quaternion[1],
                    z=move.quaternion[2],
                    w=move.quaternion[3],
                )
            )
            if move.quaternion is not None
            else body.global_pose.to_rotation_matrix()
        )
        world_T_object = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            Point3(x=position[0], y=position[1], z=position[2]),
            rotation_matrix,
            reference_frame=self.world.root,
        )
        # ``origin`` is parent-relative; express the target in the parent
        # frame (a no-op while the parent is the world root)
        parent_T_world = connection.parent.global_pose.to_homogeneous_matrix().inverse()
        parent_T_object = parent_T_world @ world_T_object
        # the matrix product drops the frames; the origin setter transforms whatever
        # frame it is handed into the parent frame, so label the result explicitly
        parent_T_object.reference_frame = connection.parent
        parent_T_object.child_frame = body
        connection.origin = parent_T_object
        self._transforms.note_viewer_write(str(connection.name))
        logger.info(
            "moved %s -> world (%.3f, %.3f, %.3f) [final=%s]",
            move.object_key,
            position[0],
            position[1],
            position[2],
            move.is_final,
        )

    # %% world discovery
    def bind(self) -> None:
        """
        Discover the robot, joints and publishable bodies of the current world.

        Re-run periodically because demos modify their world (objects get spawned and
        removed mid-run).
        """
        world = self.world
        if world is None:
            return
        self._last_bind_time = time.time()
        robots = world.get_semantic_annotations_by_type(AbstractRobot)
        if self.robot not in robots:
            self.robot = robots[0] if robots else None
        self._kinematic_connections = list(world.connections)
        self._connections = self._actuated_connections(self._kinematic_connections)
        bodies: Dict[str, Body] = {}
        if self.robot is not None:
            bodies[ROBOT_BASE_KEY] = self.robot.root
        try:
            bodies.update(self._discover_overlay_bodies())
        except Exception as error:
            # boundary guard: the world is mid-modification (a body is being spawned
            # or removed) and iterating it is not safe. Keep the previous catalog
            # rather than publishing an empty one, which would make the viewer hide
            # every object it already shows.
            logger.debug("body scan skipped this bind: %s", error)
            for key, body in self._bodies.items():
                bodies.setdefault(key, body)
        self.publish_bodies(bodies)

    def _discover_overlay_bodies(self) -> Dict[str, Body]:
        """
        Every world body the overlay renders, keyed the way it is published.

        Free objects remain tracked after attachment to the robot or a support until
        they leave the world. Mesh-named objects retain their existing publication.

        :return: Present overlay bodies keyed by their local body name.
        """
        return {
            str(body.name).split("/")[-1]: body
            for body in WorldObjects(self.world, self.robot).overlay_bodies(
                self._bodies.values()
            )
        }

    @staticmethod
    def _body_shapes(body: Body) -> List[Any]:
        """
        The shapes a body is rendered from: its visual ones, else its collision ones.

        :param body: The body whose shapes are read.
        """
        for shape_collection in (body.visual, body.collision):
            if shape_collection.shapes:
                return list(shape_collection.shapes)
        return []

    @staticmethod
    def _actuated_connections(
        connections: List[Connection],
    ) -> List[ActiveConnection1DOF]:
        """
        All 1-DOF connections — the joints published as trajectory frames.

        :param connections: The world's connections to pick the actuated ones from.
        """
        return [
            connection
            for connection in connections
            if isinstance(connection, ActiveConnection1DOF)
        ]

    def _build_object_metadata(self, bodies: Dict[str, Body]) -> None:
        """
        Rebuild the geometry catalog the viewer spawns live objects from.

        Each object gets a mesh URL (served by the bridge), its real shapes, or a
        fallback box size, so objects the viewer does not know yet can appear mid-run.

        :param bodies: The current published bodies, keyed by mesh key.
        """
        catalog: List[ObjectCatalogEntry] = []
        serve: Dict[str, str] = {}
        palette = ObjectPalette()
        for index, (key, body) in enumerate(
            item for item in bodies.items() if item[0] != ROBOT_BASE_KEY
        ):
            color = palette.color_for(index)
            object_id = Path(key).stem
            shapes = self._body_shapes(body)
            if shapes:
                catalog.append(self._shape_catalog_entry(key, shapes, color, serve))
                continue
            catalog.append(
                ObjectCatalogEntry(
                    key=key,
                    id=object_id,
                    kind=ObjectKind.BOX,
                    color=color,
                    size=list(self.DEFAULT_OBJECT_SIZE),
                )
            )
        self._mesh_serve = serve
        with self._lock:
            self.object_metadata = catalog

    def _shape_catalog_entry(
        self,
        key: str,
        shapes: List[Any],
        fallback_color: str,
        serve: Dict[str, str],
    ) -> ObjectCatalogEntry:
        """
        The catalog entry of a body published shape by shape.

        Mesh shapes are registered in the serve map under a composite key, so each of
        a body's meshes is downloadable on its own.

        :param key: The body's published key.
        :param shapes: The body's shapes, as :meth:`_body_shapes` selects them.
        :param fallback_color: Palette colour used for shapes without one of their own.
        :param serve: The serve map being built, extended with this body's mesh files.
        """
        entries: List[ShapeEntry] = []
        for shape_index, shape in enumerate(shapes):
            mesh_url = None
            mesh_file = served_mesh_file(shape)
            if mesh_file is not None:
                serve_key = "%s#%d" % (key, shape_index)
                serve[serve_key] = mesh_file
                mesh_url = "/mesh?key=" + urllib.parse.quote(serve_key, safe="")
            entries.append(
                shape_entry(
                    shape,
                    mesh_url,
                    fallback_size=list(self.DEFAULT_OBJECT_SIZE),
                    fallback_color=fallback_color,
                )
            )
        return ObjectCatalogEntry(
            key=key,
            id=Path(key).stem,
            kind=ObjectKind.SHAPES,
            color=entries[0].color,
            shapes=entries,
        )

    # %% world snapshot
    def snapshot(self) -> None:
        """
        Publish the world's joints, base pose and object poses.

        Runs on the simulation thread; rebinds the world periodically so mid-run spawns
        show up.
        """
        if self.world is None:
            return
        if time.time() - self._last_bind_time > self.REBIND_INTERVAL_SECONDS:
            self.bind()
        frames = {
            str(connection.name): round(float(connection.position), POSE_PRECISION)
            for connection in self._connections
        }
        base_pose: Optional[List[float]] = None
        object_poses: Dict[str, List[float]] = {}
        for name, body in self._bodies.items():
            if name == ROBOT_BASE_KEY:
                base_pose = rounded_pose(body)
            else:
                object_poses[name] = rounded_pose(body)
        self._refresh_marker_state()
        robot_models = RobotModels(self.world, self.robot)
        transforms = self._transforms.observe(
            self._kinematic_connections, self.world, time.monotonic()
        )
        with self._lock:
            self.transform_state = transforms
            self.sequence_number += 1
            self._robot_catalog = robot_models.catalog()
            self.state = WorldStateSnapshot(
                sequence_number=self.sequence_number,
                frames=frames,
                base=base_pose,
                objects=object_poses,
                markers_version=self.marker_state["version"],
                model_bases=robot_models.root_poses(),
            )

    def get_state(self) -> Dict[str, Any]:
        """
        The newest world snapshot (safe to call from HTTP threads).

        While the sim is idle (no recent motion tick, e.g. the Plan Builder scaffold sitting
        still), overlay the queued viewer moves so a drag / reset / rotation shows up in the
        3D view without waiting for a tick that isn't coming. During a motion the real world
        poses win, so a running plan is shown faithfully.
        """
        with self._lock:
            payload = self.state.to_payload()
        if time.monotonic() - self._last_tick_time > 0.5:
            with self._moves_lock:
                moves = dict(self._last_moves)
            objs = payload.get("objects")
            if moves and isinstance(objs, dict):
                payload["objects"] = {**objs, **moves}
        return payload

    def get_transforms(self) -> Dict[str, Any]:
        """
        The newest transform graph, aged as of now (safe to call from HTTP threads).
        """
        with self._lock:
            return self.transform_state.to_payload(time.monotonic())

    # %% plan tree
    def _live_motion_status(self, node: PlanNode) -> Optional[str]:
        """
        Status of one plan node as its plan callbacks reported it, or None.

        :param node: The plan node whose live status is looked up.
        """
        progress = self._motion_nodes.get(id(node))
        if progress is None:
            return None
        return progress.status

    def snapshot_plan(self) -> None:
        """
        Serialize the plan tree with per-node execution status.

        A node's own status wins while it says something; otherwise the live statechart
        status is used, else the aggregate of the children (running/failed bubble up; a
        parent whose children are only partly done reads as running, not succeeded).
        """
        plan = self._plan
        if plan is None:
            return
        try:
            root = plan.root
        except Exception:
            # the plan is mid-mutation and not a tree right now — next tick
            return
        nodes: List[PlanNodeEntry] = []
        order: List[str] = []
        self._serialize_plan_node(root, None, nodes, order)
        with self._lock:
            self.plan_state = PlanSnapshot(signature="|".join(order), nodes=nodes)

    def _serialize_plan_node(
        self,
        node: PlanNode,
        parent_id: Optional[str],
        nodes: List[PlanNodeEntry],
        order: List[str],
    ) -> str:
        """
        Serialize one plan node and its subtree; returns the node's status.

        :param node: The plan node to serialize.
        :param parent_id: Id of the node's parent entry, or None for the root.
        :param nodes: Output list every serialized entry is appended to.
        :param order: Output list every serialized node id is appended to, in
            traversal order, to build the tree's signature.
        """
        node_id = "plan_node_%d" % id(node)
        designator = node.designator if isinstance(node, DescribesAnAction) else None
        native_lifecycle = isinstance(node.status, LifeCycleValues)
        own_status = TaskStatusName.of_native_name(node.status.name)
        entry = PlanNodeEntry(
            id=node_id,
            parent=parent_id,
            kind=type(node).__name__,
            group=PlanNodeGroup.of_plan_node_kind(type(node).__name__),
            label=(
                type(designator).__name__
                if designator is not None
                else type(node).__name__
            ),
            status=own_status,
            derived=False,
        )
        self._add_designator_metadata(entry, designator)
        nodes.append(entry)
        order.append(node_id)

        child_best, children, done = "CREATED", 0, 0
        for child in node.children:
            child_status = self._serialize_plan_node(child, node_id, nodes, order)
            if (
                child_status == TaskStatusName.CREATED
                and PlanNodeGroup.of_plan_node_kind(type(child).__name__)
                == PlanNodeGroup.CONDITION
                and (
                    native_lifecycle
                    or (
                        isinstance(self._plan, PlanExecutionContext)
                        and self._plan.context is not None
                        and not self._plan.context.evaluate_conditions
                    )
                )
            ):
                continue
            child_best = self._max_status(child_best, child_status)
            children += 1
            if child_status == "SUCCEEDED":
                done += 1
        if own_status == "CREATED":
            if child_best == "SUCCEEDED" and done < children:
                child_best = "RUNNING"
            motion_status = None if native_lifecycle else self._live_motion_status(node)
            derived = motion_status or (child_best if child_best != "CREATED" else None)
            if derived:
                entry.status = derived
                entry.derived = True
        if native_lifecycle:
            return entry.status
        # sticky completion: once a node has run and is idle again (not running/failed),
        # keep it SUCCEEDED so the plan view shows a monotonic done-progression.
        if entry.status == "RUNNING":
            self._ever_running.add(id(node))
        elif id(node) in self._ever_running and entry.status == "CREATED":
            entry.status = "SUCCEEDED"
            entry.derived = True
        return entry.status

    def _add_designator_metadata(
        self, entry: PlanNodeEntry, designator: Optional[Any]
    ) -> None:
        """
        Add arm and target-object info from a node's designator, if any.

        :param entry: The serialized entry to fill in, mutated in place.
        :param designator: The node's designator, or None.
        """
        if designator is None:
            return
        fields = vars(designator)
        arm = fields.get("arm") or fields.get("arms")
        if arm is not None:
            entry.arm = str(arm)
        target = self._designator_target(designator)
        if target:
            entry.target = target

    @staticmethod
    def _max_status(first: str, second: str) -> str:
        """
        The higher-ranked of two statuses.

        :param first: The first status to compare.
        :param second: The second status to compare.
        """
        if TaskStatusName.rank_of(first) >= TaskStatusName.rank_of(second):
            return first
        return second

    def _designator_target(self, designator: Any) -> Optional[str]:
        """
        Published key of the object a designator refers to, if any.

        Matched by basename, because designators name world entities with their full
        prefixed name while some objects are published under a basename key.

        :param designator: The designator to search for a world-entity reference.
        """
        keys_by_basename = {key.split("/")[-1]: key for key in self._bodies}
        for value in vars(designator).values():
            if not isinstance(value, NamesAWorldEntity):
                continue
            basename = str(value.name).split("/")[-1]
            if basename in keys_by_basename:
                return keys_by_basename[basename]
        return None

    def running_step(self) -> Optional[str]:
        """
        Label of the action the plan is performing right now, or None between actions.

        The deepest running action wins: a ``Transport`` that is performing its
        ``Pickup`` is reported as the pickup, which is the step a replay of this moment
        should be labelled with.
        """
        with self._lock:
            running = [
                entry
                for entry in self.plan_state.nodes
                if entry.status == TaskStatusName.RUNNING
                and entry.group is PlanNodeGroup.ACTION
            ]
        return running[-1].label if running else None

    def get_plan(self) -> Dict[str, Any]:
        """
        The newest plan snapshot (safe to call from HTTP threads).
        """
        with self._lock:
            return self.plan_state.to_payload()

    # %% motion statechart
    def observe_chart(self, chart: Optional[MotionStatechart]) -> None:
        """
        Publish the executing statechart's structure and node states.

        Called from the tick hook. What the chart looks like is read by the observer,
        which is what a recording reads it with too; published only when it changed.

        :param chart: The motion statechart the executor is currently ticking, if any.
        """
        self._chart_observer.title = self._chart_title
        snapshot = self._chart_observer.change(chart)
        if snapshot is None:
            return
        with self._lock:
            self.chart_state = snapshot

    def executing_statechart(self) -> Optional[ChartSnapshot]:
        """
        The statechart the executor is currently ticking, or None while no motion runs.
        """
        with self._lock:
            chart = self.chart_state
        return chart if chart.nodes else None

    def get_chart(self) -> Dict[str, Any]:
        """
        The newest statechart snapshot (safe to call from HTTP threads).
        """
        with self._lock:
            chart = self.chart_state
        payload = asdict(chart)
        payload["edges"] = [edge.to_payload() for edge in chart.edges]
        return payload


BRIDGE = Bridge()
"""
The process-wide bridge instance shared by hooks and HTTP handlers.
"""
