from __future__ import annotations

import numpy as np

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, List, Any

from giskardpy.motion_statechart.context import (
    MotionStatechartContext,
    ContextExtension,
)
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from krrood.symbol_graph.symbol_graph import Symbol
from segmind.datastructures.events import MotionEvent, DetectionEvent, RotationEvent
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from segmind.event_logger import EventLogger
from semantic_digital_twin.semantic_annotations.semantic_annotations import Aperture
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    IsPerceivable,
)
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class DetectorStateChart(MotionStatechart):
    """
    Statechart responsible for running the different motion detectors.

    Currently acts as a container for the detectors and inherits the
    functionality from MotionStatechart.
    """


IndexedBodyPairs = Dict[Body, Set[Body]]
"""
Type hint for dictionaries mapping bodies to sets of bodies
"""


@dataclass
class SegmindContext(ContextExtension):
    """
    Context object shared across the motion statechart detectors.

    Stores the latest detected contact and support relationships
    between bodies in the simulation as well as the event logger.
    """

    latest_contact_bodies: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies it is currently in contact with.
    """

    latest_support: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies that currently support it.
    """

    latest_containments: IndexedBodyPairs = field(default_factory=dict)
    """
    Dictionary mapping each body to the set of bodies that currently contain it.
    """

    latest_motion_events: Dict[Body, MotionEvent] = field(default_factory=dict)
    """
    Dictionary mapping each body to its currently active motion event, if any.
    """

    latest_rotation_events: Dict[Body, RotationEvent] = field(default_factory=dict)
    """
    Dictionary mapping each body to its currently active rotation event, if any.
    """

    logger: EventLogger = field(default_factory=EventLogger)
    """
    The event logger used to record detected events.
    """

    placing_pairs: set[Any] = field(default_factory=set)
    """
    Set of placing pairs, to avoid duplicate events
    """

    holes: List[Aperture] = field(default_factory=list)
    """
    List of bodies that can be considered holes
    """

    insertion_pairs: set[Any] = field(default_factory=set)
    """
    List of insertion pairs, to avoid duplicate events
    """

    tracker_registry: ObjectTrackerFactory = field(default_factory=ObjectTrackerFactory)
    """
    The object tracker registry.    
    """

    watched_bodies: Set[Body] = field(default_factory=set)
    """
    Every body a detector has watched so far, whether or not it is still free.

    A plan that grasps something re-parents it to the gripper with a fixed joint, which
    takes it out of the world's free bodies -- while what it does in the gripper is
    exactly what a pick-up is made of. Once watched, a body stays watched.
    """


@dataclass(repr=False, eq=False)
class AbstractDetector(MotionStatechartNode, Symbol, ABC):
    """
    Abstract base class for all detectors.

    As a :class:`Symbol`, every instance is tracked in the SymbolGraph.
    """

    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    """
    :param tracked_object: Optional body that should be monitored.
    If None, all trackable objects in the world are checked.
    """

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Executes one update cycle of the detector.

        Determines the objects that should be checked for contacts,
        computes new contact relationships, and triggers events if
        contact changes are detected.

        :param context: The current motion statechart context.
        :return: ObservationStateValues.TRUE if events were triggered,
        otherwise ObservationStateValues.FALSE.
        """
        segmind_context_extension = context.require_extension(SegmindContext)

        objects_to_check = (
            [self.tracked_object]
            if self.tracked_object
            else self._bodies_to_watch(context, segmind_context_extension)
        )
        events = self.update_context_and_events(
            context, segmind_context_extension, objects_to_check
        )
        for e in events:
            segmind_context_extension.logger.log_event(
                e, segmind_context_extension.tracker_registry
            )
        return ObservationStateValues.TRUE if events else ObservationStateValues.FALSE

    @staticmethod
    def _bodies_to_watch(
        context: MotionStatechartContext, segmind_context: SegmindContext
    ) -> List[Body]:
        """
        The bodies this tick is about: the world's objects, plus the ones already being
        watched when a plan has since grasped them.

        A body counts as an object either because the world lets it move freely or
        because the world annotates it as something perceivable -- a demo may start a
        spoon off fixed inside a drawer, which says the spoon stays put rather than that
        it is part of the cabinet. Furniture and the robot's own links are neither, so
        they never enter. A body without collision geometry never does either: a world
        holds frames as well as things -- a mobile robot's ``odom`` is free to move and
        has no shape -- and whether a frame touches anything is not a question.

        :param context: The context holding the world the bodies live in.
        :param segmind_context: The context remembering what is already watched.
        """
        present = set(context.world.bodies)
        perceivable = {
            annotation.root
            for annotation in context.world.get_semantic_annotations_by_type(
                IsPerceivable
            )
            if isinstance(annotation, HasRootBody)
        }
        segmind_context.watched_bodies &= present
        segmind_context.watched_bodies |= {
            body
            for body in present
            if body.collision.shapes
            and (type(body.parent_connection) is Connection6DoF or body in perceivable)
        }
        return [
            body
            for body in context.world.bodies
            if body in segmind_context.watched_bodies
        ]

    def get_relation(
        self,
        context: MotionStatechartContext,
        tracked_objects: List[Body],
        predicate,
        reach: float,
    ) -> Dict[Body, Set[Body]]:
        """
        Get the relation between tracked objects.

        Pairs whose collision geometry cannot come within ``reach`` of each other are
        ruled out by their bounding spheres before the predicate is asked: deciding that
        is far cheaper than measuring a pair exactly, and in a world of any size almost
        every pair is far apart.

        :param context: The context containing world information.
        :param tracked_objects: List of bodies to check for contact changes.
        :param predicate: Function that returns true if the objects are related.
        :param reach: The furthest apart two bodies' surfaces can be while the predicate
            still holds.
        :return: Dictionary mapping bodies to sets of related bodies.
        """

        related_bodies: Dict[Body, Set[Body]] = {}
        bodies_with_collision = context.world.bodies_with_collision
        centre = {
            body: body.global_transform.to_np()[:3, 3]
            for body in set(bodies_with_collision) | set(tracked_objects)
        }
        for obj in tracked_objects:
            for body in bodies_with_collision:
                if body is obj:
                    continue
                separation = float(np.linalg.norm(centre[obj] - centre[body]))
                if separation > (
                    obj.collision_bounding_radius
                    + body.collision_bounding_radius
                    + reach
                ):
                    continue
                if predicate(obj, body):
                    related_bodies.setdefault(obj, set()).add(body)
        return related_bodies

    @abstractmethod
    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Core detection logic that updates the internal state and identifies new events.

        This method is called during every tick of the detector. Implementations should
        examine the current state of the world (via the context) for the given
        `tracked_objects`, update the relevant fields in `context` (e.g.,
        `latest_contact_bodies`, `latest_support`), and return a list of any
        `DetectionEvent`s that occurred since the last update.

        Specific implementations may detect:
        * State changes: e.g., a new contact (ContactEvent) or loss of contact.
        * Continuous processes: e.g., ongoing motion or containment.
        * Complex interactions: e.g., insertion or picking up objects.

        :param context: The shared SegmindContext containing the world state,
                        history of relationships, and the event logger.
        :param segmind_context: The SegmindContext extension containing additional states.
        :param tracked_objects: A list of bodies that this detector should focus on
                                during this update cycle.
        :return: A list of DetectionEvent objects representing the events detected
                 in this cycle. Returns an empty list if no events were found.
        """
        pass
