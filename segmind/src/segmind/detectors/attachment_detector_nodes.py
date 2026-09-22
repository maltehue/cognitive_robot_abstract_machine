"""
Recognizing that something was picked up from what it hangs from.

A plan that picks an object up re-parents it to the gripper, and releases it by hanging
it back on the world's root. Those two moments are the pick-up and the putting-down
themselves, so they need no inferring from an object's motion and whatever it stopped
resting on -- and there is exactly one of each per grasp, stamped when it happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from giskardpy.motion_statechart.context import MotionStatechartContext
from typing import Dict, List, Optional

from segmind.datastructures.events import DetectionEvent, PickUpEvent, PlacingEvent
from segmind.detectors.base import AbstractDetector, SegmindContext
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class AttachmentDetector(AbstractDetector):
    """
    Detector for pick-up and putting-down events.

    Reports a :class:`PickUpEvent` when a watched body comes to hang from another body,
    and a :class:`PlacingEvent` when it hangs from the world's root again.
    """

    _parents: Dict[Body, Body] = field(default_factory=dict, init=False)
    """
    What each watched body hung from when it was last looked at.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Report every watched body that has changed what it hangs from since the last
        tick.

        A body seen for the first time reports nothing: what it hung from before is not
        known, so nothing about it has changed yet.

        :param context: The context holding the world the bodies live in.
        :param segmind_context: The shared context, unused here -- what a body hangs from
            is in the world itself.
        :param tracked_objects: The bodies to look at.
        """
        events = []
        for body in tracked_objects:
            parent = body.parent_connection.parent
            previous = self._parents.get(body)
            self._parents[body] = parent
            if previous is None or parent is previous:
                continue
            event = self._event_of_change(body, parent, context.world.root)
            if event is not None:
                events.append(event)
        return events

    @staticmethod
    def _event_of_change(
        body: Body, parent: Body, world_root: Body
    ) -> Optional[DetectionEvent]:
        """
        What it means that a body now hangs from ``parent``.

        :param body: The body that was re-parented.
        :param parent: What it hangs from now.
        :param world_root: The world's root, which is what a released body hangs from.
        """
        if parent is world_root:
            return PlacingEvent(tracked_object=body)
        return PickUpEvent(tracked_object=body, with_object=parent)
