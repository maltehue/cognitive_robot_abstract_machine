"""
Tests for recognizing a pick-up from what a body hangs from.
"""

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import PickUpEvent, PlacingEvent
from segmind.detectors.attachment_detector_nodes import AttachmentDetector
from segmind.detectors.base import SegmindContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List

# %% a world whose objects a hand can take


def collidable_body(name: str) -> Body:
    """
    A body a detector can measure, and so produce events about.

    :param name: The body's name.
    """
    return Body(
        name=PrefixedName(name),
        collision=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
    )


def world_with_a_hand_and_a_cup() -> World:
    """
    A world holding a hand, a table and a loose cup, as a demo's world holds a gripper
    and the objects it moves.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    hand, table, cup = (collidable_body(name) for name in ("hand", "table", "cup"))
    with world.modify_world():
        world.add_body(root)
        world.add_connection(FixedConnection(parent=root, child=table))
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=root, child=hand)
        )
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=root, child=cup)
        )
    return world


def reattach(world: World, body: Body, new_parent: Body) -> None:
    """
    Re-parent ``body`` under ``new_parent`` the way a plan's attachment does.

    :param world: The world both bodies live in.
    :param body: The body being re-parented.
    :param new_parent: What it hangs from afterwards.
    """
    held = world.compute_forward_kinematics(new_parent, body)
    with world.modify_world():
        world.remove_connection(body.parent_connection)
        world.add_connection(
            FixedConnection(
                parent=new_parent, child=body, parent_T_connection_expression=held
            )
        )


# %% grasping and releasing, as the plan does it


class TestPickUpFromAttachment:
    """
    A plan that picks something up re-parents it to the gripper, and releases it by
    hanging it back on the world's root.

    Those two moments are the pick-up and the
    putting-down: one of each per grasp, at the instant it happens, rather than inferred
    from an object's motion and whatever it stopped resting on.
    """

    def events_of(self, world: World, moves: List[Body]) -> List:
        """
        Run the detector over a series of re-parentings and return what it detected.

        :param world: The world the bodies live in.
        :param moves: The new parent of the cup, one per tick.
        """
        cup = world.get_body_by_name("cup")
        executor = EpisodeSegmenterExecutor(
            context=MotionStatechartContext(world=world)
        )
        segmind_context = executor.context.require_extension(SegmindContext)
        executor.compile(SegmindStatechart().build_statechart([AttachmentDetector()]))
        executor.tick()
        for new_parent in moves:
            reattach(world, cup, new_parent)
            executor.tick()
        return segmind_context.logger.get_events()

    def test_attaching_a_body_to_another_body_is_a_pick_up(self):
        world = world_with_a_hand_and_a_cup()
        hand = world.get_body_by_name("hand")

        events = self.events_of(world, [hand])

        assert [type(event) for event in events] == [PickUpEvent]
        assert events[0].tracked_object is world.get_body_by_name("cup")
        assert events[0].with_object is hand

    def test_hanging_it_back_on_the_world_is_a_putting_down(self):
        world = world_with_a_hand_and_a_cup()
        hand = world.get_body_by_name("hand")

        events = self.events_of(world, [hand, world.root])

        assert [type(event) for event in events] == [PickUpEvent, PlacingEvent]

    def test_one_grasp_is_one_pick_up(self):
        world = world_with_a_hand_and_a_cup()
        hand = world.get_body_by_name("hand")

        events = self.events_of(world, [hand])

        assert len([e for e in events if isinstance(e, PickUpEvent)]) == 1

    def test_a_body_grasped_twice_is_picked_up_twice(self):
        world = world_with_a_hand_and_a_cup()
        hand = world.get_body_by_name("hand")

        events = self.events_of(world, [hand, world.root, hand])

        assert [type(event) for event in events] == [
            PickUpEvent,
            PlacingEvent,
            PickUpEvent,
        ]

    def test_a_body_nobody_touches_is_neither(self):
        world = world_with_a_hand_and_a_cup()

        assert self.events_of(world, []) == []
