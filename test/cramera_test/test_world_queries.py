"""
Standard queries of attached semantic digital twin worlds.
"""

from __future__ import annotations

import pytest

from cramera.live.bridge import Bridge
from cramera.knowledge.queryable_knowledge import QueryScope
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

from .test_live_query import CurrentStateOnlySource


# %% attached world fixture
@pytest.fixture
def annotated_bridge(world_with_two_bodies: tuple[World, Body, Body]) -> Bridge:
    """
    Attach an annotated world without registering custom query code.

    :param world_with_two_bodies: Existing empty world and its unconnected bodies.
    """
    world, root, _ = world_with_two_bodies
    with world.modify_world():
        world.add_body(root)
        Handle.create_with_new_body_in_world(
            name="drawer_handle", world=world, scale=Scale(0.2, 0.04, 0.04)
        )
    bridge = Bridge()
    bridge.attach(world)
    return bridge


# %% automatic source
class TestAttachedWorldQueries:
    """
    Every query entry point reads the world attached to the bridge.
    """

    def test_attach_enables_queries(self, annotated_bridge: Bridge) -> None:
        """
        Publish query availability without a demo-specific hook.
        """
        assert annotated_bridge.status()["query"] is True
        assert annotated_bridge.query_scopes() == [QueryScope.CURRENT_STATE]

    def test_handles_come_from_attached_annotations(
        self, annotated_bridge: Bridge
    ) -> None:
        """
        EQL ranges over the actual handle annotations.
        """
        handles = annotated_bridge.world.get_semantic_annotations_by_type(Handle)
        result = annotated_bridge.run_query("an(entity(handle))")
        assert [row["__entity__"] for row in result.rows] == [
            str(handle.name) for handle in handles
        ]

    def test_question_matching_uses_world_presets(
        self, annotated_bridge: Bridge
    ) -> None:
        """
        Typed or transcribed questions resolve to an executable live preset.
        """
        matched = annotated_bridge.match_question("show all handles")
        assert matched.matched
        assert annotated_bridge.run_query(matched.preset.code).count == 1

    def test_query_reads_annotations_added_after_attach(
        self, annotated_bridge: Bridge
    ) -> None:
        """
        New semantic entities appear without registering the source again.
        """
        world = annotated_bridge.world
        with world.modify_world():
            Handle.create_with_new_body_in_world(
                name="second_handle", world=world, scale=Scale(0.1, 0.1, 0.1)
            )
        assert annotated_bridge.run_query("an(entity(handle))").count == 2

    def test_custom_source_survives_attachment(self, annotated_bridge: Bridge) -> None:
        """
        Explicit query extensions retain their established precedence.
        """
        source = CurrentStateOnlySource()
        bridge = Bridge()
        bridge.register_query_source(source)
        bridge.attach(annotated_bridge.world)
        assert bridge.query_source is source

    def test_reattach_replaces_the_previous_world(
        self, annotated_bridge: Bridge
    ) -> None:
        """
        An automatic source follows a new world without retaining old entities.
        """
        replacement = World.create_with_root_body("replacement")
        annotated_bridge.attach(replacement)
        assert annotated_bridge.run_query("an(entity(handle))").count == 0
