"""
Live state follows external world updates without an executing plan.
"""

from __future__ import annotations

from cramera.live.bridge import Bridge
from cramera.live.visualization import WorldStateSync

from .test_live_transforms import make_kitchen


# %% externally driven joint state
class TestExternalWorldUpdates:
    """
    The world's own callbacks publish changes made outside a plan executor.
    """

    def test_joint_updates_reach_the_viewer_without_motion_ticks(self) -> None:
        """
        An external writer changes the actual world connection.
        """
        world, hinge = make_kitchen()
        bridge = Bridge()
        bridge.attach(world)
        synchronization = WorldStateSync(_world=world, bridge=bridge)

        hinge.position = 0.7

        assert bridge.get_state()["frames"][str(hinge.name)] == hinge.position
        assert bridge._tick_count == 0
        synchronization.stop()

    def test_stopped_visualization_receives_no_further_joint_updates(self) -> None:
        """
        Detachment removes the subscription to the external world.
        """
        world, hinge = make_kitchen()
        bridge = Bridge()
        bridge.attach(world)
        synchronization = WorldStateSync(_world=world, bridge=bridge)
        hinge.position = 0.7
        published = bridge.get_state()
        synchronization.stop()

        hinge.position = 1.1

        assert bridge.get_state() == published
