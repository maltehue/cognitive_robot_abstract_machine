"""
Exercise recording finalization after lazy mesh storage is first used.
"""

from unittest.mock import Mock, patch

from cramera.live.bridge import Bridge
from cramera.live.visualization import LiveVisualization
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk, Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.mesh_file_storage import MeshFileStorage
from semantic_digital_twin.world_description.world_entity import Body


def run_recording() -> None:
    """
    Leave a real tabletop recording for the process-exit safety net.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root")))
    bridge = Bridge()
    with patch("cramera.live.visualization.serve", return_value=Mock()), patch(
        "cramera.live.visualization.RosMarkerListener.start_if_available",
        return_value=None,
    ):
        LiveVisualization(world=world, bridge=bridge).start()
    with world.modify_world():
        Table.create_with_new_body_in_world(
            name="table", world=world, scale=Scale(2.0, 2.0, 0.1)
        )
        milk = Milk.create_with_new_body_in_world(
            name="milk",
            world=world,
            scale=Scale(0.08, 0.08, 0.2),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=3),
        )
        world.remove_connection(milk.root.parent_connection)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                parent=world.root, child=milk.root, world=world
            )
        )
    bridge.snapshot()
    bridge.recording.append(bridge.state)
    # A controller or semantic query may first export meshes after live startup.
    MeshFileStorage().allocate_directory()


if __name__ == "__main__":
    run_recording()
