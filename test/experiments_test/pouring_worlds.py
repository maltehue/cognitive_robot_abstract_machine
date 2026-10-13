"""
Small worlds with a source cup tilted above a receiver, shared by the pouring tests.
"""

from experiments.pouring_model_verification.poured_stream import PouredStream
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

OPENING_RADIUS = 0.035
"""
The receiver opening's radius, in metres.
"""


def box_body(name: str) -> Body:
    return Body(
        name=PrefixedName(name),
        collision=ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix(),
                    scale=Scale(0.07, 0.07, 0.16),
                )
            ]
        ),
    )


def tilted_source_above_receiver(tilt: float) -> PouredStream:
    """
    A full source cup tilted about the world's y axis, held above and beside a receiver.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
    source_body, receiver_body = box_body("source"), box_body("receiver")
    with world.modify_world():
        world.add_body(source_body)
        world.add_body(receiver_body)
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=source_body,
                axis=Vector3(0, 1, 0),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=0.4
                ),
                dof_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=-2.0, velocity=-1.0),
                    upper=DerivativeMap(position=2.0, velocity=1.0),
                ),
            )
        )
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=receiver_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.1
                ),
            )
        )
    cups = []
    for body in (source_body, receiver_body):
        cup = HasFillLevel(name=PrefixedName(body.name.name), root=body)
        with world.modify_world():
            world.add_semantic_annotation(cup)
        cups.append(cup)
    source, receiver = cups
    source.initialize_fill_level(world=world, initial_fill=1.0)
    receiver.initialize_fill_level(world=world, initial_fill=0.0)
    JointState.from_mapping({source_body.parent_connection: tilt}).apply_to(world)
    return PouredStream(source=source, receiver=receiver, world=world)
