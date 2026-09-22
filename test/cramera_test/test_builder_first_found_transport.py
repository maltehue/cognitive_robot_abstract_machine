from __future__ import annotations

import numpy as np
import pytest

from coraplex.datastructures.enums import TaskStatus
from coraplex.demonstrations import RobotDemonstration
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from cramera.live.placement_surface import PlacementSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import BoundingBox

from .test_builder_transport import builder_transport_demo, builder_transport_scenario
from .test_mobile_transport_demo import CarryTrajectory


# %% automatic table selection
@pytest.mark.parametrize(
    "builder_transport_scenario", ["builder_first_found_transport.json"], indirect=True
)
def test_builder_finishes_transport_without_a_named_table(
    builder_transport_demo: RobotDemonstration,
    apartment_meshes: None,
) -> None:
    world = builder_transport_demo.build_simulated_world()
    builder_transport_demo.populate_scene(world)
    context = builder_transport_demo.build_context(world)
    plan = builder_transport_demo.build_plan(context)
    [transport] = plan.get_nodes_by_designator_type(TransportAction)
    provider = transport.designator.target_location
    assert isinstance(provider, PlacementSurface)
    assert provider.surface_name is None
    body = transport.designator.object_designator
    trajectory = CarryTrajectory(robot=context.robot, body=body)
    plan.node_callbacks.append(trajectory)

    with simulated_robot_advanced:
        plan.perform()

    assert transport.status is TaskStatus.SUCCEEDED
    [pickup] = plan.get_nodes_by_designator_type(PickUpAction)
    assert pickup.status is TaskStatus.SUCCEEDED
    placements = plan.get_nodes_by_designator_type(PlaceAction)
    [placement] = [node for node in placements if node.status is TaskStatus.SUCCEEDED]
    assert body.parent_connection.parent is world.root
    assert min(trajectory.avoidance_counts) > 0
    assert len(trajectory.carried_positions) > 1
    assert (
        np.linalg.norm(
            trajectory.carried_positions[-1] - trajectory.carried_positions[0]
        )
        > context.motion_tolerances.default_tcp_position_threshold
    )
    [surface] = [
        annotation
        for annotation in world.get_semantic_annotations_by_type(Table)
        if annotation.supporting_surface
        is placement.designator.target_location.reference_frame
    ]
    bounds = BoundingBox.from_mesh(
        body.combined_mesh, HomogeneousTransformationMatrix(reference_frame=body)
    )
    final_pose = world.transform(body.global_pose, surface.root)
    assert provider.supports_pose(surface, final_pose, bounds)
    resting_pose = provider.supported_pose(surface, final_pose, bounds)
    assert resting_pose is not None
    assert float(final_pose.z) == pytest.approx(
        float(resting_pose.z), abs=provider.support_tolerance
    )
