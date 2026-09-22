"""
Accept a generated semantic transport with a second robot in the same world.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest

from coraplex.datastructures.enums import TaskStatus
from coraplex.demonstrations import RobotDemonstration
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.motions.navigation import MoveMotion
from cramera.live.placement_surface import PlacementSurface
from cramera.model_catalog import ModelCatalog
from cramera.paths import WEB_ROOT
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import BoundingBox

from ..coraplex_test.test_multi_robot_motion import RobotIsolationTrajectory
from .test_mobile_transport_demo import CarryTrajectory


# %% normal page generation
@pytest.fixture
def multi_robot_transport_demo(monkeypatch: pytest.MonkeyPatch) -> RobotDemonstration:
    """
    Load the actual Builder's class with the second robot selected.

    :param monkeypatch: Register the generated module for native dataclass imports.
    :return: Unmodified generated demonstration using native CRAM execution.
    """
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the browser generator")
    dataset = Path(__file__).parent / "dataset"
    scenario = json.loads((dataset / "multi_robot_transport.json").read_text())
    scenario["robots"] = ModelCatalog.installed().to_payload()["robots"]
    result = subprocess.run(
        ["node", str(dataset / "generate_builder_demo.js"), str(WEB_ROOT)],
        input=json.dumps(scenario),
        capture_output=True,
        text=True,
        check=True,
    )
    source = json.loads(result.stdout)["class"]
    module = ModuleType("multi_robot_transport_acceptance")
    module.__file__ = str(
        Path(__file__).resolve().parents[2]
        / "coraplex"
        / "demos"
        / "coraplex_generated"
        / "multi_robot_transport_acceptance.py"
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, module.__file__, "exec"), vars(module))
    demonstration_type = next(
        item
        for item in vars(module).values()
        if isinstance(item, type)
        and issubclass(item, RobotDemonstration)
        and item is not RobotDemonstration
    )
    return demonstration_type(used_robot=PR2, collision_avoidance=True)


# %% full two-robot transport
def test_selected_second_robot_carries_and_places_while_first_stays_fixed(
    multi_robot_transport_demo: RobotDemonstration,
    apartment_meshes: None,
) -> None:
    """
    Transport addresses the selected instance and retains every idle body pose.

    :param multi_robot_transport_demo: Actual generated two-robot demonstration.
    :param apartment_meshes: Existing availability guard for environment geometry.
    """
    demonstration = multi_robot_transport_demo
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    context = demonstration.build_context(world)
    robots = world.get_semantic_annotations_by_type(PR2)
    assert len(robots) == 2
    assert context.robot.root.name.prefix == "robot_2"
    idle = next(robot for robot in robots if robot is not context.robot)
    initial_idle = np.asarray([body.global_pose.to_np() for body in idle.bodies])
    body = world.get_body_by_name("milk.stl")
    assert body.global_pose.to_np()[:3, 3].tolist() == [2.37, 2.1, 1.05]
    carrying = CarryTrajectory(robot=context.robot, body=body)
    isolation = RobotIsolationTrajectory(context.robot, idle)
    plan = demonstration.build_plan(context)
    plan.node_callbacks.extend([carrying, isolation])
    with simulated_robot_advanced:
        plan.perform()
    [transport] = plan.get_nodes_by_designator_type(TransportAction)
    assert transport.status is TaskStatus.SUCCEEDED
    navigation = plan.get_nodes_by_designator_type(MoveMotion)
    assert len(navigation) == 2
    assert all(node.status is TaskStatus.SUCCEEDED for node in navigation)
    carried = np.asarray(carrying.carried_positions)
    assert np.linalg.norm(carried[-1] - carried[0]) > 1.0
    positions = np.asarray(carrying.positions)
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.02
    assert isolation.avoidance_robots == {context.robot}
    for transforms in isolation.idle_transforms:
        np.testing.assert_allclose(transforms, initial_idle, atol=1e-12)
    assert body.parent_connection.parent is world.root
    surface = PlacementSurface(
        world, body, Table, surface_name="apartment/table_area_main"
    )
    [annotation] = surface.matching_surfaces()
    bounds = BoundingBox.from_mesh(
        body.combined_mesh, HomogeneousTransformationMatrix(reference_frame=body)
    )
    final_pose = world.transform(body.global_pose, annotation.root)
    assert surface.supports_pose(annotation, final_pose, bounds)
    resting_pose = surface.supported_pose(annotation, final_pose, bounds)
    assert resting_pose is not None
    assert float(final_pose.z) == pytest.approx(float(resting_pose.z), abs=0.005)
