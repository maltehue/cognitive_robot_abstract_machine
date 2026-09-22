"""
Execute the semantic transport authored through the normal Plan Builder.
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
from typing_extensions import Any

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

from .test_mobile_transport_demo import CarryTrajectory


# %% real browser generation
@pytest.fixture(params=["builder_transport.json"])
def builder_transport_scenario(request: pytest.FixtureRequest) -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "dataset" / request.param).read_text())


@pytest.fixture
def builder_transport_demo(
    monkeypatch: pytest.MonkeyPatch,
    builder_transport_scenario: dict[str, Any],
) -> RobotDemonstration:
    """
    Generate the captured Builder scene without altering its execution policy.

    :param monkeypatch: Register the generated module during its dataclass import.
    :return: The native demonstration produced by the page's class generator.
    """
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the browser generator")
    dataset = Path(__file__).parent / "dataset"
    scenario = builder_transport_scenario
    scenario["robots"] = ModelCatalog.installed().to_payload()["robots"]
    result = subprocess.run(
        ["node", str(dataset / "generate_builder_demo.js"), str(WEB_ROOT)],
        input=json.dumps(scenario),
        capture_output=True,
        text=True,
        check=True,
    )
    source = json.loads(result.stdout)["class"]
    module = ModuleType("builder_transport_acceptance")
    module.__file__ = str(
        Path(__file__).resolve().parents[2]
        / "coraplex"
        / "demos"
        / "coraplex_generated"
        / "builder_transport_acceptance.py"
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


# %% full native acceptance
def test_builder_transports_the_captured_object_to_its_semantic_surface(
    builder_transport_demo: RobotDemonstration,
    apartment_meshes: None,
) -> None:
    """
    The generated plan carries and releases the moved object with avoidance on.

    :param builder_transport_demo: Unmodified browser-generated demonstration.
    :param apartment_meshes: Existing availability guard for apartment geometry.
    """
    world = builder_transport_demo.build_simulated_world()
    builder_transport_demo.populate_scene(world)
    context = builder_transport_demo.build_context(world)
    body = world.get_body_by_name("milk.stl")
    initial_position = body.global_pose.to_np()[:3, 3].copy()
    scenario = json.loads(
        (Path(__file__).parent / "dataset" / "builder_transport.json").read_text()
    )
    assert initial_position.tolist() == scenario["captured"]["milk.stl"][:3]
    trajectory = CarryTrajectory(robot=context.robot, body=body)
    plan = builder_transport_demo.build_plan(context)
    plan.node_callbacks.append(trajectory)

    with simulated_robot_advanced:
        plan.perform()

    [transport] = plan.get_nodes_by_designator_type(TransportAction)
    assert transport.status is TaskStatus.SUCCEEDED
    navigations = plan.get_nodes_by_designator_type(MoveMotion)
    assert len(navigations) == 2
    assert all(node.status is TaskStatus.SUCCEEDED for node in navigations)
    positions = np.asarray(trajectory.positions)
    carried = np.asarray(trajectory.carried_positions)
    assert len(carried) > 1
    displacement = np.linalg.norm(
        body.global_pose.to_np()[:2, 3] - initial_position[:2]
    )
    assert np.linalg.norm(carried[-1] - carried[0]) > displacement / 2
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.02
    assert min(trajectory.avoidance_counts) > 0
    assert body.parent_connection.parent is world.root
    target = scenario["steps"][-1]["params"]["surfaceName"]
    surface = PlacementSurface(world, body, Table, surface_name=target)
    [annotation] = surface.matching_surfaces()
    bounds = BoundingBox.from_mesh(
        body.combined_mesh, HomogeneousTransformationMatrix(reference_frame=body)
    )
    final_pose = world.transform(body.global_pose, annotation.root)
    assert surface.supports_pose(annotation, final_pose, bounds)
    resting_pose = surface.supported_pose(annotation, final_pose, bounds)
    assert resting_pose is not None
    assert float(final_pose.z) == pytest.approx(float(resting_pose.z), abs=0.005)
