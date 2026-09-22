"""
The initial Builder apartment position permits native arm preparation.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import subprocess

from lxml import html
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, TaskStatus
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramera.model_catalog import ModelCatalog
from cramera.paths import WEB_ROOT
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


# %% actual page defaults and authored overrides
@pytest.fixture
def builder_start() -> dict:
    """
    Read initial state and both generators from the real Builder page.
    """
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the browser generator")
    document = html.fromstring((WEB_ROOT / "plan_builder.html").read_text())
    scenario = {
        "robots": ModelCatalog.installed().to_payload()["robots"],
        "objects": [],
        "captured": {},
        "steps": [],
        "inspectStart": True,
        "selections": {
            "pb-robot": PR2.__name__,
            "pb-env": "apartment.urdf",
            "pb-name": "initial_park",
            **{
                identifier: document.get_element_by_id(identifier).get("value", "")
                for identifier in ("pb-rx", "pb-ry")
            },
        },
    }
    result = subprocess.run(
        [
            "node",
            str(Path(__file__).parent / "dataset" / "generate_builder_demo.js"),
            str(WEB_ROOT),
        ],
        input=json.dumps(scenario),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("style", ["script", "class"])
def test_initial_controls_and_generated_spawn_agree(
    builder_start: dict, style: str
) -> None:
    """
    Both generated demo styles use the initial position shown by the controls.

    :param builder_start: Actual initial page state and generated demos.
    :param style: Script or demonstration-class generator.
    """
    module = ast.parse(builder_start[style])
    if style == "script":
        assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "world"
                for target in node.targets
            )
        )
        position = ast.literal_eval(assignment.value.args[1])
    else:
        assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ROBOT_XY"
                for target in node.targets
            )
        )
        position = ast.literal_eval(assignment.value)
    initial = builder_start["start"]
    assert initial["inputs"] == initial["position"]
    assert position == (initial["position"]["x"], initial["position"]["y"])


# %% native PR2 apartment acceptance
def test_default_apartment_spawn_allows_parking(
    builder_start: dict, apartment_meshes: None
) -> None:
    """
    The actual default starts clear of furniture and completes collision-aware Park.

    :param builder_start: Actual initial page state and generated demos.
    :param apartment_meshes: Existing availability guard for apartment geometry.
    """
    position = builder_start["start"]["position"]
    world = WorldSpecification.from_urdf(
        str(
            Path(__file__).resolve().parents[2]
            / "coraplex"
            / "resources"
            / "worlds"
            / "apartment.urdf"
        ),
        robots=[
            RobotSpecification(
                semantic_annotation_type=PR2,
                world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(
                    position["x"], position["y"], 0
                ),
            )
        ],
    ).to_domain_object()
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    robot_bodies = set(robot.bodies_with_collision)
    world.collision_manager.update_collision_matrix()
    intersections = [
        contact
        for contact in world.collision_manager.compute_collisions().contacts
        if (contact.body_a in robot_bodies) != (contact.body_b in robot_bodies)
        and contact.distance < 0
    ]
    assert intersections == []
    plan = execute_single(ParkArmsAction(Arms.BOTH), context=Context(world, robot)).plan
    with simulated_robot_advanced:
        plan.perform()
    [park] = plan.get_nodes_by_designator_type(ParkArmsAction)
    assert park.status is TaskStatus.SUCCEEDED
