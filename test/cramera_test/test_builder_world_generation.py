"""
Generated worlds preserve authored poses and delegate robot setup to CRAM.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from cramera.model_catalog import ModelCatalog
from cramera.paths import WEB_ROOT
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from coraplex.execution_environment import simulated_robot_advanced

# %% complete generated demos


@pytest.fixture
def generated_demos() -> dict[str, str]:
    """
    Generate both demo styles with a captured object and installed robot metadata.
    """
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the browser generator")
    catalog = ModelCatalog.installed()
    scenario = {
        "robots": catalog.to_payload()["robots"],
        "objects": [
            {
                "id": "object",
                "name": "milk.stl",
                "mesh": "milk.stl",
                "x": 0,
                "y": 0,
                "z": 0,
                "color": "#ffffff",
            }
        ],
        "captured": {"milk.stl": [1.234, 2.345, 0.876, 0, 0, 0, 1]},
        "steps": [],
        "robotXY": {"x": 1.5, "y": 2.5},
        "selections": {
            "pb-robot": PR2.__name__,
            "pb-env": "apartment.urdf",
            "pb-name": "transport_demo",
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
def test_generated_world_delegates_robot_setup(
    generated_demos: dict[str, str], style: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Both styles use RobotSpecification's annotated spawn and localization frames.

    :param generated_demos: Real browser generator outputs.
    :param style: The script or demonstration-class output to inspect.
    :param monkeypatch: Fixture replacing environment parsing with a specification spy.
    """
    module = ast.parse(generated_demos[style])
    function = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"build_world", "build_simulated_world"}
    )
    create_specification = Mock()
    monkeypatch.setattr(WorldSpecification, "from_urdf", create_specification)
    namespace = {
        "WorldSpecification": WorldSpecification,
        "RobotSpecification": RobotSpecification,
        "HomogeneousTransformationMatrix": HomogeneousTransformationMatrix,
        "PR2": PR2,
        "os": os,
        "_WORLDS": "/worlds",
        "World": object,
        "ENV_FILE": "apartment.urdf",
        "ROBOT_XY": (1.5, 2.5),
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "generated_demo.py", "exec"
        ),
        namespace,
    )
    if style == "script":
        world = namespace[function.name]("apartment.urdf", (1.5, 2.5))
    else:
        demonstration = Mock(used_robot=PR2)
        world = namespace[function.name](demonstration)
    specification = create_specification.call_args.kwargs["robots"][0]
    assert specification.semantic_annotation_type is PR2
    assert specification.world_T_odom.to_position().to_np().tolist() == [
        1.5,
        2.5,
        0.0,
        1.0,
    ]
    assert world is create_specification.return_value.to_domain_object.return_value


def test_generated_demo_uses_the_captured_object_pose(
    generated_demos: dict[str, str],
) -> None:
    """
    The generated class's object specification contains the dragged position.

    :param generated_demos: Real browser generator outputs after capture.
    """
    module = ast.parse(generated_demos["class"])
    objects = next(
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "OBJECTS"
    )
    assert ast.literal_eval(objects)[0][1:4] == (1.234, 2.345, 0.876)


def test_generated_demos_enable_collision_avoidance(
    generated_demos: dict[str, str],
) -> None:
    """
    Both output styles enable collision avoidance without an explicit selection.

    :param generated_demos: Real browser generator outputs with default options.
    """
    script = ast.parse(generated_demos["script"])
    environment = next(
        node
        for node in ast.walk(script)
        if isinstance(node, ast.ImportFrom)
        and node.module == "coraplex.execution_environment"
    )
    assert [alias.name for alias in environment.names] == ["simulated_robot_advanced"]
    module = ast.parse(generated_demos["class"])
    collision_option = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.keyword) and node.arg == "collision_avoidance"
    )
    assert (
        ast.literal_eval(collision_option.value)
        is simulated_robot_advanced.collision_avoidance
    )


def test_generated_population_check_reuses_namespaced_lookup(
    generated_demos: dict[str, str],
) -> None:
    """
    A namespaced authored body must not be spawned a second time.

    :param generated_demos: Generated demonstration class from the real browser code.
    """
    world = World.create_with_root_body("map")
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=Body(name=PrefixedName("milk.stl", prefix="objects")),
            )
        )
    module = ast.parse(generated_demos["class"])
    function = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "is_scene_populated"
    )
    namespace = {"World": World, "OBJECTS": [("milk.stl",)]}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "generated_demo.py", "exec"
        ),
        namespace,
    )
    assert namespace[function.name](None, world) is True
