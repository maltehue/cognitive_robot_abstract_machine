"""
Multiple robot instances are assembled by the reusable scene API.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from cramera.model_catalog import ModelCatalog
from cramera.paths import WEB_ROOT

# %% generated source


@pytest.fixture
def multiple_robot_demos() -> dict[str, str]:
    """
    Generate both page output styles for two independently placed PR2 instances.
    """
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the browser generator")
    scenario = {
        "robots": ModelCatalog.installed().to_payload()["robots"],
        "instances": [
            {"model": "PR2", "x": 1.0, "y": 2.5, "yaw": 0.0},
            {"model": "PR2", "x": 4.0, "y": 2.0, "yaw": 1.57},
        ],
        "activeIdentifier": "robot_2",
        "objects": [],
        "captured": {},
        "steps": [{"type": "navigate", "params": {"x": 5, "y": 2, "z": 0, "yaw": 0}}],
        "selections": {
            "pb-robot": "PR2",
            "pb-env": "apartment.urdf",
            "pb-name": "two_robots",
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
def test_generated_scene_contains_all_instances(
    multiple_robot_demos: dict[str, str], style: str
) -> None:
    """
    Native scene assembly receives both namespaces and authored spawn orientations.

    :param multiple_robot_demos: Source emitted by the actual page generator.
    :param style: Generated representation under examination.
    """
    module = ast.parse(multiple_robot_demos[style])
    scene = next(
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ROBOT_SCENE"
            for target in node.targets
        )
    )
    assert isinstance(scene.func, ast.Name) and scene.func.id == "RobotScene"
    arguments = {keyword.arg: keyword.value for keyword in scene.keywords}
    assert ast.literal_eval(arguments["active_identifier"]) == "robot_2"
    instances = arguments["instances"].elts
    assert len(instances) == 2
    first = {keyword.arg: keyword.value for keyword in instances[0].keywords}
    second = {keyword.arg: keyword.value for keyword in instances[1].keywords}
    assert ast.literal_eval(first["identifier"]) == "robot_1"
    assert ast.literal_eval(second["identifier"]) == "robot_2"
    assert ast.literal_eval(second["label"]) == "PR2 2"
    assert [ast.literal_eval(argument) for argument in second["pose"].args] == [4, 2, 0]
    assert ast.literal_eval(second["pose"].keywords[0].value) == 1.57


@pytest.mark.parametrize("style", ["script", "class"])
def test_generated_context_selects_the_instance(
    multiple_robot_demos: dict[str, str], style: str
) -> None:
    """
    Repeated robot types are selected through their stable instance identifier.

    :param multiple_robot_demos: Source emitted by the actual page generator.
    :param style: Generated representation under examination.
    """
    module = ast.parse(multiple_robot_demos[style])
    methods = [
        node.func.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ROBOT_SCENE"
    ]
    assert methods.count("build_world") == 1
    assert methods.count("selected_robot") == 1
    assert "get_semantic_annotations_by_type" not in [
        node.func.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
