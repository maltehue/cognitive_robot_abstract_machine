"""
A reproducible stationary-robot semantic Transport rehearsal.
"""

from __future__ import annotations

import pytest

from cramera.semantic_transport_demo import SemanticTransportDemo
from coraplex.execution_environment import simulated_robot_advanced
from cramera.live.bridge import Bridge
from cramera.live.visualization import BridgePlanCallback
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.collision_checking.collision_matrix import CollisionCheck


def test_demo_enables_collision_avoidance() -> None:
    """
    The included rehearsal enables CRAM's collision avoidance by default.
    """
    assert SemanticTransportDemo(used_robot=Tracy).collision_avoidance is True


def test_demo_allows_only_the_intended_grasp_and_support_contacts() -> None:
    """
    Grasp and support contact retain avoidance against robot arms and the tabletop.
    """
    demonstration = SemanticTransportDemo(used_robot=Tracy)
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    robot = world.get_semantic_annotations_by_type(Tracy)[0]
    cube = world.get_body_by_name("box2.stl")
    surface = world.get_body_by_name("placement_table")
    gripper = robot.left_arm.end_effector.bodies_with_collision[0]
    other_gripper = robot.right_arm.end_effector.bodies_with_collision[0]
    forearm = world.get_body_by_name("left_forearm_link")
    manager = world.collision_manager
    manager.update_collision_matrix()
    checks = manager.collision_matrix.collision_checks

    assert CollisionCheck.create_for_bodies_with_collision(gripper, cube) not in checks
    assert CollisionCheck.create_for_bodies_with_collision(cube, surface) not in checks
    assert (
        CollisionCheck.create_for_bodies_with_collision(other_gripper, cube) in checks
    )
    assert CollisionCheck.create_for_bodies_with_collision(forearm, cube) in checks
    assert CollisionCheck.create_for_bodies_with_collision(gripper, surface) in checks
    assert (
        manager.get_buffer_zone_distance(gripper, surface)
        == demonstration.placement_clearance
    )
    assert manager.get_violated_distance(gripper, surface) == 0


def test_demo_places_on_the_named_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The actual CRAM controller places the cube inside the semantic tabletop.

    :param monkeypatch: Fixture selecting headless visualization for this simulation.
    """
    monkeypatch.setenv("CORAPLEX_VISUALIZATION", "none")
    demonstration = SemanticTransportDemo(used_robot=Tracy)
    world = demonstration.run()
    assert demonstration.is_scene_populated(world)
    position = world.get_body_by_name("box2.stl").global_pose.to_np()[:3, 3]
    assert 0.75 <= position[0] <= 0.85
    assert -0.05 <= position[1] <= 0.05
    assert position[2] == pytest.approx(1.02, abs=0.01)


def test_completed_transport_is_shown_as_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Disabled condition nodes cannot keep a completed Transport labelled running.

    :param monkeypatch: Fixture selecting headless simulated execution.
    """
    monkeypatch.setenv("CORAPLEX_VISUALIZATION", "none")
    demonstration = SemanticTransportDemo(used_robot=Tracy)
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    plan = demonstration.build_plan(demonstration.build_context(world))
    bridge = Bridge()
    bridge.attach(world)
    bridge.begin_plan(plan)
    plan.node_callbacks.append(BridgePlanCallback(bridge=bridge))
    with simulated_robot_advanced:
        plan.perform()
    [transport] = [
        node for node in bridge.plan_state.nodes if node.label == "TransportAction"
    ]
    assert transport.status == "SUCCEEDED"
