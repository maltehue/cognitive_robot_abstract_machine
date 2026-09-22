"""
Reproducible mobile-base navigation rehearsal.
"""

from cramera.navigation_demo import NavigationDemo, NavigationSceneBody
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

# %% scene and action


def test_navigation_demo_uses_the_normal_navigation_action(
    cylinder_bot_world: World,
) -> None:
    """
    The rehearsal uses the same navigation action as authored plans.

    :param cylinder_bot_world: Existing annotated robot with collision geometry.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    demo = NavigationDemo(used_robot=type(robot))
    assert demo.collision_avoidance is True
    assert demo.is_scene_populated(world) is False
    demo.populate_scene(world)
    assert demo.is_scene_populated(world) is True
    assert (
        world.get_body_by_name(NavigationSceneBody.BARRIER).global_pose.x
        == demo.distance / 2
    )
    context = demo.build_context(world)
    plan = demo.build_plan(context)
    assert context.robot is robot
    assert len(plan.get_nodes_by_designator_type(NavigateAction)) == 1
