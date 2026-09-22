"""
The action calls the Plan Builder writes into a generated demo.

The generated file hands these actions positional arguments, so their order is part of
the contract between the two packages: reordering a parameter here silently changes what
the generated demo does, and has to fail here rather than in somebody's demo run.
"""

import inspect

import pytest
from typing_extensions import ClassVar, List

from cramera.paths import WEB_ROOT

pytest.importorskip("coraplex", reason="coraplex not installed")

from coraplex.datastructures.grasp import GraspDescription  # noqa: E402
from coraplex.robot_plans.actions.core.pick_up import PickUpAction  # noqa: E402
from coraplex.robot_plans.actions.core.placing import PlaceAction  # noqa: E402
from semantic_digital_twin.robots.robot_parts import MobileBase  # noqa: E402
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase  # noqa: E402


def leading_parameters(action, count: int) -> List[str]:
    """
    The first parameters of an action, in the order a caller passes them positionally.

    :param action: The action class whose signature is read.
    :param count: How many leading parameters to name.
    """
    return list(inspect.signature(action).parameters)[:count]


# %% what a generated Pick / Place step calls


class TestGeneratedActionCalls:
    def test_pick_up_takes_the_object_then_the_arm_then_the_grasp(self):
        """
        Generated as ``PickUpAction(_pick_<id>, Arms.<arm>, _grasp_<id>)``.
        """
        assert leading_parameters(PickUpAction, 3) == [
            "object_designator",
            "arm",
            "grasp_description",
        ]

    def test_place_takes_the_object_then_the_target_then_the_arm(self):
        """
        Generated as ``PlaceAction(<body>, <target pose>, Arms.<arm>)``.
        """
        assert leading_parameters(PlaceAction, 3) == [
            "object_designator",
            "target_location",
            "arm",
        ]

    def test_the_default_grasp_takes_the_end_effector_the_pose_and_the_body(self):
        """
        Generated as ``GraspDescription.robot_relative_default(<end effector>, <pose>,
        <body>)``, leaving the side to approach from to the robot's reach.
        """
        assert leading_parameters(GraspDescription.robot_relative_default, 3) == [
            "end_effector",
            "pose",
            "body",
        ]


# %% what a generated demo writes to keep the base still


class TestGeneratedBaseControl:
    """
    A generated demo pins whole-body control by assigning the mobile base's own field,
    guarded by the mixin, so both have to be there to assign.
    """

    def test_the_mobile_base_carries_the_setting_that_is_assigned(self):
        """
        Generated as ``robot.mobile_base.full_body_controlled = ...``.
        """
        assert "full_body_controlled" in inspect.signature(MobileBase).parameters

    def test_the_guard_the_assignment_is_wrapped_in_exists(self):
        """
        Generated as ``if isinstance(robot, HasMobileBase):``, so a robot without a base
        is left alone.
        """
        assert isinstance(HasMobileBase, type)

    def test_a_robot_without_a_mobile_base_is_not_one(self):
        """
        The guard has to actually exclude such a robot, or the assignment raises.
        """
        from semantic_digital_twin.robots.tracy import Tracy

        assert not issubclass(Tracy, HasMobileBase)


# %% what the page offers


class TestPlanBuilderPalette:
    """
    The blocks the page has to offer for a plan that picks and places without a
    transport.
    """

    STEP_KINDS: ClassVar[List[str]] = ["transport", "pick", "place"]
    """
    Step kinds acting on a placed object, as ``core/plan_steps.js`` lists them.
    """

    def test_every_object_step_kind_is_an_offered_block(self):
        page_script = (WEB_ROOT / "plan_builder.js").read_text(encoding="utf-8")

        for kind in self.STEP_KINDS:
            assert "\n    %s: { name: " % kind in page_script, kind

    def test_the_step_kinds_are_the_ones_the_shared_module_names(self):
        """
        The page reads the kinds from ``core/plan_steps.js``; a kind added to one and
        not the other is a block whose object is never spawned.
        """
        module = (WEB_ROOT / "core" / "plan_steps.js").read_text(encoding="utf-8")

        assert (
            "const ACTS_ON_AN_OBJECT = ['%s']" % "', '".join(self.STEP_KINDS) in module
        )

    def test_the_page_loads_every_shared_module_it_reads(self):
        page = (WEB_ROOT / "plan_builder.html").read_text(encoding="utf-8")

        for module in ("plan_steps", "base_control", "execution_environment"):
            assert '<script src="core/%s.js">' % module in page, module

    def test_the_page_offers_the_base_control_choice(self):
        page = (WEB_ROOT / "plan_builder.html").read_text(encoding="utf-8")

        assert 'id="pb-base"' in page
