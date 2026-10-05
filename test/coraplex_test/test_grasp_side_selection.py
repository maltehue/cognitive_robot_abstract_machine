"""
Which side of an object the default grasp approaches from.

A robot that drives to the object and one that reaches from a fixed stand have to answer
this differently: the driving one has not arrived yet when the side is chosen, so its
current position must not decide it.
"""

from dataclasses import dataclass

import numpy as np
from typing_extensions import ClassVar

from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    AxisIdentifier,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import (
    GraspDescription,
    HasPreferredGraspAlignment,
    PreferredGraspAlignment,
)
from coraplex.view_manager import ViewManager
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tracy import Tracy, TracyLeftGripper
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% the object the robot is asked to grasp

OBJECT_BESIDE_THE_ROBOT = Pose.from_xyz_rpy(0.7, 2.0, 0.8)
"""
A pose well off to the robot's side, where the object's y axis, not its x axis, points
at a robot standing near the origin.
"""


def end_effector_of(world, robot_type, arm=Arms.LEFT):
    """
    One arm's end effector, as the grasp defaults are asked for it.

    :param world: The world holding the robot.
    :param robot_type: The robot's semantic annotation type.
    :param arm: Which arm to take the end effector of.
    """
    robot = world.get_semantic_annotations_by_type(robot_type)[0]
    return ViewManager.get_end_effector_view(arm, robot)


def pose_in(world, pose):
    """
    A pose expressed against the world's root, as the grasp defaults expect it.

    :param world: The world whose root the pose is expressed in.
    :param pose: The pose to re-reference.
    """
    return Pose(pose.to_position(), pose.to_quaternion(), reference_frame=world.root)


# %% a robot that drives to the object


class TestSideForADrivingRobot:
    """
    A mobile base navigates to the object after the side is chosen, so the side follows
    the object rather than the pose the robot happens to be standing in.
    """

    def test_the_object_beside_the_robot_is_still_approached_from_its_front(
        self, pr2_world_copy
    ):
        """
        Standing to the object's side, the side facing the robot is the object's y face.

        Choosing that one sends the robot around the object instead of to the front it
        could have driven to.
        """
        end_effector = end_effector_of(pr2_world_copy, PR2)

        grasp = GraspDescription.robot_relative_default(
            end_effector, pose_in(pr2_world_copy, OBJECT_BESIDE_THE_ROBOT)
        )

        assert grasp.approach_direction is ApproachDirection.FRONT
        assert grasp.vertical_alignment is VerticalAlignment.NoAlignment

    def test_a_rotated_object_is_approached_from_its_own_front(self, pr2_world_copy):
        """
        The side is the object's, so it turns with the object and the robot drives to
        wherever that is.
        """
        end_effector = end_effector_of(pr2_world_copy, PR2)
        turned_around = Pose.from_xyz_rpy(0.7, 2.0, 0.8, yaw=np.pi)

        grasp = GraspDescription.robot_relative_default(
            end_effector, pose_in(pr2_world_copy, turned_around)
        )

        assert grasp.approach_direction is ApproachDirection.FRONT

    def test_an_explicit_alignment_still_decides_the_side(self, pr2_world_copy):
        """
        A caller naming the axis to grasp along is answered from that axis, driving
        robot or not.
        """
        end_effector = end_effector_of(pr2_world_copy, PR2)
        along_the_y_axis = PreferredGraspAlignment(
            preferred_axis=AxisIdentifier.Y,
            with_vertical_alignment=False,
            with_rotated_gripper=False,
        )

        grasp = GraspDescription.robot_relative_default(
            end_effector,
            pose_in(pr2_world_copy, OBJECT_BESIDE_THE_ROBOT),
            grasp_alignment=along_the_y_axis,
        )

        assert grasp.approach_direction.axis is AxisIdentifier.Y


# %% a robot that reaches from where it stands


class TestSideForAStandingRobot:
    """
    A robot without a mobile base can only reach the side already facing it.
    """

    def test_the_side_facing_the_robot_is_approached(self, tracy_world):
        """
        The object lies off to the side, so the side facing this robot is its y face --
        the only one it can reach without a base to drive.
        """
        end_effector = end_effector_of(tracy_world, Tracy)

        grasp = GraspDescription.robot_relative_default(
            end_effector, pose_in(tracy_world, OBJECT_BESIDE_THE_ROBOT)
        )

        assert grasp.approach_direction.axis is AxisIdentifier.Y


# %% an end effector built for one alignment

FROM_ABOVE = PreferredGraspAlignment(
    preferred_axis=AxisIdentifier.Undefined,
    with_vertical_alignment=True,
    with_rotated_gripper=False,
)
"""
The alignment of an end effector that can only close over an object from above.
"""


@dataclass(eq=False)
class GripperThatGraspsOnlyFromAbove(TracyLeftGripper, HasPreferredGraspAlignment):
    """
    A gripper that declares the one alignment it can grasp with.
    """

    preferred_grasp_alignment: ClassVar[PreferredGraspAlignment] = FROM_ABOVE


def only_from_above(end_effector: TracyLeftGripper, monkeypatch) -> None:
    """
    Let the gripper declare, for the rest of the test, that it grasps only from above.
    """
    monkeypatch.setattr(end_effector, "__class__", GripperThatGraspsOnlyFromAbove)


FROM_ABOVE_WHATEVER_THE_HEIGHT = PreferredGraspAlignment(
    preferred_axis=AxisIdentifier.Undefined,
    with_vertical_alignment=True,
    with_rotated_gripper=False,
    vertical_face=VerticalAlignment.TOP,
)
"""
The alignment of an end effector that comes down from above even on an object standing
lower than the robot's root.
"""


class TestAlignmentOfAnEndEffectorBuiltForOne:
    """
    An end effector that can only grasp one way is planned with that alignment unless
    the caller names another.
    """

    def test_the_declared_alignment_is_used_when_the_caller_names_none(
        self, tracy_world, monkeypatch
    ):
        end_effector = end_effector_of(tracy_world, Tracy)
        pose = pose_in(tracy_world, OBJECT_BESIDE_THE_ROBOT)
        declared = GraspDescription.robot_relative_default(
            end_effector, pose, grasp_alignment=FROM_ABOVE
        )

        only_from_above(end_effector, monkeypatch)

        grasp = GraspDescription.robot_relative_default(end_effector, pose)

        assert grasp.approach_direction is declared.approach_direction
        assert grasp.vertical_alignment is declared.vertical_alignment

    def test_a_named_vertical_face_holds_whatever_the_robots_height(self, tracy_world):
        end_effector = end_effector_of(tracy_world, Tracy)
        root_height = float(end_effector._robot.root.global_pose.to_np()[2, 3])
        # An object well below the robot's root: judged by height it would be taken
        # from below.
        low = Pose.from_xyz_rpy(0.5, 0.0, root_height - 0.5, reference_frame=tracy_world.root)

        by_height = GraspDescription.robot_relative_default(
            end_effector, low, grasp_alignment=FROM_ABOVE
        )
        named = GraspDescription.robot_relative_default(
            end_effector, low, grasp_alignment=FROM_ABOVE_WHATEVER_THE_HEIGHT
        )

        assert by_height.vertical_alignment is VerticalAlignment.BOTTOM
        assert named.vertical_alignment is VerticalAlignment.TOP

    def test_an_alignment_the_caller_names_wins(self, tracy_world, monkeypatch):
        end_effector = end_effector_of(tracy_world, Tracy)
        pose = pose_in(tracy_world, OBJECT_BESIDE_THE_ROBOT)
        along_the_y_axis = PreferredGraspAlignment(
            preferred_axis=AxisIdentifier.Y,
            with_vertical_alignment=False,
            with_rotated_gripper=False,
        )
        named = GraspDescription.robot_relative_default(
            end_effector, pose, grasp_alignment=along_the_y_axis
        )

        only_from_above(end_effector, monkeypatch)

        grasp = GraspDescription.robot_relative_default(
            end_effector, pose, grasp_alignment=along_the_y_axis
        )

        assert grasp.approach_direction is named.approach_direction
        assert grasp.vertical_alignment is named.vertical_alignment
