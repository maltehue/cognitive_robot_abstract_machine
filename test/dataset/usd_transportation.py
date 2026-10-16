from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from typing_extensions import Any, TYPE_CHECKING
from trimesh import transform_points
from pxr import Gf, Kind, Usd, UsdGeom, UsdPhysics, UsdSemantics

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.composite.facing import FaceAndLookAtAction
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
    MoveAndPlaceAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.carrying import CarryAction
from coraplex.robot_plans.actions.core.navigation import (
    PathPlanningNavigateAction,
    FaceAtAction,
    LookAtAction,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.adapters.usd.semantics import UsdSupportLabel
from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    GripperState,
)
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.root_placement import RobotSupportPlacement
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    Floor,
    Table,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Point3,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

if TYPE_CHECKING:
    from coraplex.plans.plan_node import PlanNode
    from semantic_digital_twin.robots.robot_parts import Arm
    from semantic_digital_twin.world_description.world_entity import Body

# %% public synthetic stage


class ScenePath(StrEnum):
    """
    Objects and namespaces of the synthetic transportation scene.
    """

    ROOM = "/Room"
    FLOOR = "/Room/floor"
    SOURCE = "/Room/source"
    TARGET = "/Room/target"
    OBJECT = "/Room/object"
    CABINET = "/Room/cabinet"
    DOOR = "/Room/cabinet/door"
    JOINT = "/Room/cabinet/joint"
    VISUAL = "visual"
    COLLISION = "collision"
    TAXONOMY = "category"
    SCENE_FILE = "transportation.usda"
    ROBOT = "transport_robot"
    ENVIRONMENT = "transport_room"
    PLATFORM = "Platform_body_link"


class GroundContact(StrEnum):
    """
    Public wheel links selected to rest on the floor.
    """

    FRONT_LEFT_LEFT = "fl_caster_l_wheel_link"
    FRONT_LEFT_RIGHT = "fl_caster_r_wheel_link"
    FRONT_RIGHT_LEFT = "fr_caster_l_wheel_link"
    FRONT_RIGHT_RIGHT = "fr_caster_r_wheel_link"
    BACK_LEFT_LEFT = "bl_caster_l_wheel_link"
    BACK_LEFT_RIGHT = "bl_caster_r_wheel_link"
    BACK_RIGHT_LEFT = "br_caster_l_wheel_link"
    BACK_RIGHT_RIGHT = "br_caster_r_wheel_link"


@dataclass
class TransportationStage:
    """
    A floor, two worktops, a transport object and an articulated obstacle.
    """

    source: Pose
    """
    Authored starting object pose in stage coordinates.
    """

    target: Pose
    """
    Authored destination object pose in stage coordinates.
    """

    object_scale: Scale = field(default_factory=lambda: Scale(0.08, 0.08, 0.3))
    """
    An upright object tall enough to keep fingers above its worktop.
    """

    worktop_width: float = 0.1
    """
    Horizontal worktop extent with room for the upright object.
    """

    worktop_thickness: float = 0.05
    """
    Vertical extent of the supporting worktops.
    """

    def add_box(
        self,
        stage: Usd.Stage,
        path: ScenePath,
        scale: Scale,
        pose: Pose,
        label: str | None = None,
    ) -> Usd.Prim:
        """
        Author separate visible and guide-purpose collision geometry.
        """
        transform = UsdGeom.Xform.Define(stage, path)
        transform.AddTransformOp().Set(Gf.Matrix4d(*pose.to_np().T.flatten()))
        prim = transform.GetPrim()
        Usd.ModelAPI(prim).SetKind(Kind.Tokens.component)
        for name, purpose in (
            (ScenePath.VISUAL, UsdGeom.Tokens.default_),
            (ScenePath.COLLISION, UsdGeom.Tokens.guide),
        ):
            cube = UsdGeom.Cube.Define(stage, f"{path}/{name}")
            cube.CreateSizeAttr(1)
            cube.AddScaleOp().Set(
                Gf.Vec3d(float(scale.x), float(scale.y), float(scale.z))
            )
            cube.CreatePurposeAttr(purpose)
            if purpose == UsdGeom.Tokens.guide:
                UsdPhysics.CollisionAPI.Apply(
                    cube.GetPrim()
                ).CreateCollisionEnabledAttr(True)
        if label is not None:
            UsdSemantics.LabelsAPI.Apply(prim, ScenePath.TAXONOMY).CreateLabelsAttr(
                [label]
            )
        return prim

    def write(self, path: Path) -> None:
        """
        Export a metric Z-up stage with an explicitly anchored revolute joint.
        """
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        room = UsdGeom.Xform.Define(stage, ScenePath.ROOM)
        stage.SetDefaultPrim(room.GetPrim())
        self.add_box(
            stage,
            ScenePath.FLOOR,
            Scale(8, 8, 0.25),
            Pose.from_xyz_rpy(z=-0.125),
            UsdSupportLabel.FLOOR,
        )
        for name, object_pose in (
            (ScenePath.SOURCE, self.source),
            (ScenePath.TARGET, self.target),
        ):
            table_pose = Pose.from_xyz_rpy(
                float(object_pose.x),
                float(object_pose.y),
                float(object_pose.z)
                - float(self.object_scale.z) / 2
                - self.worktop_thickness / 2,
            )
            self.add_box(
                stage,
                name,
                Scale(self.worktop_width, self.worktop_width, self.worktop_thickness),
                table_pose,
                UsdSupportLabel.WORKBENCH,
            )
        self.add_box(stage, ScenePath.OBJECT, self.object_scale, self.source)
        self.add_box(
            stage,
            ScenePath.CABINET,
            Scale(0.3, 0.3, 1),
            Pose.from_xyz_rpy(x=2.5, y=-2.5, z=0.5),
        )
        self.add_box(
            stage, ScenePath.DOOR, Scale(0.05, 0.3, 1), Pose.from_xyz_rpy(x=0.175)
        )
        joint = UsdPhysics.RevoluteJoint.Define(stage, ScenePath.JOINT)
        joint.CreateBody0Rel().SetTargets([ScenePath.CABINET])
        joint.CreateBody1Rel().SetTargets([ScenePath.DOOR])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.175, 0, 0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateAxisAttr(UsdGeom.Tokens.z.upper())
        joint.CreateLowerLimitAttr(-90)
        joint.CreateUpperLimitAttr(90)
        stage.GetRootLayer().Save()


# %% live transport observations


@dataclass(eq=False)
class CarriedObjectSamples(StateChangeCallback):
    """
    Observe rigid attachment while the robot base moves.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The navigating robot.
    """

    object_body: Body = field(kw_only=True)
    """
    The body carried and released by the transport.
    """

    arm: Arm = field(kw_only=True)
    """
    The arm whose tool frame holds the object.
    """

    root_positions: list[np.ndarray] = field(default_factory=list, init=False)
    """
    World root positions recorded while the object is attached.
    """

    object_T_tool: list[np.ndarray] = field(default_factory=list, init=False)
    """
    Attachment transforms recorded alongside the root positions.
    """

    def on_state_change(self, **kwargs: Any) -> None:
        """
        Record only the states in which this end effector holds the object.
        """
        gripper = self.arm.end_effector
        if self.object_body.parent_kinematic_structure_entity is not gripper.tool_frame:
            return
        self.root_positions.append(
            self.robot.root.global_pose.position.to_np()[:3].copy()
        )
        self.object_T_tool.append(
            self._world.transform(
                gripper.tool_frame.global_transform, self.object_body
            ).to_np()
        )


# %% public robot transportation postures


@dataclass
class TransportationPosture:
    """
    Reachable carrying and grasp configuration for the public test robots.
    """

    worktop_distance: float = 0.45
    """
    Extra forward distance beyond the base radius.
    """

    source_lift: float = 0.15
    """
    Source object height above the parked tool frame.
    """

    carry_scale: Scale = field(default_factory=lambda: Scale(0.5, 0.5, 1))
    """
    Horizontal scaling of the source position in the robot frame.
    """

    carry_lift: float = 0.2
    """
    Carrying height above the starting object pose.
    """

    full_posture: bool = True
    """
    Whether the route requires clearance for all robot collision bodies.
    """

    upright_grasp: bool = False
    """
    Whether the grasp uses world axes instead of the parked hand orientation.
    """

    @classmethod
    def for_robot(cls, robot: AbstractRobot) -> TransportationPosture:
        """
        Select the public humanoid or two-finger compatibility configuration.
        """
        if isinstance(robot, Armar7):
            return cls()
        return cls(
            source_lift=0,
            carry_scale=Scale(0.7, 0.8, 1),
            carry_lift=0,
            full_posture=False,
            upright_grasp=True,
        )


# %% a transport assembled from core APIs


@dataclass
class TransportationScene:
    """
    Core transportation in an imported scene with an explicitly placed robot.
    """

    world: World
    """
    The imported stage and robot instance.
    """

    robot: AbstractRobot
    """
    The robot executing the transport.
    """

    graspable: Milk
    """
    The upright graspable scene object.
    """

    arm: Arm
    """
    The arm selected to carry the object.
    """

    target: Pose
    """
    The requested world-relative object release pose.
    """

    destination: Pose
    """
    The robot base pose at the destination worktop.
    """

    grasp: GraspCandidate
    """
    The object-relative grasp matched to the parked hand orientation.
    """

    posture: TransportationPosture
    """
    Grasp, carrying and navigation settings for the selected public robot.
    """

    support_bodies: list[Body]
    """
    Explicit public platform or wheel geometry in contact with the floor.
    """

    @staticmethod
    def select_supports(robot: AbstractRobot) -> list[Body]:
        """
        Select the public platform or all eight wheel contact bodies.
        """
        names = (
            [ScenePath.PLATFORM] if isinstance(robot, Armar7) else list(GroundContact)
        )
        return [
            robot._world.get_body_in_branch_by_name(robot.root, name) for name in names
        ]

    @classmethod
    def create(
        cls, directory: Path, robot_type: type[AbstractRobot]
    ) -> TransportationScene:
        """
        Describe the synthetic room using the public robot's reachable posture.
        """
        specification = RobotSpecification(
            semantic_annotation_type=robot_type, prefix=ScenePath.ROBOT
        )
        preview = World.create_with_root_body(ScenePath.ROOM)
        preview_robot = specification.spawn(preview)
        for arm in preview_robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(preview)
            arm.end_effector.get_joint_state_by_type(GripperState.OPEN).apply_to(
                preview
            )
        RobotSupportPlacement(
            preview_robot, cls.select_supports(preview_robot), 0
        ).apply()
        posture = TransportationPosture.for_robot(preview_robot)
        hand = preview_robot.all_arms[0].end_effector
        grasp_orientation = (
            Pose().orientation
            if posture.upright_grasp
            else (
                hand.tool_frame.global_pose.rotation_matrix @ hand.tool_R_grasp
            ).quaternion
        )
        source = hand.tool_frame.global_pose.position.to_np()[:3].copy()
        forward = preview_robot.mobile_base.forward_axis.to_np()[:3]
        clearance = preview_robot.mobile_base.base_radius + posture.worktop_distance
        source += forward * (clearance - np.dot(source, forward))
        source[2] += posture.source_lift
        travel = np.cross(np.array([0, 0, 1]), forward)
        target = source + travel
        path = directory / ScenePath.SCENE_FILE
        TransportationStage(
            Pose.from_xyz_rpy(*source), Pose.from_xyz_rpy(*target)
        ).write(path)
        world = WorldSpecification.from_usd_scene(
            str(path), prefix=ScenePath.ENVIRONMENT, robots=[specification]
        ).to_domain_object()
        robot = world.get_semantic_annotations_by_type(robot_type)[0]
        for arm in robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
            arm.end_effector.get_joint_state_by_type(GripperState.OPEN).apply_to(world)
        support_bodies = cls.select_supports(robot)
        RobotSupportPlacement(robot, support_bodies, 0).apply()
        arm = robot.all_arms[0]
        body = world.get_body_by_name(
            PrefixedName(ScenePath.OBJECT.lstrip("/"), ScenePath.ENVIRONMENT)
        )
        graspable = Milk(root=body)
        with world.modify_world():
            world.add_semantic_annotation(graspable)
        floor = world.get_semantic_annotations_by_type(Floor)[0]
        with world.modify_world():
            world.collision_manager.add_default_rule(
                AvoidExternalCollisions(robot=robot)
            )
            world.collision_manager.add_default_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=support_bodies,
                    body_group_b=floor.bodies_with_collision,
                )
            )
        world.collision_manager.update_collision_matrix()
        return cls(
            world,
            robot,
            graspable,
            arm,
            Pose.from_xyz_rpy(*target, reference_frame=world.root),
            Pose.from_xyz_rpy(*travel, reference_frame=world.root),
            GraspCandidate(
                graspable, Pose(orientation=grasp_orientation, reference_frame=body)
            ),
            posture,
            support_bodies,
        )

    def create_plan(self) -> PlanNode:
        """
        Execute pick-up, a carrying posture, continuous navigation and placement.
        """
        tables = self.world.get_semantic_annotations_by_type(Table)
        source_table, target_table = [
            next(
                table
                for table in tables
                if table.root.name
                == PrefixedName(path.lstrip("/"), ScenePath.ENVIRONMENT)
            )
            for path in (ScenePath.SOURCE, ScenePath.TARGET)
        ]
        face_pick = FaceAndLookAtAction(
            FaceAtAction(self.graspable.root.global_pose),
            LookAtAction(self.graspable.root.global_pose),
        )
        face_place = FaceAndLookAtAction(
            FaceAtAction(self.target), LookAtAction(self.target)
        )
        pick = MoveAndPickUpAction(
            PathPlanningNavigateAction(
                self.robot.root.global_pose, use_trajectory=True
            ),
            face_pick,
            PickUpAction(
                self.grasp,
                self.arm,
                allow_environment_contacts=False,
                support_bodies=source_table.bodies_with_collision,
                tolerate_grasp_stall=True,
            ),
        )
        hand = self.arm.end_effector
        hand_radius = max(
            np.linalg.norm(
                transform_points(
                    body.collision.combined_mesh.vertices,
                    self.world.transform(
                        body.global_transform, hand.tool_frame
                    ).to_np(),
                ),
                axis=1,
            ).max()
            for body in hand.bodies_with_collision
        )
        place = MoveAndPlaceAction(
            PathPlanningNavigateAction(
                self.destination,
                use_trajectory=True,
                include_robot_geometry=self.posture.full_posture,
            ),
            face_place,
            PlaceAction(
                self.graspable,
                self.target,
                allow_environment_contacts=False,
                support_bodies=target_table.bodies_with_collision,
                approach_clearance=hand_radius
                + HasApproachesGraspPoses.approach_clearance,
            ),
        )
        carry_position = self.world.transform(
            self.graspable.root.global_pose, self.robot.root
        ).position
        carry_position.x *= self.posture.carry_scale.x
        carry_position.y *= self.posture.carry_scale.y
        carry_position.z += self.posture.carry_lift
        carry = CarryAction(
            self.graspable,
            Pose(position=carry_position, reference_frame=self.robot.root),
        )
        return execute_single(
            TransportAction(pick, place, carry=carry),
            context=Context(self.world, self.robot),
        )
