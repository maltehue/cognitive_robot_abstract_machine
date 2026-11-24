"""
The connections that join the kinematic structure entities of a world.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from typing_extensions import TYPE_CHECKING, Any, Self

from semantic_digital_twin.world_description.connection_properties import JointDynamics
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import (
    Connection,
    KinematicStructureEntity,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.types import NpMatrix4x4
from semantic_digital_twin.exceptions import MissingConnectionAxisError
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
    Quaternion,
)
import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import from_json, to_json
from krrood.patterns.called_positionally import called_positionally
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    FloatVariable,
    Scalar,
    VariableParameters,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    FillContext,
    InflowEquation,
    PouringEquation,
)
from semantic_digital_twin.world_description.exceptions import (
    MissingFillLevelLimitsError,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


class HasUpdateState(ABC):
    """
    Mixin class for connections that need state updated which are not trivial
    integrations.

    Typically needed for connections that use active and passive degrees of freedom.
    Look at OmniDrive for an example usage.
    """

    @abstractmethod
    def update_state(self, dt: float) -> None:
        """
        Allows the connection to update the state of its dofs.

        An integration update for active dofs will have happened before this method is
        called. Write directly into self._world.state, but don't touch dofs that don't
        belong to this connection.
        :param dt: Time passed since last update.
        """
        pass


@dataclass(eq=False)
class FixedConnection(Connection):
    """
    Has 0 degrees of freedom.
    """

    @classmethod
    def create_with_dofs(
        cls,
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: PrefixedName | None = None,
        parent_T_connection_expression: HomogeneousTransformationMatrix | None = None,
        connection_T_child_expression: HomogeneousTransformationMatrix | None = None,
    ) -> Self:
        return cls(
            parent=parent,
            child=child,
            name=name,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
        )


@dataclass(eq=False)
class ActiveConnection(Connection, ABC):
    """
    Has one or more degrees of freedom that can be actively controlled, e.g., robot
    joints.
    """

    @property
    def has_hardware_interface(self) -> bool:
        """
        Whether this connection is linked to a controller and can therefore respond to
        control commands.

        E.g. the caster wheels of a PR2 are active, because they have a DOF, but they
        are not directly controlled. Instead a the omni drive connection is directly
        controlled and a low level controller translates these commands to commands for
        the caster wheels.

        A door hinge is also active but cannot be controlled.
        """
        return any(dof.has_hardware_interface for dof in self.active_dofs)

    @property
    def is_controlled(self):
        return self.has_hardware_interface


@dataclass(eq=False)
class ActiveConnection1DOF(ActiveConnection, ABC):
    """
    Superclass for active connections with 1 degree of freedom.
    """

    axis: Vector3 = field(kw_only=True)
    """
    Connection moves along this unit vector, given in the frame of the parent.

    .. warning:: Set it only before the connection is added to a world: a later
        assignment is not recorded, published or rolled back. Replace the connection.
    """

    multiplier: float = 1.0
    """
    Movement along the axis is multiplied by this value, e.g. for shared DoFs.

    .. warning:: Set it only before the connection is added to a world: a later
        assignment is not recorded, published or rolled back. Replace the connection.
    """

    offset: float = 0.0
    """
    Movement along the axis is offset by this value, e.g. for shared DoFs.

    .. warning:: Set it only before the connection is added to a world: a later
        assignment is not recorded, published or rolled back. Replace the connection.
    """

    raw_dof: DegreeOfFreedom = field(kw_only=True)
    """
    The degree of freedom whose raw, unscaled state this connection drives.

    Use the :attr:`dof` property to obtain it with ``multiplier`` and ``offset``
    applied.
    """

    dynamics: JointDynamics = field(default_factory=JointDynamics)
    """
    Dynamic properties of the joint.
    """

    @classmethod
    def create_with_dofs(
        cls,
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: PrefixedName | None = None,
        parent_T_connection_expression: HomogeneousTransformationMatrix | None = None,
        connection_T_child_expression: HomogeneousTransformationMatrix | None = None,
        multiplier: float = 1.0,
        offset: float = 0.0,
        dof_limits: DegreeOfFreedomLimits | None = None,
        axis: Vector3 | None = None,
        dynamics: JointDynamics | None = None,
    ) -> Self:
        """
        Creates and returns an instance of the class with its single degree of freedom,
        initializing a kinematic relationship between a parent and a child entity along
        ``axis``.

        :param world: The motion world in which to add the degree of freedom.
        :param parent: The parent kinematic structure entity.
        :param child: The child kinematic structure entity.
        :param name: Optional specific name for the connection. If not provided, a
            default name is generated based on the parent and child.
        :param parent_T_connection_expression: Constant pose of the connection relative
            to its parent.
        :param connection_T_child_expression: Constant pose of the connection relative
            to its child.
        :param multiplier: A scaling factor applied to the DOF's motion. Defaults to
            1.0.
        :param offset: A constant offset value applied to the DOF's motion. Defaults to
            0.0.
        :param dof_limits: Optional limits for the generated degree of freedom.
        :param axis: The axis vector defining the joint relation.
        :param dynamics: The dynamic properties of the joint. Frictionless and undamped
            if None.
        :return: An instance of the class representing the defined relationship with its
            DOF added to the world.
        """
        if axis is None:
            raise MissingConnectionAxisError(connection_type_name=cls.__name__)
        name = name or cls._generate_default_name(parent=parent, child=child)
        dof = DegreeOfFreedom(name=PrefixedName("dof", str(name)), limits=dof_limits)
        world.add_degree_of_freedom(dof)
        connection = cls(
            name=name,
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
            axis=axis,
            multiplier=multiplier,
            offset=offset,
            raw_dof=dof,
            dynamics=dynamics or JointDynamics(),
        )
        return connection

    @property
    def dof(self) -> DegreeOfFreedom:
        """
        A reference to the Degree of Freedom associated with this connection.

        .. warning:: WITH multiplier and offset applied.
        """
        result = deepcopy(self.raw_dof)
        result.variables = self.raw_dof.variables * self.multiplier
        if self.multiplier < 0:
            # if multiplier is negative, we need to swap the limits
            result.limits.lower, result.limits.upper = (
                result.limits.upper,
                result.limits.lower,
            )
        result.limits.lower = result.limits.lower * self.multiplier
        result.limits.upper = result.limits.upper * self.multiplier

        result.variables.position += self.offset
        if result.limits.lower.position is not None:
            result.limits.lower.position = result.limits.lower.position + self.offset
        if result.limits.upper.position is not None:
            result.limits.upper.position = result.limits.upper.position + self.offset
        return result

    @property
    def active_dofs(self) -> list[DegreeOfFreedom]:
        return [self.raw_dof]

    @property
    def position(self) -> float:
        return (
            self._world.state[self.raw_dof.id].position * self.multiplier + self.offset
        )

    @position.setter
    def position(self, value: float) -> None:
        self._world.state[self.raw_dof.id].position = (
            value - self.offset
        ) / self.multiplier
        self._world.notify_state_change()

    @property
    def velocity(self) -> float:
        return self._world.state[self.raw_dof.id].velocity * self.multiplier

    @velocity.setter
    def velocity(self, value: float) -> None:
        self._world.state[self.raw_dof.id].velocity = value / self.multiplier
        self._world.notify_state_change()

    @property
    def acceleration(self) -> float:
        return self._world.state[self.raw_dof.id].acceleration * self.multiplier

    @acceleration.setter
    def acceleration(self, value: float) -> None:
        self._world.state[self.raw_dof.id].acceleration = value / self.multiplier
        self._world.notify_state_change()

    @property
    def jerk(self) -> float:
        return self._world.state[self.raw_dof.id].jerk * self.multiplier

    @jerk.setter
    def jerk(self, value: float) -> None:
        self._world.state[self.raw_dof.id].jerk = value / self.multiplier
        self._world.notify_state_change()


@dataclass(eq=False)
class PrismaticConnection(ActiveConnection1DOF):
    """
    Allows translation along an axis.
    """

    def add_to_world(self, *, world: World):
        super().add_to_world(world=world)

        translation_axis = self.axis * self.dof.variables.position
        self._kinematics = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=translation_axis[0],
            y=translation_axis[1],
            z=translation_axis[2],
            child_frame=self.child,
        )


@dataclass(eq=False)
class RevoluteConnection(ActiveConnection1DOF):
    """
    Allows rotation about an axis.
    """

    def add_to_world(self, *, world: World):
        super().add_to_world(world=world)

        self._kinematics = HomogeneousTransformationMatrix.from_xyz_axis_angle(
            axis=self.axis,
            angle=self.dof.variables.position,
            child_frame=self.child,
        )


@dataclass(eq=False)
class ScrewConnection(ActiveConnection1DOF):
    """
    A screw pair: couples rotation about ``axis`` with translation along it into a
    single degree of freedom.

    Increasing the degree of freedom's position rotates the child counterclockwise about
    ``axis`` (right-hand rule) and translates it along ``axis`` by ``screw_pitch *
    position / (2 * pi)``, i.e. one full revolution advances the child by one
    ``screw_pitch``. A right-handed thread whose ``axis`` points from the parent toward
    the child therefore has a positive ``screw_pitch``: driving the degree of freedom
    toward its upper limit unscrews the child.
    """

    screw_pitch: float = field(kw_only=True)
    """
    The distance between adjacent threads, measured parallel to ``axis`` in meters.

    Assumes a single-start thread, where the child advances one ``screw_pitch`` along
    ``axis`` per full revolution. See
    https://wellfastener.com/blog/what-is-screw-pitch%EF%BC%9Fscrew-pitch-vs-lead/
    for
    the distinction between screw pitch and lead. Negative values model left-handed
    threads.

    .. warning:: Set it only before the connection is added to a world: a later
        assignment is not recorded, published or rolled back. Replace the connection.
    """

    def add_to_world(self, *, world: World):
        super().add_to_world(world=world)

        angle = self.dof.variables.position
        translation_axis = self.axis * (self.screw_pitch * angle / (2 * np.pi))
        self._kinematics = HomogeneousTransformationMatrix.from_xyz_axis_angle(
            x=translation_axis[0],
            y=translation_axis[1],
            z=translation_axis[2],
            axis=self.axis,
            angle=angle,
            child_frame=self.child,
        )

    def rotation_angle_for_travel_distance(self, travel_distance: float) -> float:
        """
        The rotation of the degree of freedom that translates the child by
        ``travel_distance`` along ``axis``.

        Allows expressing joint goals as travel distances, e.g. how far a cap should
        move along its thread.

        :param travel_distance: Signed translation along ``axis`` in meters.
        :return: The signed rotation angle in radians.
        """
        return travel_distance * 2 * np.pi / self.screw_pitch

    @classmethod
    def create_with_dofs(
        cls,
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: PrefixedName | None = None,
        parent_T_connection_expression: HomogeneousTransformationMatrix | None = None,
        connection_T_child_expression: HomogeneousTransformationMatrix | None = None,
        multiplier: float = 1.0,
        offset: float = 0.0,
        dof_limits: DegreeOfFreedomLimits | None = None,
        axis: Vector3 | None = None,
        dynamics: JointDynamics | None = None,
        screw_pitch: float,
    ) -> Self:
        """
        Creates and returns a screw connection with its single degree of freedom.

        See :meth:`ActiveConnection1DOF.create_with_dofs`; additionally requires the
        screw's ``screw_pitch``.

        :param screw_pitch: The distance between adjacent threads along ``axis`` in
            meters.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)
        dof = DegreeOfFreedom(name=PrefixedName("dof", str(name)), limits=dof_limits)
        world.add_degree_of_freedom(dof)
        connection = cls(
            name=name,
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
            axis=axis,
            multiplier=multiplier,
            offset=offset,
            raw_dof=dof,
            screw_pitch=screw_pitch,
            dynamics=dynamics or JointDynamics(),
        )
        return connection


@dataclass(eq=False)
class Connection6DoF(Connection):
    """
    Has full 6 degrees of freedom, that cannot be actively controlled.

    Useful for synchronizing with transformations from external providers.
    """

    x: DegreeOfFreedom = field(kw_only=True)
    """
    Displacement of child KinematicStructureEntity with respect to parent
    KinematicStructureEntity along the x-axis.
    """

    y: DegreeOfFreedom = field(kw_only=True)
    """
    Displacement of child KinematicStructureEntity with respect to parent
    KinematicStructureEntity along the y-axis.
    """

    z: DegreeOfFreedom = field(kw_only=True)
    """
    Displacement of child KinematicStructureEntity with respect to parent
    KinematicStructureEntity along the z-axis.
    """

    qx: DegreeOfFreedom = field(kw_only=True)
    qy: DegreeOfFreedom = field(kw_only=True)
    qz: DegreeOfFreedom = field(kw_only=True)
    qw: DegreeOfFreedom = field(kw_only=True)
    """
    Rotation of child KinematicStructureEntity with respect to parent
    KinematicStructureEntity represented as a quaternion.
    """

    def add_to_world(self, *, world: World):
        super().add_to_world(world=world)
        parent_P_child = Point3(
            x=self.x.variables.position,
            y=self.y.variables.position,
            z=self.z.variables.position,
        )
        parent_R_child = Quaternion(
            x=self.qx.variables.position,
            y=self.qy.variables.position,
            z=self.qz.variables.position,
            w=self.qw.variables.position,
        ).rotation_matrix
        self._kinematics = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=parent_P_child,
            rotation_matrix=parent_R_child,
            child_frame=self.child,
        )

    @classmethod
    def create_with_dofs(
        cls,
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: PrefixedName | None = None,
        parent_T_connection_expression: HomogeneousTransformationMatrix | None = None,
        connection_T_child_expression: HomogeneousTransformationMatrix | None = None,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated degrees of freedom
        (DoFs) for the provided parent and child kinematic entities within the specified
        world.

        This method initializes and adds the required degrees of freedom to the world,
        and sets their properties accordingly. It generates a name for the connection if
        none is provided, and ensures valid initial state for relevant degrees of freedom.

        :param world: The World object where the degrees of freedom are added and modified.
        :param parent: The KinematicStructureEntity serving as the parent.
        :param child: The KinematicStructureEntity serving as the child.
        :param name: An optional PrefixedName for the connection. If None, it will be
                     auto-generated based on the parent and child names.
        :param parent_T_connection_expression: Optional transformation matrix specifying
                                               the connection relationship between parent
                                               and the connection.
        :param connection_T_child_expression: Optional transformation matrix specifying
                                              the connection relationship between the connection
                                              and its child entities
        :return: A new instance of the class representing the parent-child connection with
                 automatically defined degrees of freedom.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)

        stringified_name = str(name)
        x = DegreeOfFreedom(name=PrefixedName("x", stringified_name))
        world.add_degree_of_freedom(x)
        y = DegreeOfFreedom(name=PrefixedName("y", stringified_name))
        world.add_degree_of_freedom(y)
        z = DegreeOfFreedom(name=PrefixedName("z", stringified_name))
        world.add_degree_of_freedom(z)
        qx = DegreeOfFreedom(name=PrefixedName("qx", stringified_name))
        world.add_degree_of_freedom(qx)
        qy = DegreeOfFreedom(name=PrefixedName("qy", stringified_name))
        world.add_degree_of_freedom(qy)
        qz = DegreeOfFreedom(name=PrefixedName("qz", stringified_name))
        world.add_degree_of_freedom(qz)
        qw = DegreeOfFreedom(name=PrefixedName("qw", stringified_name))
        world.add_degree_of_freedom(qw)
        world.state[qw.id].position = 1.0

        return cls(
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
            name=name,
            x=x,
            y=y,
            z=z,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
        )

    @property
    def passive_dofs(self) -> list[DegreeOfFreedom]:
        return [self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw]

    @property
    def origin(self) -> HomogeneousTransformationMatrix:
        return super().origin

    @origin.setter
    def origin(self, transformation: HomogeneousTransformationMatrix) -> None:
        """
        Set this connection's parent-to-child origin.

        The origin is ``parent_T_connection_expression @ _kinematics @
        connection_T_child_expression``, of which only ``_kinematics`` is DOF-backed;
        the other two are fixed at connection creation. So ``transformation`` is first
        converted into the parent frame and then un-composed with those constants, to
        find the DOF state that makes the *resulting* origin equal ``transformation``,
        rather than writing ``transformation`` in as ``_kinematics`` directly.

        The degree-of-freedom writes and the notification they trigger are held
        together under ``World._world_lock``. They form one logical pose
        change, and an observer that reads or writes the state between them --
        such as a running physics simulator syncing its own values back into
        the world -- would otherwise see, and be able to overwrite, a pose that
        is only half applied.

        :param transformation: The desired parent-to-child origin. Must carry a
            reference frame (:meth:`World.transform` raises
            :class:`~krrood_spatial_math.exceptions.MissingReferenceFrameError`
            otherwise); does not need to already be expressed in the parent frame. Other
            spatial types (e.g. ``Pose``) must be converted with their own
            ``homogeneous_matrix`` before being assigned here.
        """
        local_kinematics = self._calculate_local_kinematics(
            self._world.transform(transformation, self.parent)
        )
        position = local_kinematics.position
        orientation = local_kinematics.rotation_matrix.quaternion
        with self._world._world_lock:
            self._world.state[self.x.id].position = position[0]
            self._world.state[self.y.id].position = position[1]
            self._world.state[self.z.id].position = position[2]
            self._world.state[self.qx.id].position = orientation[0]
            self._world.state[self.qy.id].position = orientation[1]
            self._world.state[self.qz.id].position = orientation[2]
            self._world.state[self.qw.id].position = orientation[3]
            self._world.notify_state_change()


@dataclass(eq=False)
class WheeledDrive(ActiveConnection, HasUpdateState, ABC):
    """
    Superclass for connections that describe a drive, e.g., an omnidirectional drive or
    a differential drive.
    """

    x: DegreeOfFreedom = field(kw_only=True)
    """
    Passive DoFs describing the measured odometry in x with respect to parent frame.
    """

    y: DegreeOfFreedom = field(kw_only=True)
    """
    Passive DoFs describing the measured odometry in y with respect to parent frame.
    """

    roll: DegreeOfFreedom = field(kw_only=True)
    """
    Passive DoF describing the measured odometry in roll using the IMU sensor.
    """

    pitch: DegreeOfFreedom = field(kw_only=True)
    """
    Passive DoF describing the measured odometry in pitch using the IMU sensor.
    """

    yaw: DegreeOfFreedom = field(kw_only=True)
    """
    Active DoF describing rotation around the robot's z-axis.
    """

    @property
    def origin(self) -> HomogeneousTransformationMatrix:
        return super().origin

    @origin.setter
    def origin(
        self, transformation: NpMatrix4x4 | HomogeneousTransformationMatrix
    ) -> None:
        """
        Overwrites the origin of the connection.

        The origin is ``parent_T_connection_expression @ _kinematics @
        connection_T_child_expression``, of which only ``_kinematics`` is degree of
        freedom backed, so ``transformation`` is un-composed with the other two to find
        the drive's state that produces it.

        .. warning:: Ignores z position, pitch, and roll values, because a drive cannot
            reach them.

        The degree-of-freedom writes and the notification they trigger are held
        together under ``World._world_lock``. They form one logical pose
        change, and an observer that reads or writes the state between them --
        such as a running physics simulator syncing its own values back into
        the world -- would otherwise see, and be able to overwrite, a pose that
        is only half applied.

        :param transformation: The desired parent-to-child origin.
        """
        local_kinematics = self._calculate_local_kinematics(transformation)
        position = local_kinematics.position
        roll, pitch, yaw = local_kinematics.rotation_matrix.rpy
        with self._world._world_lock:
            self._world.state[self.x.id].position = position.x
            self._world.state[self.y.id].position = position.y
            self._world.state[self.yaw.id].position = yaw
            self._world.notify_state_change()


@dataclass(eq=False)
class OmniDrive(WheeledDrive):
    """
    A connection describing an omnidirectional drive.

    It can rotate about its z-axis and drive on the x-y plane simultaneously.
    - x/y: Passive dofs describing the measured odometry with respect to parent frame.
        We assume that the robot can't fly, and we can't measure its z-axis position, so z=0.
        The odometry sensors typically provide velocity measurements with respect to the child frame,
        therefore the velocity values of x/y must stay 0.
    - x_vel/y_vel: The measured and commanded velocity is represented with respect to the child frame with these
        active dofs. It must be ensured that their position values stay 0.
    - roll/pitch: Some robots, like the PR2, have sensors to measure pitch and roll using an IMU,
        we therefore have passive dofs for them.
    - yaw: Since the robot can only rotate about its z-axis, we don't need different dofs for position and velocity of yaw.
        They are combined into one active dof.
    """

    x_velocity: DegreeOfFreedom = field(kw_only=True)
    """
    Active DoF describing the measured and commanded velocity in x.

    Represented with respect to the child frame.
    """

    y_velocity: DegreeOfFreedom = field(kw_only=True)
    """
    Active DoF describing the measured and commanded velocity in y.

    Represented with respect to the child frame.
    """

    def add_to_world(self, *, world: World):
        super().add_to_world(world=world)
        odom_T_bf = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.x.variables.position,
            y=self.y.variables.position,
            yaw=self.yaw.variables.position,
        )
        bf_T_bf_vel = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.x_velocity.variables.position, y=self.y_velocity.variables.position
        )
        bf_vel_T_bf = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0,
            y=0,
            z=0,
            roll=self.roll.variables.position,
            pitch=self.pitch.variables.position,
            yaw=0,
        )
        self._kinematics = odom_T_bf @ bf_T_bf_vel @ bf_vel_T_bf
        self._kinematics.child_frame = self.child

    @classmethod
    def create_with_dofs(
        cls,
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: PrefixedName | None = None,
        parent_T_connection_expression: HomogeneousTransformationMatrix | None = None,
        connection_T_child_expression: HomogeneousTransformationMatrix | None = None,
        translation_velocity_limits: float = 0.6,
        rotation_velocity_limits: float = 0.5,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated degrees of freedom
        (DOFs) for translation on the x and y axes, rotation along roll, pitch, and yaw
        axes, and velocity limits for translation and rotation.

        This method modifies the provided world to add all required degrees of freedom
        and their limits, based on the provided settings. Names for the degrees of
        freedom are auto-generated using the stringified version of the provided name or
        its default setting.

        :param world: The world where the configuration is being applied, and degrees of
            freedom are added.
        :param parent: The parent kinematic structure entity.
        :param child: The child kinematic structure entity.
        :param name: Name of the connection. If None, it will be auto-generated.
        :param parent_T_connection_expression: Transformation matrix representing the
            relative position/orientation of the child to the parent. Default is
            Identity.
        :param connection_T_child_expression: Transformation matrix representing the
            relative position/orientation of the child to the connection. Default is
            Identity.
        :param translation_velocity_limits: The velocity limit applied to the
            translation degrees of freedom (default is 0.6).
        :param rotation_velocity_limits: The velocity limit applied to the rotation
            degrees of freedom (default is 0.5).
        :return: An instance of the class with the auto-generated DOFs incorporated.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)
        stringified_name = str(name)

        x = DegreeOfFreedom(name=PrefixedName("x", stringified_name))
        world.add_degree_of_freedom(x)
        y = DegreeOfFreedom(name=PrefixedName("y", stringified_name))
        world.add_degree_of_freedom(y)
        roll = DegreeOfFreedom(name=PrefixedName("roll", stringified_name))
        world.add_degree_of_freedom(roll)
        pitch = DegreeOfFreedom(name=PrefixedName("pitch", stringified_name))
        world.add_degree_of_freedom(pitch)
        yaw = DegreeOfFreedom(
            name=PrefixedName("yaw", stringified_name),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                maximum_speed=rotation_velocity_limits
            ),
        )
        world.add_degree_of_freedom(yaw)

        x_vel = DegreeOfFreedom(
            name=PrefixedName("x_vel", stringified_name),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                maximum_speed=translation_velocity_limits
            ),
        )
        world.add_degree_of_freedom(x_vel)
        y_vel = DegreeOfFreedom(
            name=PrefixedName("y_vel", stringified_name),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                maximum_speed=translation_velocity_limits
            ),
        )
        world.add_degree_of_freedom(y_vel)

        return cls(
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
            name=name,
            x=x,
            y=y,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            x_velocity=x_vel,
            y_velocity=y_vel,
        )

    @property
    def active_dofs(self) -> list[DegreeOfFreedom]:
        return [self.x_velocity, self.y_velocity, self.yaw]

    @property
    def passive_dofs(self) -> list[DegreeOfFreedom]:
        return [self.x, self.y, self.roll, self.pitch]

    def update_state(self, dt: float) -> None:
        state = self._world.state
        state[self.x_velocity.id].position = 0
        state[self.y_velocity.id].position = 0

        x_vel = state[self.x_velocity.id].velocity
        y_vel = state[self.y_velocity.id].velocity
        delta = state[self.yaw.id].position
        x_velocity = np.cos(delta) * x_vel - np.sin(delta) * y_vel
        state[self.x.id].position += x_velocity * dt
        y_velocity = np.sin(delta) * x_vel + np.cos(delta) * y_vel
        state[self.y.id].position += y_velocity * dt

    def get_free_variable_names(self) -> list[UUID]:
        return [self.x.id, self.y.id, self.yaw.id]


@dataclass(eq=False)
class DifferentialDrive(WheeledDrive):
    """
    A connection describing a differential drive.

    It can rotate around its z-axis and drive in x-direction. It allows movement in the
    x-y plane.
    """

    x_velocity: DegreeOfFreedom = field(kw_only=True)
    """
    Active DoF describing the measured and commanded velocity in x.

    Represented with respect to the child frame.
    """

    def add_to_world(self, *, world: World):
        super().add_to_world(world=world)
        odom_T_bf = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.x.variables.position,
            y=self.y.variables.position,
            yaw=self.yaw.variables.position,
        )
        bf_T_bf_vel = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.x_velocity.variables.position,
        )
        bf_vel_T_bf = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0,
            y=0,
            z=0,
            roll=self.roll.variables.position,
            pitch=self.pitch.variables.position,
            yaw=0,
        )
        self._kinematics = odom_T_bf @ bf_T_bf_vel @ bf_vel_T_bf
        self._kinematics.child_frame = self.child

    @classmethod
    def create_with_dofs(
        cls,
        *,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: PrefixedName | None = None,
        parent_T_connection_expression: HomogeneousTransformationMatrix | None = None,
        connection_T_child_expression: HomogeneousTransformationMatrix | None = None,
        translation_velocity_limits: float = 0.6,
        rotation_velocity_limits: float = 0.5,
    ) -> Self:
        """
        Creates an instance of the class with automatically generated DoFs for
        translation on the x-axis, rotation along roll, pitch, and yaw axes, and
        velocity limits for translation and rotation.

        :param world: The world where the configuration is being applied, and degrees of
            freedom are added.
        :param parent: The parent kinematic structure entity.
        :param child: The child kinematic structure entity.
        :param name: Name of the connection. If None, it will be auto-generated.
        :param parent_T_connection_expression: Transformation matrix representing the
            relative position/orientation of the child to the parent. Default is
            Identity.
        :param connection_T_child_expression: Transformation matrix representing the
            relative position/orientation of the child to the connection. Default is
            Identity.
        :param translation_velocity_limits: The velocity limit applied to the
            translation degrees of freedom (default is 0.6).
        :param rotation_velocity_limits: The velocity limit applied to the rotation
            degrees of freedom (default is 0.5).
        :return: An instance of the class with the auto-generated DOFs incorporated.
        """
        name = name or cls._generate_default_name(parent=parent, child=child)
        stringified_name = str(name)

        x = DegreeOfFreedom(name=PrefixedName("x", stringified_name))
        world.add_degree_of_freedom(x)
        y = DegreeOfFreedom(name=PrefixedName("y", stringified_name))
        world.add_degree_of_freedom(y)
        roll = DegreeOfFreedom(name=PrefixedName("roll", stringified_name))
        world.add_degree_of_freedom(roll)
        pitch = DegreeOfFreedom(name=PrefixedName("pitch", stringified_name))
        world.add_degree_of_freedom(pitch)
        yaw = DegreeOfFreedom(
            name=PrefixedName("yaw", stringified_name),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                maximum_speed=rotation_velocity_limits
            ),
        )
        world.add_degree_of_freedom(yaw)

        x_vel = DegreeOfFreedom(
            name=PrefixedName("x_vel", stringified_name),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                maximum_speed=translation_velocity_limits
            ),
        )
        world.add_degree_of_freedom(x_vel)

        return cls(
            parent=parent,
            child=child,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
            name=name,
            x=x,
            y=y,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            x_velocity=x_vel,
        )

    @property
    def active_dofs(self) -> list[DegreeOfFreedom]:
        return [self.x_velocity, self.yaw]

    @property
    def passive_dofs(self) -> list[DegreeOfFreedom]:
        return [self.x, self.y, self.roll, self.pitch]

    def update_state(self, dt: float) -> None:
        state = self._world.state
        state[self.x_velocity.id].position = 0

        x_vel = state[self.x_velocity.id].velocity
        delta = state[self.yaw.id].position
        x_velocity = np.cos(delta) * x_vel
        state[self.x.id].position += x_velocity * dt
        y_velocity = np.sin(delta) * x_vel
        state[self.y.id].position += y_velocity * dt

    def get_free_variable_names(self) -> list[UUID]:
        return [self.x.id, self.y.id, self.yaw.id]


# %% liquid connection


@dataclass(kw_only=True)
class CompiledFillVelocity:
    """
    The fill velocity of a liquid connection compiled once for the equations and world
    model it was built from.
    """

    outflow_equation: PouringEquation | None
    """
    The drain the function was compiled from.
    """

    inflow_equation: InflowEquation | None
    """
    The inflow the function was compiled from.
    """

    model_version: int
    """
    The world model version the function was compiled at.
    """

    variables: list[FloatVariable]
    """
    The free variables the function reads, in the order it takes them.
    """

    function: CompiledFunction
    """
    The compiled fill velocity.
    """

    def is_compiled_for(
        self,
        *,
        outflow_equation: PouringEquation | None,
        inflow_equation: InflowEquation | None,
        model_version: int,
    ) -> bool:
        """
        :return: Whether the function was compiled from these equations at this world
            model version.
        """
        return (
            self.outflow_equation is outflow_equation
            and self.inflow_equation is inflow_equation
            and self.model_version == model_version
        )

    def evaluate(self) -> float:
        """
        :return: The fill velocity at the current values of the variables.
        """
        values = np.array(
            [variable.resolve() for variable in self.variables], dtype=np.float64
        )
        return float(self.function(values)[0])


@dataclass(eq=False, kw_only=True)
class LiquidConnection(PrismaticConnection, HasUpdateState, FillContext):
    """
    Prismatic connection whose passive degree of freedom is a container's normalized
    fill level, integrated from :attr:`outflow_equation` and :attr:`inflow_equation`
    each physics step; the QP never commands it.
    """

    outflow_equation: PouringEquation | None = field(default=None, init=False)
    """
    Equation of the liquid leaving this container, e.g. as it tilts to pour.
    """

    inflow_equation: InflowEquation | None = field(default=None, init=False)
    """
    Equation of the liquid entering this container from a source.
    """

    coupled_source_drain: PouringEquation | None = field(
        default=None, init=False, repr=False, compare=False
    )
    """
    The source drain :attr:`inflow_equation` was built from; a source carrying another
    drain, e.g. one a synchronized world replaced, marks the coupling stale.
    """

    compiled_fill_velocity: CompiledFillVelocity | None = field(
        default=None, init=False, repr=False, compare=False
    )
    """
    The fill velocity compiled for :meth:`update_state`, recompiled whenever an
    equation or the world model changes.
    """

    @property
    def active_dofs(self) -> list[DegreeOfFreedom]:
        """
        :return: Empty list, so the QP solver cannot command the fill velocity directly.
        """
        return []

    @property
    def passive_dofs(self) -> list[DegreeOfFreedom]:
        """
        :return: The fill-level degree of freedom, which stays in the world state and
            forward kinematics without being a QP variable.
        """
        return [self.raw_dof]

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        :return: The JSON of this connection with its drain and without its inflow,
            which the receiving world rebuilds from the transfer coupling.
        """
        result = super().to_json(**kwargs)
        result["outflow_equation"] = to_json(self.outflow_equation)
        return result

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs: Any) -> LiquidConnection:
        """
        :return: The connection the JSON describes, with its drain.
        """
        instance = super()._from_json(data=data, **kwargs)
        raw_outflow = data.get("outflow_equation")
        if raw_outflow is not None:
            instance.outflow_equation = from_json(raw_outflow)
        return instance

    @called_positionally
    def copy_for_world(self, world: World) -> LiquidConnection:
        """
        :param world: The target world.
        :return: A copy of this connection for ``world`` with the ungated drain; the
            inflow is rebuilt there by the receiver's transfer coupling.
        """
        copy = super().copy_for_world(world)
        if self.outflow_equation is not None:
            copy.outflow_equation = self.outflow_equation.drop_gate()
        return copy

    @property
    def tilt_expression(self) -> Scalar:
        """
        :return: Symbolic tilt angle of the container from the vertical, in radians.
        """
        root_T_child = self._world.compose_forward_kinematics_expression(
            self._world.root, self.child
        )
        return FillContext.tilt_from_vertical(root_T_container=root_T_child)

    @property
    def fill_position(self) -> Scalar:
        """
        :return: Symbolic normalized fill level, the position of the fill-level degree
            of freedom.
        """
        return self.dof.variables.position

    def symbolic_fill_velocity(self) -> Scalar:
        """
        :return: The fill level's rate from :attr:`outflow_equation` and
            :attr:`inflow_equation` together, zero without either.
        """
        velocity = sm.Scalar(0.0)
        for equation in (self.outflow_equation, self.inflow_equation):
            if equation is not None:
                velocity = velocity + equation.symbolic_velocity(context=self)
        return velocity

    def current_compiled_fill_velocity(self) -> CompiledFillVelocity:
        """
        :return: The fill velocity compiled for the current equations and world model,
            compiling it only when either changed since the last call.
        """
        model_version = self._world._model_manager.version
        cached = self.compiled_fill_velocity
        if cached is not None and cached.is_compiled_for(
            outflow_equation=self.outflow_equation,
            inflow_equation=self.inflow_equation,
            model_version=model_version,
        ):
            return cached
        velocity = self.symbolic_fill_velocity()
        variables = velocity.free_variables()
        self.compiled_fill_velocity = CompiledFillVelocity(
            outflow_equation=self.outflow_equation,
            inflow_equation=self.inflow_equation,
            model_version=model_version,
            variables=variables,
            function=velocity.compile(
                parameters=VariableParameters.from_lists(variables)
            ),
        )
        return self.compiled_fill_velocity

    @called_positionally
    def update_state(self, dt: float) -> None:
        """
        Advances the fill level by one physics step, clamped to the fill degree of
        freedom's limits, which the equations do not guard themselves.

        :param dt: Time elapsed since the previous step, in seconds.
        :raises MissingFillLevelLimitsError: If the fill degree of freedom has no
            position limits to clamp to.
        """
        limits = self.raw_dof.limits
        if limits.lower.position is None or limits.upper.position is None:
            raise MissingFillLevelLimitsError(connection_name=self.name)
        state = self._world.state
        velocity = self.current_compiled_fill_velocity().evaluate()
        integrated_position = state[self.raw_dof.id].position + velocity * dt
        state[self.raw_dof.id].position = min(
            max(integrated_position, limits.lower.position), limits.upper.position
        )
