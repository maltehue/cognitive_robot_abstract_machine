"""
Containers that hold their contents as individual particles, for pouring experiments
whose ground truth is where the grains actually land.

The contents live in the physics alone. A twin describes what a robot reasons about,
which is that a container holds something and how much of it; where each grain of that
something happens to be is not that. Keeping the particles out of the twin keeps its
state from growing by a free body per grain, and keeps a simulation from converting a
pose per grain into a world that nothing reads them from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import timedelta

import mujoco
import numpy
from scipy.spatial.transform import Rotation

from typing_extensions import ClassVar, List, Optional, Self

from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    EmptyContainerCalibrationError,
    ParticlesDoNotFitError,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from physics_simulators.mujoco_simulator import (
    MujocoEntity,
    MujocoSimulator,
    NewEntity,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactParameters,
    ContactStiffness,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Scale,
    Sphere,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the container


@dataclass
class HollowCylinder:
    """
    The geometry of an upright container whose contents cannot fall through it.

    A container whose geometry is a mesh collides as that mesh's convex hull, so
    anything poured into one comes to rest on top of it rather than inside. Built from
    primitives, the cavity survives into the physics.

    The container's origin sits at the bottom of its outside, so the cavity runs from
    :attr:`base_thickness` up to :attr:`height`.
    """

    PARTICLE_SPACING: ClassVar[float] = 2.2
    """
    Distance between two neighbouring particle centres, in particle radii.

    Larger than 2 so the packing starts with the particles clear of each other: a
    packing that starts in contact resolves those contacts in its first steps, which
    costs more than the gap it saves.
    """

    inner_radius: float
    """
    Radius of the cavity, in metres.
    """

    height: float
    """
    Height of the container from the bottom of its base to its rim, in metres.
    """

    wall_thickness: float = 0.004
    """
    Thickness of the side wall, in metres.
    """

    base_thickness: float = 0.004
    """
    Thickness of the floor the contents rest on, in metres.
    """

    wall_segments: int = 16
    """
    How many flat panels stand in for the round wall.

    Each panel's inner face is tangent to the cavity, so the cavity is the circle the
    panels enclose and every extra panel makes the wall rounder.
    """

    color: Color = field(default_factory=lambda: Color(0.7, 0.7, 0.75, 1.0))
    """
    Colour of the base and the wall panels.
    """

    @property
    def cavity_volume(self) -> float:
        """
        :return: The volume the cavity encloses, in cubic metres.
        """
        return math.pi * self.inner_radius**2 * (self.height - self.base_thickness)

    def shapes(self) -> ShapeCollection:
        """
        Build the base and the wall panels of this container.

        :return: The shapes, in the container's own frame.
        """
        return ShapeCollection([self._base_shape()] + self._wall_shapes())

    def _base_shape(self) -> Cylinder:
        """
        :return: The floor of the container, spanning the full outside radius.
        """
        return Cylinder(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=self.base_thickness / 2
            ),
            width=2 * (self.inner_radius + self.wall_thickness),
            height=self.base_thickness,
            color=self.color,
        )

    def _wall_shapes(self) -> List[Box]:
        """
        :return: The panels standing in for the round wall, each tangent to the cavity.
        """
        wall_height = self.height - self.base_thickness
        panel_width = (
            2
            * (self.inner_radius + self.wall_thickness)
            * math.tan(math.pi / self.wall_segments)
        )
        centre_radius = self.inner_radius + self.wall_thickness / 2
        panels = []
        for segment in range(self.wall_segments):
            angle = 2 * math.pi * segment / self.wall_segments
            panels.append(
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=centre_radius * math.cos(angle),
                        y=centre_radius * math.sin(angle),
                        z=self.base_thickness + wall_height / 2,
                        yaw=angle,
                    ),
                    scale=Scale(self.wall_thickness, panel_width, wall_height),
                    color=self.color,
                )
            )
        return panels

    def body(self, name: PrefixedName) -> Body:
        """
        Build the body of a container with this geometry, for a caller to connect into a
        world however it stands there.

        :param name: Name of the container's body.
        :return: The container's body, not yet in any world.
        """
        return Body.from_shape_collection(name=name, shape_collection=self.shapes())

    def fill_with_particles(
        self,
        container: Body,
        world: World,
        simulator: MujocoSimulator,
        particle_radius: float,
        count: int,
    ) -> ParticleFill:
        """
        Pack a container this geometry describes with particles.

        The world is read to find where the container stands; the particles themselves
        are added to the simulation and never to the world.

        :param container: The body this geometry was built as.
        :param world: The world the container stands in.
        :param simulator: The simulation to add the particles to.
        :param particle_radius: Radius of one particle, in metres.
        :param count: How many particles to pack.
        :return: The fill holding the spawned particles.
        :raises ParticlesDoNotFitError: If the cavity holds fewer than ``count``.
        """
        return ParticleFill.spawn_in(
            simulator=simulator,
            container=container,
            world_T_container=world.compute_forward_kinematics_np(
                world.root, container
            ),
            positions=self.particle_positions(particle_radius, count),
            particle_radius=particle_radius,
        )

    def particle_capacity(
        self, particle_radius: float, fill_fraction: float = 1.0
    ) -> int:
        """
        How many particles settle to a share of the cavity's depth.

        Counted by the volume the particles take up rather than by the seats the
        packing has for them: the packing holds them clear of each other and so stands
        looser than what they settle into, and its seats deliver about half the depth
        they are asked for.

        :param particle_radius: Radius of one particle, in metres.
        :param fill_fraction: Share of the cavity to fill, in ``[0, 1]``.
        :return: The number of particles that settle to that share.
        """
        return int(
            fill_fraction
            * self.cavity_volume
            / ParticleFill.loose_bulk_volume_per_particle(particle_radius)
        )

    def particle_positions(self, particle_radius: float, count: int) -> List[Point3]:
        """
        Where ``count`` particles of this radius are released, packed from the floor up
        and clear of the walls and of each other.

        The packing stands looser than what the particles settle into, so filling the
        cavity needs a column taller than it: the layers carry on above the rim, and the
        surplus drops in as the ones below it compact.

        :param particle_radius: Radius of one particle, in metres.
        :param count: How many particles to place.
        :return: The particles' positions in the container's own frame, lowest first.
        :raises ParticlesDoNotFitError: If the cavity holds fewer than ``count`` once
            they have settled.
        """
        available = self.particle_capacity(particle_radius)
        if count > available:
            raise ParticlesDoNotFitError(
                cavity_volume=self.cavity_volume,
                particle_radius=particle_radius,
                requested=count,
                available=available,
            )
        spacing = self.PARTICLE_SPACING * particle_radius
        seat_radius = self.inner_radius - particle_radius
        height = self.base_thickness + particle_radius
        if count and not self._layer_positions(height, seat_radius, spacing):
            raise ParticlesDoNotFitError(
                cavity_volume=self.cavity_volume,
                particle_radius=particle_radius,
                requested=count,
                available=0,
            )
        positions = []
        while len(positions) < count:
            positions.extend(self._layer_positions(height, seat_radius, spacing))
            height += spacing
        return positions[:count]

    def _layer_positions(
        self, height: float, seat_radius: float, spacing: float
    ) -> List[Point3]:
        """
        One horizontal layer of the packing.

        :param height: Height of the layer above the container's origin.
        :param seat_radius: How far from the axis a particle's centre may stand.
        :param spacing: Distance between two neighbouring particle centres.
        :return: The positions in the layer, in the container's own frame.
        """
        positions = []
        steps = int(seat_radius / spacing) if spacing > 0 else 0
        for x_step in range(-steps, steps + 1):
            for y_step in range(-steps, steps + 1):
                x, y = x_step * spacing, y_step * spacing
                if math.hypot(x, y) <= seat_radius:
                    positions.append(Point3(x, y, height))
        return positions


# %% the contents


@dataclass
class ParticleFill:
    """
    A container's contents, as individual spheres in a simulation.

    Where the contents end up is whatever the physics does with them rather than a
    number integrated from a pouring equation. They are bodies of the simulation only,
    so every question about them is answered by reading it.
    """

    simulator: MujocoSimulator
    """
    The simulation the particles live in.
    """

    names: List[str]
    """
    The particles' body names in that simulation, in the order they were packed.
    """

    particle_radius: float
    """
    Radius of one particle, in metres.
    """

    bulk_volume_per_particle: Optional[float] = None
    """
    Volume one particle takes up once poured, voids included, in cubic metres.

    What makes a count of particles comparable with a fill level: a container is as full
    as the volume standing in it, and particles standing in a heap occupy more than they
    are made of. Measurable by pouring a known count into a container of known capacity,
    which is what :meth:`bulk_volume_per_particle_in` reads. Defaults to the particle
    loosely packed.
    """

    LOOSE_PACKING_FRACTION: ClassVar[float] = 0.6
    """
    Share of a poured volume that equal spheres themselves occupy, packed loosely.
    """

    def __post_init__(self) -> None:
        if self.bulk_volume_per_particle is None:
            self.bulk_volume_per_particle = self.loose_bulk_volume_per_particle(
                self.particle_radius
            )

    @property
    def particle_volume(self) -> float:
        """
        :return: The volume of one particle itself, in cubic metres.
        """
        return 4 / 3 * math.pi * self.particle_radius**3

    @classmethod
    def loose_bulk_volume_per_particle(cls, particle_radius: float) -> float:
        """
        The volume one particle of a radius takes up once poured, voids included,
        assuming it packs as loosely as equal spheres do.

        How many particles a container holds and how full a count of them makes it are
        the same conversion read in either direction, so both are read here.

        :param particle_radius: Radius of one particle, in metres.
        :return: The volume, in cubic metres.
        """
        return 4 / 3 * math.pi * particle_radius**3 / cls.LOOSE_PACKING_FRACTION

    def bulk_volume_in(self, container: Body) -> float:
        """
        The volume the particles standing in a container take up, voids included.

        :param container: The container to measure in.
        :return: The volume, in cubic metres.
        """
        return self.count_inside(container) * self.bulk_volume_per_particle

    def bulk_volume_per_particle_in(self, container: Body, capacity: float) -> float:
        """
        The volume one particle takes up, read off the contents standing in a container
        whose capacity is known.

        Measures the packing in place rather than assuming it: the contents reach a
        share of the container's height, and that share of its capacity is what the
        particles in it occupy.

        :param container: The container the contents stand in.
        :param capacity: That container's capacity, in cubic metres.
        :return: The volume one particle takes up, in cubic metres.
        :raises EmptyContainerCalibrationError: If nothing stands in the container.
        """
        inside = self.count_inside(container)
        if inside == 0:
            raise EmptyContainerCalibrationError(container=container)
        return self.filled_height_in(container) * capacity / inside

    @classmethod
    def spawn_in(
        cls,
        simulator: MujocoSimulator,
        container: Body,
        world_T_container: numpy.ndarray,
        positions: List[Point3],
        particle_radius: float,
        color: Optional[Color] = None,
        contact: Optional[ContactParameters] = None,
    ) -> Self:
        """
        Add one free body per position to a simulation.

        The positions are read in the container's frame; a free body may only hang from
        the top level, so each one is placed at the pose that frame gives it.

        :param simulator: The simulation to add the particles to.
        :param container: The container whose frame the positions are given in, whose
            name the particles are named after.
        :param world_T_container: Where that container stands, as a 4x4 pose.
        :param positions: Where the particles start, in the container's frame.
        :param particle_radius: Radius of one particle, in metres.
        :param color: Colour of the particles. Defaults to a blue.
        :param contact: What the particles' surfaces do in a contact. Defaults to
            :meth:`settling_contact`.
        :return: The fill holding the spawned particles.
        """
        color = color if color is not None else Color(0.2, 0.45, 0.9, 1.0)
        contact = contact if contact is not None else cls.settling_contact()
        names = []
        new_entities = []
        for index, position in enumerate(positions):
            name = f"{container.name.name}_particle_{index}"
            pose = world_T_container @ numpy.array(
                [float(position.x), float(position.y), float(position.z), 1.0]
            )
            new_entities.extend(
                cls._particle_entities(name, pose[:3], particle_radius, color, contact)
            )
            names.append(name)
        simulator.add_entities(new_entities)
        return cls(simulator=simulator, names=names, particle_radius=particle_radius)

    @staticmethod
    def _particle_entities(
        name: str,
        position: numpy.ndarray,
        particle_radius: float,
        color: Color,
        contact: ContactParameters,
    ) -> List[NewEntity]:
        """
        The body, sphere and free joint one particle is made of.

        :param name: Name of the particle's body.
        :param position: Where it starts, in the simulation's own frame.
        :param particle_radius: Radius of the particle, in metres.
        :param color: Colour of the particle.
        :param contact: What its surface does in a contact.
        :return: The three entities, which only compile together.
        """
        geometry_properties = {
            "type": mujoco.mjtGeom.mjGEOM_SPHERE,
            "size": [particle_radius, 0.0, 0.0],
            "rgba": color.to_rgba(),
            "friction": contact.friction.to_list(),
        }
        if contact.stiffness is not None:
            geometry_properties["solref"] = contact.stiffness.to_list()
        if contact.impedance is not None:
            geometry_properties["solimp"] = contact.impedance.to_list()
        return [
            NewEntity(
                name=name,
                kind=MujocoEntity.BODY,
                properties={"pos": position.tolist()},
            ),
            NewEntity(
                name=f"{name}_sphere",
                kind=MujocoEntity.GEOM,
                properties=geometry_properties,
                parent_name=name,
            ),
            NewEntity(
                name=f"{name}_free",
                kind=MujocoEntity.JOINT,
                properties={"type": mujoco.mjtJoint.mjJNT_FREE},
                parent_name=name,
            ),
        ]

    @staticmethod
    def settling_contact() -> ContactParameters:
        """
        Contact parameters that let poured particles come to rest where they land.

        At a physics engine's own damping a particle dropped a container's height into
        another bounces straight back out of it, so these contacts are overdamped.

        :return: The parameters.
        """
        return ContactParameters(
            friction=ContactFriction(sliding=0.6),
            stiffness=ContactStiffness(
                time_constant=timedelta(milliseconds=10), damping_ratio=3.0
            ),
        )

    def positions_in(self, container: Body) -> numpy.ndarray:
        """
        Where the particles currently stand, in a container's own frame.

        ..note:: A simulation computes a body's pose as it steps, so this reads zeros
            until the first step has run.

        :param container: The container whose frame the positions are given in.
        :return: One row of x, y and z per particle, in the order they were packed.
        """
        if not self.names:
            return numpy.empty((0, 3))
        name = container.name.name
        world_P_container = numpy.asarray(
            self.simulator.get_body_position(body_name=name).result, dtype=float
        )
        world_R_container = Rotation.from_quat(
            numpy.asarray(
                self.simulator.get_body_quaternion(body_name=name).result, dtype=float
            ),
            scalar_first=True,
        ).as_matrix()
        world_P_particles = self.simulator.get_bodies_positions(
            body_names=self.names
        ).result
        stacked = numpy.array([world_P_particles[name] for name in self.names])
        return (stacked - world_P_container) @ world_R_container

    def count_inside(self, container: Body) -> int:
        """
        How many particles stand within a container's own extent.

        ..note:: A particle counts as inside when it is within the container's bounding
            box, which also covers the volume the container's own walls occupy.

        :param container: The container to count in.
        :return: The number of particles inside it.
        """
        return int(self._inside(container).sum())

    def fraction_inside(self, container: Body) -> float:
        """
        The share of the contents standing in a container, for comparison against a fill
        level.

        :param container: The container to count in.
        :return: The fraction in ``[0, 1]``, or ``0`` for a fill with no particles.
        """
        if not self.names:
            return 0.0
        return self.count_inside(container) / len(self.names)

    def filled_height_in(self, container: Body) -> float:
        """
        How far the contents standing in a container reach up it, as a share of its own
        height.

        Upright, this is the depth of the contents, which is what a fill level is. While
        the container tilts the contents ride up its wall, so the same number says how
        close they are to its rim and reaches ``1`` as it starts pouring.

        Measured to the top of the contents, a radius above the highest particle's
        centre, so that a container packed to its rim reads as full.

        :param container: The container to measure in.
        :return: The height the contents reach over the container's height, or ``0``
            when none is inside.
        """
        inside = self._inside(container)
        if not inside.any():
            return 0.0
        lower = container.collision.min_point.to_np()[:3]
        upper = container.collision.max_point.to_np()[:3]
        highest = self.positions_in(container)[inside][:, 2].max()
        surface = highest + self.particle_radius
        return float((surface - lower[2]) / (upper[2] - lower[2]))

    def heights(self) -> numpy.ndarray:
        """
        How high each particle stands, in the world frame.

        ..note:: A simulation computes a body's pose as it steps, so this reads zeros
            until the first step has run.

        :return: One height in metres per particle, in the order they were packed.
        """
        if not self.names:
            return numpy.empty(0)
        positions = self.simulator.get_bodies_positions(body_names=self.names).result
        return numpy.array([positions[name][2] for name in self.names])

    def count_above(self, height: float) -> int:
        """
        How many particles stand above a height in the world.

        :param height: The height to count above, in metres.
        :return: The number of particles above it.
        """
        return int((self.heights() > height).sum())

    def _inside(self, container: Body) -> numpy.ndarray:
        """
        :param container: The container to test against.
        :return: One boolean per particle: whether it stands within the container's
            extent.
        """
        lower = container.collision.min_point.to_np()[:3]
        upper = container.collision.max_point.to_np()[:3]
        positions = self.positions_in(container)
        if not len(positions):
            return numpy.zeros(0, dtype=bool)
        return numpy.all((positions >= lower) & (positions <= upper), axis=1)

    @property
    def volume(self) -> float:
        """
        :return: The volume of the particles themselves, in cubic metres.
        """
        return len(self.names) * self.particle_volume


# %% reporting how full a container is


@dataclass
class MeasuredFillLevel:
    """
    How full a container is, read off the contents standing in it and written into the
    world, as a perception pipeline reporting on it would.

    A fill level in the world is otherwise integrated from a pouring equation, so a
    controller reading it reasons about the pour its own model predicts. Reported from
    the contents instead, the controller reasons about the pour that happened: it keeps
    pouring while nothing has arrived, and stops when something has.

    The level is the share of the container's own capacity that the contents standing in
    it take up, which is what a fill level means and what a scale or a depth sensor
    would report of a real container. Counting particles is how this simulation sees
    them, so the count is converted through the volume one of them occupies.
    """

    contents: ParticleFill
    """
    The particles the report is read off.
    """

    container: Body
    """
    The container being reported on.
    """

    capacity: float
    """
    That container's capacity, in cubic metres.
    """

    connection: LiquidConnection
    """
    The container's fill level, which a report replaces.
    """

    world: World
    """
    The world holding that fill level.
    """

    def measure(self) -> float:
        """
        How full the container currently is.

        :return: The share of its capacity the contents in it take up, in ``[0, 1]``.
        """
        return min(1.0, self.contents.bulk_volume_in(self.container) / self.capacity)

    def report(self) -> float:
        """
        Measure the container and write the measurement into the world, so whatever
        reads its fill level next reads this rather than what was integrated.

        :return: What was reported.
        """
        measured = self.measure()
        JointState.from_mapping({self.connection: measured}).apply_to(self.world)
        return measured


# %% how fast the contents are arriving


@dataclass
class MeasuredInflowRate:
    """
    How fast a container is filling, measured from its contents.

    A drain model says how fast a tilted container pours; this says how fast the
    container it pours into is actually filling. The two disagreeing is what a
    controller would have to act on, since correcting only the level it steers by leaves
    it predicting the same arrival from the same tilt.

    It differentiates the same measurement the controller reads rather than counting for
    itself, so the rate is in the units of the fill level it belongs to and the two
    cannot come to mean different things.
    """

    level: MeasuredFillLevel
    """
    The fill level whose change this reports.
    """

    _level: Optional[float] = field(init=False, default=None, repr=False)
    """
    The level measured at :attr:`_observed_at`.
    """

    _observed_at: Optional[float] = field(init=False, default=None, repr=False)
    """
    When that level was observed, in seconds.
    """

    def observe(self, at: float) -> float:
        """
        Measure how fast the container has been filling since the last observation.

        The first observation has nothing to compare against and reports no inflow.

        :param at: The time of this observation, in seconds.
        :return: The change in fill level per second.
        """
        level = self.level.measure()
        previous_level, previous_time = self._level, self._observed_at
        self._level, self._observed_at = level, at
        if previous_time is None or at <= previous_time:
            return 0.0
        return (level - previous_level) / (at - previous_time)


@dataclass
class MeasuredCommittedFillLevel(MeasuredFillLevel):
    """
    How full a container will be once the contents already falling towards it have
    landed, rather than how full it is at this instant.

    Pouring cannot be taken back: contents that have left the source will arrive
    whatever the controller does next, so a level counting only what has landed lags by
    everything still in the air and is steered past before it reads the goal. Counting
    what is on its way as well is what a perception pipeline watching the stream would
    report.

    Contents count as on their way when they stand above the container's opening and
    have left the source. What has already landed elsewhere lies below that opening and
    is not counted, so a spill is not mistaken for an arrival.

    ..note:: This reads the source as standing above the container it pours into, which
        is what pouring into it requires.
    """

    source: Body
    """
    The container the contents are being poured from.
    """

    opening_height: float
    """
    How high this container's opening stands in the world, in metres.
    """

    def count_on_the_way(self) -> int:
        """
        :return: How many particles have left the source and not yet landed.
        """
        return max(
            0,
            self.contents.count_above(self.opening_height)
            - self.contents.count_inside(self.source),
        )

    def measure(self) -> float:
        """
        How full the container is committed to becoming.

        :return: The share of its capacity the contents in it and on their way to it
            take up, in ``[0, 1]``.
        """
        arriving = self.count_on_the_way() * self.contents.bulk_volume_per_particle
        return min(
            1.0,
            (self.contents.bulk_volume_in(self.container) + arriving) / self.capacity,
        )
