"""
Tests for containers that hold their contents as individual particles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import timedelta

import mujoco
import numpy
import pytest

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    EmptyContainerCalibrationError,
    ParticlesDoNotFitError,
)
from semantic_digital_twin.physics.particles import (
    HollowCylinder,
    MeasuredCommittedFillLevel,
    MeasuredFillLevel,
    MeasuredInflowRate,
    ParticleFill,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
    DerivativeMap,
)
from semantic_digital_twin.world_description.contact import ContactFriction
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False)
class PourableContainer(HasFillLevel):
    """
    A container carrying the fill level a controller reasons about.
    """


CUP = HollowCylinder(inner_radius=0.035, height=0.1)
SMALL_CUP = HollowCylinder(inner_radius=0.025, height=0.05)
"""
A cavity small enough that settling a full one is quick, and still several particles
across, which a packing needs to settle the way a bulk does.
"""

PARTICLE_RADIUS = 0.005
SETTLE = timedelta(milliseconds=10)
"""
How long a test steps the physics before reading the contents back: a body has no pose
until the simulation has stepped once.
"""

SETTLING_TIME = timedelta(seconds=3)
"""
How long a test steps the physics to let a packing collapse into the contents it
becomes, measured as the point the filled height stops falling.
"""

SEATED_CONTENTS = 40
"""
A count the packing places well inside the cavity.

A packing that fills a cavity stands taller than it, since it holds the particles clear
of each other; a test that reads the contents back before they have fallen and settled
needs one that does not.
"""

GRAVITY = 9.81
"""
Free-fall acceleration, for working out how far a particle drops while a test settles.
"""


@pytest.fixture
def world_with_ground() -> World:
    """
    A world whose root carries a floor for containers to stand on.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("world"))
        world.add_body(root)
        ground = Body.from_shape_collection(
            name=PrefixedName("ground"),
            shape_collection=ShapeCollection([Box(scale=Scale(2.0, 2.0, 0.1))]),
        )
        world.add_kinematic_structure_entity(ground)
        world.add_connection(
            FixedConnection(
                parent=root,
                child=ground,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-0.05, reference_frame=root
                ),
            )
        )
    return world


# %% the container's own geometry


def test_the_wall_panels_stand_clear_of_the_cavity():
    """
    Every panel's inner face is exactly at the cavity's radius, so the cavity is the
    circle the panels enclose and nothing of the container intrudes into it.
    """
    panels = [shape for shape in CUP.shapes() if isinstance(shape, Box)]

    assert len(panels) == CUP.wall_segments
    for panel in panels:
        origin = panel.origin.to_np()
        distance_to_axis = math.hypot(origin[0, 3], origin[1, 3])
        assert distance_to_axis - panel.scale.x / 2 == pytest.approx(CUP.inner_radius)


def test_the_wall_panels_span_the_container_above_its_base():
    """
    The wall runs from the top of the base to the rim, so a container is as deep inside
    as it is tall outside, less its floor.
    """
    [panel] = [shape for shape in CUP.shapes() if isinstance(shape, Box)][:1]
    panel_centre_z = panel.origin.to_np()[2, 3]

    assert panel_centre_z - panel.scale.z / 2 == pytest.approx(CUP.base_thickness)
    assert panel_centre_z + panel.scale.z / 2 == pytest.approx(CUP.height)


def test_the_cavity_holds_the_cylinder_standing_on_the_base():
    """
    The cavity is what the container can hold, which is what a fill level is a share of.
    """
    assert CUP.cavity_volume == pytest.approx(
        math.pi * CUP.inner_radius**2 * (CUP.height - CUP.base_thickness)
    )


# %% where the particles start


def test_every_particle_starts_within_the_cavity():
    """
    A particle that starts inside a wall or below the floor is resolved by the physics
    in its first steps, which throws the contents out of the container.
    """
    positions = CUP.particle_positions(PARTICLE_RADIUS, count=60)

    for position in positions:
        x, y, z = float(position.x), float(position.y), float(position.z)
        assert math.hypot(x, y) <= CUP.inner_radius - PARTICLE_RADIUS
        assert z >= CUP.base_thickness + PARTICLE_RADIUS
        assert z <= CUP.height - PARTICLE_RADIUS


def test_the_particles_start_clear_of_each_other():
    """
    Particles that start in contact make the packing's first steps the most expensive
    ones of the whole run.
    """
    positions = [
        numpy.array([float(position.x), float(position.y), float(position.z)])
        for position in CUP.particle_positions(PARTICLE_RADIUS, count=60)
    ]

    closest = min(
        numpy.linalg.norm(one - other)
        for index, one in enumerate(positions)
        for other in positions[index + 1 :]
    )
    assert closest >= 2 * PARTICLE_RADIUS


def test_a_cavity_is_packed_from_its_floor_up():
    """
    The contents rest on the floor rather than hanging in the air, so the physics has
    nothing to settle before a run starts.
    """
    positions = CUP.particle_positions(PARTICLE_RADIUS, count=60)

    assert float(positions[0].z) == pytest.approx(CUP.base_thickness + PARTICLE_RADIUS)
    assert [float(position.z) for position in positions] == sorted(
        float(position.z) for position in positions
    )


def test_asking_for_more_particles_than_the_cavity_seats_is_refused():
    """
    Asking for more than fits would otherwise pack the surplus into the walls.
    """
    with pytest.raises(ParticlesDoNotFitError):
        CUP.particle_positions(PARTICLE_RADIUS, count=100_000)


# %% packing to a depth


def test_a_cavity_holds_the_share_of_its_volume_it_is_packed_to():
    """
    A count meant to stand as deep as a fill level has to take up that share of the
    cavity once it settles, or the two describe different cups.
    """
    fill_fraction = 0.5
    capacity = CUP.particle_capacity(PARTICLE_RADIUS, fill_fraction=fill_fraction)

    settled = capacity * ParticleFill.loose_bulk_volume_per_particle(PARTICLE_RADIUS)

    assert settled == pytest.approx(
        fill_fraction * CUP.cavity_volume,
        abs=ParticleFill.loose_bulk_volume_per_particle(PARTICLE_RADIUS),
    )


def test_a_packing_stands_taller_than_the_cavity_it_settles_into():
    """
    The packing keeps the particles clear of each other, so it is looser than what they
    settle into and a full cavity has to be released from above its own rim.
    """
    capacity = CUP.particle_capacity(PARTICLE_RADIUS)

    packed = CUP.particle_positions(PARTICLE_RADIUS, count=capacity)

    assert len(packed) == capacity
    assert max(float(position.z) for position in packed) > CUP.height


def test_a_cavity_refuses_more_than_it_holds_once_the_contents_settle():
    """
    The cavity's own volume is the limit, rather than how many seats the packing has:
    a packing can always be built taller, and what cannot be done is fit the settled
    contents in.
    """
    capacity = CUP.particle_capacity(PARTICLE_RADIUS)

    assert len(CUP.particle_positions(PARTICLE_RADIUS, count=capacity)) == capacity
    with pytest.raises(ParticlesDoNotFitError):
        CUP.particle_positions(PARTICLE_RADIUS, count=capacity * 2)


# %% the contents in the physics

pytestmark_physics = pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)


@pytest.fixture
def cup_in_a_simulation(world_with_ground):
    """
    A container standing in a world, and the simulation of that world, already stepped
    so that every body has a pose.
    """
    container = _stand_container_in(world_with_ground)
    simulation = MujocoSim(world=world_with_ground, headless=True)
    simulation.start_stepped_simulation()
    simulation.step_simulation(SETTLE)
    yield world_with_ground, container, simulation
    if simulation.is_running():
        simulation.stop_simulation()


@pytestmark_physics
def test_the_contents_are_bodies_of_the_simulation_and_not_of_the_world(
    cup_in_a_simulation,
):
    """
    A twin says a container holds something and how much; where each grain of it stands
    is the simulation's business, so the world gains nothing for the contents.
    """
    world, container, simulation = cup_in_a_simulation
    bodies_before = len(world.bodies)
    degrees_of_freedom_before = len(world.state)

    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=12,
    )

    assert len(world.bodies) == bodies_before
    assert len(world.state) == degrees_of_freedom_before
    assert set(fill.names) <= set(simulation.simulator.get_all_body_names().result)
    assert len(fill.names) == 12


@pytestmark_physics
def test_every_particle_is_free_to_move_on_its_own(cup_in_a_simulation):
    """
    Contents that share a joint, or have none, pour as one lump.
    """
    world, container, simulation = cup_in_a_simulation

    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=6,
    )

    for name in fill.names:
        [joint] = simulation.simulator.get_body_joints(body_name=name).result
        assert joint.type == mujoco.mjtJoint.mjJNT_FREE


@pytestmark_physics
def test_the_packing_stands_where_the_container_stands(cup_in_a_simulation):
    """
    The packing is laid out in the container's frame, so it has to arrive at the
    container wherever in the world that is.
    """
    world, container, simulation = cup_in_a_simulation
    count = 12
    packed = CUP.particle_positions(PARTICLE_RADIUS, count=count)

    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=count,
    )
    simulation.step_simulation(SETTLE)

    assert fill.positions_in(container) == pytest.approx(
        numpy.array(
            [
                [float(position.x), float(position.y), float(position.z)]
                for position in packed
            ]
        ),
        abs=1e-3,
    )


@pytestmark_physics
def test_a_particle_moved_out_of_a_container_stops_counting_towards_it(
    cup_in_a_simulation,
):
    """
    The count is read off where the particles are in the physics, so it follows them out
    of the container rather than staying at what was packed.
    """
    world, container, simulation = cup_in_a_simulation
    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=12,
    )
    simulation.step_simulation(SETTLE)
    assert fill.count_inside(container) == 12

    simulation.simulator.set_body_position(
        body_name=fill.names[0], position=numpy.array([1.0, 1.0, 1.0])
    )
    simulation.step_simulation(SETTLE)

    assert fill.count_inside(container) == 11
    assert fill.fraction_inside(container) == pytest.approx(11 / 12)


@pytestmark_physics
def test_the_filled_height_is_the_surface_of_the_contents(cup_in_a_simulation):
    """
    The number a fill level is compared against is where the contents end, which is a
    radius above the highest particle's centre rather than at it.
    """
    world, container, simulation = cup_in_a_simulation
    highest_packed = max(
        float(position.z)
        for position in CUP.particle_positions(PARTICLE_RADIUS, count=SEATED_CONTENTS)
    )

    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=SEATED_CONTENTS,
    )
    simulation.step_simulation(SETTLE)

    fallen = 0.5 * GRAVITY * SETTLE.total_seconds() ** 2
    assert fill.filled_height_in(container) == pytest.approx(
        (highest_packed + PARTICLE_RADIUS) / CUP.height, abs=fallen / CUP.height
    )


@pytestmark_physics
def test_a_cavity_filled_to_capacity_settles_full(world_with_ground):
    """
    What a container is asked to hold is what stands in it once the contents settle.

    The packing is looser than the settled contents, so the count that fills a cavity
    cannot be read off the packing's own seats; asking for a full cavity and getting a
    half-full one is what this pins down.
    """
    container = _stand_container_in(world_with_ground, geometry=SMALL_CUP)
    simulation = MujocoSim(world=world_with_ground, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(SETTLE)
        fill = SMALL_CUP.fill_with_particles(
            container=container,
            world=world_with_ground,
            simulator=simulation.simulator,
            particle_radius=PARTICLE_RADIUS,
            count=SMALL_CUP.particle_capacity(PARTICLE_RADIUS),
        )

        simulation.step_simulation(SETTLING_TIME)

        assert fill.filled_height_in(container) == pytest.approx(1.0, abs=0.15)
    finally:
        if simulation.is_running():
            simulation.stop_simulation()


@pytestmark_physics
def test_a_rolling_coefficient_keeps_the_contents_from_rolling_away(world_with_ground):
    """
    All three friction coefficients have to reach the solver.

    A physics engine resolves a contact in as many dimensions as it is asked for and
    reads only the coefficients that fit, so a sphere given a rolling coefficient in a
    contact resolved in the tangent plane alone is a frictionless ball bearing, and the
    coefficient may as well not have been set.
    """

    def spread_at(rolling: float) -> float:
        world = World()
        with world.modify_world():
            world.add_kinematic_structure_entity(
                Body.from_shape_collection(
                    name=PrefixedName("ground"),
                    shape_collection=ShapeCollection(
                        [
                            Box(
                                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                                    z=-0.05
                                ),
                                scale=Scale(4.0, 4.0, 0.1),
                            )
                        ]
                    ),
                )
            )
        simulation = MujocoSim(world=world, headless=True)
        simulation.start_stepped_simulation()
        try:
            simulation.step_simulation(SETTLE)
            fill = ParticleFill.spawn_in(
                simulator=simulation.simulator,
                container=world.root,
                world_T_container=numpy.eye(4),
                positions=CUP.particle_positions(PARTICLE_RADIUS, count=30),
                particle_radius=PARTICLE_RADIUS,
                contact=replace(
                    ParticleFill.settling_contact(),
                    friction=ContactFriction(sliding=0.6, rolling=rolling),
                ),
            )
            simulation.step_simulation(SETTLING_TIME)
            standing = fill.positions_in(world.root)
            return float(numpy.hypot(standing[:, 0], standing[:, 1]).max())
        finally:
            if simulation.is_running():
                simulation.stop_simulation()

    assert spread_at(rolling=0.1) < spread_at(rolling=0.0001)


@pytestmark_physics
def test_an_empty_container_is_reached_by_nothing(cup_in_a_simulation):
    """
    A container holding none of the contents reads as empty rather than as however deep
    the contents are elsewhere.
    """
    world, container, simulation = cup_in_a_simulation
    fill = ParticleFill.spawn_in(
        simulator=simulation.simulator,
        container=container,
        world_T_container=world.compute_forward_kinematics_np(world.root, container),
        positions=[],
        particle_radius=PARTICLE_RADIUS,
    )

    assert fill.filled_height_in(container) == 0.0
    assert fill.fraction_inside(container) == 0.0
    assert fill.count_inside(container) == 0


@pytestmark_physics
def test_the_particles_are_given_contact_parameters_that_let_them_settle(
    cup_in_a_simulation,
):
    """
    Particles at a physics engine's own damping bounce back out of the container they
    were poured into, so they carry their own contact parameters into the model.
    """
    world, container, simulation = cup_in_a_simulation
    settling = ParticleFill.settling_contact()

    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=4,
    )

    for name in fill.names:
        friction = simulation.simulator.get_geom_friction(
            geom_name=f"{name}_sphere"
        ).result
        assert list(friction) == pytest.approx(settling.friction.to_list())


@pytestmark_physics
def test_the_contents_stay_in_an_upright_container_and_pour_out_of_a_tilted_one(
    world_with_ground,
):
    """
    The point of holding a container's contents as particles: standing upright it keeps
    them, tilted past its rim it does not, and neither is a number anyone integrated.
    """
    container, tilt = _hang_container_on_a_tilt_in(world_with_ground)
    simulation = MujocoSim(world=world_with_ground, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(SETTLE)
        fill = CUP.fill_with_particles(
            container=container,
            world=world_with_ground,
            simulator=simulation.simulator,
            particle_radius=PARTICLE_RADIUS,
            count=40,
        )
        simulation.step_simulation(timedelta(seconds=1.0))
        held_upright = fill.count_inside(container)
        world_with_ground.state[tilt.raw_dof.id].position = math.pi
        world_with_ground.notify_state_change()
        simulation.step_simulation(timedelta(seconds=2.0))
        held_upside_down = fill.count_inside(container)
    finally:
        simulation.stop_simulation()

    assert held_upright == len(fill.names)
    assert held_upside_down == 0


def _stand_container_in(
    world: World,
    name: str = "cup",
    offset: float = 0.0,
    geometry: HollowCylinder = CUP,
) -> Body:
    """
    Add a container standing on a world's root.

    :param world: The world to add the container to.
    :param name: Name of the container's body.
    :param offset: How far along x it stands from the root.
    :param geometry: The shape of the container to stand there.
    :return: The container's body.
    """
    container = geometry.body(PrefixedName(name))
    with world.modify_world():
        world.add_kinematic_structure_entity(container)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=container,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=offset, reference_frame=world.root
                ),
            )
        )
    return container


def _hang_container_on_a_tilt_in(world: World) -> tuple[Body, RevoluteConnection]:
    """
    Add a container to a world on a connection that can tip it over.

    :param world: The world to add the container to.
    :return: The container's body and the connection that tilts it.
    """
    container = CUP.body(PrefixedName("cup"))
    with world.modify_world():
        world.add_kinematic_structure_entity(container)
        tilt = RevoluteConnection.create_with_dofs(
            world=world,
            parent=world.root,
            child=container,
            axis=Vector3(0, 1, 0),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=math.pi, velocity=2.0),
            ),
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=0.3, reference_frame=world.root
            ),
        )
        world.add_connection(tilt)
    return container, tilt


# %% reporting how full a container is


@pytestmark_physics
def test_a_measurement_reports_the_share_of_the_capacity_the_contents_take_up(
    cup_in_a_simulation,
):
    """
    What a perception pipeline would report about a container is what stands in it, not
    what an equation integrated into it, and how full it is means against its own
    capacity rather than against everything that was poured.
    """
    world, container, simulation = cup_in_a_simulation
    annotation = PourableContainer(name=PrefixedName("contents"), root=container)
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    annotation.initialize_fill_level(world=world, initial_fill=0.0)
    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=SEATED_CONTENTS,
    )
    simulation.step_simulation(SETTLE)

    measurement = MeasuredFillLevel(
        contents=fill,
        container=container,
        capacity=CUP.cavity_volume,
        connection=annotation.fill_connection,
        world=world,
    )

    assert measurement.measure() == pytest.approx(
        fill.bulk_volume_in(container) / CUP.cavity_volume
    )
    assert measurement.measure() < 1.0, "half a cup of particles is not a full cup"


@pytestmark_physics
def test_a_report_replaces_the_level_the_controller_holds(cup_in_a_simulation):
    """
    A report has to land where the tasks read the fill level from, or the controller
    keeps reasoning about the level it integrated for itself.
    """
    world, container, simulation = cup_in_a_simulation
    annotation = PourableContainer(name=PrefixedName("contents"), root=container)
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    annotation.initialize_fill_level(world=world, initial_fill=0.9)
    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=SEATED_CONTENTS,
    )
    simulation.step_simulation(SETTLE)
    measurement = MeasuredFillLevel(
        contents=fill,
        container=container,
        capacity=CUP.cavity_volume,
        connection=annotation.fill_connection,
        world=world,
    )

    reported = measurement.report()

    assert annotation.fill_level == pytest.approx(reported)
    assert annotation.fill_level == pytest.approx(measurement.measure())


@pytestmark_physics
def test_an_empty_container_is_reported_as_empty(cup_in_a_simulation):
    """
    A container nothing has reached yet reads as empty, however full the controller
    believed it was.
    """
    world, container, simulation = cup_in_a_simulation
    annotation = PourableContainer(name=PrefixedName("contents"), root=container)
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    annotation.initialize_fill_level(world=world, initial_fill=0.7)
    fill = ParticleFill.spawn_in(
        simulator=simulation.simulator,
        container=container,
        world_T_container=world.compute_forward_kinematics_np(world.root, container),
        positions=[],
        particle_radius=PARTICLE_RADIUS,
    )
    measurement = MeasuredFillLevel(
        contents=fill,
        container=container,
        capacity=CUP.cavity_volume,
        connection=annotation.fill_connection,
        world=world,
    )

    assert measurement.report() == 0.0
    assert annotation.fill_level == pytest.approx(0.0)


# %% how fast the contents are arriving


def _level_of(world: World, fill: ParticleFill, container: Body) -> MeasuredFillLevel:
    """
    A fill-level measurement of a container that carries no annotation of its own.
    """
    annotation = PourableContainer(
        name=PrefixedName(f"level_of_{container.name.name}"), root=container
    )
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    annotation.initialize_fill_level(world=world, initial_fill=0.0)
    return MeasuredFillLevel(
        contents=fill,
        container=container,
        capacity=CUP.cavity_volume,
        connection=annotation.fill_connection,
        world=world,
    )


@pytestmark_physics
def test_nothing_arriving_is_reported_as_no_inflow(cup_in_a_simulation):
    """
    A container nothing is reaching reports no inflow, however fast a drain model
    believes its source is pouring.
    """
    world, container, simulation = cup_in_a_simulation
    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=8,
    )
    elsewhere = _stand_container_in(world, name="elsewhere", offset=1.0)
    rate = MeasuredInflowRate(level=_level_of(world, fill, elsewhere))
    simulation.step_simulation(SETTLE)
    rate.observe(at=0.0)

    assert rate.observe(at=1.0) == 0.0


@pytestmark_physics
def test_the_inflow_is_the_share_that_arrived_over_the_time_it_took(
    cup_in_a_simulation,
):
    """
    The rate a controller would compare its drain model against is how much of the
    contents arrived, over how long it took them.
    """
    world, container, simulation = cup_in_a_simulation
    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=8,
    )
    catcher = _stand_container_in(world, name="catcher", offset=1.0)
    caught_level = _level_of(world, fill, catcher)
    rate = MeasuredInflowRate(level=caught_level)
    simulation.step_simulation(SETTLE)
    rate.observe(at=0.0)

    caught = fill.names[:2]
    for name in caught:
        simulation.simulator.set_body_position(
            body_name=name, position=numpy.array([1.0, 0.0, 0.02])
        )
    simulation.step_simulation(SETTLE)

    observed = rate.observe(at=0.5)

    assert fill.count_inside(catcher) == len(caught)
    assert observed == pytest.approx(
        caught_level.measure() / 0.5
    ), "the rate is the change in the very level the controller reads"


@pytestmark_physics
def test_an_observation_at_no_elapsed_time_reports_no_inflow(cup_in_a_simulation):
    """
    Two observations of the same instant say nothing about a rate, and must not divide
    by the time between them.
    """
    world, container, simulation = cup_in_a_simulation
    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=4,
    )
    rate = MeasuredInflowRate(level=_level_of(world, fill, container))
    simulation.step_simulation(SETTLE)
    rate.observe(at=1.0)

    assert rate.observe(at=1.0) == 0.0


@pytestmark_physics
def test_a_bigger_container_holding_the_same_contents_reads_emptier(
    cup_in_a_simulation,
):
    """
    A fill level is a question about the container, so the same contents answer it
    differently depending on what is holding them. This is what the share of everything
    poured could not express.
    """
    world, container, simulation = cup_in_a_simulation
    annotation = PourableContainer(name=PrefixedName("contents"), root=container)
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    annotation.initialize_fill_level(world=world, initial_fill=0.0)
    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=SEATED_CONTENTS,
    )
    simulation.step_simulation(SETTLE)

    def measured(capacity: float) -> float:
        return MeasuredFillLevel(
            contents=fill,
            container=container,
            capacity=capacity,
            connection=annotation.fill_connection,
            world=world,
        ).measure()

    assert measured(2 * CUP.cavity_volume) == pytest.approx(
        0.5 * measured(CUP.cavity_volume)
    )


@pytestmark_physics
def test_contents_filling_their_container_read_full_and_no_more(cup_in_a_simulation):
    """
    Contents taking up exactly the capacity read full, and more than it cannot read
    more than full, since a fill level is a share.
    """
    world, container, simulation = cup_in_a_simulation
    annotation = PourableContainer(name=PrefixedName("contents"), root=container)
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    annotation.initialize_fill_level(world=world, initial_fill=0.0)
    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=SEATED_CONTENTS,
    )
    simulation.step_simulation(SETTLE)
    fill.bulk_volume_per_particle = CUP.cavity_volume / fill.count_inside(container)

    def measured(capacity: float) -> float:
        return MeasuredFillLevel(
            contents=fill,
            container=container,
            capacity=capacity,
            connection=annotation.fill_connection,
            world=world,
        ).measure()

    assert measured(CUP.cavity_volume) == pytest.approx(1.0)
    assert measured(CUP.cavity_volume / 2) == pytest.approx(1.0)


@pytestmark_physics
def test_calibrating_in_a_container_makes_its_measurement_match_how_far_they_reach(
    cup_in_a_simulation,
):
    """
    The conversion from a count to a volume is measured rather than assumed: read off a
    container the contents stand in, it reproduces how far up it they reach.
    """
    world, container, simulation = cup_in_a_simulation
    annotation = PourableContainer(name=PrefixedName("contents"), root=container)
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    annotation.initialize_fill_level(world=world, initial_fill=0.0)
    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=SEATED_CONTENTS,
    )
    simulation.step_simulation(SETTLE)
    fill.bulk_volume_per_particle = fill.bulk_volume_per_particle_in(
        container, CUP.cavity_volume
    )

    measurement = MeasuredFillLevel(
        contents=fill,
        container=container,
        capacity=CUP.cavity_volume,
        connection=annotation.fill_connection,
        world=world,
    )

    assert measurement.measure() == pytest.approx(fill.filled_height_in(container))


@pytestmark_physics
def test_calibrating_needs_contents_to_measure(cup_in_a_simulation):
    """
    An empty container says nothing about how much volume a particle takes up.
    """
    world, container, simulation = cup_in_a_simulation
    fill = ParticleFill.spawn_in(
        simulator=simulation.simulator,
        container=container,
        world_T_container=world.compute_forward_kinematics_np(world.root, container),
        positions=[],
        particle_radius=PARTICLE_RADIUS,
    )

    with pytest.raises(EmptyContainerCalibrationError):
        fill.bulk_volume_per_particle_in(container, CUP.cavity_volume)


# %% counting what is still on its way


@pytestmark_physics
def test_particles_are_counted_by_how_high_they_stand(cup_in_a_simulation):
    """
    Whether contents have landed is a question about where they are, which is what
    separates what is still falling from what has arrived.
    """
    world, container, simulation = cup_in_a_simulation
    world_T_container = world.compute_forward_kinematics_np(world.root, container)
    base = world_T_container[2, 3]
    fill = ParticleFill.spawn_in(
        simulator=simulation.simulator,
        container=container,
        world_T_container=world_T_container,
        positions=[
            Point3(x=0.0, y=0.0, z=height, reference_frame=container)
            for height in (0.4, 0.6, 0.8)
        ],
        particle_radius=PARTICLE_RADIUS,
    )
    simulation.step_simulation(timedelta(milliseconds=10))

    assert fill.count_above(base + 0.5) == 2
    assert fill.count_above(base + 0.7) == 1
    assert fill.count_above(base + 1.0) == 0


@pytestmark_physics
def test_contents_still_in_the_source_are_not_counted_as_on_their_way(
    cup_in_a_simulation,
):
    """
    The source stands above the container it pours into, so its own contents are above
    the opening too; only what has left it is arriving.
    """
    world, container, simulation = cup_in_a_simulation
    annotation = PourableContainer(name=PrefixedName("contents"), root=container)
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    annotation.initialize_fill_level(world=world, initial_fill=0.0)
    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=SEATED_CONTENTS,
    )
    simulation.step_simulation(SETTLE)
    world_T_container = world.compute_forward_kinematics_np(world.root, container)

    measurement = MeasuredCommittedFillLevel(
        contents=fill,
        container=container,
        capacity=CUP.cavity_volume,
        connection=annotation.fill_connection,
        world=world,
        source=container,
        opening_height=world_T_container[2, 3] - 1.0,
    )

    assert fill.count_above(measurement.opening_height) == len(fill.names)
    assert measurement.count_on_the_way() == 0


@pytestmark_physics
def test_what_has_left_the_source_is_counted_before_it_lands(cup_in_a_simulation):
    """
    The point of the measurement: a container reads as fuller than it is by exactly
    what is still falling into it, so a controller stops before that lands.
    """
    world, container, simulation = cup_in_a_simulation
    annotation = PourableContainer(name=PrefixedName("contents"), root=container)
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    annotation.initialize_fill_level(world=world, initial_fill=0.0)
    world_T_container = world.compute_forward_kinematics_np(world.root, container)
    falling = 2
    fill = ParticleFill.spawn_in(
        simulator=simulation.simulator,
        container=container,
        world_T_container=world_T_container,
        positions=[
            Point3(x=0.0, y=0.0, z=0.5 + 0.1 * index, reference_frame=container)
            for index in range(falling)
        ],
        particle_radius=PARTICLE_RADIUS,
    )
    simulation.step_simulation(timedelta(milliseconds=10))

    arguments = dict(
        contents=fill,
        container=container,
        capacity=CUP.cavity_volume,
        connection=annotation.fill_connection,
        world=world,
    )
    landed = MeasuredFillLevel(**arguments)
    committed = MeasuredCommittedFillLevel(
        **arguments, source=container, opening_height=world_T_container[2, 3]
    )

    assert landed.measure() == 0.0, "nothing has reached the cup yet"
    assert committed.count_on_the_way() == falling
    assert committed.measure() == pytest.approx(
        falling * fill.bulk_volume_per_particle / CUP.cavity_volume
    )
