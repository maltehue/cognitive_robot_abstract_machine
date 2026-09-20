"""
Tests for containers that hold their contents as individual particles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

import mujoco
import numpy
import pytest

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ParticlesDoNotFitError
from semantic_digital_twin.physics.particles import (
    HollowCylinder,
    MeasuredFillLevel,
    MeasuredInflowRate,
    ParticleFill,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
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
PARTICLE_RADIUS = 0.005
SETTLE = timedelta(milliseconds=10)
"""
How long a test steps the physics before reading the contents back: a body has no pose
until the simulation has stepped once.
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


def test_a_cavity_packed_to_a_share_of_its_depth_stops_below_that_depth():
    """
    A packing meant to stand as deep as a fill level has to end where that level does,
    or the two describe different cups.
    """
    fill_fraction = 0.5
    surface = CUP.base_thickness + fill_fraction * (CUP.height - CUP.base_thickness)
    capacity = CUP.particle_capacity(PARTICLE_RADIUS, fill_fraction=fill_fraction)

    packed = CUP.particle_positions(PARTICLE_RADIUS, count=capacity)
    one_layer_more = CUP.particle_positions(PARTICLE_RADIUS, count=capacity + 1)

    assert max(float(position.z) for position in packed) <= surface - PARTICLE_RADIUS
    assert float(one_layer_more[-1].z) > surface - PARTICLE_RADIUS


def test_a_full_cavity_seats_every_particle_the_packing_places():
    """
    The capacity of the whole cavity is what packing it to the rim places, so the two
    ways of asking how much a container holds agree.
    """
    capacity = CUP.particle_capacity(PARTICLE_RADIUS)

    assert len(CUP.particle_positions(PARTICLE_RADIUS, count=capacity)) == capacity
    with pytest.raises(ParticlesDoNotFitError):
        CUP.particle_positions(PARTICLE_RADIUS, count=capacity + 1)


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
def test_the_filled_height_is_where_the_highest_particle_stands(cup_in_a_simulation):
    """
    The number a fill level is compared against is read off the contents themselves.
    """
    world, container, simulation = cup_in_a_simulation
    count = CUP.particle_capacity(PARTICLE_RADIUS, fill_fraction=0.5)
    highest_packed = max(
        float(position.z)
        for position in CUP.particle_positions(PARTICLE_RADIUS, count=count)
    )

    fill = CUP.fill_with_particles(
        container=container,
        world=world,
        simulator=simulation.simulator,
        particle_radius=PARTICLE_RADIUS,
        count=count,
    )
    simulation.step_simulation(SETTLE)

    fallen = 0.5 * GRAVITY * SETTLE.total_seconds() ** 2
    assert fill.filled_height_in(container) == pytest.approx(
        highest_packed / CUP.height, abs=fallen / CUP.height
    )


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


def _stand_container_in(world: World, name: str = "cup", offset: float = 0.0) -> Body:
    """
    Add a container standing on a world's root.

    :param world: The world to add the container to.
    :param name: Name of the container's body.
    :param offset: How far along x it stands from the root.
    :return: The container's body.
    """
    container = CUP.body(PrefixedName(name))
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
def test_a_measurement_reports_the_share_of_the_contents_standing_in_the_container(
    cup_in_a_simulation,
):
    """
    What a perception pipeline would report about a container is what stands in it, not
    what an equation integrated into it.
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
        count=CUP.particle_capacity(PARTICLE_RADIUS, fill_fraction=0.5),
    )
    simulation.step_simulation(SETTLE)

    measurement = MeasuredFillLevel(
        contents=fill,
        container=container,
        connection=annotation.fill_connection,
        world=world,
    )

    assert measurement.measure() == pytest.approx(fill.fraction_inside(container))
    assert measurement.measure() == pytest.approx(1.0)


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
        count=CUP.particle_capacity(PARTICLE_RADIUS, fill_fraction=0.5),
    )
    simulation.step_simulation(SETTLE)
    measurement = MeasuredFillLevel(
        contents=fill,
        container=container,
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
        connection=annotation.fill_connection,
        world=world,
    )

    assert measurement.report() == 0.0
    assert annotation.fill_level == pytest.approx(0.0)


# %% how fast the contents are arriving


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
    simulation.step_simulation(SETTLE)
    rate = MeasuredInflowRate(contents=fill, container=elsewhere)
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
    simulation.step_simulation(SETTLE)
    rate = MeasuredInflowRate(contents=fill, container=catcher)
    rate.observe(at=0.0)

    caught = fill.names[:2]
    for name in caught:
        simulation.simulator.set_body_position(
            body_name=name, position=numpy.array([1.0, 0.0, 0.02])
        )
    simulation.step_simulation(SETTLE)

    observed = rate.observe(at=0.5)

    assert fill.count_inside(catcher) == len(caught)
    assert observed == pytest.approx(len(caught) / len(fill.names) / 0.5)


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
    simulation.step_simulation(SETTLE)
    rate = MeasuredInflowRate(contents=fill, container=container)
    rate.observe(at=1.0)

    assert rate.observe(at=1.0) == 0.0
