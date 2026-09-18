"""
Tests for containers that hold their contents as individual particles.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy
import pytest

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ParticlesDoNotFitError
from semantic_digital_twin.physics.particles import HollowCylinder, ParticleFill
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
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

CUP = HollowCylinder(inner_radius=0.035, height=0.1)
PARTICLE_RADIUS = 0.005


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


# %% the particles in the world


def test_each_particle_hangs_from_the_world_root_on_a_free_connection(
    world_with_ground,
):
    """
    MuJoCo only accepts a free joint on a body of the top level, so every particle hangs
    from the root rather than from the container it stands in.
    """
    container = _stand_container_in(world_with_ground)

    fill = CUP.fill_with_particles(
        container=container,
        world=world_with_ground,
        particle_radius=PARTICLE_RADIUS,
        count=12,
    )

    assert len(fill.particles) == 12
    for particle in fill.particles:
        assert isinstance(particle.parent_connection, Connection6DoF)
        assert particle.parent_connection.parent is world_with_ground.root


def test_a_particle_moved_out_of_a_container_stops_counting_towards_it(
    world_with_ground,
):
    """
    The count is read off where the particles are, so it follows them out of the
    container rather than staying at what was packed.
    """
    container = _stand_container_in(world_with_ground)
    fill = CUP.fill_with_particles(
        container=container,
        world=world_with_ground,
        particle_radius=PARTICLE_RADIUS,
        count=12,
    )
    assert fill.count_inside(container) == 12

    escaped = fill.particles[0]
    world_with_ground.state[escaped.parent_connection.x.id].position = 1.0
    world_with_ground.notify_state_change()

    assert fill.count_inside(container) == 11
    assert fill.fraction_inside(container) == pytest.approx(11 / 12)


# %% the particles in the physics


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_contents_stay_in_an_upright_container_and_pour_out_of_a_tilted_one(
    world_with_ground,
):
    """
    The point of holding a container's contents as particles: standing upright it keeps
    them, tilted past its rim it does not, and neither is a number anyone integrated.
    """
    container, tilt = _hang_container_on_a_tilt_in(world_with_ground)
    fill = CUP.fill_with_particles(
        container=container,
        world=world_with_ground,
        particle_radius=PARTICLE_RADIUS,
        count=40,
    )

    simulation = MujocoSim(world=world_with_ground, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(timedelta(seconds=1.0))
        held_upright = fill.count_inside(container)
        world_with_ground.state[tilt.raw_dof.id].position = math.pi
        world_with_ground.notify_state_change()
        simulation.step_simulation(timedelta(seconds=2.0))
        held_upside_down = fill.count_inside(container)
    finally:
        simulation.stop_simulation()

    assert held_upright == len(fill.particles)
    assert held_upside_down == 0


def _stand_container_in(world: World) -> Body:
    """
    Add a container standing on a world's root.

    :param world: The world to add the container to.
    :return: The container's body.
    """
    container = CUP.body(PrefixedName("cup"))
    with world.modify_world():
        world.add_kinematic_structure_entity(container)
        world.add_connection(FixedConnection(parent=world.root, child=container))
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


# %% how far the contents reach


def test_the_filled_height_is_where_the_highest_particle_stands(world_with_ground):
    """
    The number a fill level is compared against is read off the contents themselves.
    """
    container = _stand_container_in(world_with_ground)
    count = CUP.particle_capacity(PARTICLE_RADIUS, fill_fraction=0.5)
    highest_packed = max(
        float(position.z)
        for position in CUP.particle_positions(PARTICLE_RADIUS, count=count)
    )

    fill = CUP.fill_with_particles(
        container=container,
        world=world_with_ground,
        particle_radius=PARTICLE_RADIUS,
        count=count,
    )

    assert fill.filled_height_in(container) == pytest.approx(
        highest_packed / CUP.height
    )


def test_an_empty_container_is_reached_by_nothing(world_with_ground):
    """
    A container holding none of the contents reads as empty rather than as however deep
    the contents are elsewhere.
    """
    container = _stand_container_in(world_with_ground)
    fill = ParticleFill.spawn(
        world=world_with_ground,
        container=container,
        positions=[],
        particle_radius=PARTICLE_RADIUS,
    )

    assert fill.filled_height_in(container) == 0.0
    assert fill.fraction_inside(container) == 0.0


# %% what the particles' surfaces do


def test_the_particles_are_given_contact_parameters_that_let_them_settle(
    world_with_ground,
):
    """
    Particles at a physics engine's own damping bounce back out of the container they
    were poured into, so they carry their own contact parameters.
    """
    container = _stand_container_in(world_with_ground)

    fill = CUP.fill_with_particles(
        container=container,
        world=world_with_ground,
        particle_radius=PARTICLE_RADIUS,
        count=4,
    )

    for particle in fill.particles:
        for geometry in particle.collision:
            assert (
                geometry.get_simulator_property_of_type(ContactParameters)
                == ParticleFill.settling_contact()
            )
