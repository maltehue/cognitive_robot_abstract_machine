"""
Pour a cup of particles into another cup in MuJoCo, and watch the pouring equation
predict the same pour beside it.

The source cup holds its contents as individual spheres, so how much leaves it and how
much lands in the receiver is whatever the physics does with them. The same cup also
carries a fill level integrated from :class:`ArticulatedPouringEquation`, so the run
prints the analytic prediction and the measured pour side by side.

Run it with ``--headless`` for the numbers alone, or without for the MuJoCo viewer.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import timedelta

import mujoco

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.particles import HollowCylinder, ParticleFill
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.joint_state import JointState
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
from semantic_digital_twin.world_description.world_entity import Body

# %% the scene

SOURCE = HollowCylinder(inner_radius=0.035, height=0.1)
RECEIVER = HollowCylinder(inner_radius=0.07, height=0.1)
PARTICLE_RADIUS = 0.005

INITIAL_FILL = 0.6
"""
How deep the source cup starts, as a share of its cavity: the particles are packed to
that depth and the analytic fill level starts at the same number, so the two describe
the same cup.
"""

SOURCE_STAND = (-0.02, 0.2)
"""
Where the source cup hangs, as x and z in metres: beside the receiver and a cup's height
above it, so its rim clears the receiver's once it tilts.
"""

RECEIVER_STAND = 0.06
"""
Where the receiving cup stands on the ground, as x in metres.
"""

TILT_RATE = 0.4
"""
How fast the source cup turns over, in radians per second.
"""

FINAL_TILT = 2.2
"""
How far the source cup turns over, in radians: past horizontal, so it empties.
"""

SETTLE_TIME = timedelta(seconds=0.5)
"""
How long the contents stand before the pour starts.
"""

DRAIN_TIME = timedelta(seconds=2.0)
"""
How long the run keeps stepping after the cup has finished turning.
"""

CONTROL_PERIOD = timedelta(milliseconds=20)
"""
How much simulated time passes between two tilt commands.
"""

OUTFLOW_RATE_CONSTANT = 1.0
"""
Outflow rate constant of the analytic drain the pour is compared against.
"""

STEP_SIZE = 2e-3
"""
Physics step, in seconds.
"""

INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
"""
Integrator the physics runs with.

The simulator's own default is Runge-Kutta, which evaluates the dynamics four times a
step. A pour is decided by the contact solver rather than by the smooth dynamics between
contacts, so the extra evaluations cost this scene about six times its speed and change
neither where the grains go nor how many arrive.
"""


@dataclass(eq=False)
class PourableContainer(HasFillLevel):
    """
    A container whose fill level the pouring equation integrates as it tilts.
    """


# %% building the world


def build_world() -> tuple[World, Body, Body, RevoluteConnection, PourableContainer]:
    """
    Build the ground, the receiving cup, and the source cup on the joint that tips it.

    :return: The world, the source and receiving cup bodies, the tilt connection, and
        the source's annotation carrying the analytic fill level.
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

    receiver = RECEIVER.body(PrefixedName("receiving_cup"))
    with world.modify_world():
        world.add_kinematic_structure_entity(receiver)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=receiver,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=RECEIVER_STAND, reference_frame=world.root
                ),
            )
        )

    source = SOURCE.body(PrefixedName("source_cup"))
    with world.modify_world():
        world.add_kinematic_structure_entity(source)
        tilt = RevoluteConnection.create_with_dofs(
            world=world,
            parent=world.root,
            child=source,
            name=PrefixedName("source_cup_tilt"),
            axis=Vector3(0, 1, 0),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=math.pi, velocity=2.0),
            ),
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=SOURCE_STAND[0], z=SOURCE_STAND[1], reference_frame=world.root
            ),
        )
        world.add_connection(tilt)

    pourable_source = PourableContainer(
        name=PrefixedName("source_cup_contents"), root=source
    )
    with world.modify_world():
        world.add_semantic_annotation(pourable_source)
    pourable_source.initialize_fill_level(
        world=world,
        initial_fill=INITIAL_FILL,
        outflow_rate_constant=OUTFLOW_RATE_CONSTANT,
    )
    return world, source, receiver, tilt, pourable_source


# %% running the pour


def run(headless: bool) -> None:
    """
    Pour the source cup out and report where its contents went.

    :param headless: Whether to run without the MuJoCo viewer.
    """
    world, source, receiver, tilt, pourable_source = build_world()
    fill = SOURCE.fill_with_particles(
        container=source,
        world=world,
        particle_radius=PARTICLE_RADIUS,
        count=SOURCE.particle_capacity(PARTICLE_RADIUS, fill_fraction=INITIAL_FILL),
    )
    print(
        f"{len(fill.particles)} particles of radius {PARTICLE_RADIUS} m "
        f"({fill.volume * 1e6:.1f} ml) in a cavity of "
        f"{SOURCE.cavity_volume * 1e6:.1f} ml"
    )

    ParticleFill.settling_contact().apply_to([source, receiver] + list(world.bodies))

    simulation = MujocoSim(
        world=world,
        headless=headless,
        step_size=STEP_SIZE,
        integrator=INTEGRATOR,
    )
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(SETTLE_TIME)
        settled_depth = fill.filled_height_in(source)
        JointState.from_mapping(
            {pourable_source.fill_connection: settled_depth}
        ).apply_to(world)
        print(
            f"the packing settled to a depth of {settled_depth:.2f} of the cup, which "
            f"the pouring equation starts from"
        )
        _report(0.0, world, fill, source, receiver, pourable_source)

        period = CONTROL_PERIOD.total_seconds()
        elapsed = 0.0
        while elapsed < FINAL_TILT / TILT_RATE + DRAIN_TIME.total_seconds():
            elapsed += period
            world.state[tilt.raw_dof.id].position = min(FINAL_TILT, TILT_RATE * elapsed)
            world.step_physics(period)
            simulation.step_simulation(CONTROL_PERIOD)
            if round(elapsed, 6) % 0.5 < period:
                _report(elapsed, world, fill, source, receiver, pourable_source)
    finally:
        simulation.stop_simulation()


def _report(
    elapsed: float,
    world: World,
    fill: ParticleFill,
    source: Body,
    receiver: Body,
    pourable_source: PourableContainer,
) -> None:
    """
    Print one line comparing the analytic fill level against the particles.

    :param elapsed: Simulated time since the pour started, in seconds.
    :param world: The world holding the tilt.
    :param fill: The contents being poured.
    :param source: The cup being emptied.
    :param receiver: The cup being filled.
    :param pourable_source: The source's annotation carrying the analytic fill level.
    """
    in_source = fill.count_inside(source)
    in_receiver = fill.count_inside(receiver)
    print(
        f"t={elapsed:5.2f}s  "
        f"tilt={float(world.state[source.parent_connection.raw_dof.id].position):4.2f}rad  "
        f"source: {in_source:3d} particles, reaching {fill.filled_height_in(source):4.2f} "
        f"vs {pourable_source.fill_level:4.2f} predicted  "
        f"receiver: {in_receiver:3d}  "
        f"spilled: {len(fill.particles) - in_source - in_receiver:3d}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="run without the MuJoCo viewer"
    )
    run(headless=parser.parse_args().headless)
