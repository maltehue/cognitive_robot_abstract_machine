"""
What both studies share: the cups, the grains in them, and the physics they run in.

The plain cup, the wide receiver and the grains are those of
``demo_pouring_transfer_particles.py``, so the open-loop rig tips the very cup and
contents the closed-loop transfer pours. The spouted cup and the narrow receiver vary
one of them each, so a difference in the results can be put down to the one that
changed.
"""

# %% imports

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import mujoco
from typing_extensions import TYPE_CHECKING, Iterator, List, Optional

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.particles import (
    HollowCylinder,
    SpoutedHollowCylinder,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, HasSpout
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

# %% the cups

CUP_INNER_RADIUS = 0.035
"""
Inner radius of both source cups, in metres.
"""

CUP_HEIGHT = 0.06
"""
Height of both source cups, in metres.
"""


class SourceCup(StrEnum):
    """
    Which cup the contents are poured from.
    """

    CUP = "cup"
    """
    The demo's plain cup, which pours over its rim.
    """

    SPOUTED_CUP = "spouted_cup"
    """
    The same cup with a slot in its wall and an open ramp climbing from it to an outlet
    at the rim's height, like a watering can's spout.
    """

    @property
    def geometry(self) -> HollowCylinder:
        """
        :return: The cup's shape.
        """
        match self:
            case SourceCup.CUP:
                return HollowCylinder(inner_radius=CUP_INNER_RADIUS, height=CUP_HEIGHT)
            case SourceCup.SPOUTED_CUP:
                return SpoutedHollowCylinder(
                    inner_radius=CUP_INNER_RADIUS, height=CUP_HEIGHT
                )

    @property
    def pour_direction(self) -> float:
        """
        :return: Which way a tilt about the cup's ``y`` axis must turn to pour it,
            ``+1`` to tip its ``+x`` side down and ``-1`` for its ``-x`` side: the
            plain cup pours either way, the spout sits on the ``-x`` side.
        """
        match self:
            case SourceCup.CUP:
                return 1.0
            case SourceCup.SPOUTED_CUP:
                return -1.0


class ReceiverCup(StrEnum):
    """
    Which cup the contents are poured into.
    """

    WIDE = "wide"
    """
    The demo's receiver, 12 cm across: hard to miss.
    """

    NARROW = "narrow"
    """
    Half as wide, 6 cm across, so a stream aimed badly misses it.
    """

    @property
    def geometry(self) -> HollowCylinder:
        """
        :return: The receiver's shape.
        """
        match self:
            case ReceiverCup.WIDE:
                return HollowCylinder(inner_radius=0.06, height=0.1)
            case ReceiverCup.NARROW:
                return HollowCylinder(inner_radius=0.03, height=0.1)


@dataclass(frozen=True)
class Scene:
    """
    One pairing of a source cup with a receiver.
    """

    source: SourceCup
    """
    The cup poured from.
    """

    receiver: ReceiverCup
    """
    The cup poured into.
    """

    @property
    def label(self) -> str:
        """
        :return: A short identifier of this scene, usable in a file name.
        """
        return f"{self.source}_into_{self.receiver}"


def all_scenes() -> List[Scene]:
    """
    :return: Every source cup with every receiver.
    """
    return [
        Scene(source=source, receiver=receiver)
        for source in SourceCup
        for receiver in ReceiverCup
    ]


# %% annotating a source cup in a world


@dataclass(eq=False)
class PourableContainer(HasFillLevel):
    """
    A container whose fill level the pouring equation integrates as it tilts.
    """


@dataclass(eq=False)
class SpoutedPourableContainer(HasSpout):
    """
    A container that pours through its spout, with a fill level the pouring equation
    integrates as it tilts.
    """


@dataclass
class SourceInWorld:
    """
    A source cup standing in a world.
    """

    annotation: HasFillLevel
    """
    The cup's annotation, which carries its fill level and its drain.
    """

    spout: Optional[Body]
    """
    The cup's spout, or ``None`` for a cup that pours over its rim.
    """

    @property
    def root(self) -> Body:
        """
        :return: The cup's body.
        """
        return self.annotation.root

    @property
    def parts(self) -> List[Body]:
        """
        :return: The bodies whose contents count as the cup's: the cup, and its spout
            if it has one.
        """
        return [self.root] + ([self.spout] if self.spout is not None else [])

    @classmethod
    def build(
        cls,
        world: World,
        source: SourceCup,
        parent: Body,
        parent_T_cup: HomogeneousTransformationMatrix,
    ) -> SourceInWorld:
        """
        Add a source cup to a world, fixed to a parent, and annotate it.

        :param world: The world to add the cup to.
        :param source: Which cup to build.
        :param parent: What the cup is fixed to.
        :param parent_T_cup: Where the cup stands relative to its parent.
        :return: The cup in the world, its fill level not yet initialized.
        """
        geometry = source.geometry
        cup = geometry.body(PrefixedName("source_cup"))
        with world.modify_world():
            world.add_kinematic_structure_entity(cup)
            world.add_connection(
                FixedConnection(
                    parent=parent,
                    child=cup,
                    parent_T_connection_expression=parent_T_cup,
                )
            )
        if not isinstance(geometry, SpoutedHollowCylinder):
            annotation = PourableContainer(name=PrefixedName("source"), root=cup)
            with world.modify_world():
                world.add_semantic_annotation(annotation)
            return cls(annotation=annotation, spout=None)
        spout = geometry.spout_body(PrefixedName("source_spout"))
        with world.modify_world():
            world.add_kinematic_structure_entity(spout)
            world.add_connection(
                FixedConnection(
                    parent=cup,
                    child=spout,
                    parent_T_connection_expression=geometry.root_T_spout,
                )
            )
        annotation = SpoutedPourableContainer(
            name=PrefixedName("source"), root=cup, spout=spout
        )
        with world.modify_world():
            world.add_semantic_annotation(annotation)
        return cls(annotation=annotation, spout=spout)


# %% the contents

PARTICLE_RADIUS = 0.005
"""
Radius of one grain, in metres.
"""

PACKED_SHARE = 0.45
"""
The share of the source cup's cavity packed with grains, which settle to about 0.72 of
its height.
"""

SETTLE_TIME = timedelta(seconds=3)
"""
How long the grains settle in the cup before anything moves.
"""

MICROSTATE_OFFSET = timedelta(milliseconds=7)
"""
How much longer each further seed lets the grains settle.

The simulation is deterministic, but which grain clips the rim is not robust to where
the grains stand, so a few milliseconds more settling is a different pour. This is the
spread a single run cannot show.
"""


def settle_time_of(seed: int, settle_time: timedelta) -> timedelta:
    """
    :param seed: Which microstate of the grains to start from.
    :param settle_time: How long seed zero settles.
    :return: How long the grains of that seed settle.
    """
    return settle_time + seed * MICROSTATE_OFFSET


# %% the physics

STEP_SIZE = 2e-3
"""
Physics step, in seconds.
"""

INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
"""
Integrator the physics runs with; a pour is decided by the contact solver rather than by
the smooth dynamics between contacts.
"""

SCENE_FILE_NAME = "scene.xml"
"""
The name the simulation's model is written under, inside the trial's own directory.
"""


class IsolatedMujocoSim(MujocoSim):
    """
    A MuJoCo simulation that writes its model and the meshes it converts into a
    directory of its own, so simulations built side by side in separate processes do not
    overwrite each other's scene.
    """

    def __init__(self, world: World, scene_directory: Path, **kwargs) -> None:
        """
        :param world: The world to simulate.
        :param scene_directory: Where the model and its assets are written.
        :param kwargs: What :class:`MujocoSim` is built with.
        """
        self.default_file_path = str(scene_directory / SCENE_FILE_NAME)
        super().__init__(world=world, **kwargs)


@contextmanager
def stepped_simulation(world: World) -> Iterator[MujocoSim]:
    """
    Simulate a world headless, stepped only when told to, for as long as the context
    lasts; the model's directory is removed afterwards.

    :param world: The world to simulate.
    :return: The started simulation.
    """
    scene_directory = Path(tempfile.mkdtemp(prefix="pouring_model_verification_"))
    simulation = IsolatedMujocoSim(
        world=world,
        scene_directory=scene_directory,
        headless=True,
        step_size=STEP_SIZE,
        integrator=INTEGRATOR,
    )
    simulation.start_stepped_simulation()
    try:
        yield simulation
    finally:
        simulation.stop_simulation()
        shutil.rmtree(scene_directory, ignore_errors=True)
