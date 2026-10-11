"""
The open-loop study: the source cup of the transfer scene, full of grains, tipped along a
fixed tilt profile on a joint of its own, with no robot and no controller.

What the grains do along a known tilt is what the pouring equation claims to predict, so
this is the data its fidelity is judged on, and the data its parameters are fitted to.
"""

# %% imports

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

import pandas
from typing_extensions import Any, Dict, List, Self

from experiments.pouring_model_verification.parameters import (
    DEMO_PARAMETERS,
    ModelParameters,
)
from experiments.pouring_model_verification.records import (
    Phase,
    Sample,
    Study,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.scene import (
    PACKED_SHARE,
    PARTICLE_RADIUS,
    SETTLE_TIME,
    SourceCup,
    SourceInWorld,
    settle_time_of,
    stepped_simulation,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.particles import MeasuredFillLevel, ParticleFill
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
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
from semantic_digital_twin.world_description.world_entity import Body

# %% the rig

SAMPLE_PERIOD = timedelta(milliseconds=20)
"""
How much simulated time passes between two tilt commands, and between two samples.
"""

CUP_MOUNT_HEIGHT = 0.2
"""
How high above the ground the cup's tilt axis stands, in metres: high enough that the
grains it pours never pile up to its rim.
"""

MAXIMUM_TILT_RATE = 1.0
"""
How fast the rig turns the cup when a profile asks it to step, in radians per second.
"""


@dataclass
class TiltRig:
    """
    The source cup on a joint that tips it, above a ground the grains fall onto.
    """

    world: World
    """
    The world the rig stands in.
    """

    cup: SourceInWorld
    """
    The cup being tipped, carrying the fill level the model integrates.
    """

    @property
    def source(self) -> HasFillLevel:
        """
        :return: The cup's annotation.
        """
        return self.cup.annotation

    tilt: RevoluteConnection
    """
    The joint that tips it.
    """

    @classmethod
    def build(cls, parameters: ModelParameters, source: SourceCup) -> Self:
        """
        :param parameters: What the fill level the rig carries is integrated with.
        :param source: Which cup the rig tips.
        :return: The rig, the cup upright and empty.
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
        mount = Body(name=PrefixedName("cup_mount"))
        with world.modify_world():
            world.add_kinematic_structure_entity(mount)
            tilt = RevoluteConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=mount,
                name=PrefixedName("source_cup_tilt"),
                axis=Vector3(0, source.pour_direction, 0),
                dof_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=0.0, velocity=-2.0),
                    upper=DerivativeMap(position=math.pi, velocity=2.0),
                ),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=CUP_MOUNT_HEIGHT, reference_frame=world.root
                ),
            )
            world.add_connection(tilt)
        cup = SourceInWorld.build(
            world=world,
            source=source,
            parent=mount,
            parent_T_cup=HomogeneousTransformationMatrix(reference_frame=mount),
        )
        cup.annotation.initialize_fill_level(world=world, initial_fill=PACKED_SHARE)
        with world.modify_world():
            cup.annotation.add_fill_equation(
                parameters.applied_to(cup.annotation.fill_equation)
            )
        return cls(world=world, cup=cup, tilt=tilt)

    def set_tilt(self, tilt: float) -> None:
        """
        :param tilt: How far to turn the cup from upright, in radians.
        """
        self.world.state[self.tilt.raw_dof.id].position = tilt


# %% tilt profiles


class ProfileKey(StrEnum):
    """
    The keys a tilt profile is stored under.
    """

    KIND = "kind"
    TILT = "tilt"
    RATE = "rate"
    HOLD = "hold"


class ProfileKind(StrEnum):
    """
    Which kind of tilt profile a stored one is.
    """

    STEP = "step"
    RAMP = "ramp"


@dataclass(frozen=True)
class TiltProfile(ABC):
    """
    How the rig turns the cup over time.
    """

    hold: float
    """
    How long the final tilt is held, in seconds.
    """

    @property
    @abstractmethod
    def final_tilt(self) -> float:
        """
        :return: The tilt the profile ends at, in radians.
        """

    @property
    @abstractmethod
    def rate(self) -> float:
        """
        :return: How fast the profile turns the cup until it reaches its final tilt,
            in radians per second.
        """

    @property
    @abstractmethod
    def label(self) -> str:
        """
        :return: A short identifier of this profile, usable in a file name.
        """

    @abstractmethod
    def to_json(self) -> Dict[str, Any]: ...

    @property
    def duration(self) -> float:
        """
        :return: How long the profile runs, in seconds.
        """
        return self.final_tilt / self.rate + self.hold

    def tilt_at(self, elapsed: float) -> float:
        """
        :param elapsed: Time since the profile started, in seconds.
        :return: The tilt the profile asks for then, in radians.
        """
        return min(self.final_tilt, self.rate * elapsed)

    @staticmethod
    def from_json(data: Dict[str, Any]) -> TiltProfile:
        match ProfileKind(data[ProfileKey.KIND]):
            case ProfileKind.STEP:
                return StepAndHold(
                    tilt=data[ProfileKey.TILT], hold=data[ProfileKey.HOLD]
                )
            case ProfileKind.RAMP:
                return Ramp(
                    tilt_rate=data[ProfileKey.RATE],
                    end_tilt=data[ProfileKey.TILT],
                    hold=data[ProfileKey.HOLD],
                )


@dataclass(frozen=True)
class StepAndHold(TiltProfile):
    """
    Turn the cup to one tilt as fast as the rig turns and hold it there, so the contents
    relax to what that tilt leaves behind.
    """

    tilt: float = 0.0
    """
    The tilt that is held, in radians.
    """

    @property
    def final_tilt(self) -> float:
        return self.tilt

    @property
    def rate(self) -> float:
        return MAXIMUM_TILT_RATE

    @property
    def label(self) -> str:
        return f"step_{math.degrees(self.tilt):05.1f}deg"

    def to_json(self) -> Dict[str, Any]:
        return {
            ProfileKey.KIND: ProfileKind.STEP,
            ProfileKey.TILT: self.tilt,
            ProfileKey.HOLD: self.hold,
        }


@dataclass(frozen=True)
class Ramp(TiltProfile):
    """
    Turn the cup at a steady rate until it is nearly upside down, so the contents never
    come to rest on the way.
    """

    tilt_rate: float = 0.2
    """
    How fast the cup turns, in radians per second.
    """

    end_tilt: float = 2.0
    """
    The tilt the ramp stops at, in radians.
    """

    @property
    def final_tilt(self) -> float:
        return self.end_tilt

    @property
    def rate(self) -> float:
        return self.tilt_rate

    @property
    def label(self) -> str:
        return f"ramp_{self.tilt_rate:.2f}rads"

    def to_json(self) -> Dict[str, Any]:
        return {
            ProfileKey.KIND: ProfileKind.RAMP,
            ProfileKey.TILT: self.end_tilt,
            ProfileKey.RATE: self.tilt_rate,
            ProfileKey.HOLD: self.hold,
        }


STEP_HOLD_DURATION = 20.0
"""
How long a step's tilt is held, in seconds: about three times the relaxation the grains
of this scene show at 85 degrees, so a step ends close to what its tilt leaves behind.
"""


def fitting_profiles() -> List[TiltProfile]:
    """
    :return: The profiles the model's parameters are fitted to: one held tilt every five
        degrees across the range the grains start and finish flowing in.
    """
    return [
        StepAndHold(tilt=math.radians(degrees), hold=STEP_HOLD_DURATION)
        for degrees in range(55, 110, 5)
    ]


def held_out_profiles() -> List[TiltProfile]:
    """
    :return: The profiles a fit is tested on and never fitted to: steady turns at three
        rates, fast enough that the contents never settle on the way.
    """
    return [Ramp(tilt_rate=rate, hold=4.0) for rate in (0.1, 0.2, 0.4)]


# %% running one profile


@dataclass(frozen=True)
class TiltRigTrial:
    """
    One tilt profile run on one microstate of the grains.
    """

    source: SourceCup
    """
    Which cup is tipped.
    """

    profile: TiltProfile
    """
    How the cup is turned.
    """

    seed: int
    """
    Which microstate the grains start from; see :func:`settle_time_of`.
    """

    @property
    def name(self) -> str:
        """
        :return: The directory name this trial writes its record into.
        """
        return f"{Study.TILT_RIG}_{self.source}_{self.profile.label}_seed{self.seed}"

    def run(self) -> TrialRecord:
        """
        Fill the cup, let the grains settle, turn the cup along the profile and sample
        how much of them it keeps.

        :return: The trial's record.
        """
        started = time.monotonic()
        rig = TiltRig.build(DEMO_PARAMETERS, self.source)
        ParticleFill.settling_contact().apply_to(rig.world.bodies)
        with stepped_simulation(rig.world) as simulation:
            simulation.step_simulation(SAMPLE_PERIOD)
            geometry = self.source.geometry
            fill = geometry.fill_with_particles(
                container=rig.source.root,
                world=rig.world,
                simulator=simulation.simulator,
                particle_radius=PARTICLE_RADIUS,
                count=geometry.particle_capacity(
                    PARTICLE_RADIUS, fill_fraction=PACKED_SHARE
                ),
            )
            simulation.step_simulation(settle_time_of(self.seed, SETTLE_TIME))
            fill.bulk_volume_per_particle = fill.bulk_volume_per_particle_in(
                rig.source.root, rig.source.capacity
            )
            measured = MeasuredFillLevel(
                contents=fill,
                container=rig.source.root,
                capacity=rig.source.capacity,
                connection=rig.source.fill_connection,
                world=rig.world,
                spout=rig.cup.spout,
            )
            JointState.from_mapping(
                {rig.source.fill_connection: measured.measure()}
            ).apply_to(rig.world)
            samples = self._tip(rig, simulation, fill, measured)
        return TrialRecord(
            description={
                TrialKey.STUDY: Study.TILT_RIG,
                TrialKey.NAME: self.name,
                TrialKey.SOURCE: self.source,
                TrialKey.SEED: self.seed,
                TrialKey.PROFILE: self.profile.to_json(),
                TrialKey.PARAMETERS: DEMO_PARAMETERS.to_json(),
                TrialKey.EQUATION: rig.source.fill_equation.ungated().to_json(),
                TrialKey.WALL_TIME: time.monotonic() - started,
            },
            samples=samples,
        )

    def _tip(
        self,
        rig: TiltRig,
        simulation: MujocoSim,
        fill: ParticleFill,
        measured: MeasuredFillLevel,
    ) -> pandas.DataFrame:
        """
        Turn the cup along the profile, sampling once per command.

        :param rig: The rig being run.
        :param simulation: The physics it runs in.
        :param fill: The grains in the cup.
        :param measured: What reads the cup's fill level off them.
        :return: One row per sample, the first before the cup moves.
        """
        period = SAMPLE_PERIOD.total_seconds()
        steps = math.ceil(self.profile.duration / period)
        rows = [self._sample(0.0, rig, fill, measured)]
        for step in range(1, steps + 1):
            elapsed = step * period
            rig.set_tilt(self.profile.tilt_at(elapsed))
            rig.world.step_physics(period)
            simulation.step_simulation(SAMPLE_PERIOD)
            rows.append(self._sample(elapsed, rig, fill, measured))
        return pandas.DataFrame(rows)

    @staticmethod
    def _sample(
        elapsed: float, rig: TiltRig, fill: ParticleFill, measured: MeasuredFillLevel
    ) -> Dict[Sample, Any]:
        """
        :return: One row of the rig's samples, taken now.
        """
        return {
            Sample.TIME: elapsed,
            Sample.PHASE: Phase.TILT,
            Sample.TILT: rig.tilt.position,
            Sample.MEASURED_SOURCE_FILL: measured.measure(),
            Sample.MODEL_SOURCE_FILL: rig.source.fill_level,
            Sample.GRAINS_IN_SOURCE: fill.count_inside(*rig.cup.parts),
        }
