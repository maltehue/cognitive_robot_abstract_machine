"""
The qualitative rule that a tilted cup out of which nothing comes must be tilted
further, added to the model-predictive controller.

A model that expects the pour to start sooner than it does holds the cup where it
predicts a flow, and the pour stalls. The rule notices that nothing has arrived for a
while and responds in one of two ways: it overrides the model with a slow tilt until
grains arrive, or it revises the model's repose angle so its predicted onset lies just
below the tilt nothing came out at, which lets the model drive the tilt again.
"""

# %% imports

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Optional

from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable
from typing_extensions import Any, Dict, List

from experiments.pouring_model_verification.pid import signed_tilt_about_pour_axis
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
)
from semantic_digital_twin.spatial_types.spatial_types import RotationMatrix
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from experiments.pouring_model_verification.concept_revision import Diagnosis

# %% the rule

OVERRIDE_WEIGHT = float(DefaultWeights.WEIGHT_COLLISION_AVOIDANCE)
"""
How much the override counts in the controller by default: below the fill row and the
aim, so it only acts where they leave room.
"""

FILL_ROW_WEIGHT = float(DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE)
"""
The weight of the model's fill row, which an override weighted as much can outvote.
"""


class OnsetResponse(StrEnum):
    """
    What the rule does about a stalled pour.
    """

    OVERRIDE = "override"
    """
    A low-priority task tilts the cup further at a set speed until grains arrive.
    """

    REVISION = "revision"
    """
    The repose angle is raised so the model's onset lies just below the current tilt,
    and the model drives the tilt again.
    """

    CONCEPT_AWARE_REVISION = "concept_aware_revision"
    """
    The source is weighed as well: a stall while it holds its grains raises the repose
    angle as :attr:`REVISION` does, a stall while it loses them corrects the discharge
    coefficient to where the stream was seen to land.
    """

    @property
    def revises_repose_angle(self) -> bool:
        """
        :return: Whether this response revises the model's repose angle.
        """
        return self is not OnsetResponse.OVERRIDE


class TwoSidedOverrideError(Exception):
    """
    Raised for an override told to lower the repose angle: only a revision has one to
    lower.
    """

    def __init__(self) -> None:
        super().__init__("only a revision can lower the repose angle")


class OnsetRuleKey(StrEnum):
    """
    A key of a rule written to JSON.
    """

    RESPONSE = "response"
    PATIENCE = "patience"
    OVERRIDE_VELOCITY = "override_velocity"
    OVERRIDE_WEIGHT = "override_weight"
    LOWERS = "lowers"


@dataclass(frozen=True)
class OnsetRule:
    """
    How long the rule waits for grains, and how it responds when none arrive.
    """

    response: OnsetResponse
    """
    What it does about a stall.
    """

    patience: float
    """
    How long nothing may arrive before the pour counts as stalled, in seconds.
    """

    override_velocity: float = 0.0
    """
    How fast the override tilts the cup, in radians per second.
    """

    override_weight: float = OVERRIDE_WEIGHT
    """
    How much the override counts in the controller.
    """

    lowers: bool = False
    """
    Whether a revision also lowers the repose angle when grains arrive while the model
    predicts none, so it is revised in both directions.
    """

    def __post_init__(self) -> None:
        if self.lowers and not self.response.revises_repose_angle:
            raise TwoSidedOverrideError()

    @property
    def label(self) -> str:
        """
        :return: A short identifier of the rule, usable in a file name.
        """
        if self.response is OnsetResponse.OVERRIDE:
            weight = (
                ""
                if self.override_weight == OVERRIDE_WEIGHT
                else f"_w{self.override_weight:g}"
            )
            return (
                f"{self.response}_t{self.patience:g}_v{self.override_velocity:g}"
                f"{weight}"
            )
        two_sided = "_two_sided" if self.lowers else ""
        return f"{self.response}_t{self.patience:g}{two_sided}"

    def to_json(self) -> Dict[str, Any]:
        return {
            OnsetRuleKey.RESPONSE: self.response,
            OnsetRuleKey.PATIENCE: self.patience,
            OnsetRuleKey.OVERRIDE_VELOCITY: self.override_velocity,
            OnsetRuleKey.OVERRIDE_WEIGHT: self.override_weight,
            OnsetRuleKey.LOWERS: self.lowers,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> OnsetRule:
        return cls(
            response=OnsetResponse(data[OnsetRuleKey.RESPONSE]),
            patience=data[OnsetRuleKey.PATIENCE],
            override_velocity=data[OnsetRuleKey.OVERRIDE_VELOCITY],
            override_weight=data[OnsetRuleKey.OVERRIDE_WEIGHT],
            lowers=data.get(OnsetRuleKey.LOWERS, False),
        )


# %% noticing a stall

ARRIVAL = 1.0
"""
How many grains the reading has to rise by to count as grains arriving; less is noise.
"""

TIMING_TOLERANCE = 1e-6
"""
How much earlier than the patience a stall may be called, in seconds, so readings on a
fixed period are not lost to rounding.
"""


@dataclass
class StallWatch:
    """
    Watches the receiver's reading for grains arriving.
    """

    rule: OnsetRule
    """
    How long it waits.
    """

    goal: float
    """
    How many grains the receiver is to hold.
    """

    tolerance: float
    """
    How many grains short of the goal count as reached.
    """

    _level: Optional[float] = field(init=False, default=None, repr=False)
    """
    The reading grains last arrived at.
    """

    _arrived_at: Optional[float] = field(init=False, default=None, repr=False)
    """
    When grains last arrived, or the watch started, in seconds.
    """

    def observe(self, reading: float, at: float) -> bool:
        """
        :param reading: How many grains the receiver is read to hold.
        :param at: When it was read, in seconds.
        :return: Whether the pour is stalled: short of the goal, with nothing arrived
            for the rule's patience.
        """
        if self._level is None or reading >= self._level + ARRIVAL:
            self._level, self._arrived_at = reading, at
        if reading >= self.goal - self.tolerance:
            return False
        return at - self._arrived_at >= self.rule.patience - TIMING_TOLERANCE

    def restart(self, at: float) -> None:
        """
        Wait a full patience again from now, after a response has changed the pour.

        :param at: The time now, in seconds.
        """
        self._arrived_at = at


ARRIVAL_NOISE_MULTIPLE = 2.0
"""
How many standard deviations of the sensor's noise a reading has to rise by, on top of
one grain, to count as evidence that grains arrived.
"""

ARRIVAL_CONFIRMATIONS = 3
"""
How many readings in a row have to show that evidence before grains count as arrived, so
one noisy reading does not.
"""


@dataclass
class ArrivalWatch:
    """
    Watches the receiver's reading for the first grains of the pour arriving.
    """

    noise: float
    """
    The standard deviation of the sensor's noise, in grains.
    """

    reference: float
    """
    How many grains the receiver held before the pour.
    """

    _first_above: Optional[float] = field(init=False, default=None, repr=False)
    """
    When the current run of readings above the evidence began, in seconds.
    """

    _above: int = field(init=False, default=0, repr=False)
    """
    How many readings in a row have shown the evidence.
    """

    _arrived: bool = field(init=False, default=False, repr=False)
    """
    Whether the arrival was already reported.
    """

    @property
    def evidence(self) -> float:
        """
        :return: How many grains above the reference count as grains arriving.
        """
        return ARRIVAL + ARRIVAL_NOISE_MULTIPLE * self.noise

    def observe(self, reading: float, at: float) -> Optional[float]:
        """
        :param reading: How many grains the receiver is read to hold.
        :param at: When it was read, in seconds.
        :return: When the first reading showing the arrival was taken, in seconds, at
            the reading that confirms it, else ``None``; reported once per pour.
        """
        if self._arrived:
            return None
        if reading < self.reference + self.evidence:
            self._above, self._first_above = 0, None
            return None
        if self._above == 0:
            self._first_above = at
        self._above += 1
        if self._above < ARRIVAL_CONFIRMATIONS:
            return None
        self._arrived = True
        return self._first_above


# %% responding to it


@dataclass
class OnsetResponder(ABC):
    """
    What the rule does on each reading, stalled or not.
    """

    def observe(self, at: float, source_grains: float) -> None:
        """
        Take in what is measured on each reading, stalled or not.

        :param at: The time now, in seconds.
        :param source_grains: How many grains the source is read to hold.
        """

    @abstractmethod
    def respond(
        self, stalled: bool, at: float, tilt: float, source_fill: float
    ) -> bool:
        """
        :param stalled: Whether the pour is stalled.
        :param at: The time now, in seconds.
        :param tilt: The cup's tilt now, in radians.
        :param source_fill: The source's reported fill level.
        :return: Whether the response changed the pour, so the watch waits anew.
        """

    def lower(self, at: float, tilt: float, source_fill: float) -> bool:
        """
        Respond to grains that arrived although the model predicted none.

        :param at: The time now, in seconds.
        :param tilt: The tilt the arriving grains left the cup at, in radians.
        :param source_fill: The source's reported fill level.
        :return: Whether the response changed the model.
        """
        return False

    @property
    def revisions(self) -> List[Revision]:
        """
        :return: Every revision of the model this response made, in order.
        """
        return []

    @property
    def diagnoses(self) -> List[Diagnosis]:
        """
        :return: Every stall this response blamed on a concept of the model, in order.
        """
        return []


@dataclass
class OnsetOverrideSwitch(OnsetResponder):
    """
    Switches the override task on while the pour is stalled and off once grains arrive.
    """

    active: FloatVariable
    """
    The variable the override task is scaled by: one while it runs, else zero.
    """

    variables: FloatVariableData
    """
    Where that variable's value lives.
    """

    def respond(
        self, stalled: bool, at: float, tilt: float, source_fill: float
    ) -> bool:
        self.variables.set_value(self.active, 1.0 if stalled else 0.0)
        return False


REVISION_MARGIN = 0.02
"""
How far below the current tilt a revision puts the model's onset, in radians, so the
model predicts a small flow and keeps a slope to tilt along.
"""


class RevisionCause(StrEnum):
    """
    What made the rule revise the repose angle.
    """

    STALL = "stall"
    """
    Nothing arrived for the patience; the onset is moved just below the tilt.
    """

    ARRIVAL = "arrival"
    """
    Grains arrived while the model predicted none; the onset is lowered to the tilt
    they left at.
    """


class RevisionKey(StrEnum):
    """
    A key of a logged revision written to JSON.
    """

    TIME = "time"
    TILT = "tilt"
    OLD_REPOSE_ANGLE = "old_repose_angle"
    NEW_REPOSE_ANGLE = "new_repose_angle"
    CAUSE = "cause"


@dataclass(frozen=True)
class Revision:
    """
    One change the rule made to the repose angle.
    """

    time: float
    """
    When it was made, in seconds.
    """

    tilt: float
    """
    The tilt nothing had come out at, in radians.
    """

    old_repose_angle: float
    """
    The repose angle before, in radians.
    """

    new_repose_angle: float
    """
    The repose angle after, in radians.
    """

    cause: RevisionCause = RevisionCause.STALL
    """
    What made the rule revise it.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            RevisionKey.TIME: self.time,
            RevisionKey.TILT: self.tilt,
            RevisionKey.OLD_REPOSE_ANGLE: self.old_repose_angle,
            RevisionKey.NEW_REPOSE_ANGLE: self.new_repose_angle,
            RevisionKey.CAUSE: self.cause,
        }


@dataclass
class OnsetRevision(OnsetResponder):
    """
    Moves the model's onset to just below the tilt nothing came out at, by correcting
    its repose angle, and logs every change.
    """

    equation: ArticulatedPouringEquation
    """
    The source's drain, whose fitted repose angle is corrected.
    """

    correction: FloatVariable
    """
    The angle added to the fitted repose angle.
    """

    variables: FloatVariableData
    """
    Where that angle's value lives.
    """

    log: List[Revision] = field(init=False, default_factory=list)
    """
    Every revision made, in order.
    """

    @property
    def revisions(self) -> List[Revision]:
        return self.log

    @property
    def repose_angle(self) -> float:
        """
        :return: The repose angle the model currently pours with, in radians.
        """
        return self.equation.repose_angle + self.variables.get_value(self.correction)

    def respond(
        self, stalled: bool, at: float, tilt: float, source_fill: float
    ) -> bool:
        if not stalled:
            return False
        self._move_onset_to(
            tilt - REVISION_MARGIN, at, tilt, source_fill, RevisionCause.STALL
        )
        return True

    def lower(self, at: float, tilt: float, source_fill: float) -> bool:
        """
        Lower the model's onset to the tilt the arriving grains left at, if the model
        put it higher.
        """
        if self._uncorrected_onset(source_fill) + self._correction <= tilt:
            return False
        self._move_onset_to(tilt, at, tilt, source_fill, RevisionCause.ARRIVAL)
        return True

    @property
    def _correction(self) -> float:
        """
        :return: The angle currently added to the fitted repose angle, in radians.
        """
        return self.variables.get_value(self.correction)

    def _uncorrected_onset(self, source_fill: float) -> float:
        """
        :param source_fill: The source's reported fill level.
        :return: The tilt the fitted model predicts the contents reach the lip at.
        """
        uncorrected = replace(self.equation, repose_correction=None)
        return float(uncorrected.onset_tilt(source_fill).evaluate()[0])

    def _move_onset_to(
        self,
        onset: float,
        at: float,
        tilt: float,
        source_fill: float,
        cause: RevisionCause,
    ) -> None:
        """
        Correct the repose angle so the model's onset lies at a tilt, and log it.

        :param onset: Where the onset is to lie, in radians.
        :param at: The time now, in seconds.
        :param tilt: The tilt the evidence was seen at, in radians.
        :param source_fill: The source's reported fill level.
        :param cause: What made the rule revise it.
        """
        old = self.repose_angle
        self.variables.set_value(
            self.correction, onset - self._uncorrected_onset(source_fill)
        )
        self.log.append(
            Revision(
                time=at,
                tilt=tilt,
                old_repose_angle=old,
                new_repose_angle=self.repose_angle,
                cause=cause,
            )
        )


# %% the override in the controller


@dataclass(eq=False, repr=False)
class TiltOverride(Task):
    """
    Asks the hand to turn about the pour axis at a set speed, scaled by a switch read
    anew every cycle, so it acts only while the switch is on.
    """

    root_link: Body = field(kw_only=True)
    """
    The frame the carry orientation is given in.
    """

    tip_link: Body = field(kw_only=True)
    """
    The hand holding the cup.
    """

    root_R_carry: RotationMatrix = field(kw_only=True)
    """
    The orientation the hand carries the cup upright in.
    """

    active: FloatVariable = field(kw_only=True)
    """
    One while the override runs, else zero.
    """

    velocity: float = field(kw_only=True)
    """
    How fast the cup is turned, in radians per second.
    """

    weight: float = field(default=OVERRIDE_WEIGHT, kw_only=True)
    """
    How much the override counts in the controller.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        :param context: The build context.
        :return: One velocity row on the tilt, weighted by the switch.
        """
        artifacts = NodeArtifacts()
        tilt = signed_tilt_about_pour_axis(
            context.world.compose_forward_kinematics_expression(
                self.root_link, self.tip_link
            ),
            self.root_R_carry,
        )
        artifacts.constraints.add_velocity_eq_constraint(
            velocity_goal=self.active * self.velocity,
            quadratic_weight=self.active * self.weight,
            task_expression=tilt,
            velocity_limit=self.velocity,
            name=f"{self.name}_tilt_velocity",
        )
        return artifacts
