"""
Choosing which concept of the model to revise when a pour stalls.

A receiver that gains nothing has two explanations the model can tell apart once the
source is weighed as well: the contents have not started to flow, so the onset is wrong
and the repose angle is raised; or they flow and land elsewhere, so the aim is wrong and
the discharge coefficient, which sets where the stream lands, is solved anew from where
the stream was seen to cross the receiver's opening.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy
from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable
from scipy.optimize import brentq
from typing_extensions import Any, Dict, List, Optional, Protocol, Tuple

from experiments.pouring_model_verification.onset_rule import (
    ARRIVAL,
    ARRIVAL_NOISE_MULTIPLE,
    OnsetResponder,
    OnsetRevision,
    Revision,
)

# %% what was revised, and why


class RevisedConcept(StrEnum):
    """
    A concept of the pouring model a stall can be blamed on.
    """

    REPOSE_ANGLE = "repose_angle"
    """
    Where the pour starts: the contents have not reached the lip.
    """

    DISCHARGE_COEFFICIENT = "discharge_coefficient"
    """
    Where the stream lands: the contents flow but miss the receiver.
    """


class StallEvidence(StrEnum):
    """
    What the source's reading said when the receiver stalled.
    """

    SOURCE_HOLDING = "source_holding"
    """
    The source lost no more grains than its reading's noise explains.
    """

    SOURCE_LOSING = "source_losing"
    """
    The source lost grains the receiver did not gain.
    """


class DiagnosisKey(StrEnum):
    """
    A key of a logged diagnosis written to JSON.
    """

    TIME = "time"
    CONCEPT = "concept"
    EVIDENCE = "evidence"
    SOURCE_LOSS = "source_loss"
    REQUIRED_LOSS = "required_loss"
    OLD_VALUE = "old_value"
    NEW_VALUE = "new_value"


@dataclass(frozen=True)
class Diagnosis:
    """
    One stall, the concept it was blamed on, and how that concept was changed.
    """

    time: float
    """
    When the stall was diagnosed, in seconds.
    """

    concept: RevisedConcept
    """
    The concept blamed.
    """

    evidence: StallEvidence
    """
    What the source's reading said.
    """

    source_loss: float
    """
    How many grains the source was read to lose over the patience.
    """

    required_loss: float
    """
    How many it had to lose to count as losing them.
    """

    old_value: float
    """
    The concept's value before: the repose angle in radians, or the discharge
    coefficient.
    """

    new_value: float
    """
    Its value after; the same as before where nothing could be revised.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            DiagnosisKey.TIME: self.time,
            DiagnosisKey.CONCEPT: self.concept,
            DiagnosisKey.EVIDENCE: self.evidence,
            DiagnosisKey.SOURCE_LOSS: self.source_loss,
            DiagnosisKey.REQUIRED_LOSS: self.required_loss,
            DiagnosisKey.OLD_VALUE: self.old_value,
            DiagnosisKey.NEW_VALUE: self.new_value,
        }


# %% telling whether the source is losing grains


@dataclass
class SourceWatch:
    """
    Watches the source's weighed content for grains leaving it.
    """

    noise: float
    """
    The standard deviation of the weighing's noise, in grains.
    """

    patience: float
    """
    How far back the loss is measured, in seconds: the time the receiver has to gain
    nothing for a stall.
    """

    _times: List[float] = field(init=False, default_factory=list, repr=False)
    """
    When each reading was taken, in seconds, oldest first.
    """

    _grains: List[float] = field(init=False, default_factory=list, repr=False)
    """
    How many grains each reading put in the source.
    """

    def observe(self, grains: float, at: float) -> None:
        """
        :param grains: How many grains the source is read to hold.
        :param at: When it was read, in seconds.
        """
        self._times.append(at)
        self._grains.append(grains)

    def _window(self, at: float) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        :return: The times and readings of the last patience before a time.
        """
        times = numpy.asarray(self._times)
        within = times >= at - self.patience - 1e-9
        return times[within], numpy.asarray(self._grains)[within]

    def lost(self, at: float) -> float:
        """
        :param at: The time now, in seconds.
        :return: How many grains the source lost over the last patience, by the slope
            of a line through its readings; nothing with fewer than two readings.
        """
        times, grains = self._window(at)
        if len(times) < 2:
            return 0.0
        slope = numpy.polyfit(times, grains, 1)[0]
        return float(-slope * self.patience)

    def evidence(self, at: float) -> float:
        """
        :param at: The time now, in seconds.
        :return: How many grains the source has to lose over the patience to count as
            losing them: a grain plus the noise the line's slope carries.
        """
        times, _ = self._window(at)
        spread = float(numpy.sum((times - times.mean()) ** 2)) if len(times) else 0.0
        if spread == 0.0:
            return math.inf
        slope_noise = self.noise / math.sqrt(spread)
        return ARRIVAL + ARRIVAL_NOISE_MULTIPLE * slope_noise * self.patience

    def is_losing(self, at: float) -> bool:
        """
        :param at: The time now, in seconds.
        :return: Whether the source lost more grains than its noise explains.
        """
        return self.lost(at) >= self.evidence(at)


# %% where the model throws the stream


@dataclass(frozen=True)
class Launch:
    """
    The stream as the model predicts it leaves the lip now: its speed and direction, and
    how far it falls to the receiver's opening.
    """

    speed: float
    """
    The predicted exit speed, in metres per second.
    """

    horizontal_share: float
    """
    The share of the exit speed that is horizontal.
    """

    vertical_share: float
    """
    The share of the exit speed that is upwards; negative for a stream launched
    downwards.
    """

    drop: float
    """
    How far the lip stands above the receiver's opening, in metres.
    """

    gravity: float
    """
    Gravitational acceleration, in metres per second squared.
    """

    def distance(self, speed: float) -> float:
        """
        :param speed: An exit speed, in metres per second.
        :return: How far from the lip, horizontally, a stream of that speed crosses the
            receiver's opening, in metres.
        """
        upwards = speed * self.vertical_share
        flight_time = (
            upwards + math.sqrt(upwards**2 + 2 * self.gravity * self.drop)
        ) / self.gravity
        return speed * self.horizontal_share * flight_time

    def speed_for(self, distance: float) -> float:
        """
        :param distance: How far from the lip the stream was seen to cross the
            receiver's opening, in metres.
        :return: The exit speed that throws it that far, in metres per second.
        """
        if distance <= 0.0:
            return 0.0
        upper = max(self.speed, 1e-3)
        while self.distance(upper) < distance:
            upper *= 2.0
        return float(brentq(lambda speed: self.distance(speed) - distance, 0.0, upper))


@dataclass
class DischargeRevision:
    """
    Corrects the discharge coefficient so the model throws the stream where it was seen
    to land.
    """

    coefficient: float
    """
    The discharge coefficient the model was given.
    """

    scale: FloatVariable
    """
    The factor the model's coefficient is multiplied by.
    """

    variables: FloatVariableData
    """
    Where that factor's value lives.
    """

    @property
    def value(self) -> float:
        """
        :return: The discharge coefficient the model currently pours with.
        """
        return self.coefficient * self.variables.get_value(self.scale)

    def revise(self, launch: Launch, seen_distance: float) -> Tuple[float, float]:
        """
        :param launch: The stream as the model predicts it now.
        :param seen_distance: How far from the lip it was seen to cross the receiver's
            opening, in metres.
        :return: The coefficient before and after; the exit speed is proportional to
            it, so it is scaled by the speed that throws the stream that far.
        """
        old = self.value
        if launch.speed <= 0.0:
            return old, old
        factor = launch.speed_for(seen_distance) / launch.speed
        self.variables.set_value(
            self.scale, self.variables.get_value(self.scale) * factor
        )
        return old, self.value


class SeesStream(Protocol):
    """
    Sees where the stream crosses the receiver's opening, and knows where the model
    throws it.
    """

    def record(self, at: float) -> None:
        """
        Note where the grains crossing the receiver's opening stand now.

        :param at: The time now, in seconds.
        """

    def observed_distance(self, since: float) -> Optional[float]:
        """
        :param since: The earliest time to count, in seconds.
        :return: How far from the lip, along the stream, the grains seen since then
            crossed the receiver's opening, in metres, or ``None`` if none were seen.
        """

    def launch(self) -> Launch:
        """
        :return: The stream as the model predicts it leaves the lip now.
        """


# %% the concept-aware response


@dataclass
class ConceptAwareRevision(OnsetResponder):
    """
    Blames a stall on the onset while the source holds its grains, and on the aim while
    it loses them, and revises that concept.
    """

    onset: OnsetRevision
    """
    What revises the repose angle.
    """

    discharge: DischargeRevision
    """
    What revises the discharge coefficient.
    """

    source: SourceWatch
    """
    What tells whether the source is losing grains.
    """

    stream: SeesStream
    """
    What sees where the stream lands.
    """

    patience: float
    """
    How long the receiver gained nothing before the stall, in seconds.
    """

    log: List[Diagnosis] = field(init=False, default_factory=list)
    """
    Every stall diagnosed, in order.
    """

    @property
    def revisions(self) -> List[Revision]:
        return self.onset.revisions

    @property
    def diagnoses(self) -> List[Diagnosis]:
        return self.log

    def lower(self, at: float, tilt: float, source_fill: float) -> bool:
        return self.onset.lower(at, tilt, source_fill)

    def observe(self, at: float, source_grains: float) -> None:
        self.source.observe(source_grains, at)
        self.stream.record(at)

    def respond(
        self, stalled: bool, at: float, tilt: float, source_fill: float
    ) -> bool:
        if not stalled:
            return False
        loss, required = self.source.lost(at), self.source.evidence(at)
        if loss < required:
            old = self.onset.repose_angle
            self.onset.respond(stalled, at, tilt, source_fill)
            self._log(
                at,
                RevisedConcept.REPOSE_ANGLE,
                StallEvidence.SOURCE_HOLDING,
                loss,
                required,
                old,
                self.onset.repose_angle,
            )
            return True
        old = self.discharge.value
        seen = self.stream.observed_distance(since=at - self.patience)
        new = (
            old
            if seen is None
            else self.discharge.revise(self.stream.launch(), seen)[1]
        )
        self._log(
            at,
            RevisedConcept.DISCHARGE_COEFFICIENT,
            StallEvidence.SOURCE_LOSING,
            loss,
            required,
            old,
            new,
        )
        return seen is not None

    def _log(
        self,
        at: float,
        concept: RevisedConcept,
        evidence: StallEvidence,
        loss: float,
        required: float,
        old: float,
        new: float,
    ) -> None:
        """
        Log one diagnosis.
        """
        self.log.append(
            Diagnosis(
                time=at,
                concept=concept,
                evidence=evidence,
                source_loss=loss,
                required_loss=required,
                old_value=old,
                new_value=new,
            )
        )
