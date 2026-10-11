"""
Coupling the hypothesis bank to the controller: which hypotheses a cup is weighed
against, what the bank knows of the sensors, which model the controller runs on after
each reading, and how far short of the goal it stops.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy

from typing_extensions import Any, Dict, List, Optional, Tuple

from experiments.pouring_model_verification.hypothesis_bank import (
    OUTSIDE_SHARE,
    OnsetSpread,
    BankSettings,
    Commit,
    FreeParameter,
    Hypothesis,
    HypothesisBank,
    ModelPoint,
    ModelSwitch,
    Reading,
    SensorKnowledge,
)
from experiments.pouring_model_verification.parameters import (
    ModelParameter,
    Perturbation,
)
from experiments.pouring_model_verification.scene import SourceCup
from experiments.pouring_model_verification.sensor import SensorSet

# %% which hypotheses a cup is weighed against


def hypotheses_for(
    source: SourceCup, forces_second_lip: bool = False, outflow_capacity: bool = False
) -> Tuple[Hypothesis, ...]:
    """
    :param source: The cup poured from.
    :param forces_second_lip: Whether the second lip is weighed even for a cup without
        a slot, as a negative control.
    :param outflow_capacity: Whether the twin has the concept of how much can pass the
        lip.
    :return: The hypotheses built from the concepts the twin has for that cup: the
        second lip only where the cup has a slot edge.
    """
    has_slot = source is SourceCup.SPOUTED_CUP or forces_second_lip
    left_out = set()
    if not has_slot:
        left_out.add(Hypothesis.SECOND_LIP)
    if not outflow_capacity:
        left_out.add(Hypothesis.OUTFLOW_CAPACITY)
    return tuple(hypothesis for hypothesis in Hypothesis if hypothesis not in left_out)


def injected_hypothesis(perturbation: Perturbation) -> Optional[Hypothesis]:
    """
    :param perturbation: The error set into the model on purpose.
    :return: The hypothesis that names it, the nominal one for none, or ``None`` for an
        error no single hypothesis names.
    """
    if perturbation.nominal:
        return None
    coupled = {perturbation.parameter, perturbation.second_parameter}
    if coupled == {ModelParameter.REPOSE_ANGLE, ModelParameter.DISCHARGE_COEFFICIENT}:
        return Hypothesis.COUPLED
    if perturbation.second_parameter is not None:
        return None
    match perturbation.parameter:
        case None:
            return Hypothesis.NOMINAL
        case ModelParameter.REPOSE_ANGLE:
            return Hypothesis.ONSET
        case ModelParameter.DISCHARGE_COEFFICIENT:
            return Hypothesis.AIM
        case _:
            return None


def sensor_knowledge(
    sensors: SensorSet, fall_time: float, camera_noise: Optional[float] = None
) -> SensorKnowledge:
    """
    :param sensors: The sensors the pour reads.
    :param fall_time: How long a grain falls from the lip to the receiver's opening, in
        seconds.
    :param camera_noise: The camera's noise to assume, or ``None`` for its own.
    :return: What the bank knows about those sensors.
    """
    receiver, wrist, camera = (
        sensors.receiver.model,
        sensors.wrist.model,
        sensors.camera.model,
    )
    if camera is not None and camera_noise is None:
        camera_noise = camera.noise
    return SensorKnowledge(
        receiver_noise=receiver.noise,
        receiver_delay=receiver.delay,
        counts_falling_grains=receiver.counts_falling_grains,
        fall_time=fall_time,
        source_noise=wrist.noise,
        source_delay=wrist.delay,
        crossing_noise=None if camera is None else camera_noise,
    )


# %% what is logged after each reading


class TraceColumn(StrEnum):
    """
    A column of the bank's trace, one row per reading.
    """

    TIME = "time"
    ACTIVE = "active"
    MARGIN = "margin"
    POURED_SHARE = "poured_share"


CAUTION_MARGIN_FACTOR = 2.0
"""
How much further short of the goal a pour stops while no hypothesis explains it.
"""

# %% the controller's model after each reading


@dataclass
class BankCoupling:
    """
    The hypothesis bank weighing each reading, and the model the controller runs on as a
    result.
    """

    bank: HypothesisBank
    """
    The hypotheses, weighed against each reading.
    """

    switch: ModelSwitch = field(init=False)
    """
    Which hypothesis the controller runs on.
    """

    commits: List[Commit] = field(init=False, default_factory=list)
    """
    Every switch of the controller's model, in order.
    """

    trace: List[Dict[str, Any]] = field(init=False, default_factory=list)
    """
    One row per reading: the time, every hypothesis's probability, the hypothesis the
    controller runs on, and the margin it stops short by.
    """

    _tilt: float = field(init=False, default=0.0)
    """
    The cup's tilt at the latest reading, in radians.
    """

    _trickle: float = field(init=False, default=0.0)
    """
    The least flow the controller's model was held to expect at the latest reading, in
    grains per second.
    """

    _margin: float = field(init=False, default=0.0)
    """
    The margin the controller stops short of the goal by, in grains.
    """

    def __post_init__(self) -> None:
        settings = self.bank.settings
        self.switch = ModelSwitch(
            threshold=settings.commit_threshold, confirmations=settings.confirmations
        )

    @property
    def active(self) -> Hypothesis:
        """
        :return: The hypothesis the controller runs on now.
        """
        return self.switch.active

    @property
    def cautious(self) -> bool:
        """
        :return: Whether no hypothesis explains the pour, so it is finished slowly.
        """
        return self.active is Hypothesis.CATCH_ALL

    @property
    def point(self) -> ModelPoint:
        """
        :return: The parameters the controller runs on: those the active hypothesis
            gives, or the model as given for a hypothesis that neither revises it nor
            adapts its repose angle to the pour.
        """
        if not self.bank.started:
            return self.bank.base
        active = self.bank.filters[self.active]
        if not self.active.revises and not active.adapts_onset:
            return self.bank.base
        return active.controller_point(self._tilt, self._trickle)

    @property
    def margin(self) -> float:
        """
        :return: How many grains short of the goal the controller stops.
        """
        return self._margin

    def observe(
        self, reading: Reading, poured_share: float, trickle: float = 0.0
    ) -> Optional[Commit]:
        """
        Weigh one reading, switch the controller's model if a hypothesis has become
        probable enough, and update the margin.

        :param reading: The reading just taken.
        :param poured_share: The share of the goal the receiver holds now, for the log.
        :param trickle: The least flow the controller's model is held to expect at the
            tilt, in grains per second, so it keeps tilting where no grains flow yet.
        :return: The switch made, or ``None``.
        """
        self.bank.observe(reading)
        self._tilt = reading.tilt
        self._trickle = trickle
        previous = self.active
        vetoed = self.vetoed
        _, support = self.switch.target(self.bank.probabilities, vetoed)
        chosen = self.switch.decide(self.bank.probabilities, vetoed)
        commit = None
        if chosen is not None:
            commit = Commit(
                time=reading.time,
                hypothesis=chosen,
                previous=previous,
                probability=support,
                point=self.point,
                estimate=self.bank.filters[chosen].mean(),
                poured_share=poured_share,
            )
            self.commits.append(commit)
        settings = self.bank.settings
        factor = settings.margin_factor * (
            CAUTION_MARGIN_FACTOR if self.cautious else 1.0
        )
        self._margin = factor * self.bank.filters[self.active].final_fill_spread(
            reading.tilt
        )
        self.trace.append(
            {
                TraceColumn.TIME: reading.time,
                **{
                    hypothesis.label: probability
                    for hypothesis, probability in self.bank.probabilities.items()
                },
                TraceColumn.ACTIVE: self.active.label,
                TraceColumn.MARGIN: self._margin,
                TraceColumn.POURED_SHARE: poured_share,
            }
        )
        return commit

    @property
    def vetoed(self) -> frozenset[FreeParameter]:
        """
        :return: The onset, while the pour's repose angle may still lie within the
            twin's spread: only a pour clearly outside it shows the twin's centre wrong.
        """
        share = self.bank.onset_share_inside()
        if share is None or share <= 1.0 - OUTSIDE_SHARE:
            return frozenset()
        return frozenset({FreeParameter.REPOSE_ANGLE})

    def summary(self) -> Dict[str, Any]:
        """
        :return: What the bank concluded at the end of the pour: each hypothesis's
            probability and posterior mean, and how often a parameter was held at its
            bound; the model as given for a bank that saw no reading.
        """
        if not self.bank.started:
            return {
                hypothesis: {
                    "probability": probability,
                    "mean": self.bank.base.to_json(),
                    "spread": {},
                    "bound_share": {},
                    "clipped_share": 0.0,
                }
                for hypothesis, probability in self.bank.probabilities.items()
            }
        return {
            hypothesis: {
                "probability": self.bank.probabilities[hypothesis],
                "mean": hypothesis_filter.mean().to_json(),
                "spread": {
                    parameter: hypothesis_filter.spread(parameter)
                    for parameter in hypothesis_filter.free
                },
                "bound_share": {
                    parameter: hypothesis_filter.bound_share(parameter)
                    for parameter in hypothesis_filter.free
                },
                "clipped_share": hypothesis_filter.clipped
                / max(1, hypothesis_filter.walked),
            }
            for hypothesis, hypothesis_filter in self.bank.filters.items()
        }


# %% a cautious start while the aim is unconfirmed

NORMAL_TILT_RATE = 0.1
"""
How fast the cup turns in a pour, in radians per second: the 95th percentile of the tilt
rate over the spouted cup's pours into the narrow receiver with the bank.
"""


@dataclass(frozen=True)
class CautionSettings:
    """
    How a pour starts cautiously while nothing has confirmed where its stream lands.
    """

    cap_fraction: float
    """
    The share of :data:`NORMAL_TILT_RATE` the tilt speed is capped at.
    """

    release_time: float
    """
    How long the camera must see the stream land inside the opening without a break
    before the cap is released, in seconds.
    """

    @property
    def tilt_rate_cap(self) -> float:
        """
        :return: The capped tilt speed, in radians per second.
        """
        return self.cap_fraction * NORMAL_TILT_RATE

    @property
    def label(self) -> str:
        """
        :return: A short identifier of these settings, usable in a file name.
        """
        return f"caution{self.cap_fraction:g}_{self.release_time:g}s"

    def to_json(self) -> Dict[str, float]:
        return {"cap_fraction": self.cap_fraction, "release_time": self.release_time}


@dataclass
class CautiousStart:
    """
    Whether the tilt speed is capped now: from the start of a pour until the aim concept
    is confirmed or the camera has seen the stream inside the opening for long enough.
    """

    settings: CautionSettings
    """
    The cap and when it is released.
    """

    opening_radius: float
    """
    The receiver opening's radius, in metres.
    """

    released_at: Optional[float] = field(init=False, default=None)
    """
    When the cap was released, in seconds, or ``None`` while it holds.
    """

    inside_since: Optional[float] = field(init=False, default=None)
    """
    Since when the camera has seen the stream inside the opening without a break.
    """

    def observe(
        self, at: float, crossing_distance: Optional[float], aim_confirmed: bool
    ) -> bool:
        """
        :param at: The time of the reading, in seconds.
        :param crossing_distance: Where the camera saw the stream cross the opening
            plane, as a distance from its centre, or ``None`` for no stream seen.
        :param aim_confirmed: Whether the bank has committed to a hypothesis revising
            the aim.
        :return: Whether the tilt speed is capped after this reading.
        """
        if self.released_at is not None:
            return False
        if crossing_distance is not None:
            inside = crossing_distance <= self.opening_radius
            self.inside_since = (
                (self.inside_since if self.inside_since is not None else at)
                if inside
                else None
            )
        seen_long_enough = (
            self.inside_since is not None
            and at - self.inside_since >= self.settings.release_time - 1e-9
        )
        if aim_confirmed or seen_long_enough:
            self.released_at = at
            return False
        return True


# %% writing a new onset back only after repeated evidence

WRITE_BACK_RUN = 3
"""
How many pours in a row have to land outside the twin's interval on the same side before
its centre is moved.
"""


@dataclass(frozen=True)
class OnsetWriteBack:
    """
    The twin's spread of the repose angle across a sequence of pours, and the pours
    lately found outside it: one pour outside is variation, several in a row on the
    same side are the twin's centre being wrong.
    """

    onset: OnsetSpread
    """
    The twin's centre and spread now.
    """

    run: Tuple[float, ...] = ()
    """
    The repose angles of the latest pours in a row outside the interval on one side.
    """

    written: bool = False
    """
    Whether the pour just weighed moved the twin's centre.
    """

    def after(self, estimate: float, explained: bool) -> OnsetWriteBack:
        """
        :param estimate: The pour's repose angle, in radians.
        :param explained: Whether the pour's explanation held up against what it
            showed; one that did not is never written back and breaks the run.
        :return: The state after the pour.
        """
        side = self.onset.side_of(estimate)
        if not explained or side == 0:
            return OnsetWriteBack(onset=self.onset)
        same_side = self.run and self.onset.side_of(self.run[0]) == side
        run = (self.run if same_side else ()) + (estimate,)
        if len(run) < WRITE_BACK_RUN:
            return OnsetWriteBack(onset=self.onset, run=run)
        moved = OnsetSpread(
            centre=float(numpy.mean(run)),
            spread=max(self.onset.spread, float(numpy.std(run, ddof=1))),
        )
        return OnsetWriteBack(onset=moved, written=True)
