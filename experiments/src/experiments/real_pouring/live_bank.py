"""
The hypothesis bank on the real robot: it reads what the camera saw of the stream,
weighs it against its versions of the pouring model, and writes the drain corrections
and the fill levels the controller acts on into the world, which carries them to
Giskard.

Nothing here depends on ROS: a node hands each reading to :meth:`LiveBank.observe`.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy
from typing_extensions import Any, Dict, List, Optional

from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    DrainGeometry,
    HypothesisBank,
    ModelPoint,
    Reading,
    SensorKnowledge,
)
from experiments.pouring_model_verification.poured_stream import PouredStream
from experiments.pouring_model_verification.recovery import (
    BankCoupling,
    DrainCorrectionValues,
    hypotheses_for,
)
from experiments.pouring_model_verification.scene import SourceCup
from experiments.pouring_model_verification.sensor import CameraModel
from experiments.pouring_model_verification.transfer_trial import ModelBlend
from experiments.real_pouring.receiving_cup import ReceiverFillReport
from experiments.real_pouring.receiver_evidence import (
    ReceiverEvidence,
    ReceiverObservation,
    InvalidReceiverEvidenceError,
)
from experiments.real_pouring.stream_detection import StreamReading
from krrood.adapters.json_serializer import SubclassJSONSerializer
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.world import World

# %% one step of the bank


class BankTraceField(StrEnum):
    """Receiver evidence and source state recorded with each bank update."""

    RECEIVER_OBSERVATION = "receiver_observation"
    """Receiver observation admitted to the likelihood, in grains."""
    RECEIVER_NOISE = "receiver_noise"
    """Error scale of the admitted receiver observation, in grains."""
    SOURCE_GRAINS = "source_grains"
    """Posterior estimate of remaining source contents, in grains."""


@dataclass(frozen=True)
class LiveBankStep(SubclassJSONSerializer):
    """
    What the bank concluded from one reading, and what it told the controller.
    """

    time: float
    """
    When the reading was taken, in seconds.
    """

    tilt: float
    """
    The source cup's tilt then, in radians.
    """

    stream_seen: bool
    """
    Whether the camera saw a stream.
    """

    counted_grains: float
    """
    How many grains the camera had counted crossing.
    """

    active: str
    """
    The label of the hypothesis the controller runs on.
    """

    probabilities: Dict[str, float]
    """
    How probable each hypothesis is, by label.
    """

    corrections: DrainCorrectionValues
    """
    The drain corrections written for the controller.
    """

    received_grains: float
    """
    How many grains the bank estimates reached the receiver.
    """

    in_flight_grains: float = 0.0
    """
    How many grains the controller was told are still to arrive after a stop; zero where
    the receiver is reported from the bank's estimate instead.
    """

    told_grains: float = 0.0
    """
    How many grains the controller was told the receiver holds, in flight or margin
    included.
    """

    receiver_observation: Optional[float] = None
    """Measured contents admitted to the likelihood, or no observation this step."""

    receiver_noise: Optional[float] = None
    """Uncertainty of the admitted receiver observation, in grains."""

    source_grains: Optional[float] = None
    """Posterior source contents, or unavailable in older traces."""

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "time": self.time,
            "tilt": self.tilt,
            "stream_seen": self.stream_seen,
            "counted_grains": self.counted_grains,
            "active": self.active,
            "probabilities": self.probabilities,
            "corrections": vars(self.corrections),
            "received_grains": self.received_grains,
            "in_flight_grains": self.in_flight_grains,
            "told_grains": self.told_grains,
            BankTraceField.RECEIVER_OBSERVATION: self.receiver_observation,
            BankTraceField.RECEIVER_NOISE: self.receiver_noise,
            BankTraceField.SOURCE_GRAINS: self.source_grains,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> LiveBankStep:
        return cls(
            time=data["time"],
            tilt=data["tilt"],
            stream_seen=data["stream_seen"],
            counted_grains=data["counted_grains"],
            active=data["active"],
            probabilities=data["probabilities"],
            corrections=DrainCorrectionValues(**data["corrections"]),
            received_grains=data["received_grains"],
            in_flight_grains=data.get("in_flight_grains", 0.0),
            told_grains=data.get("told_grains", data["received_grains"]),
            receiver_observation=data.get(BankTraceField.RECEIVER_OBSERVATION),
            receiver_noise=data.get(BankTraceField.RECEIVER_NOISE),
            source_grains=data.get(BankTraceField.SOURCE_GRAINS),
        )


# %% the bank on the robot


NEGLIGIBLE_ARRIVAL_RATE = 0.1
"""
The arrival rate below which the grains are taken to have stopped, in grains per second:
a grain every ten seconds, well under a stray kernel a second.

Below it the smoothed rate is set to zero rather than left decaying, so a receiver
nothing arrives in is told nothing new.
"""


@dataclass
class LiveBank:
    """
    Weighs the camera's readings of a pour against the bank's hypotheses and keeps the
    controller's drain and fill levels at what they call for.
    """

    source: HasFillLevel
    """
    The cup poured from.
    """

    receiver: HasFillLevel
    """
    The cup poured into.
    """

    world: World
    """
    The world the controller's model lives in; what is written into it reaches Giskard.
    """

    coupling: BankCoupling
    """
    The bank, and the model the controller is to run on.
    """

    built: ModelPoint
    """
    The parameters the source's drain was built with.
    """

    camera: CameraModel
    """
    How the camera departs from what crosses the opening.
    """

    grains_per_fill: float
    """
    How many of the bank's grains fill the source to the top.
    """

    loaded_grains: float
    """
    How many grains the source was loaded with.
    """

    goal_grains: float
    """
    How many grains the receiver is to hold.
    """

    opening_radius: float
    """
    The receiver opening's radius, in metres.
    """

    prediction_window: float
    """
    How far ahead the fill task predicts, in seconds.
    """

    fill_report: ReceiverFillReport = field(
        default=ReceiverFillReport.ESTIMATED, kw_only=True
    )
    """
    What the controller is told the receiver holds: the bank's estimate, what the camera
    counted crossing, or what it sees through the receiver's wall.
    """

    receiver_evidence: Optional[ReceiverEvidence] = field(default=None, kw_only=True)
    """Receiver likelihood settings; absent when the bank learns from the count."""

    _last_receiver_stamp: Optional[float] = field(init=False, default=None)
    """Timestamp of the most recent receiver likelihood update."""

    stop_lead_time: float = field(default=0.0, kw_only=True)
    """
    How long grains keep arriving once the controller stops, in seconds: the rate the
    grains arrive at, held for this time, is reported as already in the receiver, so the
    controller stops that much short of the goal.

    The rate is smoothed with this time as its time constant, so one avalanche in the
    count raises it gradually.
    """

    in_flight_limit: float = field(default=math.inf, kw_only=True)
    """
    The most grains ever reported as still in flight: what the controller is to resolve,
    so no burst of the count can push the receiver past the goal by more.
    """

    trace: List[LiveBankStep] = field(default_factory=list)
    """
    Every step the bank took, in order.
    """

    _blend: Optional[ModelBlend] = field(init=False, default=None, repr=False)
    """
    The model the controller runs on, moving towards the one the bank calls for.
    """

    _arrival_rate: float = field(init=False, default=0.0)
    """
    How fast the grains arrive by the camera's count, smoothed, in grains per second.
    """

    _measured_grains: Optional[float] = field(init=False, default=None)
    """
    Most recent measured receiver contents, excluding late arrivals.
    """

    _count_at_measurement: float = field(init=False, default=0.0)
    """
    Cumulative crossing count at the most recent contents measurement.
    """

    @classmethod
    def for_cups(
        cls,
        source: HasFillLevel,
        receiver: HasFillLevel,
        world: World,
        built: ModelPoint,
        settings: BankSettings,
        camera: CameraModel,
        grains_per_fill: float,
        loaded_grains: float,
        goal_grains: float,
        opening_radius: float,
        prediction_window: float,
        seed: int = 0,
        fill_report: ReceiverFillReport = ReceiverFillReport.ESTIMATED,
        stop_lead_time: float = 0.0,
        in_flight_limit: float = math.inf,
        receiver_evidence: Optional[ReceiverEvidence] = None,
    ) -> LiveBank:
        """
        :return: A bank over the hypotheses a plain cup is weighed against, reading only
            the camera.
        """
        equation = source.fill_equation
        bank = HypothesisBank(
            hypotheses=hypotheses_for(
                SourceCup.CUP,
                outflow_capacity=settings.outflow_capacity,
                tracks_onset=settings.tracks_onset,
            ),
            geometry=DrainGeometry(
                container_height=equation.container_height,
                lip_offset=equation.lip_offset,
                grains_per_fill=grains_per_fill,
                slot_offset=equation.lip_offset,
            ),
            base=built,
            sensors=SensorKnowledge(
                receiver_noise=None,
                receiver_delay=(
                    0.0 if receiver_evidence is None else receiver_evidence.delay
                ),
                counts_falling_grains=False,
                fall_time=0.0,
                source_noise=None,
                source_delay=0.0,
                crossing_noise=camera.noise,
                stream_dropout=camera.dropout,
                count_noise=camera.count_noise if receiver_evidence is None else None,
                conditions_receiver=receiver_evidence is not None,
            ),
            settings=settings,
            seed=seed,
        )
        return cls(
            source=source,
            receiver=receiver,
            world=world,
            coupling=BankCoupling(bank=bank),
            built=built,
            camera=camera,
            grains_per_fill=grains_per_fill,
            loaded_grains=loaded_grains,
            goal_grains=goal_grains,
            opening_radius=opening_radius,
            prediction_window=prediction_window,
            fill_report=fill_report,
            stop_lead_time=stop_lead_time,
            in_flight_limit=in_flight_limit,
            receiver_evidence=receiver_evidence,
        )

    @property
    def stream(self) -> PouredStream:
        """
        :return: The stream the model throws from the source's lip into the receiver.
        """
        return PouredStream(
            source=self.source, receiver=self.receiver, world=self.world
        )

    def observe(self, seen: StreamReading) -> LiveBankStep:
        """
        Weigh one reading of the camera, then write the drain corrections and the fill
        levels the bank calls for into the world.

        :param seen: What the camera saw of the stream.
        :return: What the bank concluded and wrote.
        """
        stream = self.stream
        tilt = float(self.source.fill_connection.tilt_expression.evaluate()[0])
        observation = self._receiver_observation(seen)
        reading = Reading(
            time=seen.stamp,
            tilt=tilt,
            receiver_grains=None if observation is None else observation.grains,
            receiver_noise=None if observation is None else observation.noise,
            source_grains=self.loaded_grains if not self.trace else None,
            crossing_distance=(
                None
                if seen.crossing is None
                else stream.distance_from_lip(
                    [numpy.array([seen.crossing.x, seen.crossing.y])]
                )
            ),
            stream=stream.geometry(self.opening_radius),
            stream_seen=seen.stream_seen,
            counted_grains=(
                seen.counted_grains if self.receiver_evidence is None else None
            ),
        )
        received = self.coupling.received_grains if self.trace else 0.0
        self.coupling.observe(
            reading,
            poured_share=received / self.goal_grains,
            trickle=min(
                self.coupling.bank.settings.trickle,
                (self.goal_grains - received - self.coupling.margin)
                / (2 * self.prediction_window),
            ),
        )
        corrections = DrainCorrectionValues.between(
            self._blended(seen.stamp), self.built, self.coupling.cautious
        )
        corrections.write_to(self.world, self.source.fill_connection.drain_correction)
        in_flight = self._in_flight_grains(seen)
        told = self._write_fill_levels(seen.counted_grains, in_flight, seen.heap_fill)
        step = LiveBankStep(
            time=seen.stamp,
            tilt=tilt,
            stream_seen=seen.stream_seen,
            counted_grains=seen.counted_grains,
            active=self.coupling.active.label,
            probabilities={
                hypothesis.label: probability
                for hypothesis, probability in self.coupling.bank.probabilities.items()
            },
            corrections=corrections,
            received_grains=self.coupling.received_grains,
            in_flight_grains=in_flight,
            told_grains=told,
            receiver_observation=reading.receiver_grains,
            receiver_noise=reading.receiver_noise,
            source_grains=self.coupling.source_grains,
        )
        self.trace.append(step)
        return step

    def _receiver_observation(
        self, seen: StreamReading
    ) -> Optional[ReceiverObservation]:
        """Admit a finite receiver measurement after the preceding window has elapsed."""
        evidence = self.receiver_evidence
        if evidence is None or seen.heap_fill is None:
            return None
        if not math.isfinite(seen.heap_fill) or not 0 <= seen.heap_fill <= 1:
            raise InvalidReceiverEvidenceError()
        if (
            self._last_receiver_stamp is not None
            and seen.stamp - self._last_receiver_stamp < evidence.interval
        ):
            return None
        self._last_receiver_stamp = seen.stamp
        conversion = (
            self.grains_per_fill * self.receiver.capacity / self.source.capacity
        )
        noise = (
            evidence.calibrated_noise
            if seen.wall_calibrated
            else evidence.geometric_noise
        )
        return ReceiverObservation(seen.heap_fill * conversion, noise * conversion)

    def count_calibration_factor(self, weighed_grains: float) -> float:
        """
        :param weighed_grains: How many grains left the source, by the scale.
        :return: The factor the detector's area per grain is to be multiplied by so its
            count matches the scale: above one where it counted too many.
        """
        return (
            self.camera.grains_crossed(self.trace[-1].counted_grains) / weighed_grains
        )

    def _blended(self, at: float) -> ModelPoint:
        """
        :param at: The time of the reading, in seconds.
        :return: The model the controller runs on now, moving towards the one the bank
            calls for.
        """
        if self._blend is None:
            self._blend = ModelBlend(point=self.coupling.bank.base, time=at)
        return self._blend.towards(self.coupling.point, at)

    def _in_flight_grains(self, seen: StreamReading) -> float:
        """
        :param seen: The reading just taken.
        :return: How many grains are still to arrive after a stop now: the smoothed
            rate the camera's count rises at, held for :attr:`stop_lead_time` and
            capped at :attr:`in_flight_limit`; zero where the receiver is reported
            from the bank.
        """
        if self.fill_report is ReceiverFillReport.ESTIMATED or not self.trace:
            return 0.0
        previous = self.trace[-1]
        interval = seen.stamp - previous.time
        if interval > 0.0:
            instant = (
                max(
                    0.0,
                    self.camera.grains_crossed(
                        seen.counted_grains - previous.counted_grains
                    ),
                )
                / interval
            )
            share = (
                1.0
                if self.stop_lead_time <= 0.0
                else min(1.0, interval / self.stop_lead_time)
            )
            self._arrival_rate += share * (instant - self._arrival_rate)
            if self._arrival_rate < NEGLIGIBLE_ARRIVAL_RATE:
                self._arrival_rate = 0.0
        return min(self._arrival_rate * self.stop_lead_time, self.in_flight_limit)

    def _write_fill_levels(
        self, counted_grains: float, in_flight: float, heap_fill: Optional[float]
    ) -> float:
        """
        Report source contents from the receiver-conditioned bank or crossing count,
        receiver contents according to :attr:`fill_report`, and observed arrival rate.

        :param counted_grains: How many grains the camera has counted crossing.
        :param in_flight: How many grains are still to arrive after a stop now.
        :param heap_fill: The receiver's fill by its contents seen lying in it, or
            ``None``.
        :return: How many grains the receiver was told to hold.
        """
        crossed = self.camera.grains_crossed(counted_grains)
        source_grains = (
            self.loaded_grains - crossed
            if self.receiver_evidence is None
            else min(self.loaded_grains, self.coupling.source_grains)
        )
        capacity_ratio = self.source.capacity / self.receiver.capacity
        receiver_grains = self._receiver_grains(
            crossed=crossed,
            in_flight=in_flight,
            lying=(
                None
                if heap_fill is None
                else heap_fill * self.grains_per_fill / capacity_ratio
            ),
        )
        values = {
            self.source.fill_connection: max(0.0, source_grains) / self.grains_per_fill,
            self.receiver.fill_connection: min(
                1.0, receiver_grains / self.grains_per_fill * capacity_ratio
            ),
        }
        if self.fill_report is not ReceiverFillReport.ESTIMATED:
            values[
                self.receiver.fill_connection.observed_inflow_connection_in(self.world)
            ] = (
                self._arrival_rate
                * self.coupling.inside_share
                / self.grains_per_fill
                * capacity_ratio
            )
        JointState.from_mapping(values).apply_to(self.world)
        return receiver_grains

    def _receiver_grains(
        self, crossed: float, in_flight: float, lying: Optional[float]
    ) -> float:
        """
        :param crossed: How many grains the camera counted crossing the opening.
        :param in_flight: How many grains are still to arrive after a stop now.
        :param lying: How many grains the camera sees lying in the receiver, or
            ``None`` where it sees too few to tell.
        :return: How many grains the receiver is told to hold, by
            :attr:`fill_report`.
        """
        counted = crossed * self.coupling.inside_share
        match self.fill_report:
            case ReceiverFillReport.MEASURED:
                if lying is not None:
                    self._measured_grains = lying
                    self._count_at_measurement = crossed
                if self._measured_grains is None:
                    return counted + in_flight
                return (
                    self._measured_grains
                    + max(0.0, crossed - self._count_at_measurement)
                    * self.coupling.inside_share
                    + in_flight
                )
            case ReceiverFillReport.ESTIMATED:
                reported = self.coupling.received_grains + self.coupling.margin
            case ReceiverFillReport.COUNTED:
                reported = counted + in_flight
            case ReceiverFillReport.SEEN_THROUGH_WALL:
                return (counted if lying is None else max(counted, lying)) + in_flight
        return reported if lying is None else max(reported, lying)
