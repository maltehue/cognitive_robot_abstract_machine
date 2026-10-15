"""
The hypothesis bank on the real robot: it reads what the camera saw of the stream,
weighs it against its versions of the pouring model, and writes the drain corrections
and the fill levels the controller acts on into the world, which carries them to
Giskard.

Nothing here depends on ROS: a node hands each reading to :meth:`LiveBank.observe`.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass, field

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
from experiments.real_pouring.stream_detection import StreamReading
from krrood.adapters.json_serializer import SubclassJSONSerializer
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.world import World

# %% one step of the bank


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
        )


# %% the bank on the robot

COUNT_RATE_WINDOW = 0.5
"""
Over how many seconds the camera's count is differenced for the rate the grains arrive
at: long enough to smooth the frame-by-frame steps of the count at ten readings a
second.
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

    receiver_from_count: bool = field(default=False, kw_only=True)
    """
    Whether the receiver's fill is reported from what the camera counted crossing, with
    the grains still in flight added, rather than from the bank's estimate with its
    margin.
    """

    stop_lead_time: float = field(default=0.0, kw_only=True)
    """
    How long grains keep arriving once the controller stops, in seconds: the count's
    rate over this time is reported as already in the receiver, so the controller stops
    that much short of the goal.
    """

    trace: List[LiveBankStep] = field(default_factory=list)
    """
    Every step the bank took, in order.
    """

    _blend: Optional[ModelBlend] = field(init=False, default=None, repr=False)
    """
    The model the controller runs on, moving towards the one the bank calls for.
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
        receiver_from_count: bool = False,
        stop_lead_time: float = 0.0,
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
                receiver_delay=0.0,
                counts_falling_grains=False,
                fall_time=0.0,
                source_noise=None,
                source_delay=0.0,
                crossing_noise=camera.noise,
                stream_dropout=camera.dropout,
                count_noise=camera.count_noise,
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
            receiver_from_count=receiver_from_count,
            stop_lead_time=stop_lead_time,
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
        reading = Reading(
            time=seen.stamp,
            tilt=tilt,
            receiver_grains=None,
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
            counted_grains=seen.counted_grains,
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
        )
        self.trace.append(step)
        return step

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
        :return: How many grains are still to arrive after a stop now: the rate the
            camera's count rose at over :data:`COUNT_RATE_WINDOW`, held for
            :attr:`stop_lead_time`; zero where the receiver is reported from the bank.
        """
        if not self.receiver_from_count:
            return 0.0
        earlier = [
            step for step in self.trace if step.time <= seen.stamp - COUNT_RATE_WINDOW
        ]
        if not earlier:
            return 0.0
        reference = earlier[-1]
        interval = seen.stamp - reference.time
        if interval <= 0.0:
            return 0.0
        rate = (
            self.camera.grains_crossed(seen.counted_grains - reference.counted_grains)
            / interval
        )
        return max(0.0, rate) * self.stop_lead_time

    def _write_fill_levels(
        self, counted_grains: float, in_flight: float, heap_fill: Optional[float]
    ) -> float:
        """
        Tell the controller what both cups hold: the source what was loaded less what
        the camera counted leaving; the receiver what crossed and is still in flight, or
        what the bank estimates arrived with the margin it stops short by, or what the
        heap seen in it shows, whichever is more.

        :param counted_grains: How many grains the camera has counted crossing.
        :param in_flight: How many grains are still to arrive after a stop now.
        :param heap_fill: The receiver's fill by the heap seen in it, or ``None``.
        :return: How many grains the receiver was told to hold.
        """
        crossed = self.camera.grains_crossed(counted_grains)
        source_grains = self.loaded_grains - crossed
        if self.receiver_from_count:
            receiver_grains = crossed * self.coupling.inside_share + in_flight
        else:
            receiver_grains = self.coupling.received_grains + self.coupling.margin
        capacity_ratio = self.source.capacity / self.receiver.capacity
        if heap_fill is not None:
            receiver_grains = max(
                receiver_grains, heap_fill * self.grains_per_fill / capacity_ratio
            )
        JointState.from_mapping(
            {
                self.source.fill_connection: max(0.0, source_grains)
                / self.grains_per_fill,
                self.receiver.fill_connection: min(
                    1.0, receiver_grains / self.grains_per_fill * capacity_ratio
                ),
            }
        ).apply_to(self.world)
        return receiver_grains
