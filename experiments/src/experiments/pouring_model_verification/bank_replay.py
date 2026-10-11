"""
Running the hypothesis bank over pours that were already recorded, without running the
physics again.

A record holds what the grains really did at each sample: how many stood in the source,
how many had landed and how many were on their way. The readings a sensor set would have
reported are drawn from that, with the sensor's noise and delay, and handed to the bank
as if the pour were running. Where the stream crossed the receiver's opening was not
recorded, so a replay reads no camera.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy
import pandas
from typing_extensions import Any, Dict, List, Optional, Tuple

from experiments.pouring_model_verification.concept_revision import DiagnosisKey
from experiments.pouring_model_verification.consistency import (
    PREDICTIVE_SHARE,
    Consistency,
    ConsistencyCheck,
    Feature,
    RateResiduals,
)
from experiments.pouring_model_verification.hypothesis_bank import (
    DrainRateSpread,
    OnsetSpread,
    BankSettings,
    DrainGeometry,
    FreeParameter,
    Hypothesis,
    HypothesisBank,
    ModelPoint,
    Reading,
    StreamGeometry,
)
from experiments.pouring_model_verification.onset_rule import RevisionKey
from experiments.pouring_model_verification.parameters import (
    ModelParameters,
    Perturbation,
)
from experiments.pouring_model_verification.records import (
    Phase,
    Sample,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.recovery import (
    BankCoupling,
    hypotheses_for,
    injected_hypothesis,
    sensor_knowledge,
)
from experiments.pouring_model_verification.scene import ReceiverCup, SourceCup
from experiments.pouring_model_verification.sensor import (
    CameraCondition,
    SensorChannel,
    SensorSet,
)
from experiments.pouring_model_verification.transfer_trial import (
    GOAL_GRAINS,
    PERCEPTION_FREQUENCY,
    BankReadingKey,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    STANDARD_GRAVITY,
)

# %% how the stream is taken to leave the lip


REPLAY_DROP = 0.11
"""
How far the lip is taken to stand above the receiver's opening in a replay, in metres:

about what the rim clearance the transfer keeps puts it at.
"""

SPOUT_PITCH = math.pi / 4
"""
How steeply the spouted cup's trough climbs above the cup's floor, in radians.
"""


def stream_direction(source: SourceCup, tilt: float) -> Tuple[float, float]:
    """
    :param source: The cup poured from.
    :param tilt: The cup's tilt, in radians.
    :return: The horizontal and the upward share of the stream's exit direction: a cup
        throws it horizontally, a spout along its trough.
    """
    if source is SourceCup.CUP:
        return 1.0, 0.0
    angle = SPOUT_PITCH - tilt
    return abs(math.cos(angle)), math.sin(angle)


# %% one recorded pour, as the bank would have read it


@dataclass
class LoggedPour:
    """
    A recorded transfer pour, with the readings a sensor set would have reported along
    it.
    """

    record: TrialRecord
    """
    The recorded pour.
    """

    baseline: ModelParameters
    """
    The parameters fitted to the pour's source cup, which its error is taken from.
    """

    transfer: pandas.DataFrame = field(init=False)
    """
    The samples taken while the pouring tasks drove the pour.
    """

    def __post_init__(self) -> None:
        samples = self.record.samples
        self.transfer = samples[samples[Sample.PHASE] == Phase.TRANSFER].reset_index(
            drop=True
        )

    @property
    def description(self) -> Dict[str, Any]:
        return self.record.description

    @property
    def source(self) -> SourceCup:
        return SourceCup(self.description[TrialKey.SOURCE])

    @property
    def receiver(self) -> ReceiverCup:
        return ReceiverCup(self.description[TrialKey.RECEIVER])

    @property
    def perturbation(self) -> Perturbation:
        return Perturbation.from_json(self.description[TrialKey.PERTURBATION])

    @property
    def truth(self) -> Optional[Hypothesis]:
        """
        :return: The hypothesis naming the error set into the model, the nominal one
            for none.
        """
        return injected_hypothesis(self.perturbation)

    @property
    def given(self) -> ModelPoint:
        """
        :return: The parameters the controller's model was given, before any revision;
            where the baseline holds values of its own while turning, the error set
            against those.
        """
        if self.baseline.has_dynamic_values:
            return self._point(self.perturbation.around(self.baseline).while_turning())
        parameters = ModelParameters.from_json(self.description[TrialKey.PARAMETERS])
        return ModelPoint(
            repose_angle=self.description.get(
                TrialKey.REPOSE_ANGLE_AT_START, parameters.repose_angle
            ),
            discharge_coefficient=parameters.discharge_coefficient,
            outflow_rate_constant=parameters.outflow_rate_constant,
        )

    @property
    def true_point(self) -> ModelPoint:
        """
        :return: The parameters fitted to the cup, which the error is set against, as
            they hold while it turns.
        """
        return self._point(self.baseline.while_turning())

    @staticmethod
    def _point(parameters: ModelParameters) -> ModelPoint:
        """
        :param parameters: A parameter set.
        :return: The same values as a point the bank weighs.
        """
        return ModelPoint(
            repose_angle=parameters.repose_angle,
            discharge_coefficient=parameters.discharge_coefficient,
            outflow_rate_constant=parameters.outflow_rate_constant,
        )

    @property
    def source_grains_per_fill(self) -> float:
        """
        :return: How many grains fill the source, as its fill readings count them.
        """
        first = self.transfer.iloc[0]
        return first[Sample.GRAINS_IN_SOURCE] / first[Sample.MEASURED_SOURCE_FILL]

    @property
    def receiver_grains_per_fill(self) -> float:
        """
        :return: How many grains fill the receiver, as its fill readings count them.
        """
        samples = self.record.samples
        landed = samples[samples[Sample.GRAINS_IN_RECEIVER] > 0]
        if not len(landed):
            return math.nan
        last = landed.iloc[-1]
        return last[Sample.GRAINS_IN_RECEIVER] / last[Sample.MEASURED_RECEIVER_FILL]

    def geometry(self) -> DrainGeometry:
        """
        :return: The source cup as the controller's model saw it.
        """
        equation = self.record.equation
        return DrainGeometry(
            container_height=equation.container_height,
            lip_offset=equation.lip_offset,
            grains_per_fill=self.source_grains_per_fill,
            slot_offset=equation.container_width / 2,
        )

    def controller_points(self) -> List[Tuple[float, ModelPoint]]:
        """
        :return: When the controller's model changed and to what, starting with the
            model it was given: a rule's revisions of the repose angle and of the
            discharge coefficient.
        """
        changes: List[Tuple[float, Dict[FreeParameter, float]]] = []
        for revision in self.description.get(TrialKey.REVISIONS) or []:
            changes.append(
                (
                    revision[RevisionKey.TIME],
                    {
                        FreeParameter.REPOSE_ANGLE: revision[
                            RevisionKey.NEW_REPOSE_ANGLE
                        ]
                    },
                )
            )
        for diagnosis in self.description.get(TrialKey.DIAGNOSES) or []:
            if diagnosis[DiagnosisKey.CONCEPT] == FreeParameter.DISCHARGE_COEFFICIENT:
                changes.append(
                    (
                        diagnosis[DiagnosisKey.TIME],
                        {
                            FreeParameter.DISCHARGE_COEFFICIENT: diagnosis[
                                DiagnosisKey.NEW_VALUE
                            ]
                        },
                    )
                )
        points = [(-math.inf, self.given)]
        for at, values in sorted(changes, key=lambda change: change[0]):
            points.append((at, points[-1][1].with_values(values)))
        return points

    @property
    def logged_streams(self) -> Dict[float, Dict[str, Any]]:
        """
        :return: The stream and the true crossing a bank logged at each reading, by the
            reading's time rounded to the millisecond; nothing for a pour without one.
        """
        return {
            round(reading[BankReadingKey.TIME], 3): reading
            for reading in self.description.get(TrialKey.BANK_READINGS) or []
        }

    @property
    def fall_time(self) -> float:
        """
        :return: How long a grain falls from the lip to the receiver's opening: from
            the drop the bank logged, or the replay's assumed one.
        """
        streams = self.logged_streams
        drops = [reading[BankReadingKey.STREAM]["drop"] for reading in streams.values()]
        drop = float(numpy.median(drops)) if drops else REPLAY_DROP
        return math.sqrt(2 * drop / STANDARD_GRAVITY)

    def readings(self, sensors: SensorSet, seed: int) -> List[Reading]:
        """
        :param sensors: The sensors to draw the readings of.
        :param seed: Where their noise is drawn from.
        :return: One reading per perception period of the transfer: the cups' weights
            drawn from what the grains did, and, where the pour logged the stream, the
            stream and where the camera would have seen it cross.
        """
        period = 1.0 / PERCEPTION_FREQUENCY
        transfer = self.transfer
        start = transfer[Sample.TIME].iloc[0]
        steps = numpy.round((transfer[Sample.TIME] - start) / period, 6)
        taken = transfer[steps == numpy.round(steps)]
        receiver_channel = SensorChannel(
            model=sensors.receiver.model,
            grain_share=1.0,
            random=numpy.random.default_rng([seed, 1]),
        )
        source_channel = SensorChannel(
            model=sensors.wrist.model,
            grain_share=1.0,
            random=numpy.random.default_rng([seed, 0]),
        )
        receiver_column = (
            Sample.COMMITTED_RECEIVER_FILL
            if sensors.receiver.model.counts_falling_grains
            else Sample.MEASURED_RECEIVER_FILL
        )
        receiver_scale = self.receiver_grains_per_fill
        if math.isnan(receiver_scale):
            receiver_scale = 0.0
        source_scale = self.source_grains_per_fill
        geometry = self.geometry()
        points = self.controller_points()
        opening = self.receiver.geometry.inner_radius
        streams = self.logged_streams
        camera = sensors.camera.model
        camera_random = numpy.random.default_rng([seed, 2])
        readings = []
        for _, sample in taken.iterrows():
            at = float(sample[Sample.TIME])
            tilt = float(sample[Sample.TILT])
            controller = [point for since, point in points if since <= at][-1]
            logged = streams.get(round(at, 3))
            if logged is not None:
                stream = StreamGeometry(**logged[BankReadingKey.STREAM])
                crossing = logged[BankReadingKey.TRUE_CROSSING]
                if camera is None or crossing is None:
                    crossing = None
                elif camera_random.random() < camera.dropout:
                    crossing = None
                else:
                    crossing = crossing + camera_random.normal(0.0, camera.noise)
            else:
                stream = self._stream(
                    geometry,
                    controller,
                    tilt,
                    sample[Sample.MODEL_SOURCE_FILL] * source_scale,
                    opening,
                )
                crossing = None
            readings.append(
                Reading(
                    time=at,
                    tilt=tilt,
                    receiver_grains=_unclipped(
                        receiver_channel, sample[receiver_column] * receiver_scale, at
                    ),
                    source_grains=_unclipped(
                        source_channel,
                        sample[Sample.MEASURED_SOURCE_FILL] * source_scale,
                        at,
                    ),
                    crossing_distance=crossing,
                    stream=stream,
                )
            )
        return readings

    def _stream(
        self,
        geometry: DrainGeometry,
        controller: ModelPoint,
        tilt: float,
        believed_grains: float,
        opening: float,
    ) -> StreamGeometry:
        """
        :return: The stream as the controller aimed it: the receiver's centre where the
            controller's own model throws it.
        """
        horizontal, upward = stream_direction(self.source, tilt)
        parameters = numpy.array(
            [
                [
                    controller.repose_angle,
                    controller.discharge_coefficient,
                    controller.outflow_rate_constant,
                    math.nan,
                ]
            ]
        )
        head = geometry.head(tilt, numpy.array([believed_grains]), parameters)
        aimed = StreamGeometry(
            horizontal_share=horizontal,
            vertical_share=upward,
            drop=REPLAY_DROP,
            centre_distance=0.0,
            opening_radius=opening,
        )
        centre = float(aimed.distance(geometry.exit_speed(head, parameters))[0])
        return StreamGeometry(
            horizontal_share=horizontal,
            vertical_share=upward,
            drop=REPLAY_DROP,
            centre_distance=centre,
            opening_radius=opening,
        )

    def received_share(self, at: float) -> float:
        """
        :param at: A time of the transfer, in seconds.
        :return: The share of the goal the receiver held then, counting the grains on
            their way.
        """
        transfer = self.transfer
        before = transfer[transfer[Sample.TIME] <= at + 1e-9]
        if not len(before):
            return 0.0
        scale = self.receiver_grains_per_fill
        if math.isnan(scale):
            return 0.0
        return float(
            before[Sample.COMMITTED_RECEIVER_FILL].iloc[-1] * scale / GOAL_GRAINS
        )


def _unclipped(channel: SensorChannel, grains: float, at: float) -> float:
    """
    :return: What a sensor reports of a grain count, late and noisy.
    """
    channel._readings.append((at, grains))
    delayed = channel._readings[0][1]
    for taken_at, value in channel._readings:
        if taken_at > at - channel.model.delay + 1e-6:
            break
        delayed = value
    return delayed + channel.random.normal(0.0, channel.model.noise)


# %% replaying the bank


@dataclass
class ReplayResult:
    """
    What the bank concluded over one recorded pour.
    """

    pour: LoggedPour
    """
    The pour replayed.
    """

    sensors: SensorSet
    """
    The sensors whose readings were drawn.
    """

    coupling: BankCoupling
    """
    The bank, coupled to a controller that only logs which model it would run on.
    """

    consistency: Optional[Consistency] = None
    """
    How the explanation the controller ended on compares with what the pour showed, or
    ``None`` where it was not checked.
    """

    @property
    def onset_estimate(self) -> Optional[float]:
        """
        :return: The pour's repose angle as the onset hypothesis estimates it, free of
            the twin's spread, or ``None`` without one.
        """
        hypothesis_filter = self.coupling.bank.filters.get(Hypothesis.ONSET)
        if hypothesis_filter is None:
            return None
        return hypothesis_filter.mean().repose_angle

    @property
    def drain_rate_estimate(self) -> Optional[float]:
        """
        :return: The pour's drain rate as the outflow capacity hypothesis estimates it,
            free of the twin's spread, or ``None`` without one.
        """
        hypothesis_filter = self.coupling.bank.filters.get(Hypothesis.OUTFLOW_CAPACITY)
        if hypothesis_filter is None:
            return None
        return hypothesis_filter.mean().outflow_rate_constant

    def row(self) -> Dict[str, Any]:
        """
        :return: One row of the replay table.
        """
        pour, coupling = self.pour, self.coupling
        description = pour.description
        first = next(
            (
                commit
                for commit in coupling.commits
                if commit.hypothesis.revises
                or commit.hypothesis is Hypothesis.CATCH_ALL
            ),
            None,
        )
        start = pour.transfer[Sample.TIME].iloc[0]
        summary = coupling.summary()
        truth = pour.truth
        row = {
            "name": description[TrialKey.NAME],
            "source": pour.source,
            "receiver": pour.receiver,
            "perturbation": pour.perturbation.label,
            "seed": description[TrialKey.SEED],
            "sensors": self.sensors,
            "truth": None if truth is None else truth.label,
            "committed": None if first is None else first.hypothesis.label,
            "commit_time": None if first is None else first.time - start,
            "poured_share_at_commit": None if first is None else first.poured_share,
            "final_most_probable": coupling.bank.most_probable.label,
            "commits": len(coupling.commits),
            "duration": pour.transfer[Sample.TIME].iloc[-1] - start,
        }
        true_point = pour.true_point
        for parameter in (
            FreeParameter.REPOSE_ANGLE,
            FreeParameter.DISCHARGE_COEFFICIENT,
            FreeParameter.OUTFLOW_RATE_CONSTANT,
        ):
            row[f"true_{parameter}"] = true_point.value_of(parameter)
            row[f"given_{parameter}"] = pour.given.value_of(parameter)
            row[f"committed_{parameter}"] = (
                None if first is None else first.estimate.value_of(parameter)
            )
        for hypothesis, values in summary.items():
            row[f"p_{hypothesis.label}"] = values["probability"]
            for parameter, value in values["mean"].items():
                row[f"{hypothesis.label}_{parameter}"] = value
            row[f"{hypothesis.label}_bound"] = max(
                values["bound_share"].values(), default=0.0
            )
        row["final_active"] = coupling.active.label
        row["onset_estimate"] = self.onset_estimate
        row["onset_share_inside"] = coupling.bank.share_inside(
            FreeParameter.REPOSE_ANGLE
        )
        row["drain_rate_estimate"] = self.drain_rate_estimate
        row["drain_rate_share_inside"] = coupling.bank.share_inside(
            FreeParameter.OUTFLOW_RATE_CONSTANT
        )
        if self.consistency is not None:
            row["consistent"] = self.consistency.consistent
            for feature, outside in self.consistency.outside.items():
                row[f"outside_{feature}"] = outside
            row["observed_first_tilt"] = self.consistency.observed.first_tilt
            row["observed_early_rate"] = self.consistency.observed.early_rate
            for feature, median in self.consistency.medians.items():
                row[f"predicted_{feature}"] = median
            for feature, position in self.consistency.positions.items():
                row[f"position_{feature}"] = position
            row["approach_rate"] = self.consistency.observed.approach_rate
        return row


def replay(
    pour: LoggedPour,
    sensors: SensorSet,
    settings: BankSettings,
    seed: int = 0,
    forces_second_lip: bool = False,
    onset: Optional[OnsetSpread] = None,
    outflow_capacity: bool = False,
    drain_rate: Optional[DrainRateSpread] = None,
    checks_consistency: bool = False,
    residuals: Optional[Dict[Feature, RateResiduals]] = None,
    predictive_share: float = PREDICTIVE_SHARE,
) -> ReplayResult:
    """
    :param pour: A recorded pour.
    :param sensors: The sensors whose readings to draw.
    :param settings: The bank's settings.
    :param seed: Where the readings' noise and the filters' randomness are drawn from.
    :param forces_second_lip: Whether the second lip is weighed even for a plain cup.
    :param onset: How the twin's repose angle scatters from pour to pour, whose centre
        the model is given, or ``None`` for one fixed angle.
    :param outflow_capacity: Whether the twin has the concept of how much can pass the
        lip.
    :param drain_rate: How the twin's drain rate scatters from pour to pour, whose
        centre the model is given, or ``None`` for one fixed rate.
    :param checks_consistency: Whether the explanation the controller ends on is checked
        against what the pour showed.
    :param residuals: How far each feature lands from its prediction on sound pours, by
        tilt rate, which the check adds to its prediction.
    :param predictive_share: The share of the predictions an observed feature has to
        fall within.
    :return: What the bank concluded along it.
    """
    readings = pour.readings(sensors, seed)
    base = pour.given
    if onset is not None:
        base = base.with_values({FreeParameter.REPOSE_ANGLE: onset.centre})
    if drain_rate is not None:
        base = base.with_values(
            {FreeParameter.OUTFLOW_RATE_CONSTANT: drain_rate.centre}
        )
    knowledge = sensor_knowledge(sensors, fall_time=pour.fall_time)
    bank = HypothesisBank(
        hypotheses=hypotheses_for(pour.source, forces_second_lip, outflow_capacity),
        geometry=pour.geometry(),
        base=base,
        sensors=knowledge,
        settings=settings,
        seed=seed,
        onset=onset,
        drain_rate=drain_rate,
    )
    coupling = BankCoupling(bank=bank)
    for reading in readings:
        coupling.observe(reading, pour.received_share(reading.time))
    consistency = None
    if checks_consistency and bank.started:
        explanation = bank.filters[coupling.active]
        consistency = ConsistencyCheck(
            geometry=bank.geometry,
            sensors=knowledge,
            settings=settings,
            seed=seed,
            residuals=residuals or {},
            predictive_share=predictive_share,
        ).verdict(explanation.parameters, explanation.log_weights, readings)
    return ReplayResult(
        pour=pour, sensors=sensors, coupling=coupling, consistency=consistency
    )
