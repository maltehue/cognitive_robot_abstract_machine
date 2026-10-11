"""
Tuning the PID baselines so a winner is not merely lucky: every candidate is screened on
one pour, the best few are poured on every tuning seed, and a winner on the edge of the
grid has the grid extended past it. The tuning seeds are new to every other study, and
the winners are scored again on the held-out seeds.

Each cup is tuned for two sensors: the ideal one, under which every PID variant behaves
the same, and the realistic scale, once for every variant.
"""

# %% imports

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property
from pathlib import Path

import pandas
from typing_extensions import Dict, List, Optional, Sequence, Tuple

from experiments.pouring_model_verification.controller_comparison import (
    ComparisonColumn,
    misplaced_grains,
    succeeded,
)
from experiments.pouring_model_verification.inflow_model import InflowModels
from experiments.pouring_model_verification.parameters import BASELINE, Baselines
from experiments.pouring_model_verification.pid import PidGains, PidVariant
from experiments.pouring_model_verification.records import (
    Outcome,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.scene import (
    ReceiverCup,
    Scene,
    SourceCup,
)
from experiments.pouring_model_verification.sensor import SensorCondition
from experiments.pouring_model_verification.transfer_trial import (
    MAXIMUM_PID_TILT,
    FillController,
    LoopMode,
    TransferTrial,
)

# %% what is tuned

FAIR_TUNING_SEEDS = (10, 11, 12, 13, 14, 15)
"""
The microstates the tuning pours on: new to the transfer study, the first PID tuning
and the held-out seeds.
"""

SHORTLIST_SIZE = 8
"""
How many of the screened settings are poured on every tuning seed.
"""

EDGE_ROUNDS = 1
"""
How many times the grid is extended past a winner on its edge.
"""

TUNING_RECEIVER = ReceiverCup.WIDE
"""
The receiver every tuning pour goes into.
"""

TUNED_SENSORS = (SensorCondition.IDEAL, SensorCondition.SCALE)
"""
The sensors the baselines are tuned for.
"""


@dataclass(frozen=True)
class TuningTarget:
    """
    One PID variant on one cup, tuned for one sensor.
    """

    source: SourceCup
    """
    The cup tuned on.
    """

    sensor: SensorCondition
    """
    The sensor tuned for.
    """

    variant: PidVariant
    """
    The PID variant tuned.
    """

    @property
    def label(self) -> str:
        """
        :return: A short identifier of the target, usable as a key.
        """
        return f"{self.source}/{self.sensor}/{self.variant}"

    @classmethod
    def from_label(cls, label: str) -> TuningTarget:
        """
        :param label: A label written by :attr:`label`.
        :return: The target it names.
        """
        source, sensor, variant = label.split("/")
        return cls(
            source=SourceCup(source),
            sensor=SensorCondition(sensor),
            variant=PidVariant(variant),
        )

    @property
    def variants(self) -> List[PidVariant]:
        """
        :return: The variants the target's winner stands for: every variant for a sensor
            without lag, under which they all behave the same.
        """
        if has_lag(self.sensor):
            return [self.variant]
        return list(PidVariant)


def has_lag(sensor: SensorCondition) -> bool:
    """
    :return: Whether a reading of the sensor shows grains later than they leave the cup,
        so the PID variants can differ under it.
    """
    model = sensor.model
    return model.delay > 0.0 or not model.counts_falling_grains


def tuning_targets(sources: List[SourceCup]) -> List[TuningTarget]:
    """
    :param sources: The cups tuned on.
    :return: For each cup, the plain PID for the ideal sensor, standing for every
        variant, and every variant for the realistic scale.
    """
    return [
        TuningTarget(source=source, sensor=sensor, variant=variant)
        for source in sources
        for sensor in TUNED_SENSORS
        for variant in (PidVariant if has_lag(sensor) else [PidVariant.PLAIN])
    ]


# %% the grid


class GainAxis(StrEnum):
    """
    One of the three gains a tuning grid spans, each halved or doubled past its edge.
    """

    PROPORTIONAL = "proportional"
    INTEGRAL = "integral"
    DERIVATIVE = "derivative"

    def value_of(self, gains: PidGains) -> float:
        """
        :return: This gain's value in a setting.
        """
        match self:
            case GainAxis.PROPORTIONAL:
                return gains.proportional
            case GainAxis.INTEGRAL:
                return gains.integral
            case GainAxis.DERIVATIVE:
                return gains.derivative

    def with_value(self, gains: PidGains, value: float) -> PidGains:
        """
        :return: The setting with this gain set to a value.
        """
        match self:
            case GainAxis.PROPORTIONAL:
                return replace(gains, proportional=value)
            case GainAxis.INTEGRAL:
                return replace(gains, integral=value)
            case GainAxis.DERIVATIVE:
                return replace(gains, derivative=value)


START_TILT_STEP = 0.3
"""
How far past the edge of the tried start tilts the next one lies, in radians.
"""


class TunedGainsKey(StrEnum):
    """
    A key of a target's entry in the file of winners.
    """

    GAINS = "gains"
    POURS = "pours"


@dataclass(frozen=True)
class TuningGrid:
    """
    The gains tried while tuning, one tuple of values per gain.
    """

    proportional: Tuple[float, ...]
    """
    Tilt per missing grain, in radians.
    """

    integral: Tuple[float, ...]
    """
    Tilt per missing grain and second, in radians.
    """

    derivative: Tuple[float, ...]
    """
    Tilt taken back per grain arriving each second, in radians.
    """

    start_tilt: Tuple[float, ...]
    """
    The tilt the cup starts from, in radians.
    """

    def points(self) -> List[PidGains]:
        """
        :return: Every combination of the values.
        """
        return [
            PidGains(
                proportional=proportional,
                integral=integral,
                derivative=derivative,
                start_tilt=start_tilt,
            )
            for proportional, integral, derivative, start_tilt in itertools.product(
                self.proportional, self.integral, self.derivative, self.start_tilt
            )
        ]

    @staticmethod
    def outward_neighbours(
        gains: PidGains, tried: Sequence[PidGains]
    ) -> List[PidGains]:
        """
        :param gains: A winner.
        :param tried: Every setting tried so far.
        :return: For every gain the winner holds at the edge of what was tried, the same
            setting with that gain one step further out: halved or doubled for the
            gains, 0.3 rad for the start tilt; nothing below no gain or upright, nor past
            the largest tilt.
        """
        neighbours = []
        for axis in GainAxis:
            value = axis.value_of(gains)
            values = {axis.value_of(other) for other in tried}
            if value == max(values) and value > 0.0:
                neighbours.append(axis.with_value(gains, value * 2))
            if value == min(values) and value > 0.0:
                neighbours.append(axis.with_value(gains, value / 2))
        starts = {other.start_tilt for other in tried}
        further = round(gains.start_tilt + START_TILT_STEP, 6)
        if gains.start_tilt == max(starts) and further < MAXIMUM_PID_TILT:
            neighbours.append(replace(gains, start_tilt=further))
        if gains.start_tilt == min(starts) and gains.start_tilt > 0.0:
            nearer = round(max(0.0, gains.start_tilt - START_TILT_STEP), 6)
            neighbours.append(replace(gains, start_tilt=nearer))
        return [
            neighbour
            for neighbour in neighbours
            if neighbour != gains and neighbour not in tried
        ]


FAIR_TUNING_GRID = TuningGrid(
    proportional=(0.0025, 0.005, 0.01, 0.02),
    integral=(0.002, 0.005, 0.01),
    derivative=(0.0, 0.02),
    start_tilt=(0.3, 0.6, 0.9),
)
"""
The grid every target is screened on: the first tuning's grid extended past the edges
its winners sat on, without the integral-free settings that never reached the goal
there.
"""


# %% scoring the pours


@dataclass
class TuningScores:
    """
    How each setting poured for each target.
    """

    records: List[TrialRecord]
    """
    Every tuning pour recorded so far.
    """

    @classmethod
    def from_directory(cls, directory: Path) -> TuningScores:
        """
        :param directory: A directory holding one directory per tuning pour.
        :return: The scores of those pours.
        """
        if not directory.exists():
            return cls(records=[])
        return cls(records=TrialRecord.read_all(directory))

    @cached_property
    def summary(self) -> pandas.DataFrame:
        """
        :return: One row per pour: its target, gains, seed and score.
        """
        rows = []
        for record in self.records:
            description = record.description
            outcome = description[TrialKey.OUTCOME]
            delivered, spilled = outcome[Outcome.DELIVERED], outcome[Outcome.SPILLED]
            target = TuningTarget(
                source=SourceCup(description[TrialKey.SOURCE]),
                sensor=SensorCondition(description[TrialKey.SENSOR]),
                variant=PidVariant(description[TrialKey.PID_VARIANT]),
            )
            rows.append(
                {
                    ComparisonColumn.TARGET: target.label,
                    ComparisonColumn.GAINS: PidGains.from_json(
                        description[TrialKey.GAINS]
                    ),
                    ComparisonColumn.SEED: description[TrialKey.SEED],
                    ComparisonColumn.MISPLACED: misplaced_grains(delivered, spilled),
                    ComparisonColumn.SUCCEEDED: succeeded(delivered, spilled),
                }
            )
        return pandas.DataFrame(
            rows,
            columns=[
                ComparisonColumn.TARGET,
                ComparisonColumn.GAINS,
                ComparisonColumn.SEED,
                ComparisonColumn.MISPLACED,
                ComparisonColumn.SUCCEEDED,
            ],
        )

    def ranking(self, target: TuningTarget) -> pandas.DataFrame:
        """
        :param target: A tuning target.
        :return: One row per setting poured for it: mean grains misplaced, runs within
            tolerance and pours; fewest misplaced first, most within tolerance among
            equals.
        """
        of_target = self.summary[self.summary[ComparisonColumn.TARGET] == target.label]
        return (
            of_target.groupby(ComparisonColumn.GAINS, as_index=False, sort=False)
            .agg(
                **{
                    ComparisonColumn.MISPLACED: (ComparisonColumn.MISPLACED, "mean"),
                    ComparisonColumn.SUCCESS_RATE: (ComparisonColumn.SUCCEEDED, "sum"),
                    ComparisonColumn.RUNS: (ComparisonColumn.SEED, "count"),
                }
            )
            .sort_values(
                [ComparisonColumn.MISPLACED, ComparisonColumn.SUCCESS_RATE],
                ascending=[True, False],
                kind="stable",
            )
        )

    def best(self, target: TuningTarget) -> Optional[PidGains]:
        """
        :param target: A tuning target.
        :return: Its best setting among those poured on every tuning seed, or ``None``
            before any was.
        """
        ranking = self.ranking(target)
        complete = ranking[ranking[ComparisonColumn.RUNS] >= len(FAIR_TUNING_SEEDS)]
        if complete.empty:
            return None
        return complete.iloc[0][ComparisonColumn.GAINS]

    def pours(self, target: TuningTarget) -> int:
        """
        :param target: A tuning target.
        :return: How many pours its tuning took.
        """
        return int((self.summary[ComparisonColumn.TARGET] == target.label).sum())

    def tried(self, target: TuningTarget) -> List[PidGains]:
        """
        :param target: A tuning target.
        :return: Every setting poured for it.
        """
        return list(self.ranking(target)[ComparisonColumn.GAINS])


# %% the tuning


def _pid_trial(
    scene: Scene,
    gains: PidGains,
    variant: PidVariant,
    sensor: SensorCondition,
    seed: int,
    baselines: Baselines,
    inflow_models: InflowModels,
) -> TransferTrial:
    """
    :return: One aimed pour of a PID variant.
    """
    return TransferTrial(
        scene=scene,
        baseline=baselines.of(scene.source),
        perturbation=BASELINE,
        loop_mode=LoopMode.FILL_FEEDBACK,
        seed=seed,
        controller=FillController.PID,
        gains=gains,
        sensor=sensor,
        pid_variant=variant,
        inflow_model=(
            inflow_models.of(scene.source) if variant is PidVariant.SMITH else None
        ),
    )


@dataclass
class FairTuning:
    """
    The stages of the tuning, each a set of trials decided by the pours before it.
    """

    grid: TuningGrid
    """
    The grid every target is screened on.
    """

    targets: List[TuningTarget]
    """
    What is tuned.
    """

    baselines: Baselines
    """
    The parameters the aim's model is evaluated with.
    """

    inflow_models: InflowModels
    """
    The inflow model each cup's Smith predictor runs with.
    """

    def _trial(self, target: TuningTarget, gains: PidGains, seed: int) -> TransferTrial:
        return _pid_trial(
            Scene(source=target.source, receiver=TUNING_RECEIVER),
            gains,
            target.variant,
            target.sensor,
            seed,
            self.baselines,
            self.inflow_models,
        )

    def screening_trials(self) -> List[TransferTrial]:
        """
        :return: Every grid point of every target, poured on the first tuning seed.
        """
        return [
            self._trial(target, gains, FAIR_TUNING_SEEDS[0])
            for target in self.targets
            for gains in self.grid.points()
        ]

    def shortlist_trials(self, scores: TuningScores) -> List[TransferTrial]:
        """
        :param scores: The screening's scores.
        :return: Each target's best screened settings, poured on every further seed.
        """
        return [
            self._trial(target, gains, seed)
            for target in self.targets
            for gains in self.grid_ranking(scores, target)[:SHORTLIST_SIZE]
            for seed in FAIR_TUNING_SEEDS[1:]
        ]

    def grid_ranking(
        self, scores: TuningScores, target: TuningTarget
    ) -> List[PidGains]:
        """
        :return: The target's screened grid points, best first, by their screening pour.
        """
        points = set(self.grid.points())
        screened = scores.summary[
            (scores.summary[ComparisonColumn.TARGET] == target.label)
            & (scores.summary[ComparisonColumn.SEED] == FAIR_TUNING_SEEDS[0])
        ]
        ranked = screened.sort_values(
            [ComparisonColumn.MISPLACED, ComparisonColumn.SUCCEEDED],
            ascending=[True, False],
            kind="stable",
        )
        return [gains for gains in ranked[ComparisonColumn.GAINS] if gains in points]

    def edge_trials(self, scores: TuningScores) -> List[TransferTrial]:
        """
        :param scores: The scores so far.
        :return: For each target whose winner sits on the edge of what was tried, the
            settings one step further out, poured on every tuning seed.
        """
        trials = []
        for target in self.targets:
            best = scores.best(target)
            if best is None:
                continue
            for gains in self.grid.outward_neighbours(best, scores.tried(target)):
                trials += [
                    self._trial(target, gains, seed) for seed in FAIR_TUNING_SEEDS
                ]
        return trials

    def result(self, scores: TuningScores) -> FairlyTunedGains:
        """
        :param scores: The scores of every stage.
        :return: Each target's winner and what its tuning cost.
        """
        return FairlyTunedGains(
            gains={target: scores.best(target) for target in self.targets},
            pours={target: scores.pours(target) for target in self.targets},
        )


# %% the winners


@dataclass(frozen=True)
class FairlyTunedGains:
    """
    Each target's winning gains, and how many pours finding them took.
    """

    gains: Dict[TuningTarget, PidGains]
    """
    The winners.
    """

    pours: Dict[TuningTarget, int]
    """
    How many tuning pours each target took.
    """

    def write(self, path: Path) -> None:
        """
        :param path: The file to write the winners to.
        """
        path.write_text(
            json.dumps(
                {
                    target.label: {
                        TunedGainsKey.GAINS: gains.to_json(),
                        TunedGainsKey.POURS: self.pours[target],
                    }
                    for target, gains in self.gains.items()
                },
                indent=2,
            )
        )

    @classmethod
    def read(cls, path: Path) -> FairlyTunedGains:
        """
        :param path: A file written by :meth:`write`.
        :return: The winners it holds.
        """
        data = json.loads(path.read_text())
        targets = {label: TuningTarget.from_label(label) for label in data}
        return cls(
            gains={
                targets[label]: PidGains.from_json(entry[TunedGainsKey.GAINS])
                for label, entry in data.items()
            },
            pours={
                targets[label]: entry[TunedGainsKey.POURS]
                for label, entry in data.items()
            },
        )


@dataclass
class PidSelection:
    """
    Which tuned PID meets the model under each sensor, decided by the tuning pours
    alone: the plain PID tuned for the ideal sensor without lag, and with lag the
    variant whose winner misplaced fewest grains while tuning for the realistic scale.
    """

    tuned: FairlyTunedGains
    """
    The tuning's winners.
    """

    scores: TuningScores
    """
    The tuning pours.
    """

    def ideal_tuned(self, source: SourceCup) -> Tuple[PidVariant, PidGains]:
        """
        :return: The plain PID tuned for the ideal sensor.
        """
        target = TuningTarget(
            source=source, sensor=SensorCondition.IDEAL, variant=PidVariant.PLAIN
        )
        return target.variant, self.tuned.gains[target]

    def scale_tuned(self, source: SourceCup) -> Tuple[PidVariant, PidGains]:
        """
        :return: The variant, with its winner, that tuned best for the realistic scale.
        """
        candidates = [
            target
            for target in self.tuned.gains
            if target.source == source and target.sensor is SensorCondition.SCALE
        ]
        best = min(candidates, key=self._tuning_score)
        return best.variant, self.tuned.gains[best]

    def for_sensor(
        self, source: SourceCup, sensor: SensorCondition
    ) -> Tuple[PidVariant, PidGains]:
        """
        :return: The PID that meets the model under a sensor.
        """
        if has_lag(sensor):
            return self.scale_tuned(source)
        return self.ideal_tuned(source)

    def _tuning_score(self, target: TuningTarget) -> float:
        """
        :return: The mean grains a target's winner misplaced over its tuning pours.
        """
        ranking = self.scores.ranking(target)
        winner = ranking[ranking[ComparisonColumn.GAINS] == self.tuned.gains[target]]
        return float(winner[ComparisonColumn.MISPLACED].iloc[0])


def fair_rescoring_trials(
    seeds: List[int],
    scenes: List[Scene],
    baselines: Baselines,
    selection: PidSelection,
    inflow_models: InflowModels,
) -> List[TransferTrial]:
    """
    The PID tuned for the ideal sensor under every sensor, and the variant chosen for a
    lagging sensor under every sensor with lag, in every scene.

    :param seeds: The held-out microstates.
    :param scenes: Which source cups pour into which receivers.
    :param baselines: The parameters the aim's model is evaluated with.
    :param selection: Which tuned PIDs are rescored.
    :param inflow_models: Each cup's inflow model, for the Smith predictor.
    :return: The trials of the rescoring, each once.
    """
    trials: Dict[str, TransferTrial] = {}
    for scene in scenes:
        source = SourceCup(scene.source)
        rescored = [
            (selection.ideal_tuned(source), list(SensorCondition)),
            (
                selection.scale_tuned(source),
                [sensor for sensor in SensorCondition if has_lag(sensor)],
            ),
        ]
        for (variant, gains), sensors in rescored:
            for sensor in sensors:
                for seed in seeds:
                    trial = _pid_trial(
                        scene, gains, variant, sensor, seed, baselines, inflow_models
                    )
                    trials[trial.name] = trial
    return list(trials.values())
