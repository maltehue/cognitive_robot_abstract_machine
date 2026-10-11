"""
Running a study's trials side by side, each in a process of its own, and keeping every
finished one on disk so an interrupted campaign picks up where it stopped.
"""

# %% imports

from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Dict, List, Protocol

from experiments.pouring_model_verification.parameters import (
    BASELINE,
    Baselines,
    closed_loop_perturbations,
)
from experiments.pouring_model_verification.inflow_model import InflowModels
from experiments.pouring_model_verification.pid import (
    PidVariant,
    SOFTER_GAINS,
    STIFFER_GAINS,
    TunedPidGains,
    tuning_grid,
)
from experiments.pouring_model_verification.scene import (
    ReceiverCup,
    Scene,
    SourceCup,
)
from experiments.pouring_model_verification.sensor import SensorCondition
from experiments.pouring_model_verification.records import TrialRecord
from experiments.pouring_model_verification.tilt_rig import (
    TiltRigTrial,
    fitting_profiles,
    held_out_profiles,
)
from experiments.pouring_model_verification.transfer_trial import (
    FillController,
    LoopMode,
    TransferTrial,
)

logger = logging.getLogger(__name__)

# %% what a campaign runs


class Trial(Protocol):
    """
    One run of a study that writes a record of its own.
    """

    @property
    def name(self) -> str:
        """
        :return: The directory name the trial writes its record into.
        """

    def run(self) -> TrialRecord:
        """
        :return: The record of the finished run.
        """


def tilt_rig_trials(seeds: int, sources: List[SourceCup]) -> List[TiltRigTrial]:
    """
    :param seeds: How many microstates of the grains each profile runs on.
    :param sources: Which cups to tip.
    :return: Every fitting and held-out profile on every seed, for every cup.
    """
    return [
        TiltRigTrial(source=source, profile=profile, seed=seed)
        for source in sources
        for profile in fitting_profiles() + held_out_profiles()
        for seed in range(seeds)
    ]


def transfer_trials(
    seeds: int, scenes: List[Scene], baselines: Baselines
) -> List[TransferTrial]:
    """
    :param seeds: How many microstates of the grains each combination runs on.
    :param scenes: Which source cups pour into which receivers.
    :param baselines: The parameters each source cup's perturbations are taken from.
    :return: Every perturbation under every loop mode on every seed, in every scene.
    :raises MissingBaselineError: If a scene's source cup has not been fitted.
    """
    return [
        TransferTrial(
            scene=scene,
            baseline=baselines.of(scene.source),
            perturbation=perturbation,
            loop_mode=loop_mode,
            seed=seed,
        )
        for scene in scenes
        for perturbation in closed_loop_perturbations()
        for loop_mode in LoopMode
        for seed in range(seeds)
    ]


PID_TUNING_RECEIVER = ReceiverCup.WIDE
"""
The receiver the PID baseline is tuned on; the narrow one is left to test how its gains
carry over, as the narrow receiver tests how a fit carries over.
"""

PID_GAIN_FACTORS = (SOFTER_GAINS, STIFFER_GAINS)
"""
What the tuned PID gains are scaled by to see how sensitive the baseline is to them,
like a wrong model parameter.
"""

MODEL_PREDICTIVE_FEEDBACK = (
    LoopMode.FILL_FEEDBACK,
    LoopMode.FILL_FEEDBACK_AND_CALIBRATION,
)
"""
The loop modes in which the model-predictive controller reads a sensor at all.
"""


def pid_tuning_trials(
    seeds: int, sources: List[SourceCup], baselines: Baselines
) -> List[TransferTrial]:
    """
    :param seeds: How many microstates of the grains each set of gains runs on.
    :param sources: Which cups to tune on.
    :param baselines: The parameters the aim's model is evaluated with.
    :return: Every set of gains of the tuning grid on every seed, for every cup, poured
        into the tuning receiver with an ideal sensor.
    """
    return [
        TransferTrial(
            scene=Scene(source=source, receiver=PID_TUNING_RECEIVER),
            baseline=baselines.of(source),
            perturbation=BASELINE,
            loop_mode=LoopMode.FILL_FEEDBACK,
            seed=seed,
            controller=FillController.PID,
            gains=gains,
        )
        for source in sources
        for gains in tuning_grid()
        for seed in range(seeds)
    ]


FIRST_ROBUSTNESS_SEED = 4
"""
The first microstate of the robustness study, after those the transfer study and the PID
tuning ran on, so neither controller is scored on the pours it was fitted or tuned on.
"""


def robustness_seeds(count: int) -> List[int]:
    """
    :param count: How many microstates of the grains each combination runs on.
    :return: Those microstates, none of them run by an earlier study.
    """
    return list(range(FIRST_ROBUSTNESS_SEED, FIRST_ROBUSTNESS_SEED + count))


def robustness_trials(
    seeds: List[int], scenes: List[Scene], baselines: Baselines, tuned: TunedPidGains
) -> List[TransferTrial]:
    """
    The model-predictive controller at its fitted parameters and the tuned PID baseline,
    each under every sensor; and the PID baseline with its gains scaled, or holding the
    cup still instead of aiming it, with an ideal sensor.

    :param seeds: The microstates of the grains each combination runs on.
    :param scenes: Which source cups pour into which receivers.
    :param baselines: Each source cup's fitted parameters.
    :param tuned: Each source cup's tuned PID gains.
    :return: The trials of the robustness study.
    """
    trials: List[TransferTrial] = []
    for scene in scenes:
        common = dict(
            scene=scene,
            baseline=baselines.of(scene.source),
            perturbation=BASELINE,
        )
        gains = tuned.of(scene.source)
        for seed in seeds:
            trials += [
                TransferTrial(**common, loop_mode=loop_mode, seed=seed, sensor=sensor)
                for loop_mode in MODEL_PREDICTIVE_FEEDBACK
                for sensor in SensorCondition
            ]
            trials += [
                TransferTrial(
                    **common,
                    loop_mode=LoopMode.FILL_FEEDBACK,
                    seed=seed,
                    controller=FillController.PID,
                    gains=gains,
                    sensor=sensor,
                )
                for sensor in SensorCondition
            ]
            trials += [
                TransferTrial(
                    **common,
                    loop_mode=LoopMode.FILL_FEEDBACK,
                    seed=seed,
                    controller=FillController.PID,
                    gains=gains.scaled(factor),
                )
                for factor in PID_GAIN_FACTORS
            ]
            trials.append(
                TransferTrial(
                    **common,
                    loop_mode=LoopMode.FILL_FEEDBACK,
                    seed=seed,
                    controller=FillController.PID_FIXED_POSE,
                    gains=gains,
                )
            )
    return trials


COMPENSATED_VARIANTS = (PidVariant.STOP_EARLY, PidVariant.SMITH)
"""
The PID variants that make up for a late reading.
"""

HELD_STILL_RECEIVER = ReceiverCup.WIDE
"""
The receiver a held-still cup pours into, where it has the best chance of hitting.
"""

HELD_STILL_SENSOR = SensorCondition.SCALE
"""
The sensor a held-still cup is poured behind: one with a lag to make up for, so the
variants can differ at all.
"""


def delay_compensation_trials(
    seeds: List[int],
    scenes: List[Scene],
    baselines: Baselines,
    tuned: TunedPidGains,
    inflow_models: InflowModels,
) -> List[TransferTrial]:
    """
    Each compensated PID variant at the tuned gains, aimed, under every sensor; and
    every PID variant once per cup held still, behind a scale.

    :param seeds: The microstates of the grains each combination runs on.
    :param scenes: Which source cups pour into which receivers.
    :param baselines: The parameters the aim's model is evaluated with.
    :param tuned: Each source cup's tuned PID gains.
    :param inflow_models: Each source cup's inflow model, for the Smith predictor.
    :return: The trials of the delay compensation study.
    """

    def trial(
        scene: Scene, seed: int, variant: PidVariant, **variation
    ) -> TransferTrial:
        return TransferTrial(
            scene=scene,
            baseline=baselines.of(scene.source),
            perturbation=BASELINE,
            loop_mode=LoopMode.FILL_FEEDBACK,
            seed=seed,
            gains=tuned.of(scene.source),
            pid_variant=variant,
            inflow_model=(
                inflow_models.of(scene.source) if variant is PidVariant.SMITH else None
            ),
            **variation,
        )

    aimed = [
        trial(scene, seed, variant, controller=FillController.PID, sensor=sensor)
        for scene in scenes
        for seed in seeds
        for variant in COMPENSATED_VARIANTS
        for sensor in SensorCondition
    ]
    sources = list(dict.fromkeys(scene.source for scene in scenes))
    held_still = [
        trial(
            Scene(source=source, receiver=HELD_STILL_RECEIVER),
            seed,
            variant,
            controller=FillController.PID_FIXED_POSE,
            sensor=HELD_STILL_SENSOR,
        )
        for source in sources
        for seed in seeds
        for variant in PidVariant
    ]
    return aimed + held_still


# %% running them


def _run_and_write(trial: Trial, output: Path) -> str:
    """
    Run one trial and write its record, in a worker process.

    :param trial: The trial to run.
    :param output: The directory holding one directory per trial.
    :return: The trial's name.
    """
    trial.run().write(output / trial.name)
    return trial.name


@dataclass
class Campaign:
    """
    A set of trials run into one output directory.
    """

    output: Path
    """
    The directory holding one directory per trial.
    """

    workers: int
    """
    How many trials run at once.
    """

    def pending(self, trials: List[Trial]) -> List[Trial]:
        """
        :param trials: The trials the campaign is asked to run.
        :return: Those that have no finished record yet.
        """
        return [
            trial
            for trial in trials
            if not TrialRecord.is_complete(self.output / trial.name)
        ]

    def run(self, trials: List[Trial]) -> List[str]:
        """
        Run every trial that has no finished record yet.

        A trial that fails is logged and left without a record, so the next run of the
        campaign tries it again.

        :param trials: The trials to run.
        :return: The names of the trials that failed.
        """
        self.output.mkdir(parents=True, exist_ok=True)
        pending = self.pending(trials)
        logger.info(
            f"{len(trials) - len(pending)} of {len(trials)} trials already recorded, "
            f"running {len(pending)} on {self.workers} workers"
        )
        failed: List[str] = []
        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=1,
        ) as pool:
            futures: Dict[Future, Trial] = {
                pool.submit(_run_and_write, trial, self.output): trial
                for trial in pending
            }
            for finished, future in enumerate(as_completed(futures), start=1):
                trial = futures[future]
                error = future.exception()
                if error is not None:
                    failed.append(trial.name)
                    logger.error(f"{trial.name} failed: {error!r}")
                    continue
                logger.info(f"[{finished}/{len(pending)}] {trial.name}")
        return failed
