"""
Recovering a pour from one execution: the hypothesis bank weighing each reading and
switching the controller's model, set against the model alone, the concept-aware rule,
an oracle and the best PID, at three levels of model error: one wrong parameter, two
wrong together, and a missing lip.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from typing_extensions import Dict, FrozenSet, List, Optional, Sequence, Tuple

from experiments.pouring_model_verification.diagnosis_study import DIAGNOSIS_SENSORS
from experiments.pouring_model_verification.consistency import Feature, RateResiduals
from experiments.pouring_model_verification.recovery import OnsetWriteBack
from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    ModelPoint,
    OnsetSpread,
)
from experiments.pouring_model_verification.onset_rule import (
    OnsetResponse,
    OnsetRule,
)
from experiments.pouring_model_verification.persistence_study import (
    POUR_SEED_STRIDE,
    SEQUENCE_LENGTH,
)
from experiments.pouring_model_verification.records import TrialKey, TrialRecord
from experiments.pouring_model_verification.parameters import (
    BASELINE,
    Baselines,
    ModelParameter,
    Perturbation,
)
from experiments.pouring_model_verification.scene import (
    ReceiverCup,
    Scene,
    SourceCup,
)
from experiments.pouring_model_verification.sensor import SensorCondition, SensorSet
from experiments.pouring_model_verification.transfer_trial import (
    PourFault,
    LoopMode,
    SequencePosition,
    TransferTrial,
)

# %% the levels of model error


class Level(StrEnum):
    """
    How deep the model's error runs.
    """

    ONE_PARAMETER = "one_parameter"
    """
    L1: one parameter of the model is wrong.
    """

    COUPLED = "coupled"
    """
    L2: two parameters are wrong together.
    """

    MISSING_LIP = "missing_lip"
    """
    L3: the model lacks a concept, the slot edge of the spouted cup.
    """


ONE_PARAMETER_PERTURBATIONS = (
    Perturbation(parameter=ModelParameter.REPOSE_ANGLE, factor=0.0),
    Perturbation(parameter=ModelParameter.REPOSE_ANGLE, factor=0.5),
    Perturbation(parameter=ModelParameter.REPOSE_ANGLE, factor=1.25),
    Perturbation(parameter=ModelParameter.DISCHARGE_COEFFICIENT, factor=0.33),
    Perturbation(parameter=ModelParameter.DISCHARGE_COEFFICIENT, factor=2.0),
    Perturbation(parameter=ModelParameter.OUTFLOW_RATE_CONSTANT, factor=0.25),
    Perturbation(parameter=ModelParameter.OUTFLOW_RATE_CONSTANT, factor=4.0),
)
"""
The single wrong parameters of L1.
"""

COUPLED_PERTURBATIONS = (
    Perturbation(
        parameter=ModelParameter.REPOSE_ANGLE,
        factor=0.5,
        second_parameter=ModelParameter.DISCHARGE_COEFFICIENT,
        second_factor=0.33,
    ),
    Perturbation(
        parameter=ModelParameter.REPOSE_ANGLE,
        factor=1.25,
        second_parameter=ModelParameter.DISCHARGE_COEFFICIENT,
        second_factor=2.0,
    ),
)
"""
The pairs of wrong parameters of L2.
"""

SECOND_LIP_FIT = ModelPoint(
    repose_angle=1.217720674069427,
    discharge_coefficient=0.3,
    outflow_rate_constant=1.0980679240625195,
    slot_height=0.011082387580239923,
)
"""
The spouted cup's model with the slot edge as a second lip, its height, the repose
angle and the drain rate fitted offline to the hinge's step-and-hold pours by least
squares, on the spout whose trough stops at the wall: the oracle for the missing lip.
"""

COMPARISON_SENSORS = (SensorCondition.IDEAL, SensorCondition.SCALE)
"""
The receiver's sensors the arms without a bank run with.
"""


@dataclass(frozen=True)
class RecoveryCondition:
    """
    One scene and one wrong model a pour is recovered from.
    """

    scene: Scene
    """
    Which cup pours into which receiver.
    """

    perturbation: Perturbation
    """
    How the model's parameters differ from the cup's fit.
    """

    level: Level
    """
    How deep the error runs.
    """

    forces_second_lip: bool = False
    """
    Whether the bank weighs the second lip even for a cup without a slot.
    """

    fault: Optional[PourFault] = None
    """
    A fault in the grains no hypothesis covers, or ``None``.
    """


def one_parameter_conditions() -> List[RecoveryCondition]:
    """
    :return: Every single wrong parameter for both cups into the wide receiver, and a
        wrong discharge coefficient into the narrow one as well.
    """
    conditions = []
    for perturbation in ONE_PARAMETER_PERTURBATIONS:
        receivers = [ReceiverCup.WIDE]
        if perturbation.parameter is ModelParameter.DISCHARGE_COEFFICIENT:
            receivers.append(ReceiverCup.NARROW)
        conditions += [
            RecoveryCondition(
                scene=Scene(source=source, receiver=receiver),
                perturbation=perturbation,
                level=Level.ONE_PARAMETER,
            )
            for source in SourceCup
            for receiver in receivers
        ]
    return conditions


def coupled_conditions() -> List[RecoveryCondition]:
    """
    :return: Both pairs of wrong parameters for the spouted cup into both receivers,
        and the spouted cup at its fit into the narrow receiver, where the onset and
        the aim were seen to interfere.
    """
    return [
        RecoveryCondition(
            scene=Scene(source=SourceCup.SPOUTED_CUP, receiver=receiver),
            perturbation=perturbation,
            level=Level.COUPLED,
        )
        for perturbation in COUPLED_PERTURBATIONS
        for receiver in ReceiverCup
    ] + [
        RecoveryCondition(
            scene=Scene(source=SourceCup.SPOUTED_CUP, receiver=ReceiverCup.NARROW),
            perturbation=BASELINE,
            level=Level.COUPLED,
        )
    ]


def missing_lip_conditions() -> List[RecoveryCondition]:
    """
    :return: The spouted cup at its fit into both receivers, the plain cup at its fit
        with the second lip forced into the bank, and both cups with rough grains.
    """
    return (
        [
            RecoveryCondition(
                scene=Scene(source=SourceCup.SPOUTED_CUP, receiver=receiver),
                perturbation=BASELINE,
                level=Level.MISSING_LIP,
            )
            for receiver in ReceiverCup
        ]
        + [
            RecoveryCondition(
                scene=Scene(source=SourceCup.CUP, receiver=receiver),
                perturbation=BASELINE,
                level=Level.MISSING_LIP,
                forces_second_lip=True,
            )
            for receiver in ReceiverCup
        ]
        + [
            RecoveryCondition(
                scene=Scene(source=source, receiver=ReceiverCup.WIDE),
                perturbation=BASELINE,
                level=Level.MISSING_LIP,
                fault=PourFault.ROUGH_GRAINS,
            )
            for source in SourceCup
        ]
    )


def nominal_conditions() -> List[RecoveryCondition]:
    """
    :return: Both cups at their fit into both receivers, with nothing set wrong: where
        a switch is a false commit.
    """
    return [
        RecoveryCondition(
            scene=Scene(source=source, receiver=receiver),
            perturbation=BASELINE,
            level=Level.ONE_PARAMETER,
        )
        for source in SourceCup
        for receiver in ReceiverCup
    ]


# %% the pours


def bank_trial(
    condition: RecoveryCondition,
    baselines: Baselines,
    seed: int,
    settings: BankSettings,
    sensors: SensorSet,
    sequence: Optional[SequencePosition] = None,
) -> TransferTrial:
    """
    :return: One pour of a condition with the hypothesis bank switching the controller's
        model.
    """
    return TransferTrial(
        scene=condition.scene,
        baseline=baselines.of(condition.scene.source),
        perturbation=condition.perturbation,
        loop_mode=LoopMode.FILL_FEEDBACK,
        seed=seed,
        sensor=sensors.receiver,
        wrist_sensor=sensors.wrist,
        bank=settings,
        sensor_set=sensors,
        forces_second_lip=condition.forces_second_lip,
        fault=condition.fault,
        sequence=sequence,
    )


def bank_trials(
    conditions: Sequence[RecoveryCondition],
    baselines: Baselines,
    seeds: Sequence[int],
    settings: BankSettings,
    sensor_sets: Sequence[SensorSet],
    recorded: FrozenSet[str] = frozenset(),
) -> List[TransferTrial]:
    """
    :return: Every condition with the bank, under every sensor set and seed, except the
        pours already recorded.
    """
    trials = [
        bank_trial(condition, baselines, seed, settings, sensors)
        for condition in conditions
        for sensors in sensor_sets
        for seed in seeds
    ]
    return [trial for trial in trials if trial.name not in recorded]


def comparison_trials(
    conditions: Sequence[RecoveryCondition],
    baselines: Baselines,
    seeds: Sequence[int],
    revision: OnsetRule,
    recorded: FrozenSet[str] = frozenset(),
) -> List[TransferTrial]:
    """
    Under every condition and both receivers' sensors: the model alone, the
    concept-aware rule, and the oracle: the model at the cup's fit, or for the spouted
    cup at its fit the model with the second lip fitted offline.

    :param conditions: The conditions.
    :param baselines: Each cup's fitted parameters.
    :param seeds: The microstates.
    :param revision: The tuned onset rule, whose patience the concept-aware rule uses.
    :param recorded: The names of pours an earlier study already recorded.
    :return: The pours not recorded yet, each once.
    """
    concept_aware = replace(revision, response=OnsetResponse.CONCEPT_AWARE_REVISION)
    trials = {}
    for condition in conditions:
        if condition.forces_second_lip:
            continue
        for sensor in COMPARISON_SENSORS:
            common = dict(
                scene=condition.scene,
                baseline=baselines.of(condition.scene.source),
                loop_mode=LoopMode.FILL_FEEDBACK,
                sensor=sensor,
                fault=condition.fault,
            )
            for seed in seeds:
                arms = [
                    TransferTrial(
                        **common, perturbation=condition.perturbation, seed=seed
                    ),
                    TransferTrial(
                        **common,
                        perturbation=condition.perturbation,
                        seed=seed,
                        onset_rule=concept_aware,
                        start_margin=0.0,
                        wrist_sensor=DIAGNOSIS_SENSORS[sensor],
                    ),
                    TransferTrial(**common, perturbation=BASELINE, seed=seed),
                ]
                if condition.scene.source is SourceCup.SPOUTED_CUP:
                    arms.append(
                        TransferTrial(
                            **common,
                            perturbation=BASELINE,
                            seed=seed,
                            fixed_model=SECOND_LIP_FIT,
                            fixed_label="second_lip_fit",
                        )
                    )
                for trial in arms:
                    trials[trial.name] = trial
    return [trial for name, trial in trials.items() if name not in recorded]


# %% sequences of pours that write back what they learned


@dataclass(frozen=True)
class BankSequence:
    """
    One cup poured :data:`SEQUENCE_LENGTH` times with the bank, refilled before every
    pour, each pour starting from the model the previous one wrote back.
    """

    condition: RecoveryCondition
    """
    The scene and model every pour runs under.
    """

    sensors: SensorSet
    """
    What every pour reads.
    """

    sequence_seed: int
    """
    The microstate the first pour starts from.
    """

    settings: BankSettings
    """
    The bank's settings.
    """

    check_residuals: Optional[Dict[Feature, RateResiduals]] = None
    """
    How far each feature the consistency check reads landed from its prediction on the
    cup's sound pours, or ``None`` for an uncalibrated check.
    """

    def trial(
        self,
        pour: int,
        baselines: Baselines,
        kept: Optional[ModelPoint],
        kept_onset: Optional[OnsetSpread] = None,
    ) -> TransferTrial:
        """
        :param pour: Which pour, counted from one.
        :param baselines: Each cup's fitted parameters.
        :param kept: The model written back so far, or ``None`` for the model as
            given.
        :param kept_onset: The twin's spread of the repose angle written back so far, or
            ``None`` for the twin's own.
        :return: The pour.
        """
        return replace(
            bank_trial(
                self.condition,
                baselines,
                self.sequence_seed + POUR_SEED_STRIDE * (pour - 1),
                self.settings,
                self.sensors,
                sequence=SequencePosition(
                    pour=pour,
                    sequence_seed=self.sequence_seed,
                    kept_model=kept,
                    kept_onset=kept_onset,
                ),
            ),
            check_residuals=self.check_residuals,
        )

    def ready_trial(
        self, baselines: Baselines, records: Dict[str, TrialRecord]
    ) -> Optional[TransferTrial]:
        """
        :param baselines: Each cup's fitted parameters.
        :param records: The pours recorded so far, by name.
        :return: The first pour not recorded yet, starting from what the ones before it
            wrote back, or ``None`` once the sequence is poured. A bank that adapts the
            onset writes back only the twin's spread of the repose angle, and only
            after enough pours in a row land outside it on one side.
        """
        kept: Optional[ModelPoint] = None
        write_back: Optional[OnsetWriteBack] = None
        for pour in range(1, SEQUENCE_LENGTH + 1):
            kept_onset = (
                None
                if write_back is None or write_back.onset == first_onset
                else write_back.onset
            )
            trial = self.trial(pour, baselines, kept, kept_onset)
            if pour == 1:
                first_onset = trial.onset
                write_back = (
                    None if first_onset is None else OnsetWriteBack(onset=first_onset)
                )
            record = records.get(trial.name)
            if record is None:
                return trial
            description = record.description
            if write_back is not None:
                consistency = description.get(TrialKey.CONSISTENCY)
                write_back = write_back.after(
                    description[TrialKey.ONSET_ESTIMATE],
                    explained=consistency is None or consistency["consistent"],
                )
            elif description[TrialKey.WRITES_BACK]:
                kept = ModelPoint.from_json(description[TrialKey.FINAL_MODEL])
        return None


def bank_sequences(
    seeds: Sequence[int], settings: BankSettings, sensor_sets: Sequence[SensorSet]
) -> List[BankSequence]:
    """
    :return: A sequence of the spouted cup at its fit into either receiver, under every
        sensor set, from every seed.
    """
    return [
        BankSequence(
            condition=condition,
            sensors=sensors,
            sequence_seed=seed,
            settings=settings,
        )
        for condition in missing_lip_conditions()[:2]
        for sensors in sensor_sets
        for seed in seeds
    ]
