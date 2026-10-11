"""
Run the pouring model verification experiment, and draw its figures from what it
recorded.

The closed-loop study perturbs each source cup's parameters around what its own
open-loop trials fit to, so the tilt rig is recorded and fitted first. The robustness
study compares the model-predictive controller with a model-free PID baseline under
imperfect sensors, so the PID baseline is tuned before it. ``run --study all`` does
every step in order; each can also run on its own.

Examples::

    # Record everything (resumes: trials already on disk are skipped):
    python experiments/scripts/run_pouring_model_verification.py run --study all

    # Only the spouted cup's open-loop trials, then fit every recorded cup:
    python experiments/scripts/run_pouring_model_verification.py run --study tilt_rig \\
        --sources spouted_cup
    python experiments/scripts/run_pouring_model_verification.py fit

    # Tune the PID baseline, then compare it with the model under imperfect sensors:
    python experiments/scripts/run_pouring_model_verification.py run --study pid_tuning
    python experiments/scripts/run_pouring_model_verification.py tune
    python experiments/scripts/run_pouring_model_verification.py run --study robustness

    # Fit, compare and draw everything recorded so far:
    python experiments/scripts/run_pouring_model_verification.py analyze
"""

# %% imports

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import replace
from enum import StrEnum
from pathlib import Path

import pandas
from typing_extensions import List, Optional

from experiments.pouring_model_verification.analysis import Analysis
from experiments.pouring_model_verification.campaign import (
    Campaign,
    Trial,
    delay_compensation_trials,
    pid_tuning_trials,
    robustness_seeds,
    robustness_trials,
    tilt_rig_trials,
    transfer_trials,
)
from experiments.pouring_model_verification.controller_comparison import (
    ComparisonColumn,
    ControllerComparison,
    PidTuning,
)
from experiments.pouring_model_verification.fair_comparison import (
    FairColumn,
    FairComparison,
)
from experiments.pouring_model_verification.figures import (
    draw_all,
    draw_controller_comparison,
    write_frames,
)
from experiments.pouring_model_verification.parameters import Baselines
from experiments.pouring_model_verification.pid import TunedPidGains
from experiments.pouring_model_verification.fair_tuning import (
    EDGE_ROUNDS,
    FAIR_TUNING_GRID,
    FairlyTunedGains,
    FairTuning,
    PidSelection,
    TuningScores,
    fair_rescoring_trials,
    tuning_targets,
)
from experiments.pouring_model_verification.onset_study import (
    OnsetEvaluation,
    OnsetTuning,
    TunedOnsetRules,
    onset_evaluation_trials,
    onset_tuning_trials,
)
from experiments.pouring_model_verification.diagnosis_study import (
    DiagnosisEvaluation,
    diagnosis_trials,
)
from experiments.pouring_model_verification.persistence_study import (
    PersistenceEvaluation,
    PourSequence,
    pour_sequences,
    ready_trials,
)
from experiments.pouring_model_verification.onset_rule import (
    OnsetResponse,
    OnsetRule,
)
from experiments.pouring_model_verification.inflow_model import (
    InflowModels,
    identify_inflow_model,
)
from experiments.pouring_model_verification.hypothesis_bank import BankSettings
from experiments.pouring_model_verification.records import Study, TrialKey, TrialRecord
from experiments.pouring_model_verification.recovery_evaluation import (
    RecoveryEvaluation,
)
from experiments.pouring_model_verification.recovery_study import (
    BankSequence,
    bank_sequences,
    bank_trials,
    comparison_trials,
    coupled_conditions,
    missing_lip_conditions,
    nominal_conditions,
    one_parameter_conditions,
)
from experiments.pouring_model_verification.sensor import SensorSet
from experiments.pouring_model_verification.scene import (
    ReceiverCup,
    Scene,
    SourceCup,
)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "results" / "pouring_model_verification"
)
"""
Where the trials and the figures are written unless told otherwise.
"""

TRIALS_DIRECTORY = "trials"
"""
The subdirectory of the output holding one directory per trial.
"""

FIGURES_DIRECTORY = "figures"
"""
The subdirectory of the output holding the figures and the summary tables.
"""

BASELINES_FILE = "baselines.json"
"""
The file in the output holding each source cup's fitted parameters.
"""

PID_TUNING_DIRECTORY = "trials_pid_tuning"
"""
The subdirectory of the output holding the PID baseline's tuning trials.
"""

ROBUSTNESS_DIRECTORY = "trials_robustness"
"""
The subdirectory of the output holding the robustness study's trials.
"""

PID_GAINS_FILE = "pid_gains.json"
"""
The file in the output holding each source cup's tuned PID gains.
"""

DELAY_COMPENSATION_DIRECTORY = "trials_delay_compensation"
"""
The subdirectory of the output holding the delay compensation study's trials.
"""

INFLOW_MODELS_FILE = "inflow_models.json"
"""
The file in the output holding each source cup's inflow model, identified from the PID
tuning's readings.
"""

FAIR_TUNING_DIRECTORY = "trials_fair_tuning"
"""
The subdirectory of the output holding the fair tuning's pours.
"""

FAIR_RESCORING_DIRECTORY = "trials_fair_rescoring"
"""
The subdirectory of the output holding the fairly tuned winners' held-out pours.
"""

FAIR_GAINS_FILE = "fair_pid_gains.json"
"""
The file in the output holding each fair tuning target's winner and its cost.
"""

ONSET_TUNING_DIRECTORY = "trials_onset_tuning"
"""
The subdirectory of the output holding the onset rule's tuning pours.
"""

ONSET_DIRECTORY = "trials_onset"
"""
The subdirectory of the output holding the onset study's held-out pours.
"""

ONSET_RULES_FILE = "onset_rules.json"
"""
The file in the output holding the tuned onset rules.
"""

PERSISTENCE_DIRECTORY = "trials_persistence"
"""
The subdirectory of the output holding the persistence study's sequences of pours.
"""

DIAGNOSIS_DIRECTORY = "trials_diagnosis"
"""
The subdirectory of the output holding the diagnosis study's pours.
"""

RECOVERY_TUNING_DIRECTORY = "trials_recovery_tuning"
"""
The subdirectory of the output holding the hypothesis bank's tuning pours.
"""

RECOVERY_DIRECTORY = "trials_recovery"
"""
The subdirectory of the output holding the recovery study's held-out pours.
"""

BANK_SETTINGS_FILE = "bank_settings.json"
"""
The file in the output holding the hypothesis bank's tuned settings.
"""

RECOVERY_TUNING_SEEDS = (10, 11, 12)
"""
The microstates the hypothesis bank is tuned on.
"""

PID_TUNING_SEEDS = 2
"""
Microstates of the grains per set of gains while tuning, unless told otherwise.
"""

TILT_RIG_SEEDS = 3
"""
Microstates of the grains per tilt profile, unless told otherwise.
"""

TRANSFER_SEEDS = 4
"""
Microstates of the grains per closed-loop combination, unless told otherwise.
"""


class Command(StrEnum):
    """
    What the script is asked to do.
    """

    RUN = "run"
    FIT = "fit"
    TUNE = "tune"
    IDENTIFY = "identify"
    ANALYZE = "analyze"


class StudySelection(StrEnum):
    """
    Which studies a run records.
    """

    TILT_RIG = Study.TILT_RIG.value
    TRANSFER = Study.TRANSFER.value
    PID_TUNING = "pid_tuning"
    ROBUSTNESS = "robustness"
    DELAY_COMPENSATION = "delay_compensation"
    FAIR_TUNING = "fair_tuning"
    FAIR_RESCORING = "fair_rescoring"
    ONSET_TUNING = "onset_tuning"
    ONSET = "onset"
    PERSISTENCE = "persistence"
    DIAGNOSIS = "diagnosis"
    RECOVERY_TUNING = "recovery_tuning"
    RECOVERY = "recovery"
    ALL = "all"

    def includes(self, study: StudySelection) -> bool:
        """
        :param study: One study.
        :return: Whether this selection records it.
        """
        return self is study or self is StudySelection.ALL


# %% argument parsing


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=list(Command), type=Command)
    parser.add_argument(
        "--study",
        choices=list(StudySelection),
        type=StudySelection,
        default=StudySelection.ALL,
        help="Which study to record.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help=f"Microstates of the grains per combination (default {TILT_RIG_SEEDS} "
        f"for the tilt rig, {PID_TUNING_SEEDS} for the PID tuning, {TRANSFER_SEEDS} "
        f"for the transfer and the robustness study).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(SourceCup),
        type=SourceCup,
        default=list(SourceCup),
        help="Which source cups to record.",
    )
    parser.add_argument(
        "--receivers",
        nargs="+",
        choices=list(ReceiverCup),
        type=ReceiverCup,
        default=list(ReceiverCup),
        help="Which receivers the closed-loop study pours into.",
    )
    parser.add_argument(
        "--workers", type=int, default=12, help="How many trials run at once."
    )
    parser.add_argument(
        "--two-sided",
        action="store_true",
        help="Let the diagnosis study's revisions also lower the repose angle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where trials and figures are written.",
    )
    return parser.parse_args()


# %% entry point


def _fit(output: Path) -> Baselines:
    """
    Fit every source cup's open-loop trials and keep the result as its baseline.

    :param output: The experiment's output directory.
    :return: The baselines written.
    """
    baselines = Analysis.from_directory(output / TRIALS_DIRECTORY).baselines()
    baselines.write(output / BASELINES_FILE)
    logging.info(f"baselines written to {output / BASELINES_FILE}: {baselines}")
    return baselines


def _tune(output: Path) -> TunedPidGains:
    """
    Pick each source cup's best gains from its tuning trials and keep them.

    :param output: The experiment's output directory.
    :return: The gains written.
    """
    tuned = PidTuning.from_directory(output / PID_TUNING_DIRECTORY).best()
    tuned.write(output / PID_GAINS_FILE)
    logging.info(f"PID gains written to {output / PID_GAINS_FILE}: {tuned}")
    return tuned


def _identify(output: Path) -> InflowModels:
    """
    Identify each source cup's inflow model from the readings of its PID tuning pours,
    and keep them.

    :param output: The experiment's output directory.
    :return: The models written.
    """
    records = TrialRecord.read_all(output / PID_TUNING_DIRECTORY)
    sources = list(
        dict.fromkeys(record.description[TrialKey.SOURCE] for record in records)
    )
    models = InflowModels(
        models={
            source: identify_inflow_model(
                [
                    record
                    for record in records
                    if record.description[TrialKey.SOURCE] == source
                ]
            )
            for source in sources
        }
    )
    models.write(output / INFLOW_MODELS_FILE)
    logging.info(f"inflow models written to {output / INFLOW_MODELS_FILE}: {models}")
    return models


def _fair_tuning(campaign: Campaign, tuning: FairTuning, output: Path) -> None:
    """
    Screen the grid, pour the shortlist on every tuning seed, extend the grid past a
    winner on its edge, and keep the winners.

    :param campaign: Where the tuning pours are recorded.
    :param tuning: The tuning's stages.
    :param output: The experiment's output directory.
    """
    _record(campaign, tuning.screening_trials())
    _record(
        campaign, tuning.shortlist_trials(TuningScores.from_directory(campaign.output))
    )
    for _ in range(EDGE_ROUNDS):
        trials = tuning.edge_trials(TuningScores.from_directory(campaign.output))
        if not trials:
            break
        _record(campaign, trials)
    winners = tuning.result(TuningScores.from_directory(campaign.output))
    winners.write(output / FAIR_GAINS_FILE)
    logging.info(f"fairly tuned gains written to {output / FAIR_GAINS_FILE}: {winners}")


def _pour_sequences(
    campaign: Campaign, sequences: List[PourSequence], baselines: Baselines
) -> None:
    """
    Pour every sequence to its end, each pour as soon as the one it starts from is
    recorded.

    :param campaign: Where the pours are recorded.
    :param sequences: The sequences to pour.
    :param baselines: Each cup's fitted parameters.
    """
    campaign.output.mkdir(parents=True, exist_ok=True)
    while True:
        records = {
            record.description[TrialKey.NAME]: record
            for record in TrialRecord.read_all(campaign.output)
        }
        trials = ready_trials(sequences, baselines, records)
        if not trials:
            return
        failed = campaign.run(trials)
        if len(failed) == len(trials):
            logging.error(f"no pour of this round finished: {', '.join(failed)}")
            return


def _bank_sequences(
    campaign: Campaign, sequences: List[BankSequence], baselines: Baselines
) -> None:
    """
    Pour every sequence with the bank to its end, each pour as soon as the one before it
    has written back what it learned.

    :param campaign: Where the pours are recorded.
    :param sequences: The sequences to pour.
    :param baselines: Each cup's fitted parameters.
    """
    campaign.output.mkdir(parents=True, exist_ok=True)
    while True:
        records = {
            record.description[TrialKey.NAME]: record
            for record in TrialRecord.read_all(campaign.output)
        }
        trials = {
            trial.name: trial
            for sequence in sequences
            if (trial := sequence.ready_trial(baselines, records)) is not None
        }
        if not trials:
            return
        failed = campaign.run(list(trials.values()))
        if len(failed) == len(trials):
            logging.error(f"no pour of this round finished: {', '.join(failed)}")
            return


def _recovery_conditions():
    """
    :return: Every condition of the recovery study: one wrong parameter, two together,
        a missing lip, and both cups at their fit.
    """
    return (
        one_parameter_conditions()
        + coupled_conditions()
        + missing_lip_conditions()
        + nominal_conditions()
    )


def _record(campaign: Campaign, trials: List[Trial]) -> None:
    """
    Run a campaign's trials and say which failed.
    """
    failed = campaign.run(trials)
    if failed:
        logging.error(f"{len(failed)} trials failed: {', '.join(failed)}")


def _analyze(output: Path) -> None:
    """
    Draw the figures and write the tables of everything recorded so far.

    :param output: The experiment's output directory.
    """
    figures = output / FIGURES_DIRECTORY
    analysis = Analysis.from_directory(output / TRIALS_DIRECTORY)
    draw_all(analysis, figures)
    _analyze_persistence(output, figures, analysis)
    selection = _analyze_fair_tuning(output, figures)
    if selection is not None:
        _analyze_onset(output, figures, selection)
        _analyze_diagnosis(output, figures, selection)
        _analyze_recovery(output, figures, selection)
    if not (output / ROBUSTNESS_DIRECTORY).exists():
        return
    tuning = PidTuning.from_directory(output / PID_TUNING_DIRECTORY)
    draw_controller_comparison(
        ControllerComparison(
            records=[
                record
                for directory in (ROBUSTNESS_DIRECTORY, DELAY_COMPENSATION_DIRECTORY)
                if (output / directory).exists()
                for record in TrialRecord.read_all(output / directory)
            ],
            tuned=tuning.best(),
        ),
        tuning.table,
        figures,
    )


def _analyze_fair_tuning(output: Path, figures: Path) -> Optional[PidSelection]:
    """
    Compare the fairly tuned PIDs with the model on the held-out seeds, and write the
    verdicts and the tuning costs.

    :return: Which PID meets the model under each sensor, or ``None`` before the fair
        rescoring was recorded.
    """
    if not (output / FAIR_RESCORING_DIRECTORY).exists():
        return None
    tuned = FairlyTunedGains.read(output / FAIR_GAINS_FILE)
    scores = TuningScores.from_directory(output / FAIR_TUNING_DIRECTORY)
    selection = PidSelection(tuned=tuned, scores=scores)
    comparison = FairComparison(
        model=TrialRecord.read_all(output / ROBUSTNESS_DIRECTORY),
        pid=TrialRecord.read_all(output / FAIR_RESCORING_DIRECTORY),
        selection=selection,
    )
    costs = pandas.DataFrame(
        [
            {
                FairColumn.SOURCE: target.source,
                FairColumn.SENSOR: target.sensor,
                FairColumn.PID_VARIANT: target.variant,
                ComparisonColumn.GAINS: gains.label,
                ComparisonColumn.MISPLACED: float(
                    scores.ranking(target)
                    .set_index(ComparisonColumn.GAINS)
                    .loc[[gains], ComparisonColumn.MISPLACED]
                    .iloc[0]
                ),
                ComparisonColumn.RUNS: tuned.pours[target],
            }
            for target, gains in tuned.gains.items()
        ]
    )
    write_frames(
        {"fair_comparison": comparison.table, "fair_tuning_costs": costs},
        "fair_comparison",
        figures,
    )
    return selection


class HingeColumn(StrEnum):
    """
    A column of the table of repose angles the hinge gives each cup.
    """

    SOURCE = "source"
    FITTED = "fitted"
    STEADY_TURNS = "steady_turns"


def _analyze_persistence(output: Path, figures: Path, analysis: Analysis) -> None:
    """
    Summarise every pour index of the sequences, keep each pour's repose angles, and
    write the repose angles the hinge gives each cup to compare them with, in degrees.
    """
    if not (output / PERSISTENCE_DIRECTORY).exists():
        return
    evaluation = PersistenceEvaluation(
        records=TrialRecord.read_all(output / PERSISTENCE_DIRECTORY)
    )
    hinge = pandas.DataFrame(
        [
            {
                HingeColumn.SOURCE: source,
                HingeColumn.FITTED: math.degrees(
                    source_analysis.fitted_parameters.repose_angle
                ),
                HingeColumn.STEADY_TURNS: math.degrees(
                    source_analysis.steady_turn_repose_angle
                ),
            }
            for source, source_analysis in analysis.fitted.items()
        ]
    )
    write_frames(
        {
            "persistence_summary": evaluation.summary,
            "persistence_pours": evaluation.rows,
            "hinge_repose_angles": hinge,
        },
        "persistence_study",
        figures,
    )


def _analyze_diagnosis(output: Path, figures: Path, selection: PidSelection) -> None:
    """
    Evaluate every arm of the diagnosis study on the held-out seeds, with whichever
    revision, one-sided or two-sided, its concept-aware pours ran.
    """
    if not (output / DIAGNOSIS_DIRECTORY).exists():
        return
    records = TrialRecord.read_all(output / DIAGNOSIS_DIRECTORY)
    concept_aware = next(
        OnsetRule.from_json(record.description[TrialKey.ONSET_RULE])
        for record in records
        if record.description[TrialKey.WRIST_SENSOR] is not None
    )
    evaluation = DiagnosisEvaluation(
        records=records
        + TrialRecord.read_all(output / ONSET_DIRECTORY)
        + TrialRecord.read_all(output / ROBUSTNESS_DIRECTORY),
        pid_records=TrialRecord.read_all(output / FAIR_RESCORING_DIRECTORY),
        revision=replace(concept_aware, response=OnsetResponse.REVISION),
        selection=selection,
    )
    write_frames(
        {"diagnosis_summary": evaluation.summary, "diagnosis_pours": evaluation.rows},
        "diagnosis_study",
        figures,
    )


def _analyze_recovery(output: Path, figures: Path, selection: PidSelection) -> None:
    """
    Score every arm of the recovery study on the held-out seeds, and the bank's tuning
    pours on their own.
    """
    if not (output / RECOVERY_DIRECTORY).exists():
        return
    baselines = Baselines.read(output / BASELINES_FILE)
    conditions = _recovery_conditions()
    records = [
        record
        for directory in (
            RECOVERY_DIRECTORY,
            DIAGNOSIS_DIRECTORY,
            ONSET_DIRECTORY,
            ROBUSTNESS_DIRECTORY,
        )
        if (output / directory).exists()
        for record in TrialRecord.read_all(output / directory)
        if record.description[TrialKey.SEED] in range(4, 4 + TRANSFER_SEEDS)
    ]
    evaluation = RecoveryEvaluation(
        records=records,
        conditions=conditions,
        baselines=baselines,
        pid_records=TrialRecord.read_all(output / FAIR_RESCORING_DIRECTORY),
        selection=selection,
    )
    tuning = RecoveryEvaluation(
        records=TrialRecord.read_all(output / RECOVERY_TUNING_DIRECTORY),
        conditions=conditions,
        baselines=baselines,
        pid_records=[],
        selection=None,
    )
    write_frames(
        {"recovery_pours": evaluation.rows, "recovery_tuning_pours": tuning.rows},
        "recovery_study",
        figures,
    )


def _analyze_onset(output: Path, figures: Path, selection: PidSelection) -> None:
    """
    Evaluate the onset rule's arms on the held-out seeds, and write the rows and the
    rule's tuning.
    """
    if not (output / ONSET_DIRECTORY).exists():
        return
    tuning = OnsetTuning(records=TrialRecord.read_all(output / ONSET_TUNING_DIRECTORY))
    evaluation = OnsetEvaluation(
        records=TrialRecord.read_all(output / ONSET_DIRECTORY)
        + TrialRecord.read_all(output / ROBUSTNESS_DIRECTORY),
        pid_records=TrialRecord.read_all(output / FAIR_RESCORING_DIRECTORY),
        tuned=TunedOnsetRules.read(output / ONSET_RULES_FILE),
        selection=selection,
    )
    tuning_table = tuning.table.assign(
        **{
            ComparisonColumn.RULE: tuning.table[ComparisonColumn.RULE].map(
                lambda rule: rule.label
            )
        }
    )
    write_frames(
        {"onset_summary": evaluation.summary, "onset_tuning": tuning_table},
        "onset_study",
        figures,
    )


def _run(arguments: argparse.Namespace) -> None:
    """
    Record the selected studies, fitting and tuning in between where they depend on it.

    :param arguments: The parsed command line.
    """
    output = arguments.output
    study = arguments.study
    scenes = [
        Scene(source=source, receiver=receiver)
        for source in arguments.sources
        for receiver in arguments.receivers
    ]

    def campaign(directory: str) -> Campaign:
        return Campaign(output=output / directory, workers=arguments.workers)

    if study.includes(StudySelection.TILT_RIG):
        _record(
            campaign(TRIALS_DIRECTORY),
            tilt_rig_trials(
                seeds=arguments.seeds or TILT_RIG_SEEDS, sources=arguments.sources
            ),
        )
    if study is StudySelection.ALL:
        _fit(output)
    if study.includes(StudySelection.TRANSFER):
        _record(
            campaign(TRIALS_DIRECTORY),
            transfer_trials(
                seeds=arguments.seeds or TRANSFER_SEEDS,
                scenes=scenes,
                baselines=Baselines.read(output / BASELINES_FILE),
            ),
        )
    if study.includes(StudySelection.PID_TUNING):
        _record(
            campaign(PID_TUNING_DIRECTORY),
            pid_tuning_trials(
                seeds=arguments.seeds or PID_TUNING_SEEDS,
                sources=arguments.sources,
                baselines=Baselines.read(output / BASELINES_FILE),
            ),
        )
    if study is StudySelection.ALL:
        _tune(output)
    if study.includes(StudySelection.ROBUSTNESS):
        _record(
            campaign(ROBUSTNESS_DIRECTORY),
            robustness_trials(
                seeds=robustness_seeds(arguments.seeds or TRANSFER_SEEDS),
                scenes=scenes,
                baselines=Baselines.read(output / BASELINES_FILE),
                tuned=TunedPidGains.read(output / PID_GAINS_FILE),
            ),
        )
    if study is StudySelection.ALL:
        _identify(output)
    if study.includes(StudySelection.DELAY_COMPENSATION):
        _record(
            campaign(DELAY_COMPENSATION_DIRECTORY),
            delay_compensation_trials(
                seeds=robustness_seeds(arguments.seeds or TRANSFER_SEEDS),
                scenes=scenes,
                baselines=Baselines.read(output / BASELINES_FILE),
                tuned=TunedPidGains.read(output / PID_GAINS_FILE),
                inflow_models=InflowModels.read(output / INFLOW_MODELS_FILE),
            ),
        )
    if study.includes(StudySelection.FAIR_TUNING):
        _fair_tuning(
            campaign(FAIR_TUNING_DIRECTORY),
            FairTuning(
                grid=FAIR_TUNING_GRID,
                targets=tuning_targets(arguments.sources),
                baselines=Baselines.read(output / BASELINES_FILE),
                inflow_models=InflowModels.read(output / INFLOW_MODELS_FILE),
            ),
            output,
        )
    if study.includes(StudySelection.ONSET_TUNING):
        _record(
            campaign(ONSET_TUNING_DIRECTORY),
            onset_tuning_trials(Baselines.read(output / BASELINES_FILE)),
        )
        rules = OnsetTuning(
            records=TrialRecord.read_all(output / ONSET_TUNING_DIRECTORY)
        ).tuned()
        rules.write(output / ONSET_RULES_FILE)
        logging.info(f"onset rules written to {output / ONSET_RULES_FILE}: {rules}")
    if study.includes(StudySelection.ONSET):
        _record(
            campaign(ONSET_DIRECTORY),
            onset_evaluation_trials(
                seeds=robustness_seeds(arguments.seeds or TRANSFER_SEEDS),
                baselines=Baselines.read(output / BASELINES_FILE),
                tuned=TunedOnsetRules.read(output / ONSET_RULES_FILE),
            ),
        )
    if study.includes(StudySelection.PERSISTENCE):
        _pour_sequences(
            campaign(PERSISTENCE_DIRECTORY),
            pour_sequences(
                seeds=robustness_seeds(arguments.seeds or TRANSFER_SEEDS),
                revision=TunedOnsetRules.read(output / ONSET_RULES_FILE).revision,
            ),
            Baselines.read(output / BASELINES_FILE),
        )
    if study.includes(StudySelection.DIAGNOSIS):
        _record(
            campaign(DIAGNOSIS_DIRECTORY),
            diagnosis_trials(
                seeds=robustness_seeds(arguments.seeds or TRANSFER_SEEDS),
                baselines=Baselines.read(output / BASELINES_FILE),
                revision=TunedOnsetRules.read(output / ONSET_RULES_FILE).revision,
                lowers=arguments.two_sided,
                recorded=frozenset(
                    trial.name
                    for directory in (ONSET_DIRECTORY, ROBUSTNESS_DIRECTORY)
                    for trial in (output / directory).iterdir()
                    if TrialRecord.is_complete(trial)
                ),
            ),
        )
    if study.includes(StudySelection.RECOVERY_TUNING):
        tuning_conditions = {
            (condition.scene, condition.perturbation): condition
            for condition in one_parameter_conditions()
            + coupled_conditions()
            + nominal_conditions()
            + missing_lip_conditions()[:2]
        }
        _record(
            campaign(RECOVERY_TUNING_DIRECTORY),
            bank_trials(
                list(tuning_conditions.values()),
                Baselines.read(output / BASELINES_FILE),
                list(RECOVERY_TUNING_SEEDS),
                BankSettings(label="probe"),
                [SensorSet.IDEAL, SensorSet.REALISTIC],
            ),
        )
    if study.includes(StudySelection.RECOVERY):
        baselines = Baselines.read(output / BASELINES_FILE)
        settings = BankSettings.from_json(
            json.loads((output / BANK_SETTINGS_FILE).read_text())
        )
        seeds = robustness_seeds(arguments.seeds or TRANSFER_SEEDS)
        conditions = _recovery_conditions()
        _record(
            campaign(RECOVERY_DIRECTORY),
            bank_trials(conditions, baselines, seeds, settings, list(SensorSet))
            + comparison_trials(
                conditions,
                baselines,
                seeds,
                revision=TunedOnsetRules.read(output / ONSET_RULES_FILE).revision,
                recorded=frozenset(
                    trial.name
                    for directory in (
                        ONSET_DIRECTORY,
                        ROBUSTNESS_DIRECTORY,
                        DIAGNOSIS_DIRECTORY,
                    )
                    for trial in (output / directory).iterdir()
                    if TrialRecord.is_complete(trial)
                ),
            ),
        )
        _bank_sequences(
            campaign(RECOVERY_DIRECTORY),
            bank_sequences(seeds, settings, list(SensorSet)),
            baselines,
        )
    if study.includes(StudySelection.FAIR_RESCORING):
        _record(
            campaign(FAIR_RESCORING_DIRECTORY),
            fair_rescoring_trials(
                seeds=robustness_seeds(arguments.seeds or TRANSFER_SEEDS),
                scenes=scenes,
                baselines=Baselines.read(output / BASELINES_FILE),
                selection=PidSelection(
                    tuned=FairlyTunedGains.read(output / FAIR_GAINS_FILE),
                    scores=TuningScores.from_directory(output / FAIR_TUNING_DIRECTORY),
                ),
                inflow_models=InflowModels.read(output / INFLOW_MODELS_FILE),
            ),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    arguments = _parse_arguments()
    match arguments.command:
        case Command.FIT:
            _fit(arguments.output)
        case Command.TUNE:
            _tune(arguments.output)
        case Command.IDENTIFY:
            _identify(arguments.output)
        case Command.ANALYZE:
            _analyze(arguments.output)
        case Command.RUN:
            _run(arguments)


if __name__ == "__main__":
    main()
