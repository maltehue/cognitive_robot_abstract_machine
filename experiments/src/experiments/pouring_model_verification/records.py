"""
What a trial records, and where on disk it keeps it.

A trial writes two files into a directory of its own: a table of samples taken while it
ran, one row per sample, and a description of what was run and how it ended.
"""

# %% imports

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas
from typing_extensions import Any, Dict, List

from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
)

# %% the columns of a trial's samples


class Sample(StrEnum):
    """
    A column of the table of samples a trial records.
    """

    TIME = "time"
    """
    Simulated time since the contents came to rest, in seconds.
    """

    PHASE = "phase"
    """
    Which part of the trial the sample was taken in, a :class:`Phase`.
    """

    TILT = "tilt"
    """
    How far the source cup is turned from upright, in radians.
    """

    MEASURED_SOURCE_FILL = "measured_source_fill"
    """
    The source's fill level, read off the grains standing in it.
    """

    MODEL_SOURCE_FILL = "model_source_fill"
    """
    The source's fill level the controller holds.
    """

    MEASURED_RECEIVER_FILL = "measured_receiver_fill"
    """
    The receiver's fill level, read off the grains that have landed in it.
    """

    COMMITTED_RECEIVER_FILL = "committed_receiver_fill"
    """
    The receiver's fill level once the grains falling towards it have landed.
    """

    MODEL_RECEIVER_FILL = "model_receiver_fill"
    """
    The receiver's fill level the controller holds.
    """

    DRAIN_SCALE = "drain_scale"
    """
    The factor the source's drain is currently scaled by.
    """

    GRAINS_IN_SOURCE = "grains_in_source"
    """
    How many grains stand in the source.
    """

    GRAINS_IN_RECEIVER = "grains_in_receiver"
    """
    How many grains stand in the receiver.
    """

    GRAINS_IN_SPOUT = "grains_in_spout"
    """
    How many grains stand in the source's spout, past the slot edge but short of the
    outlet; nothing for a cup without one.
    """


class Phase(StrEnum):
    """
    A part of a trial.
    """

    TILT = "tilt"
    """
    The rig turns the cup along its tilt profile.
    """

    PRE_TILT = "pre_tilt"
    """
    The robot turns the cup to where its model says the pour starts.
    """

    TRANSFER = "transfer"
    """
    The pouring tasks drive the pour.
    """

    RETURN = "return"
    """
    The robot turns the cup back upright once the pour has ended.
    """

    SETTLE = "settle"
    """
    Nothing moves while the last grains land.
    """


# %% how a closed-loop trial ended


class Outcome(StrEnum):
    """
    A number a closed-loop trial reports about how its pour ended.
    """

    GOAL_GRAINS = "goal_grains"
    """
    How many grains the receiver's goal fill level amounts to.
    """

    TOTAL_GRAINS = "total_grains"
    """
    How many grains the source held before the pour.
    """

    DELIVERED_AT_END = "delivered_at_end"
    """
    Grains in the receiver when the pouring motion ended.
    """

    DELIVERED = "delivered"
    """
    Grains in the receiver once the cup is upright and everything has landed.
    """

    RETAINED = "retained"
    """
    Grains still in the source at the end.
    """

    SPILLED = "spilled"
    """
    Grains that ended neither in the source nor in the receiver.
    """

    PEAK_TILT = "peak_tilt"
    """
    The furthest the cup was turned during the transfer, in radians.
    """

    TRANSFER_DURATION = "transfer_duration"
    """
    How long the transfer motion ran, in seconds.
    """

    REACHED_GOAL = "reached_goal"
    """
    Whether the transfer motion ended by reaching its goal rather than timing out.
    """

    ENDING = "ending"
    """
    How the transfer motion came to an end, a
    :class:`~experiments.pouring_model_verification.transfer_trial.MotionEnding`.
    """

    PRE_TILT = "pre_tilt"
    """
    The tilt the cup was carried to before the transfer, from the model's onset.
    """


# %% what a trial file says


class TrialKey(StrEnum):
    """
    A key of a trial's description.
    """

    STUDY = "study"
    NAME = "name"
    SOURCE = "source"
    RECEIVER = "receiver"
    SEED = "seed"
    PROFILE = "profile"
    PERTURBATION = "perturbation"
    PARAMETERS = "parameters"
    LOOP_MODE = "loop_mode"
    CONTROLLER = "controller"
    SENSOR = "sensor"
    GAINS = "gains"
    PID_VARIANT = "pid_variant"
    INFLOW_MODEL = "inflow_model"
    ONSET_RULE = "onset_rule"
    START_MARGIN = "start_margin"
    SEQUENCE = "sequence"
    REPOSE_ANGLE_AT_START = "repose_angle_at_start"
    REPOSE_ANGLE_AT_END = "repose_angle_at_end"
    WRIST_SENSOR = "wrist_sensor"
    DIAGNOSES = "diagnoses"
    DISCHARGE_COEFFICIENT_AT_END = "discharge_coefficient_at_end"
    STALLS = "stalls"
    REVISIONS = "revisions"
    EQUATION = "equation"
    SENSOR_SET = "sensor_set"
    BANK = "bank"
    BANK_TRACE = "bank_trace"
    COMMITS = "commits"
    BANK_SUMMARY = "bank_summary"
    BANK_READINGS = "bank_readings"
    FINAL_MODEL = "final_model"
    WRITES_BACK = "writes_back"
    FORCES_SECOND_LIP = "forces_second_lip"
    CAUTION = "caution"
    CAUTION_RELEASED_AT = "caution_released_at"
    TWIN_ONSET = "twin_onset"
    ONSET_ESTIMATE = "onset_estimate"
    ONSET_SHARE_INSIDE = "onset_share_inside"
    TWIN_DRAIN_RATE = "twin_drain_rate"
    DRAIN_RATE_SHARE_INSIDE = "drain_rate_share_inside"
    CONSISTENCY = "consistency"
    FAULT = "fault"
    OUTCOME = "outcome"
    WALL_TIME = "wall_time"


class Study(StrEnum):
    """
    Which of the two experiments a trial belongs to.
    """

    TILT_RIG = "tilt_rig"
    """
    The source cup tipped along a fixed tilt profile, with no controller.
    """

    TRANSFER = "transfer"
    """
    Tracy pouring into a receiver under the pouring tasks.
    """


class TrialFile(StrEnum):
    """
    The files a trial writes into its directory.
    """

    DESCRIPTION = "trial.json"
    SAMPLES = "samples.csv"


@dataclass
class TrialRecord:
    """
    Everything a finished trial wrote: what was run, how it ended, and its samples.
    """

    description: Dict[str, Any]
    """
    What was run and how it ended, keyed by :class:`TrialKey`.
    """

    samples: pandas.DataFrame
    """
    The samples taken while it ran, with :class:`Sample` columns.
    """

    @property
    def equation(self) -> ArticulatedPouringEquation:
        """
        :return: The source's drain as the trial built it from the cup's geometry.
        """
        return ArticulatedPouringEquation.from_json(self.description[TrialKey.EQUATION])

    def write(self, directory: Path) -> None:
        """
        Write this record into a directory, creating it.

        :param directory: Where the trial keeps its files.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self.samples.to_csv(directory / TrialFile.SAMPLES, index=False)
        (directory / TrialFile.DESCRIPTION).write_text(
            json.dumps(self.description, indent=2)
        )

    @classmethod
    def read(cls, directory: Path) -> TrialRecord:
        """
        :param directory: Where a trial kept its files.
        :return: The record the trial wrote there.
        """
        return cls(
            description=json.loads((directory / TrialFile.DESCRIPTION).read_text()),
            samples=pandas.read_csv(directory / TrialFile.SAMPLES),
        )

    @staticmethod
    def is_complete(directory: Path) -> bool:
        """
        :param directory: Where a trial keeps its files.
        :return: Whether the trial finished and wrote both of them.
        """
        return all((directory / file).exists() for file in TrialFile)

    @classmethod
    def read_all(cls, directory: Path) -> List[TrialRecord]:
        """
        :param directory: A directory holding one directory per trial.
        :return: The records of every trial in it that finished, in name order.
        """
        return [
            cls.read(trial)
            for trial in sorted(directory.iterdir())
            if trial.is_dir() and cls.is_complete(trial)
        ]
