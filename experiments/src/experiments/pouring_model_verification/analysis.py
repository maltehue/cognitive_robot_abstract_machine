"""
What the recorded trials say: how closely the pouring equation follows the grains, how
that moves with each parameter, which parameters fit best, and how much each loop mode
makes up for parameters that are wrong.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from pathlib import Path

import numpy
import pandas
from scipy.optimize import least_squares
from typing_extensions import Dict, List

from experiments.pouring_model_verification.model_replay import (
    FillModelReplay,
    RecordedPour,
    root_mean_square_errors,
)
from experiments.pouring_model_verification.parameters import (
    NOMINAL_PARAMETERS,
    DEMO_PARAMETERS,
    Baselines,
    ModelParameter,
    ModelParameters,
    Perturbation,
)
from experiments.pouring_model_verification.records import (
    Outcome,
    Phase,
    Sample,
    Study,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.tilt_rig import (
    ProfileKey,
    ProfileKind,
    TiltProfile,
)
from experiments.pouring_model_verification.scene import SourceCup
from experiments.pouring_model_verification.transfer_trial import LoopMode
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
)

# %% the tables an analysis produces


class Column(StrEnum):
    """
    A column of a table the analysis produces.
    """

    SOURCE = "source"
    RECEIVER = "receiver"
    PARAMETER_SET = "parameter_set"
    PARAMETER = "parameter"
    FACTOR = "factor"
    PERTURBATION = "perturbation"
    LOOP_MODE = "loop_mode"
    SEED = "seed"
    PROFILE = "profile"
    TILT = "tilt"
    TIME = "time"
    FITTING_ERROR = "fitting_error"
    HELD_OUT_ERROR = "held_out_error"
    IN_SITU_ERROR = "in_situ_error"
    MEASURED = "measured"
    PREDICTED = "predicted"
    MEASURED_ONSET = "measured_onset"
    PREDICTED_ONSET = "predicted_onset"
    DELIVERY_ERROR = "delivery_error"
    RELATIVE_DELIVERY_ERROR = "relative_delivery_error"
    BELIEF_ERROR = "belief_error"
    MEAN = "mean"
    STANDARD_DEVIATION = "standard_deviation"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    OUTCOME = "outcome"
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    COMPENSATION = "compensation"
    RUNS = "runs"


class ParameterSet(StrEnum):
    """
    A named set of parameter values the replay compares.
    """

    DEMO = "demo"
    """
    The values the demo runs with, fitted in an earlier round of the evaluation note.
    """

    NOMINAL = "nominal"
    """
    The model before any parameter of the contents is estimated.
    """

    FITTED = "fitted"
    """
    Repose angle and rate constant fitted here to the step-and-hold trials.
    """

    FITTED_WITH_LIP = "fitted_with_lip"
    """
    As :attr:`FITTED`, with the lip offset freed as well.
    """


SWEEP_FACTORS = (0.25, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 2.0, 2.8, 4.0)
"""
The factors each parameter is scaled by in the one-at-a-time sensitivity of the replay.
"""

SWEPT_PARAMETERS = (
    ModelParameter.REPOSE_ANGLE,
    ModelParameter.OUTFLOW_RATE_CONSTANT,
    ModelParameter.LIP_OFFSET,
)
"""
The parameters the fill prediction depends on; the discharge coefficient only moves
where the stream lands, so the fill replay cannot see it.
"""

LANDSCAPE_REPOSE_ANGLES = numpy.radians(numpy.linspace(0.0, 60.0, 16))
"""
The repose angles of the error landscape, in radians.
"""

LANDSCAPE_RATE_CONSTANTS = numpy.geomspace(0.03, 3.0, 16)
"""
The outflow rate constants of the error landscape, in 1/s.
"""

REPLAYED_TRANSFER_PHASES = (Phase.PRE_TILT, Phase.TRANSFER, Phase.RETURN)
"""
The parts of a closed-loop trial whose tilt the replay follows: from the settled cup
until it stands upright again.
"""

EXAMPLE_PROFILES = ("step_085.0deg", "ramp_0.20rads")
"""
The tilt profiles whose trajectories the analysis keeps for plotting.
"""


# %% loading and splitting the trials


@dataclass
class SourceAnalysis:
    """
    Everything the analysis derives about one source cup: its open-loop trials, and the
    closed-loop trials it poured in, into either receiver.
    """

    records: List[TrialRecord]
    """
    Every finished trial of any source cup.
    """

    source: SourceCup
    """
    The source cup analysed.
    """

    rig: List[TrialRecord] = field(init=False)
    """
    The open-loop tilt rig trials.
    """

    transfer: List[TrialRecord] = field(init=False)
    """
    The closed-loop transfer trials.
    """

    def __post_init__(self) -> None:
        of_source = [
            record
            for record in self.records
            if record.description[TrialKey.SOURCE] == self.source
        ]
        self.rig = [
            record
            for record in of_source
            if record.description[TrialKey.STUDY] == Study.TILT_RIG
        ]
        self.transfer = [
            record
            for record in of_source
            if record.description[TrialKey.STUDY] == Study.TRANSFER
        ]

    @cached_property
    def equation(self) -> ArticulatedPouringEquation:
        """
        :return: The source's drain as built from the cup's geometry, shared by every
            trial of both studies.
        """
        return (self.rig or self.transfer)[0].equation

    @staticmethod
    def profile_of(record: TrialRecord) -> TiltProfile:
        """
        :return: The tilt profile a rig trial ran.
        """
        return TiltProfile.from_json(record.description[TrialKey.PROFILE])

    def _rig_of_kind(self, kind: ProfileKind) -> List[TrialRecord]:
        return [
            record
            for record in self.rig
            if record.description[TrialKey.PROFILE][ProfileKey.KIND] == kind
        ]

    @cached_property
    def fitting_pours(self) -> List[RecordedPour]:
        """
        :return: The step-and-hold trials, which the parameters are fitted to.
        """
        return [
            RecordedPour.from_samples(record.samples)
            for record in self._rig_of_kind(ProfileKind.STEP)
        ]

    @cached_property
    def held_out_pours(self) -> List[RecordedPour]:
        """
        :return: The ramps, which a fit is tested on and never fitted to.
        """
        return [
            RecordedPour.from_samples(record.samples)
            for record in self._rig_of_kind(ProfileKind.RAMP)
        ]

    @cached_property
    def in_situ_pours(self) -> List[RecordedPour]:
        """
        :return: The source's tilt and fill in every closed-loop trial, from the settled
            cup until it is upright again: the cup swung by an arm rather than turned
            on a fixed axis.
        """
        return [
            RecordedPour.from_samples(
                record.samples[
                    record.samples[Sample.PHASE].isin(REPLAYED_TRANSFER_PHASES)
                ]
            )
            for record in self.transfer
        ]

    @property
    def pour_groups(self) -> List[List[RecordedPour]]:
        """
        :return: The fitting, held-out and in-situ pours, in that order.
        """
        return [self.fitting_pours, self.held_out_pours, self.in_situ_pours]

    # %% fitting the parameters

    @cached_property
    def fitted_parameters(self) -> ModelParameters:
        """
        :return: Repose angle and rate constant fitted to the step-and-hold trials by
            least squares, with the geometry pinned to what the cup measures.
        """
        return self._fit(free_lip=False)

    @cached_property
    def fitted_parameters_with_lip(self) -> ModelParameters:
        """
        :return: Repose angle, rate constant and lip offset fitted together.
        """
        return self._fit(free_lip=True)

    def _fit(self, free_lip: bool) -> ModelParameters:
        """
        Least squares of the replayed fill against the step-and-hold trials, started
        from the lowest point of the error landscape.

        :param free_lip: Whether the lip offset is fitted as well.
        :return: The fitted parameters.
        """
        landscape = self.error_landscape
        start = landscape.loc[landscape[Column.FITTING_ERROR].idxmin()]
        initial = [
            start[ModelParameter.REPOSE_ANGLE],
            math.log(start[ModelParameter.OUTFLOW_RATE_CONSTANT]),
        ]
        lower, upper = [0.0, math.log(0.01)], [1.3, math.log(10.0)]
        if free_lip:
            initial.append(1.0)
            lower.append(0.3)
            upper.append(3.0)

        def parameters_of(vector: numpy.ndarray) -> ModelParameters:
            return ModelParameters(
                repose_angle=float(vector[0]),
                outflow_rate_constant=float(math.exp(vector[1])),
                lip_offset_scale=float(vector[2]) if free_lip else 1.0,
            )

        solution = least_squares(
            lambda vector: FillModelReplay(
                equation=self.equation, parameters=parameters_of(vector)
            ).residuals(self.fitting_pours),
            x0=numpy.array(initial),
            bounds=(numpy.array(lower), numpy.array(upper)),
            diff_step=1e-3,
        )
        return parameters_of(solution.x)

    @cached_property
    def parameter_sets(self) -> Dict[ParameterSet, ModelParameters]:
        """
        :return: Every named set of parameters the replay compares.
        """
        return {
            ParameterSet.DEMO: DEMO_PARAMETERS,
            ParameterSet.NOMINAL: NOMINAL_PARAMETERS,
            ParameterSet.FITTED: self.fitted_parameters,
            ParameterSet.FITTED_WITH_LIP: self.fitted_parameters_with_lip,
        }

    # %% how closely the model follows the grains

    @cached_property
    def fits(self) -> pandas.DataFrame:
        """
        :return: One row per named parameter set: its values and its fill error on the
            fitting, held-out and in-situ pours.
        """
        rows = []
        for name, parameters in self.parameter_sets.items():
            fitting, held_out, in_situ = root_mean_square_errors(
                self.equation, parameters, self.pour_groups
            )
            rows.append(
                {
                    Column.PARAMETER_SET: name,
                    **parameters.to_json(),
                    Column.FITTING_ERROR: fitting,
                    Column.HELD_OUT_ERROR: held_out,
                    Column.IN_SITU_ERROR: in_situ,
                }
            )
        return pandas.DataFrame(rows)

    @cached_property
    def replay_consistency(self) -> float:
        """
        :return: The largest difference between the replay at the reference parameters
            and the fill the rig's world integrated along the same tilt, which checks
            the replay integrates the drain the way the world does.
        """
        replay = FillModelReplay(equation=self.equation, parameters=DEMO_PARAMETERS)
        return max(
            float(
                numpy.max(
                    numpy.abs(
                        replay.predict(RecordedPour.from_samples(record.samples))
                        - record.samples[Sample.MODEL_SOURCE_FILL].to_numpy()
                    )
                )
            )
            for record in self.rig
        )

    @cached_property
    def sensitivity(self) -> pandas.DataFrame:
        """
        :return: One row per parameter and factor: the fill error on each set of pours
            with that one parameter of the fitted set scaled.
        """
        rows = []
        for parameter in SWEPT_PARAMETERS:
            for factor in SWEEP_FACTORS:
                fitting, held_out, in_situ = root_mean_square_errors(
                    self.equation,
                    self.fitted_parameters.scaled(parameter, factor),
                    self.pour_groups,
                )
                rows.append(
                    {
                        Column.PARAMETER: parameter,
                        Column.FACTOR: factor,
                        Column.FITTING_ERROR: fitting,
                        Column.HELD_OUT_ERROR: held_out,
                        Column.IN_SITU_ERROR: in_situ,
                    }
                )
        return pandas.DataFrame(rows)

    @cached_property
    def error_landscape(self) -> pandas.DataFrame:
        """
        :return: One row per repose angle and rate constant of a grid: the fill error on
            the fitting and the held-out pours.
        """
        rows = []
        for repose_angle in LANDSCAPE_REPOSE_ANGLES:
            for rate_constant in LANDSCAPE_RATE_CONSTANTS:
                fitting, held_out = root_mean_square_errors(
                    self.equation,
                    ModelParameters(
                        repose_angle=float(repose_angle),
                        outflow_rate_constant=float(rate_constant),
                    ),
                    [self.fitting_pours, self.held_out_pours],
                )
                rows.append(
                    {
                        ModelParameter.REPOSE_ANGLE: float(repose_angle),
                        ModelParameter.OUTFLOW_RATE_CONSTANT: float(rate_constant),
                        Column.FITTING_ERROR: fitting,
                        Column.HELD_OUT_ERROR: held_out,
                    }
                )
        return pandas.DataFrame(rows)

    @cached_property
    def equilibrium(self) -> pandas.DataFrame:
        """
        :return: One row per step trial and parameter set: the fill left at the end of
            the hold, measured and predicted.
        """
        replays = {
            name: FillModelReplay(equation=self.equation, parameters=parameters)
            for name, parameters in self.parameter_sets.items()
        }
        rows = []
        for record in self._rig_of_kind(ProfileKind.STEP):
            pour = RecordedPour.from_samples(record.samples)
            for name, replay in replays.items():
                rows.append(
                    {
                        Column.TILT: self.profile_of(record).final_tilt,
                        Column.SEED: record.description[TrialKey.SEED],
                        Column.PARAMETER_SET: name,
                        Column.MEASURED: float(pour.measured_fill[-1]),
                        Column.PREDICTED: float(replay.predict(pour)[-1]),
                    }
                )
        return pandas.DataFrame(rows)

    @cached_property
    def onsets(self) -> pandas.DataFrame:
        """
        :return: One row per ramp trial and parameter set: the tilt the contents started
            flowing at, measured and predicted.
        """
        rows = []
        for record in self._rig_of_kind(ProfileKind.RAMP):
            pour = RecordedPour.from_samples(record.samples)
            for name, parameters in self.parameter_sets.items():
                predicted = FillModelReplay(
                    equation=self.equation, parameters=parameters
                ).predict(pour)
                rows.append(
                    {
                        Column.PROFILE: self.profile_of(record).label,
                        Column.SEED: record.description[TrialKey.SEED],
                        Column.PARAMETER_SET: name,
                        Column.MEASURED_ONSET: pour.onset_tilt(pour.measured_fill),
                        Column.PREDICTED_ONSET: pour.onset_tilt(predicted),
                    }
                )
        return pandas.DataFrame(rows)

    @cached_property
    def steady_turn_repose_angle(self) -> float:
        """
        :return: The repose angle that puts the fitted model's onset where the steady
            turns measured it, on average, in radians.
        """
        return self.repose_angle_matching(
            self.onsets, self.fitted_parameters.repose_angle
        )

    @staticmethod
    def repose_angle_matching(onsets: pandas.DataFrame, fitted: float) -> float:
        """
        :param onsets: Measured and predicted onsets, as :attr:`onsets` gives them.
        :param fitted: The fitted repose angle, in radians.
        :return: The fitted repose angle moved by the mean amount the fitted model
            predicted the onset too early; the onset moves one for one with it.
        """
        rows = onsets[onsets[Column.PARAMETER_SET] == ParameterSet.FITTED]
        error = rows[Column.MEASURED_ONSET] - rows[Column.PREDICTED_ONSET]
        return fitted + float(error.mean())

    @cached_property
    def example_trajectories(self) -> pandas.DataFrame:
        """
        :return: For each example profile and seed, the measured fill over time and
            the fill each named parameter set predicts along the same tilt.
        """
        frames = []
        for record in self.rig:
            label = self.profile_of(record).label
            if label not in EXAMPLE_PROFILES:
                continue
            pour = RecordedPour.from_samples(record.samples)
            frame = pandas.DataFrame(
                {
                    Column.PROFILE: label,
                    Column.SEED: record.description[TrialKey.SEED],
                    Column.TIME: pour.time,
                    Column.TILT: pour.tilt,
                    Column.MEASURED: pour.measured_fill,
                }
            )
            for name, parameters in self.parameter_sets.items():
                frame[name] = FillModelReplay(
                    equation=self.equation, parameters=parameters
                ).predict(pour)
            frames.append(frame)
        return pandas.concat(frames, ignore_index=True)

    # %% how the closed loop fared

    @cached_property
    def transfer_summary(self) -> pandas.DataFrame:
        """
        :return: One row per closed-loop trial: what was run and how the pour ended.
        """
        rows = []
        for record in self.transfer:
            description = record.description
            perturbation = Perturbation.from_json(description[TrialKey.PERTURBATION])
            outcome = description[TrialKey.OUTCOME]
            transfer = record.samples[record.samples[Sample.PHASE] == Phase.TRANSFER]
            final = transfer.iloc[-1]
            goal = outcome[Outcome.GOAL_GRAINS]
            rows.append(
                {
                    Column.SOURCE: self.source,
                    Column.RECEIVER: description[TrialKey.RECEIVER],
                    Column.PERTURBATION: perturbation.label,
                    Column.PARAMETER: (
                        None
                        if perturbation.parameter is None
                        else perturbation.parameter.value
                    ),
                    Column.FACTOR: perturbation.factor,
                    Column.LOOP_MODE: description[TrialKey.LOOP_MODE],
                    Column.SEED: description[TrialKey.SEED],
                    **outcome,
                    Column.DELIVERY_ERROR: outcome[Outcome.DELIVERED] - goal,
                    Column.RELATIVE_DELIVERY_ERROR: (outcome[Outcome.DELIVERED] - goal)
                    / goal,
                    Column.BELIEF_ERROR: final[Sample.MODEL_RECEIVER_FILL]
                    - final[Sample.COMMITTED_RECEIVER_FILL],
                }
            )
        return pandas.DataFrame(rows)

    @cached_property
    def transfer_aggregate(self) -> pandas.DataFrame:
        """
        :return: One row per perturbation, loop mode and outcome: the mean, spread and
            range of that outcome over the seeds.
        """
        summary = self.transfer_summary
        outcomes = [
            Outcome.DELIVERED,
            Outcome.RETAINED,
            Outcome.SPILLED,
            Outcome.PEAK_TILT,
            Outcome.TRANSFER_DURATION,
            Outcome.REACHED_GOAL,
            Outcome.PRE_TILT,
            Column.DELIVERY_ERROR,
            Column.RELATIVE_DELIVERY_ERROR,
        ]
        keys = [
            Column.SOURCE,
            Column.RECEIVER,
            Column.PERTURBATION,
            Column.PARAMETER,
            Column.FACTOR,
            Column.LOOP_MODE,
        ]
        rows = []
        for group_key, group in summary.groupby(keys, dropna=False):
            for outcome in outcomes:
                values = group[outcome].astype(float)
                rows.append(
                    {
                        **dict(zip(keys, group_key)),
                        Column.OUTCOME: outcome,
                        Column.MEAN: values.mean(),
                        Column.STANDARD_DEVIATION: values.std(ddof=1),
                        Column.MINIMUM: values.min(),
                        Column.MAXIMUM: values.max(),
                        Column.RUNS: len(values),
                    }
                )
        return pandas.DataFrame(rows)

    @cached_property
    def compensation(self) -> pandas.DataFrame:
        """
        :return: One row per perturbation and closing loop mode: the share of the open
            loop's mean absolute delivery error that closing the loop removes; one is
            a perfect correction, zero none, negative a worse pour.
        """
        summary = self.transfer_summary
        mean_absolute_error = (
            summary.assign(
                **{Column.DELIVERY_ERROR: summary[Column.DELIVERY_ERROR].abs()}
            )
            .groupby([Column.RECEIVER, Column.PERTURBATION, Column.LOOP_MODE])[
                Column.DELIVERY_ERROR
            ]
            .mean()
        )
        rows = []
        pairs = summary[[Column.RECEIVER, Column.PERTURBATION]].drop_duplicates()
        for receiver, perturbation in pairs.itertuples(index=False):
            open_loop = mean_absolute_error.get(
                (receiver, perturbation, LoopMode.OPEN_LOOP)
            )
            for loop_mode in (
                LoopMode.FILL_FEEDBACK,
                LoopMode.FILL_FEEDBACK_AND_CALIBRATION,
            ):
                closed = mean_absolute_error.get((receiver, perturbation, loop_mode))
                if open_loop is None or closed is None:
                    continue
                rows.append(
                    {
                        Column.SOURCE: self.source,
                        Column.RECEIVER: receiver,
                        Column.PERTURBATION: perturbation,
                        Column.LOOP_MODE: loop_mode,
                        Column.MEAN_ABSOLUTE_ERROR: closed,
                        Column.COMPENSATION: 1.0 - closed / open_loop,
                    }
                )
        return pandas.DataFrame(rows)

    def transfer_traces(self, perturbations: List[str]) -> pandas.DataFrame:
        """
        :param perturbations: Labels of the perturbations to keep traces of.
        :return: The samples of every closed-loop trial of those perturbations, labelled
            with what was run.
        """
        frames = []
        for record in self.transfer:
            perturbation = Perturbation.from_json(
                record.description[TrialKey.PERTURBATION]
            )
            if perturbation.label not in perturbations:
                continue
            frames.append(
                record.samples.assign(
                    **{
                        Column.SOURCE: self.source,
                        Column.RECEIVER: record.description[TrialKey.RECEIVER],
                        Column.PERTURBATION: perturbation.label,
                        Column.LOOP_MODE: record.description[TrialKey.LOOP_MODE],
                        Column.SEED: record.description[TrialKey.SEED],
                    }
                )
            )
        return pandas.concat(frames, ignore_index=True)


# %% every source cup at once

TRACE_PERTURBATIONS = (
    "baseline",
    "repose_angle_x0",
    "outflow_rate_constant_x0.25",
    "outflow_rate_constant_x4",
)
"""
The perturbations whose closed-loop pours are kept over time.
"""

TRACE_SEED = 0
"""
The seed whose closed-loop pours are kept over time.
"""


class Table(StrEnum):
    """
    A table the analysis writes, named as its file is.
    """

    FITS = "fits"
    SENSITIVITY = "sensitivity"
    ERROR_LANDSCAPE = "error_landscape"
    EQUILIBRIUM = "equilibrium"
    ONSETS = "onsets"
    EXAMPLE_TRAJECTORIES = "example_trajectories"
    REPLAY_CONSISTENCY = "replay_consistency"
    TRANSFER_SUMMARY = "transfer_summary"
    TRANSFER_AGGREGATE = "transfer_aggregate"
    COMPENSATION = "compensation"
    TRANSFER_TRACES = "transfer_traces"

    @property
    def is_open_loop(self) -> bool:
        """
        :return: Whether the table is drawn from the open-loop trials.
        """
        return self in (
            Table.FITS,
            Table.SENSITIVITY,
            Table.ERROR_LANDSCAPE,
            Table.EQUILIBRIUM,
            Table.ONSETS,
            Table.EXAMPLE_TRAJECTORIES,
            Table.REPLAY_CONSISTENCY,
        )

    def of(self, analysis: SourceAnalysis) -> pandas.DataFrame:
        """
        :param analysis: The analysis of one source cup.
        :return: This table for that cup.
        """
        match self:
            case Table.FITS:
                return analysis.fits
            case Table.SENSITIVITY:
                return analysis.sensitivity
            case Table.ERROR_LANDSCAPE:
                return analysis.error_landscape
            case Table.EQUILIBRIUM:
                return analysis.equilibrium
            case Table.ONSETS:
                return analysis.onsets
            case Table.EXAMPLE_TRAJECTORIES:
                return analysis.example_trajectories
            case Table.REPLAY_CONSISTENCY:
                return pandas.DataFrame({Column.MAXIMUM: [analysis.replay_consistency]})
            case Table.TRANSFER_SUMMARY:
                return analysis.transfer_summary
            case Table.TRANSFER_AGGREGATE:
                return analysis.transfer_aggregate
            case Table.COMPENSATION:
                return analysis.compensation
            case Table.TRANSFER_TRACES:
                traces = analysis.transfer_traces(list(TRACE_PERTURBATIONS))
                return traces[traces[Column.SEED] == TRACE_SEED]


@dataclass
class Analysis:
    """
    Everything the analysis derives from one directory of trial records, one source cup
    at a time.
    """

    records: List[TrialRecord]
    """
    Every finished trial.
    """

    @classmethod
    def from_directory(cls, directory: Path) -> Analysis:
        """
        :param directory: A directory holding one directory per trial.
        :return: The analysis of every finished trial in it.
        """
        return cls(records=TrialRecord.read_all(directory))

    @cached_property
    def sources(self) -> Dict[SourceCup, SourceAnalysis]:
        """
        :return: The analysis of each source cup that has any trial, in enum order.
        """
        present = {record.description[TrialKey.SOURCE] for record in self.records}
        return {
            source: SourceAnalysis(records=self.records, source=source)
            for source in SourceCup
            if source in present
        }

    @property
    def fitted(self) -> Dict[SourceCup, SourceAnalysis]:
        """
        :return: The analyses of the source cups whose open-loop trials were recorded.
        """
        return {
            source: analysis
            for source, analysis in self.sources.items()
            if analysis.rig
        }

    @property
    def poured(self) -> Dict[SourceCup, SourceAnalysis]:
        """
        :return: The analyses of the source cups whose closed-loop trials were recorded.
        """
        return {
            source: analysis
            for source, analysis in self.sources.items()
            if analysis.transfer
        }

    def baselines(self) -> Baselines:
        """
        :return: Each fitted source cup's repose angle and rate constant, fitted to its
            own open-loop trials, as the baseline of its closed-loop perturbations.
        """
        return Baselines(
            parameters={
                source.value: analysis.fitted_parameters
                for source, analysis in self.fitted.items()
            }
        )

    def table(self, table: Table) -> pandas.DataFrame:
        """
        :param table: Which table.
        :return: That table for every source cup it has data for, with a source column.
        """
        analyses = self.fitted if table.is_open_loop else self.poured
        return pandas.concat(
            [
                table.of(analysis).assign(**{Column.SOURCE: source.value})
                for source, analysis in analyses.items()
            ],
            ignore_index=True,
        )
