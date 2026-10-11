"""
Drawing an analysis as figures, and writing the tables they are drawn from.

The colours are the validated default categorical palette of the data visualisation
guidelines, taken in its fixed order: three slots at most per chart, which is where the
order stays distinguishable for every kind of colour vision.
"""

# %% imports

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as pyplot
import numpy
import pandas
from typing_extensions import Dict, List, Tuple

from experiments.pouring_model_verification.analysis import (
    TRACE_PERTURBATIONS,
    TRACE_SEED,
    Analysis,
    Column,
    ParameterSet,
    SourceAnalysis,
    Table,
)
from experiments.pouring_model_verification.controller_comparison import (
    ComparisonColumn,
    ControllerArm,
    ControllerComparison,
)
from experiments.pouring_model_verification.parameters import (
    BASELINE,
    ModelParameter,
)
from experiments.pouring_model_verification.records import Outcome, Phase, Sample
from experiments.pouring_model_verification.sensor import SensorCondition
from experiments.pouring_model_verification.transfer_trial import (
    GOAL_GRAINS,
    LoopMode,
)

# %% the look

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
"""
The first four slots of the categorical palette, in order.
"""

MEASURED_COLOR = "#52514e"
"""
The ink the measured grains are drawn in: data, but not one of the compared series.
"""

MUTED_COLOR = "#898781"
"""
Axis labels and ticks.
"""

GRID_COLOR = "#e1e0d9"
"""
Gridlines.
"""

SURFACE_COLOR = "#fcfcfb"
"""
The chart surface.
"""

LOOP_MODE_COLORS: Dict[LoopMode, str] = dict(zip(LoopMode, SERIES_COLORS))
"""
Each loop mode's colour, the same in every chart.
"""

PARAMETER_SET_COLORS: Dict[ParameterSet, str] = {
    ParameterSet.DEMO: SERIES_COLORS[0],
    ParameterSet.FITTED: SERIES_COLORS[1],
    ParameterSet.NOMINAL: SERIES_COLORS[2],
}
"""
Each drawn parameter set's colour, the same in every chart.
"""

POUR_SET_COLORS: Dict[Column, str] = {
    Column.FITTING_ERROR: SERIES_COLORS[0],
    Column.HELD_OUT_ERROR: SERIES_COLORS[1],
    Column.IN_SITU_ERROR: SERIES_COLORS[2],
}
"""
Each set of pours' colour in the error charts.
"""

LABELS: Dict[str, str] = {
    ParameterSet.DEMO: "demo values",
    ParameterSet.FITTED: "fresh fit",
    ParameterSet.NOMINAL: "nominal (unestimated)",
    ParameterSet.FITTED_WITH_LIP: "fitted, lip freed",
    Column.FITTING_ERROR: "step-and-hold (fitted on)",
    Column.HELD_OUT_ERROR: "ramps (held out)",
    Column.IN_SITU_ERROR: "closed-loop pours (held out)",
    LoopMode.OPEN_LOOP: "open loop",
    LoopMode.FILL_FEEDBACK: "fill feedback",
    LoopMode.FILL_FEEDBACK_AND_CALIBRATION: "feedback + drain calibration",
    ModelParameter.REPOSE_ANGLE: "repose angle",
    ModelParameter.OUTFLOW_RATE_CONSTANT: "outflow rate constant k",
    ModelParameter.DISCHARGE_COEFFICIENT: "discharge coefficient",
    ModelParameter.LIP_OFFSET: "lip offset",
}
"""
What each series and parameter is called on a chart.
"""

CLOSED_LOOP_PARAMETERS = (
    ModelParameter.REPOSE_ANGLE,
    ModelParameter.OUTFLOW_RATE_CONSTANT,
    ModelParameter.DISCHARGE_COEFFICIENT,
)
"""
The parameters the closed-loop study perturbs.
"""


def _style(axis: pyplot.Axes) -> None:
    """
    Recessive axes: hairline grid, no box, muted ticks.
    """
    axis.set_facecolor(SURFACE_COLOR)
    axis.grid(color=GRID_COLOR, linewidth=1.0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID_COLOR)
    axis.tick_params(colors=MUTED_COLOR, labelcolor=MEASURED_COLOR)


def _figure(columns: int, rows: int = 1, width: float = 4.2, height: float = 3.2):
    figure, axes = pyplot.subplots(
        rows,
        columns,
        figsize=(width * columns, height * rows),
        squeeze=False,
        facecolor=SURFACE_COLOR,
    )
    for axis in axes.flat:
        _style(axis)
    return figure, axes


def _legend_below(figure: pyplot.Figure, axis: pyplot.Axes) -> None:
    """
    One legend for every panel, under them, so it never covers data.
    """
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.subplots_adjust(bottom=0.25)


def _save(figure: pyplot.Figure, directory: Path, name: str) -> None:
    figure.tight_layout(rect=(0, 0.1 if figure.legends else 0, 1, 1))
    figure.savefig(directory / f"{name}.png", dpi=150, facecolor=SURFACE_COLOR)
    pyplot.close(figure)


# %% the open-loop figures


def draw_trajectories(analysis: SourceAnalysis, directory: Path) -> None:
    """
    The fill over time along the example profiles: each seed's grains against what each
    parameter set predicts along the same tilt.
    """
    trajectories = analysis.example_trajectories
    profiles = list(trajectories[Column.PROFILE].unique())
    figure, axes = _figure(len(profiles))
    for axis, profile in zip(axes.flat, profiles):
        of_profile = trajectories[trajectories[Column.PROFILE] == profile]
        for seed, run in of_profile.groupby(Column.SEED):
            axis.plot(
                run[Column.TIME],
                run[Column.MEASURED],
                color=MEASURED_COLOR,
                linewidth=1.0,
                alpha=0.6,
                label="grains (one line per seed)" if seed == 0 else None,
            )
        first_seed = of_profile[
            of_profile[Column.SEED] == of_profile[Column.SEED].min()
        ]
        for name, color in PARAMETER_SET_COLORS.items():
            axis.plot(
                first_seed[Column.TIME],
                first_seed[name],
                color=color,
                linewidth=2.0,
                label=LABELS[name],
            )
        axis.set_title(profile, color=MEASURED_COLOR)
        axis.set_xlabel("time [s]")
        axis.set_ylabel("source fill level")
    _legend_below(figure, axes.flat[0])
    _save(figure, directory, f"open_loop_trajectories_{analysis.source}")


def draw_equilibrium(analysis: SourceAnalysis, directory: Path) -> None:
    """
    The fill left at the end of each held tilt, measured on every seed and predicted by
    each parameter set.
    """
    equilibrium = analysis.equilibrium
    figure, axes = _figure(1, width=5.5)
    axis = axes[0, 0]
    measured = equilibrium[equilibrium[Column.PARAMETER_SET] == ParameterSet.DEMO]
    axis.scatter(
        numpy.degrees(measured[Column.TILT]),
        measured[Column.MEASURED],
        color=MEASURED_COLOR,
        s=18,
        label="grains, end of 20 s hold",
        zorder=3,
    )
    for name, color in PARAMETER_SET_COLORS.items():
        predicted = (
            equilibrium[equilibrium[Column.PARAMETER_SET] == name]
            .groupby(Column.TILT)[Column.PREDICTED]
            .mean()
        )
        axis.plot(
            numpy.degrees(predicted.index),
            predicted.values,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=LABELS[name],
        )
    axis.set_xlabel("held tilt [deg]")
    axis.set_ylabel("fill left after the hold")
    axis.legend(frameon=False)
    _save(figure, directory, f"open_loop_equilibrium_{analysis.source}")


def draw_sensitivity(analysis: SourceAnalysis, directory: Path) -> None:
    """
    How the fill error moves when one parameter of the fitted set is scaled, on each set
    of pours.
    """
    sensitivity = analysis.sensitivity
    parameters = list(sensitivity[Column.PARAMETER].unique())
    figure, axes = _figure(len(parameters))
    for axis, parameter in zip(axes.flat, parameters):
        of_parameter = sensitivity[sensitivity[Column.PARAMETER] == parameter]
        for column, color in POUR_SET_COLORS.items():
            axis.plot(
                of_parameter[Column.FACTOR],
                of_parameter[column],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=LABELS[column],
            )
        axis.axvline(1.0, color=GRID_COLOR, linewidth=1.0)
        axis.set_xscale("log")
        axis.set_title(LABELS[parameter], color=MEASURED_COLOR)
        axis.set_xlabel("factor on the fitted value")
        axis.set_ylabel("fill RMSE")
    _legend_below(figure, axes.flat[0])
    _save(figure, directory, f"open_loop_sensitivity_{analysis.source}")


def draw_landscape(analysis: SourceAnalysis, directory: Path) -> None:
    """
    The fill error over repose angle and rate constant together, with the fitted and the
    reference sets marked.
    """
    landscape = analysis.error_landscape
    grid = landscape.pivot(
        index=ModelParameter.OUTFLOW_RATE_CONSTANT,
        columns=ModelParameter.REPOSE_ANGLE,
        values=Column.FITTING_ERROR,
    )
    figure, axes = _figure(1, width=5.5, height=4.2)
    axis = axes[0, 0]
    mesh = axis.pcolormesh(
        numpy.degrees(grid.columns),
        grid.index,
        grid.values,
        cmap="Blues_r",
        shading="nearest",
    )
    figure.colorbar(mesh, ax=axis, label="fill RMSE on step-and-hold")
    for name in (ParameterSet.DEMO, ParameterSet.FITTED):
        parameters = analysis.parameter_sets[name]
        axis.scatter(
            math.degrees(parameters.repose_angle),
            parameters.outflow_rate_constant,
            color=PARAMETER_SET_COLORS[name],
            edgecolor=SURFACE_COLOR,
            linewidth=2,
            s=70,
            label=LABELS[name],
            zorder=3,
        )
    axis.set_yscale("log")
    axis.set_xlabel("repose angle [deg]")
    axis.set_ylabel("outflow rate constant k [1/s]")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(False)
    _save(figure, directory, f"open_loop_landscape_{analysis.source}")


# %% the closed-loop figures


def draw_delivery(analysis: SourceAnalysis, receiver: str, directory: Path) -> None:
    """
    Grains delivered against the factor each parameter was set wrong by, one line per
    loop mode, the band spanning the seeds.
    """
    summary = analysis.transfer_summary
    summary = summary[summary[Column.RECEIVER] == receiver]
    goal = summary[Outcome.GOAL_GRAINS].mean()
    reference = summary[summary[Column.PERTURBATION] == BASELINE.label].assign(
        **{Column.FACTOR: 1.0}
    )
    figure, axes = _figure(len(CLOSED_LOOP_PARAMETERS))
    for axis, parameter in zip(axes.flat, CLOSED_LOOP_PARAMETERS):
        of_parameter = pandas.concat(
            [summary[summary[Column.PARAMETER] == parameter], reference]
        )
        for loop_mode, color in LOOP_MODE_COLORS.items():
            runs = of_parameter[of_parameter[Column.LOOP_MODE] == loop_mode]
            statistics = runs.groupby(Column.FACTOR)[Outcome.DELIVERED].agg(
                ["mean", "min", "max"]
            )
            axis.fill_between(
                statistics.index,
                statistics["min"],
                statistics["max"],
                color=color,
                alpha=0.1,
                linewidth=0,
            )
            axis.plot(
                statistics.index,
                statistics["mean"],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=LABELS[loop_mode],
            )
        axis.axhline(goal, color=MEASURED_COLOR, linewidth=1.0, linestyle="--")
        axis.text(axis.get_xlim()[0], goal, " goal", va="bottom", color=MEASURED_COLOR)
        axis.set_xscale("symlog", linthresh=0.25)
        axis.set_title(LABELS[parameter], color=MEASURED_COLOR)
        axis.set_xlabel("factor on the baseline value")
        axis.set_ylabel("grains delivered")
    _legend_below(figure, axes.flat[0])
    _save(figure, directory, f"closed_loop_delivery_{analysis.source}_into_{receiver}")


def draw_traces(analysis: SourceAnalysis, receiver: str, directory: Path) -> None:
    """
    The receiver's fill over time on seed zero: what the controller believed and what
    was committed, for a few perturbations under every loop mode.
    """
    traces = analysis.transfer_traces(list(TRACE_PERTURBATIONS))
    traces = traces[
        (traces[Column.SEED] == TRACE_SEED)
        & (traces[Column.RECEIVER] == receiver)
        & traces[Sample.PHASE].isin([Phase.PRE_TILT, Phase.TRANSFER, Phase.RETURN])
    ]
    figure, axes = _figure(len(TRACE_PERTURBATIONS), rows=2, width=3.8, height=2.8)
    for column, perturbation in enumerate(TRACE_PERTURBATIONS):
        of_perturbation = traces[traces[Column.PERTURBATION] == perturbation]
        for loop_mode, color in LOOP_MODE_COLORS.items():
            run = of_perturbation[of_perturbation[Column.LOOP_MODE] == loop_mode]
            axes[0, column].plot(
                run[Sample.TIME],
                run[Sample.COMMITTED_RECEIVER_FILL],
                color=color,
                linewidth=2.0,
                label=LABELS[loop_mode],
            )
            axes[0, column].plot(
                run[Sample.TIME],
                run[Sample.MODEL_RECEIVER_FILL],
                color=color,
                linewidth=1.0,
                linestyle=":",
            )
            axes[1, column].plot(
                run[Sample.TIME],
                numpy.degrees(run[Sample.TILT]),
                color=color,
                linewidth=2.0,
            )
        axes[0, column].set_title(
            f"{perturbation}\nsolid: grains, dotted: model", color=MEASURED_COLOR
        )
        axes[0, column].set_ylabel("receiver fill")
        axes[1, column].set_ylabel("cup tilt [deg]")
        axes[1, column].set_xlabel("time [s]")
    _legend_below(figure, axes[0, 0])
    _save(figure, directory, f"closed_loop_traces_{analysis.source}_into_{receiver}")


# %% the tables


REPORT_FILE = "report.json"
"""
The file holding every table at once, in the form a report page reads.
"""

TRACE_SAMPLE_EVERY = 2
"""
Which share of a closed-loop trace's samples the report keeps: every second one.
"""


def write_tables(analysis: Analysis, directory: Path) -> None:
    """
    Write every table the figures are drawn from as a CSV file each, and all of them
    together as one JSON report.
    """
    tables: Dict[Table, pandas.DataFrame] = {}
    for table in Table:
        if analysis.fitted if table.is_open_loop else analysis.poured:
            tables[table] = analysis.table(table)
    for table, frame in tables.items():
        frame.to_csv(directory / f"{table}.csv", index=False)
    if Table.TRANSFER_TRACES in tables:
        traces = tables[Table.TRANSFER_TRACES].reset_index(drop=True)
        tables[Table.TRANSFER_TRACES] = traces[traces.index % TRACE_SAMPLE_EVERY == 0]
    (directory / REPORT_FILE).write_text(
        json.dumps(
            {
                table.value: json.loads(frame.to_json(orient="records"))
                for table, frame in tables.items()
            }
        )
    )


def draw_all(analysis: Analysis, directory: Path) -> List[Path]:
    """
    Draw every figure and write every table of an analysis into a directory.

    :param analysis: The analysis to draw.
    :param directory: Where the figures and tables are written, created if missing.
    :return: The figures written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    write_tables(analysis, directory)
    for source_analysis in analysis.fitted.values():
        draw_trajectories(source_analysis, directory)
        draw_equilibrium(source_analysis, directory)
        draw_sensitivity(source_analysis, directory)
        draw_landscape(source_analysis, directory)
    for source_analysis in analysis.poured.values():
        summary = source_analysis.transfer_summary
        for receiver in summary[Column.RECEIVER].unique():
            draw_delivery(source_analysis, receiver, directory)
            draw_traces(source_analysis, receiver, directory)
    return sorted(directory.glob("*.png"))


# %% the controller comparison

COMPARED_ARMS = (
    ControllerArm.MODEL_FILL_FEEDBACK,
    ControllerArm.MODEL_CALIBRATION,
    ControllerArm.PID,
)
"""
The controllers every sensor is compared on, one colour each.
"""

COMPENSATION_ARMS = (
    ControllerArm.MODEL_FILL_FEEDBACK,
    ControllerArm.PID,
    ControllerArm.PID_STOP_EARLY,
    ControllerArm.PID_SMITH,
)
"""
The controllers the delay compensation study compares, one colour each.
"""

ARM_COLORS: Dict[ControllerArm, str] = {
    **dict(zip(COMPARED_ARMS, SERIES_COLORS)),
    ControllerArm.PID_STOP_EARLY: SERIES_COLORS[1],
    ControllerArm.PID_SMITH: SERIES_COLORS[3],
}
"""
Each compared controller's colour within one chart; the stop-early PID takes the slot
calibration has elsewhere, as the two never share a chart.
"""

PID_VARIANT_ARMS = (
    ControllerArm.PID_SOFTER,
    ControllerArm.PID,
    ControllerArm.PID_STIFFER,
    ControllerArm.PID_FIXED_POSE,
)
"""
The PID baseline's variants, compared with an ideal sensor.
"""

ARM_LABELS: Dict[ControllerArm, str] = {
    ControllerArm.MODEL_FILL_FEEDBACK: "model, fill feedback",
    ControllerArm.MODEL_CALIBRATION: "model, + calibration",
    ControllerArm.PID: "PID (tuned)",
    ControllerArm.PID_SOFTER: "PID gains x0.5",
    ControllerArm.PID_STIFFER: "PID gains x2",
    ControllerArm.PID_FIXED_POSE: "PID, cup held still",
    ControllerArm.PID_STOP_EARLY: "PID, stop early",
    ControllerArm.PID_SMITH: "PID, Smith predictor",
    ControllerArm.PID_STOP_EARLY_FIXED_POSE: "PID, stop early, held still",
    ControllerArm.PID_SMITH_FIXED_POSE: "PID, Smith, held still",
}
"""
What each controller is called on a chart.
"""

TRACED_SENSORS = (SensorCondition.IDEAL, SensorCondition.SCALE)
"""
The sensors whose pours the trace figure shows.
"""

COMPARISON_REPORT_FILE = "controller_comparison.json"
"""
The file holding the comparison's tables at once, in the form a report page reads.
"""

COMPARISON_TRACE_EVERY = 3
"""
Which share of a comparison trace's samples the report keeps: every third one.
"""


def draw_sensor_robustness(
    comparison: ControllerComparison,
    source: str,
    receiver: str,
    directory: Path,
    arms: Tuple[ControllerArm, ...] = COMPARED_ARMS,
    name: str = "sensor_robustness",
) -> None:
    """
    Grains misplaced under each sensor, one series per controller, every seed a dot and
    the mean a bar.
    """
    summary = comparison.summary
    scene = summary[
        (summary[ComparisonColumn.SOURCE] == source)
        & (summary[ComparisonColumn.RECEIVER] == receiver)
    ]
    figure, axes = _figure(1, width=8.4, height=3.4)
    axis = axes[0, 0]
    sensors = list(SensorCondition)
    width = 0.8 / len(arms)
    for index, arm in enumerate(arms):
        of_arm = scene[scene[ComparisonColumn.ARM] == arm]
        offsets = [
            position - 0.4 + width * (index + 0.5) for position in range(len(sensors))
        ]
        means = [
            of_arm[of_arm[ComparisonColumn.SENSOR] == sensor][
                ComparisonColumn.MISPLACED
            ].mean()
            for sensor in sensors
        ]
        axis.bar(offsets, means, width=width * 0.9, color=ARM_COLORS[arm], alpha=0.35)
        for offset, sensor in zip(offsets, sensors):
            runs = of_arm[of_arm[ComparisonColumn.SENSOR] == sensor]
            axis.plot(
                [offset] * len(runs),
                runs[ComparisonColumn.MISPLACED],
                linestyle="none",
                marker="o",
                markersize=4,
                color=ARM_COLORS[arm],
            )
        axis.plot([], [], color=ARM_COLORS[arm], linewidth=6, label=ARM_LABELS[arm])
    axis.set_xticks(range(len(sensors)))
    axis.set_xticklabels([sensor.replace("_", " ") for sensor in sensors])
    axis.set_ylabel("grains misplaced")
    axis.set_title(f"{source} into {receiver}", color=MEASURED_COLOR)
    _legend_below(figure, axis)
    _save(figure, directory, f"{name}_{source}_into_{receiver}")


def draw_pid_variants(comparison: ControllerComparison, directory: Path) -> None:
    """
    Grains delivered by each variant of the PID baseline with an ideal sensor, next to
    the model with fill feedback, one panel per scene.
    """
    summary = comparison.summary
    ideal = summary[summary[ComparisonColumn.SENSOR] == SensorCondition.IDEAL]
    scenes = ideal[
        [ComparisonColumn.SOURCE, ComparisonColumn.RECEIVER]
    ].drop_duplicates()
    arms = (ControllerArm.MODEL_FILL_FEEDBACK,) + PID_VARIANT_ARMS
    figure, axes = _figure(len(scenes), width=3.6, height=3.4)
    for axis, (_, scene) in zip(axes.flat, scenes.iterrows()):
        of_scene = ideal[
            (ideal[ComparisonColumn.SOURCE] == scene[ComparisonColumn.SOURCE])
            & (ideal[ComparisonColumn.RECEIVER] == scene[ComparisonColumn.RECEIVER])
        ]
        for position, arm in enumerate(arms):
            runs = of_scene[of_scene[ComparisonColumn.ARM] == arm]
            axis.plot(
                [position] * len(runs),
                runs[ComparisonColumn.DELIVERED],
                linestyle="none",
                marker="o",
                markersize=4,
                color=MEASURED_COLOR,
            )
        axis.axhline(GOAL_GRAINS, color=MEASURED_COLOR, linewidth=1.0, linestyle="--")
        axis.set_xticks(range(len(arms)))
        axis.set_xticklabels([ARM_LABELS[arm] for arm in arms], rotation=60, ha="right")
        axis.set_title(
            f"{scene[ComparisonColumn.SOURCE]} into {scene[ComparisonColumn.RECEIVER]}",
            color=MEASURED_COLOR,
        )
        axis.set_ylabel("grains delivered")
    _save(figure, directory, "pid_variants")


def draw_comparison_traces(
    comparison: ControllerComparison, source: str, receiver: str, directory: Path
) -> None:
    """
    Tilt and grains in the receiver over time on seed zero, for the model with fill
    feedback and the PID baseline, with an ideal sensor and with a scale.
    """
    traces = comparison.traces
    traces = traces[
        (traces[ComparisonColumn.SOURCE] == source)
        & (traces[ComparisonColumn.RECEIVER] == receiver)
        & (traces[ComparisonColumn.SEED] == TRACE_SEED)
    ]
    arms = (ControllerArm.MODEL_FILL_FEEDBACK, ControllerArm.PID)
    figure, axes = _figure(len(TRACED_SENSORS), rows=2, width=4.2, height=2.8)
    for column, sensor in enumerate(TRACED_SENSORS):
        for arm in arms:
            run = traces[
                (traces[ComparisonColumn.ARM] == arm)
                & (traces[ComparisonColumn.SENSOR] == sensor)
            ]
            axes[0, column].plot(
                run[ComparisonColumn.TIME],
                run[ComparisonColumn.GRAINS_IN_RECEIVER],
                color=ARM_COLORS[arm],
                linewidth=2.0,
                label=ARM_LABELS[arm],
            )
            axes[1, column].plot(
                run[ComparisonColumn.TIME],
                numpy.degrees(run[ComparisonColumn.TILT]),
                color=ARM_COLORS[arm],
                linewidth=2.0,
            )
        axes[0, column].axhline(
            GOAL_GRAINS, color=MEASURED_COLOR, linewidth=1.0, linestyle="--"
        )
        axes[0, column].set_title(
            f"sensor: {sensor.replace('_', ' ')}", color=MEASURED_COLOR
        )
        axes[0, column].set_ylabel("grains in receiver")
        axes[1, column].set_ylabel("cup tilt [deg]")
        axes[1, column].set_xlabel("time [s]")
    _legend_below(figure, axes[0, 0])
    _save(figure, directory, f"comparison_traces_{source}_into_{receiver}")


def write_comparison_tables(comparison: ControllerComparison, directory: Path) -> None:
    """
    Write the comparison's tables as a CSV file each, and together, with the tuning and
    the tuned gains, as one JSON report.
    """
    tables = {
        "comparison_summary": comparison.summary,
        "comparison_aggregate": comparison.aggregate,
    }
    for name, frame in tables.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    traces = comparison.traces.reset_index(drop=True)
    tables["comparison_traces"] = traces[traces.index % COMPARISON_TRACE_EVERY == 0]
    report = {
        name: json.loads(frame.to_json(orient="records"))
        for name, frame in tables.items()
    }
    report["tuned_gains"] = {
        source: gains.to_json() for source, gains in comparison.tuned.gains.items()
    }
    (directory / COMPARISON_REPORT_FILE).write_text(json.dumps(report))


def write_tuning_table(tuning: pandas.DataFrame, directory: Path) -> None:
    """
    Write the PID tuning, one row per cup and gains, as a CSV file and a JSON report.
    """
    tuning.to_csv(directory / "pid_tuning.csv", index=False)
    (directory / "pid_tuning.json").write_text(tuning.to_json(orient="records"))


def draw_controller_comparison(
    comparison: ControllerComparison, tuning: pandas.DataFrame, directory: Path
) -> List[Path]:
    """
    Draw every figure and write every table of the controller comparison.

    :param comparison: The comparison to draw.
    :param tuning: The PID baseline's tuning table.
    :param directory: Where the figures and tables are written, created if missing.
    :return: The figures written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    write_comparison_tables(comparison, directory)
    write_tuning_table(tuning, directory)
    scenes = comparison.summary[
        [ComparisonColumn.SOURCE, ComparisonColumn.RECEIVER]
    ].drop_duplicates()
    for _, scene in scenes.iterrows():
        source, receiver = (
            scene[ComparisonColumn.SOURCE],
            scene[ComparisonColumn.RECEIVER],
        )
        draw_sensor_robustness(comparison, source, receiver, directory)
        if comparison.summary[ComparisonColumn.ARM].isin(COMPENSATION_ARMS[2:]).any():
            draw_sensor_robustness(
                comparison,
                source,
                receiver,
                directory,
                arms=COMPENSATION_ARMS,
                name="delay_compensation",
            )
        draw_comparison_traces(comparison, source, receiver, directory)
    draw_pid_variants(comparison, directory)
    return sorted(directory.glob("*.png"))


# %% the fair comparison and the onset study


def write_frames(
    frames: Dict[str, pandas.DataFrame], name: str, directory: Path
) -> None:
    """
    Write tables as a CSV file each and together as one JSON report.

    :param frames: The tables, by name.
    :param name: The JSON report's name, without its suffix.
    :param directory: Where they are written.
    """
    for table, frame in frames.items():
        frame.to_csv(directory / f"{table}.csv", index=False)
    (directory / f"{name}.json").write_text(
        json.dumps(
            {
                table: json.loads(frame.to_json(orient="records"))
                for table, frame in frames.items()
            }
        )
    )
