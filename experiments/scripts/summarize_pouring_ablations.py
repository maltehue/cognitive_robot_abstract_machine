"""
Produce recording-balanced tables and figures from saved paired ablations.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
import json
from pathlib import Path

import matplotlib.pyplot as pyplot
import numpy

from experiments.real_pouring.forecast_evaluation import RecordingForecasts
from experiments.real_pouring.pouring_ablation import ForecastModel
from krrood.adapters.json_serializer import from_json

# %% report artifacts


class SummaryFile(StrEnum):
    """
    Saved ablation results and generated summaries.
    """

    DIRECTORY = "directory"
    """
    Root of the ablation artifacts.
    """

    FORECASTS = "forecasts"
    """
    Per-recording frozen predictions.
    """

    COMMANDS = "commands"
    """
    Paired full-arm optimizer results.
    """

    RESULTS = "results.txt"
    """
    Recording-balanced numerical results.
    """

    FIGURE = "ablations.png"
    """
    Forecast errors and paired command sensitivity.
    """

    PATTERN = "*.json"
    """
    Serialized recording results.
    """


def main() -> None:
    """
    Report mean per-recording errors without treating windows as independent trials.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(SummaryFile.DIRECTORY, type=Path)
    directory = parser.parse_args().directory
    recordings: list[RecordingForecasts] = [
        from_json(json.loads(path.read_text()))
        for path in sorted(
            (directory / SummaryFile.FORECASTS).glob(SummaryFile.PATTERN)
        )
    ]
    lines = [
        "Frozen forecast errors: receiver percentage points; mean of recording MAEs.",
        "Windows are paired; incomplete or perception-mode-changing intervals are excluded.",
    ]
    figure, axes = pyplot.subplots(1, 2, figsize=(12, 4.8))
    for validation in (False, True):
        lines.append(f"\nCalibration-excluded recordings: {validation}")
        for delay in (0.0, 0.33):
            for flowing in (False, True):
                for horizon in (0.5, 1.5, 3.0):
                    lines.append(
                        f"delay={delay}, stream-only={flowing}, horizon={horizon}"
                    )
                    for model in (
                        ForecastModel.ADAPTED,
                        ForecastModel.INITIAL,
                        ForecastModel.RECENT_RATE,
                        ForecastModel.NO_FLOW,
                        ForecastModel.FASTER,
                    ):
                        errors = [
                            [
                                row.predicted - row.observed
                                for row in recording.errors
                                if row.model is model
                                and row.delay == delay
                                and row.horizon == horizon
                                and (not flowing or row.flowing)
                            ]
                            for recording in recordings
                            if recording.validation == validation
                        ]
                        errors = [values for values in errors if values]
                        if not errors:
                            continue
                        mean_error = float(
                            numpy.mean(
                                [numpy.mean(numpy.abs(values)) for values in errors]
                            )
                        )
                        bias = float(
                            numpy.mean([numpy.mean(values) for values in errors])
                        )
                        lines.append(
                            f"  {model}: MAE={100 * mean_error:.3f}, bias={100 * bias:.3f}, recordings={len(errors)}, windows={sum(map(len, errors))}"
                        )
    lines.append("\nAll recordings: stream origins, delay=0.33, horizon=1.5:")
    for calibrated_only in (False, True):
        for model in (
            ForecastModel.ADAPTED,
            ForecastModel.INITIAL,
            ForecastModel.RECENT_RATE,
            ForecastModel.NO_FLOW,
        ):
            groups = [
                [
                    row.predicted - row.observed
                    for row in recording.errors
                    if row.model is model
                    and row.delay == 0.33
                    and row.horizon == 1.5
                    and row.flowing
                    and (not calibrated_only or row.calibrated)
                ]
                for recording in recordings
            ]
            groups = [group for group in groups if group]
            lines.append(
                f"calibrated-only={calibrated_only}, {model}: MAE={100 * numpy.mean([numpy.mean(numpy.abs(group)) for group in groups]):.3f}; recordings={len(groups)}, windows={sum(map(len, groups))}"
            )
    lines.append("\nPer recording, stream origins, delay=0.33, horizon=1.5:")
    for recording in recordings:
        cells = []
        for model in (
            ForecastModel.ADAPTED,
            ForecastModel.INITIAL,
            ForecastModel.RECENT_RATE,
        ):
            rows = [
                row
                for row in recording.errors
                if row.model is model
                and row.delay == 0.33
                and row.horizon == 1.5
                and row.flowing
            ]
            if rows:
                cells.append(
                    f"{model}={100 * numpy.mean([abs(row.predicted - row.observed) for row in rows]):.3f} ({len(rows)})"
                )
        lines.append(f"{recording.recording}: " + ", ".join(cells))
    lines.append(
        "\nIndependent final weighing, rollouts from 25%, 50%, 75% of recording:"
    )
    for validation in (False, True):
        for model in (ForecastModel.ADAPTED, ForecastModel.INITIAL):
            errors = [
                [
                    row.predicted - row.weighed
                    for row in recording.endpoints
                    if row.model is model
                ]
                for recording in recordings
                if recording.validation == validation
            ]
            lines.append(
                f"validation={validation}, {model}: MAE={100 * numpy.mean([numpy.mean(numpy.abs(values)) for values in errors]):.3f}"
            )
    for model in (
        ForecastModel.ADAPTED,
        ForecastModel.INITIAL,
        ForecastModel.RECENT_RATE,
        ForecastModel.NO_FLOW,
    ):
        values = []
        for horizon in (0.5, 1.5, 3.0):
            groups = [
                [
                    abs(row.predicted - row.observed)
                    for row in recording.errors
                    if row.model is model
                    and row.delay == 0.33
                    and row.horizon == horizon
                    and row.flowing
                ]
                for recording in recordings
            ]
            values.append(
                100 * numpy.mean([numpy.mean(group) for group in groups if group])
            )
        axes[0].plot((0.5, 1.5, 3.0), values, marker="o", label=model.value)
    axes[0].set(
        xlabel="Forecast horizon (s)",
        ylabel="Increment MAE (receiver percentage points)",
        title="Frozen predictions: stream origins, recording-balanced",
    )
    axes[0].legend()
    lines.append(
        "\nFull-arm commands; differences relative to adapted, degrees per second:"
    )
    for path in sorted((directory / SummaryFile.COMMANDS).glob(SummaryFile.PATTERN)):
        rows = from_json(json.loads(path.read_text()))
        adapted = {row.time: row for row in rows if row.model is ForecastModel.ADAPTED}
        lines.append(
            f"{path.stem}: {len(adapted)} states; max pose error={1000 * max(row.pose_error for row in rows):.3f} mm; max tilt error={numpy.degrees(max(row.angle_error for row in rows)):.3f} deg; max joint age={max(row.joint_age for row in rows):.4f} s"
        )
        for model in (
            ForecastModel.INITIAL,
            ForecastModel.FASTER,
            ForecastModel.PACING_ONLY,
        ):
            alternatives = [row for row in rows if row.model is model]
            if not alternatives:
                continue
            immediate = numpy.degrees(
                [
                    row.tilt_velocity - adapted[row.time].tilt_velocity
                    for row in alternatives
                ]
            )
            planned = numpy.degrees(
                [
                    numpy.mean(row.planned_tilt_velocities)
                    - numpy.mean(adapted[row.time].planned_tilt_velocities)
                    for row in alternatives
                ]
            )
            lines.append(
                f"  {model}: next-command mean/max |delta|={numpy.mean(abs(immediate)):.4f}/{max(abs(immediate)):.4f}; mean-plan mean/max |delta|={numpy.mean(abs(planned)):.3f}/{max(abs(planned)):.3f}; plan |delta|>0.1={numpy.mean(abs(planned) > 0.1):.1%}"
            )
            if model is ForecastModel.INITIAL:
                axes[1].plot(
                    [row.time for row in alternatives], planned, label=path.stem[-8:]
                )
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set(
        xlabel="Time from transfer start (s)",
        ylabel="Initial − adapted mean planned tilt speed (°/s)",
        title="Production full-arm optimizer, identical states",
    )
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(directory / SummaryFile.FIGURE, dpi=160)
    report = "\n".join(lines) + "\n"
    (directory / SummaryFile.RESULTS).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
