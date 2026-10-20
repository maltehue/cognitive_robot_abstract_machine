"""
Compare stopping influence at two fill forecast windows on paired robot states.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
import json
from pathlib import Path

import matplotlib.pyplot as pyplot
import numpy

from experiments.real_pouring.pouring_ablation import ForecastModel
from experiments.real_pouring.stopping_ablation import (
    StoppingAnalysis,
    UnpairedCommandsError,
)
from krrood.adapters.json_serializer import from_json

# %% paired horizon reports


class HorizonFile(StrEnum):
    """
    Artifact paths for a paired forecast-window experiment.
    """

    DIRECTORY = "directory"
    """
    Root containing both sets of saved solves.
    """

    SHORT = "1_5_seconds"
    """
    The default forecast-window solves.
    """

    LONG = "3_seconds"
    """
    The extended forecast-window solves.
    """

    PATTERN = "*.json"
    """
    One serialized result per recording.
    """

    RESULTS = "results.txt"
    """
    Numerical stopping and model-influence comparisons.
    """

    FIGURE = "stopping_comparison.png"
    """
    Paired planned velocities and parameter influence over time.
    """


def main() -> None:
    """
    Write metrics and a figure after verifying exact snapshot pairing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(HorizonFile.DIRECTORY, type=Path)
    directory = parser.parse_args().directory
    paths = sorted((directory / HorizonFile.SHORT).glob(HorizonFile.PATTERN))
    figure, axes = pyplot.subplots(2, len(paths), figsize=(15, 8), squeeze=False)
    lines = [
        "Paired forecast windows: only fill prediction duration changes.",
        "Speeds are degrees/second; plan speed is averaged over the 0.35-second control horizon.",
        "Reversal: mean planned speed < -0.1 deg/s. Sustained: two consecutive snapshots.",
        "Suppressed forward: pacing-only > +0.1 deg/s, model plan <= +0.1 deg/s.",
    ]
    for column, path in enumerate(paths):
        short = StoppingAnalysis(from_json(json.loads(path.read_text())))
        long = StoppingAnalysis(
            from_json(
                json.loads((directory / HorizonFile.LONG / path.name).read_text())
            )
        )
        lines.append(f"\n{path.stem}")
        for model in ForecastModel:
            first, second = short.select(model), long.select(model)
            if first.keys() != second.keys():
                raise UnpairedCommandsError()
            for time in first:
                short.check_pair(first[time], second[time])
        for analysis in (short, long):
            duration = analysis.rows[0].prediction_duration
            lines.append(f"forecast={duration:g} s")
            adapted = analysis.select(ForecastModel.ADAPTED)
            initial = analysis.select(ForecastModel.INITIAL)
            times = sorted(adapted)
            delta = numpy.degrees(
                [
                    analysis.mean_speed(initial[time])
                    - analysis.mean_speed(adapted[time])
                    for time in times
                ]
            )
            commands = numpy.degrees(
                [
                    initial[time].tilt_velocity - adapted[time].tilt_velocity
                    for time in times
                ]
            )
            lines.append(
                f"  initial vs adapted: mean |planned delta|={numpy.mean(abs(delta)):.3f}, max={max(abs(delta)):.3f}, |delta|>0.1={numpy.mean(abs(delta) > 0.1):.1%}; mean |next-command delta|={numpy.mean(abs(commands)):.4f}"
            )
            for model in (
                ForecastModel.ADAPTED,
                ForecastModel.INITIAL,
                ForecastModel.FASTER,
                ForecastModel.PACING_ONLY,
            ):
                summary = analysis.summarize(model)
                first = summary.first_sustained_reversal
                first_text = "none" if first is None else f"{first:.3f} s"
                lines.append(
                    f"  {model}: suppress-forward={summary.suppressed_forward}/{summary.states}; reverse={summary.reversals}/{summary.states}; first sustained reversal={first_text}; mean |plan - pacing|={numpy.degrees(summary.mean_plan_difference):.3f}; mean |command - pacing|={numpy.degrees(summary.mean_command_difference):.4f}"
                )
            speeds = numpy.degrees(
                [analysis.mean_speed(adapted[time]) for time in times]
            )
            axes[0, column].plot(times, speeds, label=f"Adapted, {duration:g} s")
            axes[1, column].plot(times, delta, label=f"{duration:g} s forecast")
        short_adapted, long_adapted = short.select(ForecastModel.ADAPTED), long.select(
            ForecastModel.ADAPTED
        )
        times = sorted(short_adapted)
        difference = numpy.degrees(
            [
                long.mean_speed(long_adapted[time])
                - short.mean_speed(short_adapted[time])
                for time in times
            ]
        )
        first_pacing, second_pacing = short.select(
            ForecastModel.PACING_ONLY
        ), long.select(ForecastModel.PACING_ONLY)
        pacing_difference = max(
            abs(
                short.mean_speed(first_pacing[time])
                - long.mean_speed(second_pacing[time])
            )
            for time in times
        )
        new_returns = sum(
            long.mean_speed(long_adapted[time]) < -long.threshold
            and short.mean_speed(short_adapted[time]) >= -short.threshold
            for time in times
        )
        lost_returns = sum(
            short.mean_speed(short_adapted[time]) < -short.threshold
            and long.mean_speed(long_adapted[time]) >= -long.threshold
            for time in times
        )
        lines.append(
            f"  3 vs 1.5 s adapted: mean |planned delta|={numpy.mean(abs(difference)):.3f}, signed mean={numpy.mean(difference):.3f}; newly reversing states={new_returns}, no-longer reversing={lost_returns}; max pacing-only difference={numpy.degrees(pacing_difference):.9f}"
        )
        axes[0, column].plot(
            times,
            numpy.degrees([short.mean_speed(first_pacing[time]) for time in times]),
            color="gray",
            linestyle="--",
            label="Fill objective disabled",
        )
        axes[0, column].set_title(path.stem[-8:])
        axes[0, column].set_ylabel("Mean planned tilt speed (°/s)")
        axes[1, column].set_ylabel("Initial − adapted planned speed (°/s)")
        for axis in axes[:, column]:
            axis.axhline(0, color="black", linewidth=0.6)
            axis.set_xlabel("Time from transfer start (s)")
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / HorizonFile.FIGURE, dpi=160)
    report = "\n".join(lines) + "\n"
    (directory / HorizonFile.RESULTS).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
