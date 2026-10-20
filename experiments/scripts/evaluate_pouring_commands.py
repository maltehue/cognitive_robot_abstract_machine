"""
Solve the current pouring controller on recorded arm states without hardware.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
import json
from pathlib import Path

from experiments.real_pouring.command_evaluation import ArmSample, CommandEvaluation
from experiments.real_pouring.controller_ablation import ControllerProbe
from experiments.real_pouring.pour_replay import PourRecording
from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.tracy import Tracy

# %% command-line entry point


class CommandArgument(StrEnum):
    """
    Files and configuration required for counterfactual command solves.
    """

    BAG = "bag"
    """
    Extracted recording directory.
    """

    REPLAY = "replay"
    """
    Complete causal receiver-model replay.
    """

    OUTPUT = "output"
    """
    Destination for every paired solve.
    """

    GOAL = "--goal"
    """
    Known receiver goal for this recording.
    """

    PREDICTION_DURATION = "--prediction-duration"
    """
    Optional fill forecast window in seconds.
    """


def main() -> None:
    """
    Reconstruct a disconnected robot and evaluate paired commands.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in (
        CommandArgument.BAG,
        CommandArgument.REPLAY,
        CommandArgument.OUTPUT,
    ):
        parser.add_argument(argument, type=Path)
    parser.add_argument(CommandArgument.GOAL, type=float, required=True)
    parser.add_argument(CommandArgument.PREDICTION_DURATION, type=float)
    arguments = parser.parse_args()
    world = URDFParser.from_file(Tracy.get_ros_file_path()).parse()
    Tracy.from_world(world)
    probe = ControllerProbe.build(
        world, arguments.goal, prediction_duration=arguments.prediction_duration
    )
    recording = PourRecording.read(arguments.bag)
    steps = from_json(json.loads(arguments.replay.read_text()))
    evaluation = CommandEvaluation(probe, recording, ArmSample.read(arguments.bag))
    results = evaluation.evaluate(steps)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(to_json(results), indent=2, allow_nan=False))
    print(
        f"{len(results)} full-arm solves; source dimensions "
        f"{probe.source.fill_equation.container_width}, {probe.source.fill_equation.container_height}",
        flush=True,
    )


if __name__ == "__main__":
    main()
