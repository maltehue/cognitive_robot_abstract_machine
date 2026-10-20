"""
Run frozen forecasts on complete saved bag replays.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
import json
from pathlib import Path

from experiments.real_pouring.forecast_evaluation import (
    ForecastEvaluation,
    RecordingForecasts,
)
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup
from experiments.real_pouring.pour_replay import GRAINS_PER_FILL
from experiments.real_pouring.wall_fill_validation import WallFillDataset
from giskardpy.middleware.ros2.particle_weighing import WeighedParticles
from krrood.adapters.json_serializer import from_json, to_json

# %% artifact interface


class ForecastFile(StrEnum):
    """
    Inputs and outputs of the reproducible forecast comparison.
    """

    DATASET = "dataset.json"
    """
    Declared development and validation recording identities.
    """

    RECORDINGS = "recordings"
    """
    Directory containing the dataset and weighing fixtures.
    """

    REPLAYS = "replays"
    """
    Complete receiver-model replay directory.
    """

    OUTPUT = "output"
    """
    Directory for scored forecast samples.
    """

    SUFFIX = ".json"
    """
    Serialized artifact extension.
    """

    REPLAY_SUFFIX = "_receiver.json"
    """
    Receiver-evidence replay filename suffix.
    """


def main() -> None:
    """
    Evaluate every declared recording, leaving source traces untouched.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in (
        ForecastFile.RECORDINGS,
        ForecastFile.REPLAYS,
        ForecastFile.OUTPUT,
    ):
        parser.add_argument(argument, type=Path)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    dataset: WallFillDataset = from_json(
        json.loads((arguments.recordings / ForecastFile.DATASET).read_text())
    )
    conversion = GRAINS_PER_FILL * ReceivingCup.TRANSLUCENT_CUP.source_cups_held(
        PouredMaterial.CORN
    )
    for name in (*dataset.development, *dataset.validation):
        steps = from_json(
            json.loads(
                (arguments.replays / (name + ForecastFile.REPLAY_SUFFIX)).read_text()
            )
        )
        weighing = WeighedParticles.from_json(
            json.loads(
                (arguments.recordings / (name + ForecastFile.SUFFIX)).read_text()
            )
        )
        evaluation = ForecastEvaluation(steps, conversion)
        result = RecordingForecasts(
            name,
            name in dataset.validation,
            evaluation.evaluate(),
            evaluation.endpoints(weighing.received_fill_level),
        )
        (arguments.output / (name + ForecastFile.SUFFIX)).write_text(
            json.dumps(to_json(result), indent=2, allow_nan=False)
        )
        print(f"{name}: {len(result.errors)} paired forecast samples", flush=True)


if __name__ == "__main__":
    main()
