"""
Where the bulk of a pour starts and where it stops on the way back, per weighed pour,
and how far below the starting repose angle the contents kept flowing in the model's lip
geometry: the repose hysteresis in `poured_material.py`.

The bags are replayed through the production detector. The bulk starts where the count
first passes START_COUNT grains; it stops at the last reading after the highest tilt at
which the count still rose by more than STILL_RISING grains. The cup holds what the
scale says did not leave it.

Usage: python stop_angles.py <material>::<bag dir or zip>::<grams out>::<full cup grams> ...
Writes stop_angles.json next to this script.
"""

import json
import math
import sys
from pathlib import Path

from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    ModelPoint,
)
from experiments.real_pouring.pour_replay import (
    GRAINS_PER_FILL,
    OPENING_HEIGHT,
    PourRecording,
    PourReplay,
)
from experiments.real_pouring.poured_material import PouredMaterial

START_COUNT = 1.0
STILL_RISING = 0.1
LIP_OFFSET = 0.035
DRAIN = ModelPoint(
    repose_angle=0.0, discharge_coefficient=0.2, outflow_rate_constant=0.8
)


def lip_angle(left_share: float) -> float:
    """
    The model's lip angle, in degrees, for a cup holding this share.
    """
    return math.degrees(math.atan2(OPENING_HEIGHT * (1.0 - left_share), LIP_OFFSET))


results = {}
for argument in sys.argv[1:]:
    material, bag, out, full = argument.split("::")
    material, out, full = PouredMaterial(material), float(out), float(full)
    steps = PourReplay.of(
        PourRecording.read(Path(bag)),
        material,
        DRAIN,
        BankSettings(tracks_onset=True, trickle=0.0),
    ).steps
    counts = [step.reading.counted_grains for step in steps]
    tilts = [math.degrees(step.tilt) for step in steps]
    start = next(index for index, count in enumerate(counts) if count > START_COUNT)
    peak = max(range(len(tilts)), key=tilts.__getitem__)
    stop = max(
        index
        for index in range(peak, len(steps) - 1)
        if counts[index + 1] - counts[index] > STILL_RISING
    )
    start_left = 1.0 - counts[start] / GRAINS_PER_FILL
    stop_left = 1.0 - out / full
    starting_repose = tilts[start] - lip_angle(start_left)
    stopping_repose = tilts[stop + 1] - lip_angle(stop_left)
    results[Path(bag).name] = dict(
        material=material.value,
        start_tilt=tilts[start],
        stop_tilt=tilts[stop + 1],
        left_share=stop_left,
        hysteresis=starting_repose - stopping_repose,
    )
    print(Path(bag).name, json.dumps(results[Path(bag).name]))
Path(__file__).with_name("stop_angles.json").write_text(json.dumps(results, indent=1))
