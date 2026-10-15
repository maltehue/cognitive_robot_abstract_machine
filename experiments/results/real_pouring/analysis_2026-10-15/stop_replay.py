"""
Replay weighed pours with the demo's live-bank settings and the opening moved onto the
rim, and report where the controller would have been told the goal was reached, against
what the scale says arrived.

Usage: python stop_replay.py <material>::<bag dir or zip>::<received grams>::<full cup grams> ...
Writes stop_replay.json next to this script.
"""

import json, math, sys
from pathlib import Path
from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    ModelPoint,
)
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.pour_replay import (
    GOAL_FILL,
    GRAINS_PER_FILL,
    PourRecording,
    PourReplay,
)

DEMO_DRAIN = ModelPoint(
    repose_angle=0.0, discharge_coefficient=0.2, outflow_rate_constant=0.8
)
GOAL_LOWER_EDGE = (GOAL_FILL - 0.05) * GRAINS_PER_FILL
results = {}
for argument in sys.argv[1:]:
    material, bag, received, full = argument.split("::")
    material, received, full = PouredMaterial(material), float(received), float(full)
    replay = PourReplay.of(
        PourRecording.read(Path(bag)),
        material=material,
        drain=DEMO_DRAIN,
        settings=BankSettings(tracks_onset=True, trickle=0.0),
        receiver_from_count=True,
        stop_lead_time=1.0,
    )
    steps = replay.steps
    stop = next((s for s in steps if s.told_grains >= GOAL_LOWER_EDGE), None)
    first_seen = next((s for s in steps if s.reading.stream_seen), None)
    results[Path(bag).name] = dict(
        material=str(material),
        weighed_received=round(received / full * GRAINS_PER_FILL, 1),
        final_count=round(steps[-1].reading.counted_grains, 1),
        final_told=round(steps[-1].told_grains, 1),
        stream_first_seen=(
            None
            if first_seen is None
            else dict(
                time=round(first_seen.time, 1),
                tilt=round(math.degrees(first_seen.tilt), 1),
            )
        ),
        told_reaches_goal=(
            None
            if stop is None
            else dict(
                time=round(stop.time, 1),
                tilt=round(math.degrees(stop.tilt), 1),
                count=round(stop.reading.counted_grains, 1),
            )
        ),
    )
    print(Path(bag).name, json.dumps(results[Path(bag).name]), flush=True)
json.dump(results, open(Path(__file__).parent / "stop_replay.json", "w"), indent=1)
