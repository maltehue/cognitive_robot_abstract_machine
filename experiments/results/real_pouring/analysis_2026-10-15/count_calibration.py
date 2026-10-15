"""
Count calibration of the stream detector on the opening moved onto the receiver's rim
(BLUE_CUP_OUTLINE): pixel-seconds per bank grain received, per weighed pour, with the
segmentation each material uses.

Usage: python count_calibration.py <material>::<bag dir or zip>::<received grams>::<full cup grams> ...
Writes count_calibration.json next to this script.
"""

import json, sys
from pathlib import Path
import numpy
from experiments.pouring_model_verification.sensor import CameraCondition
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.pour_replay import (
    BACKGROUND_FRAMES,
    FRAME_PERIOD,
    GRAINS_PER_FILL,
    OPENING_HEIGHT,
    OPENING_RADIUS,
    PourRecording,
)
from experiments.real_pouring.stream_detection import (
    BLUE_CUP_OUTLINE,
    CameraView,
    OpeningRegion,
    StreamDetector,
)

DROPOUT = CameraCondition.POLE_COUNTING.model.dropout
results = {}
for argument in sys.argv[1:]:
    material, bag, received, full = argument.split("::")
    material, received, full = PouredMaterial(material), float(received), float(full)
    recording = PourRecording.read(Path(bag))
    detector = StreamDetector(
        view=CameraView(
            intrinsics=recording.intrinsics, world_T_camera=recording.world_T_camera
        ),
        opening=OpeningRegion(
            centre=recording.receiver_position + numpy.array([0, 0, OPENING_HEIGHT]),
            radius=OPENING_RADIUS,
        ),
        segmentation=material.segmentation,
        pixel_seconds_per_grain=1.0,
        frame_period=FRAME_PERIOD,
        outline=BLUE_CUP_OUTLINE,
    )
    first = recording.first_pour_frame
    detector.learn_background(
        [
            frame.image()
            for frame in recording.frames[max(0, first - BACKGROUND_FRAMES) : first]
        ]
    )
    reading = None
    for frame in recording.frames[first:]:
        reading = detector.observe(
            frame.image(), stamp=frame.stamp - recording.pour_start
        )
    received_grains = received / full * GRAINS_PER_FILL
    calibration = reading.counted_grains / ((1 - DROPOUT) * received_grains)
    results[Path(bag).name] = dict(
        material=str(material),
        pixel_seconds=round(reading.counted_grains, 1),
        received_grains=round(received_grains, 2),
        pixel_seconds_per_grain=round(calibration, 2),
    )
    print(Path(bag).name, results[Path(bag).name], flush=True)
for material in {r["material"] for r in results.values()}:
    values = [
        r["pixel_seconds_per_grain"]
        for r in results.values()
        if r["material"] == material
    ]
    print(
        f"{material}: mean {numpy.mean(values):.2f}, spread {min(values):.2f}-{max(values):.2f} ({(max(values) - min(values)) / numpy.mean(values) * 50:.0f} % about the mean)"
    )
json.dump(
    results, open(Path(__file__).parent / "count_calibration.json", "w"), indent=1
)
