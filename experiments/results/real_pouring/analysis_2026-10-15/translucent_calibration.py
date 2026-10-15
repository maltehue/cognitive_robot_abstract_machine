"""
Calibration of the translucent receiving cup on a weighed corn pour: the moving yellow
area in its opening, times the time it shows, per bank grain received, and the share of
the corn's projected area the camera picks out through the frosted wall.

The stream is counted from the pour's start until the cup is back below RETURNED_TILT;
the wall's still pixels are read there too. The count over the seconds after the receiver
stopped rising (NO_STREAM_FROM, given per bag) shows how much the count creeps without a
stream.

Usage: python translucent_calibration.py <bag dir or zip>::<received grams>::<no stream from s> ...
Writes translucent_calibration.json next to this script.
"""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy

from experiments.real_pouring.pour_replay import (
    FRAME_PERIOD,
    BACKGROUND_FRAMES,
    GRAINS_PER_FILL,
    PourRecording,
)
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup
from experiments.real_pouring.stream_detection import CameraView

RETURNED_TILT = math.radians(10.0)
MATERIAL = PouredMaterial.CORN
CUP = ReceivingCup.TRANSLUCENT_CUP

results = {}
for argument in sys.argv[1:]:
    bag, received, no_stream_from = argument.split("::")
    received, no_stream_from = float(received), float(no_stream_from)
    recording = PourRecording.read(Path(bag))
    detector = CUP.detector_for(
        material=MATERIAL,
        view=CameraView(
            intrinsics=recording.intrinsics, world_T_camera=recording.world_T_camera
        ),
        opening_centre=recording.receiver_position
        + numpy.array([0.0, 0.0, CUP.height]),
        frame_period=FRAME_PERIOD,
    )
    detector = replace(
        detector,
        pixel_seconds_per_grain=1.0,
        wall_gauge=replace(detector.wall_gauge, detected_share=1.0),
    )
    first = recording.first_pour_frame
    detector.learn_background(
        [
            frame.image()
            for frame in recording.frames[max(0, first - BACKGROUND_FRAMES) : first]
        ]
    )
    tilts = [pose.tilt for pose in recording.cup_poses]
    peak = int(numpy.argmax(tilts))
    returned = next(
        (index for index in range(peak, len(tilts)) if tilts[index] < RETURNED_TILT),
        len(tilts) - 1,
    )
    at_no_stream = None
    for index in range(first, returned + 1):
        frame = recording.frames[index]
        reading = detector.observe(
            frame.image(), stamp=frame.stamp - recording.pour_start
        )
        if at_no_stream is None and reading.stamp >= no_stream_from:
            at_no_stream = (reading.stamp, reading.counted_grains)
    end = recording.frames[returned].stamp - recording.pour_start
    pixel_seconds = reading.counted_grains
    creep = (pixel_seconds - at_no_stream[1]) / (end - at_no_stream[0])
    source_grains = received / MATERIAL.full_cup_mass_grams * GRAINS_PER_FILL
    stream_pixel_seconds = pixel_seconds - creep * (end - no_stream_from)
    fill = received / CUP.full_mass_grams(MATERIAL)
    results[Path(bag).name] = dict(
        received_grams=received,
        pour_seconds=end,
        pixel_seconds=pixel_seconds,
        creep_pixel_seconds_per_second=creep,
        pixel_seconds_per_grain=stream_pixel_seconds / source_grains,
        wall_pixels=detector.wall_pixels,
        received_fill=fill,
        wall_detected_share=detector.wall_pixels
        / detector.wall_gauge.projected_area(fill),
    )
    print(Path(bag).name, json.dumps(results[Path(bag).name], indent=1))
Path(__file__).with_name("translucent_calibration.json").write_text(
    json.dumps(results, indent=1)
)
