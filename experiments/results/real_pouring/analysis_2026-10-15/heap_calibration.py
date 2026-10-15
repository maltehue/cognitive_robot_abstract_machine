"""
Heap gauge calibration: the heap pixels the production detector sees at the end of a
weighed pour, over the share of the surface the camera sees at the weighed fill, gives
the pixels the whole surface would show for that material.

Usage: python heap_calibration.py <material>::<bag dir or zip>::<received grams>::<full cup grams> ...
Writes heap_calibration.json next to this script.
"""

import json, sys
from pathlib import Path
import numpy
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.pour_replay import (
    BACKGROUND_FRAMES,
    FRAME_PERIOD,
    OPENING_HEIGHT,
    OPENING_RADIUS,
    PourRecording,
)
from experiments.real_pouring.receiver_heap import HeapGauge
from experiments.real_pouring.stream_detection import (
    BLUE_CUP_OUTLINE,
    CameraView,
    OpeningRegion,
    StreamDetector,
)

SETTLED_FRAMES = 30
"""
The last frames of a bag whose heap is averaged.
"""

results = {}
for argument in sys.argv[1:]:
    material, bag, received, full = argument.split("::")
    material, fill = PouredMaterial(material), float(received) / float(full)
    recording = PourRecording.read(Path(bag))
    opening = OpeningRegion(
        centre=recording.receiver_position + numpy.array([0, 0, OPENING_HEIGHT]),
        radius=OPENING_RADIUS,
    )
    detector = StreamDetector(
        view=CameraView(
            intrinsics=recording.intrinsics, world_T_camera=recording.world_T_camera
        ),
        opening=opening,
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
    heap = []
    for frame in recording.frames[first:]:
        detector.observe(frame.image(), stamp=frame.stamp - recording.pour_start)
        heap.append(detector.heap_pixels)
    settled = float(numpy.mean(heap[-SETTLED_FRAMES:]))
    gauge = HeapGauge.for_receiver(
        recording.world_T_camera[:3, 3],
        opening.centre,
        OPENING_HEIGHT,
        OPENING_RADIUS,
        pixels_per_share=1.0,
    )
    share = gauge.visible_share(fill)
    results[Path(bag).name] = dict(
        material=str(material),
        weighed_fill=round(fill, 3),
        heap_pixels=round(settled, 1),
        visible_share=round(share, 3),
        pixels_per_share=None if share == 0 else round(settled / share, 1),
    )
    print(Path(bag).name, results[Path(bag).name], flush=True)
json.dump(results, open(Path(__file__).parent / "heap_calibration.json", "w"), indent=1)
