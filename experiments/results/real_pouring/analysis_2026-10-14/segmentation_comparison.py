"""
Count a recorded pour with both grain segmentations: whiteness, as used for the
styrofoam balls, and any change from the empty view, as meant for corn.

Usage: python segmentation_comparison.py <bag dir> ... Each bag dir needs extracted.pkl
and kin.pkl (see README.md). Prints, per bag, what each segmentation counted before the
stream started, in all, and how often it saw a stream.
"""

import json, pickle, sys
from pathlib import Path
import cv2, numpy
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.stream_detection import (
    CameraView,
    ChangeSegmentation,
    OpeningRegion,
    StreamDetector,
    WhiteSegmentation,
)

decode = lambda b: cv2.imdecode(numpy.frombuffer(b, numpy.uint8), cv2.IMREAD_COLOR)
results = {}
for bag in map(Path, sys.argv[1:]):
    o = pickle.load(open(bag / "extracted.pkl", "rb"))
    k = pickle.load(open(bag / "kin.pkl", "rb"))
    rows = [r for r in k["rows"] if "receiver_p" in r]
    cam_T_map = numpy.eye(4)
    cam_T_map[:3, :3] = k["rows"][0]["map_to_cam_R"]
    cam_T_map[:3, 3] = k["rows"][0]["map_to_cam_p"]
    trace_file = bag / "live_bank_trace.json"
    start = (
        json.load(open(trace_file))[0]["time"] if trace_file.exists() else rows[0]["t"]
    )
    detectors = {
        name: StreamDetector(
            view=CameraView(
                intrinsics=k["K"], world_T_camera=numpy.linalg.inv(cam_T_map)
            ),
            opening=OpeningRegion(
                centre=rows[0]["receiver_p"] + [0, 0, 0.16], radius=0.035
            ),
            segmentation=segmentation,
            pixel_seconds_per_grain=PouredMaterial.STYROFOAM.pixel_seconds_per_grain,
            frame_period=1 / 30,
        )
        for name, segmentation in (
            ("white", WhiteSegmentation()),
            ("change", ChangeSegmentation()),
        )
    }
    empty = [decode(im[2]) for im in o["images"] if im[0] < start][-15:]
    for detector in detectors.values():
        detector.learn_background(empty)
    summary = {
        name: dict(seen=0, frames=0, first_seen=None, before_white_first=None)
        for name in detectors
    }
    for im in o["images"]:
        if im[0] < start:
            continue
        image = decode(im[2])
        for name, detector in detectors.items():
            reading = detector.observe(image, stamp=im[0] - start)
            entry = summary[name]
            entry["frames"] += 1
            entry["seen"] += int(reading.stream_seen)
            entry["counted"] = round(reading.counted_grains, 1)
            if reading.stream_seen and entry["first_seen"] is None:
                entry["first_seen"] = round(reading.stamp, 2)
        if (
            summary["white"]["first_seen"] is not None
            and summary["change"]["before_white_first"] is None
        ):
            summary["change"]["before_white_first"] = summary["change"]["counted"]
    weighed = json.load(open(bag / "weighed_particles.json"))
    results[bag.name] = dict(
        summary,
        weighed_out=round(
            (weighed["received_mass_grams"] + weighed["spilled_mass_grams"])
            / 2.65
            * 100,
            1,
        ),
    )
    print(bag.name, json.dumps(results[bag.name]), flush=True)
json.dump(
    results, open(Path(__file__).parent / "segmentation_comparison.json", "w"), indent=1
)
