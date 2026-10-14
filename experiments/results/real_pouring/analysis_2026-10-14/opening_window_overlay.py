"""
Draw the opening window the stream detector watches onto frames of a recorded pour: one
before the pour, one with the most white grains in the window, and one late in the pour.

Usage: python opening_window_overlay.py <bag dir> Writes opening_window_<bag>.png next
to this script.
"""

import json, pickle, sys
from pathlib import Path
import cv2, numpy
from experiments.real_pouring.stream_detection import (
    CameraView,
    OpeningRegion,
    WhiteSegmentation,
)

HERE = Path(__file__).parent
decode = lambda b: cv2.imdecode(numpy.frombuffer(b, numpy.uint8), cv2.IMREAD_COLOR)
bag = Path(sys.argv[1])
o = pickle.load(open(bag / "extracted.pkl", "rb"))
k = pickle.load(open(bag / "kin.pkl", "rb"))
rows = [r for r in k["rows"] if "receiver_p" in r]
cam_T_map = numpy.eye(4)
cam_T_map[:3, :3] = k["rows"][0]["map_to_cam_R"]
cam_T_map[:3, 3] = k["rows"][0]["map_to_cam_p"]
view = CameraView(intrinsics=k["K"], world_T_camera=numpy.linalg.inv(cam_T_map))
start = json.load(open(bag / "live_bank_trace.json"))[0]["time"]
images = [(t, decode(b)) for t, _, b in o["images"]]
before = [im for t, im in images if t < start]
background = numpy.median(numpy.stack(before[-15:]), axis=0).astype(numpy.uint8)
mask = OpeningRegion(centre=rows[0]["receiver_p"] + [0, 0, 0.16], radius=0.035).mask_in(
    view, background.shape[:2]
)
white = WhiteSegmentation()
during = [
    (t, im, int((white.grains(im, background) & mask).sum()))
    for t, im in images
    if t >= start
]
busiest = max(during, key=lambda f: f[2])
late = during[int(len(during) * 0.9)]
contours, _ = cv2.findContours(
    mask.astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
panels = []
for label, im in (
    ("before", before[-1]),
    (f"most grains t={busiest[0] - start:.1f}s", busiest[1]),
    (f"late t={late[0] - start:.1f}s", late[1]),
):
    drawn = im.copy()
    cv2.drawContours(drawn, contours, -1, (0, 0, 255), 2)
    ys, xs = numpy.nonzero(mask)
    crop = drawn[
        max(0, ys.min() - 150) : ys.max() + 150, max(0, xs.min() - 150) : xs.max() + 150
    ]
    cv2.putText(crop, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    panels.append(crop)
height = min(p.shape[0] for p in panels)
cv2.imwrite(
    str(HERE / f"opening_window_{bag.name}.png"),
    numpy.hstack([p[:height] for p in panels]),
)
