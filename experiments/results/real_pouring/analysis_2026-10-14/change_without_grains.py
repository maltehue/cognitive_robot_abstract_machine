"""
Split what the change segmentation counts in a recorded styrofoam pour into frames that
also show white grains and frames that show none, to see whether the extra it counts
comes with the grains or with something else moving through the opening, such as the
cup, the arm or their shadows.

Usage: python change_without_grains.py <bag dir> ... Prints, per bag, the changed area
inside the opening in frames with and without white grains, binned by the cup's tilt,
and saves crops of the frames with the most change but no white grains next to this
script.
"""

import json, pickle, sys
from pathlib import Path
import cv2, numpy
from experiments.real_pouring.stream_detection import (
    CameraView,
    ChangeSegmentation,
    OpeningRegion,
    WhiteSegmentation,
)

HERE = Path(__file__).parent
decode = lambda b: cv2.imdecode(numpy.frombuffer(b, numpy.uint8), cv2.IMREAD_COLOR)
for bag in map(Path, sys.argv[1:]):
    o = pickle.load(open(bag / "extracted.pkl", "rb"))
    k = pickle.load(open(bag / "kin.pkl", "rb"))
    rows = [r for r in k["rows"] if "receiver_p" in r]
    T = numpy.array([r["t"] for r in k["rows"]])
    tilt = numpy.degrees(
        numpy.arccos(numpy.clip([r["tool_R"][2, 1] for r in k["rows"]], -1, 1))
    )
    cam_T_map = numpy.eye(4)
    cam_T_map[:3, :3] = k["rows"][0]["map_to_cam_R"]
    cam_T_map[:3, 3] = k["rows"][0]["map_to_cam_p"]
    view = CameraView(intrinsics=k["K"], world_T_camera=numpy.linalg.inv(cam_T_map))
    start = (
        json.load(open(bag / "live_bank_trace.json"))[0]["time"]
        if (bag / "live_bank_trace.json").exists()
        else rows[0]["t"]
    )
    images = [(t, decode(b)) for t, _, b in o["images"]]
    background = numpy.median(
        numpy.stack([im for t, im in images if t < start][-15:]), axis=0
    ).astype(numpy.uint8)
    mask = OpeningRegion(
        centre=rows[0]["receiver_p"] + [0, 0, 0.16], radius=0.035
    ).mask_in(view, background.shape[:2])
    white, change = WhiteSegmentation(), ChangeSegmentation()
    frames = []
    for t, im in images:
        if t < start:
            continue
        w = int((white.grains(im, background) & mask).sum())
        c = int((change.grains(im, background) & mask).sum())
        frames.append((t - start, float(numpy.interp(t, T, tilt)), w, c, im))
    with_grains = [f for f in frames if f[2] > 0]
    without = [f for f in frames if f[2] == 0]
    print(
        f"{bag.name}: opening {int(mask.sum())} px; frames with white grains {len(with_grains)}, change/white area there {sum(f[3] for f in with_grains) / max(1, sum(f[2] for f in with_grains)):.2f}; "
        f"frames without white grains {len(without)}, change area there {sum(f[3] for f in without)} px in {sum(1 for f in without if f[3] > 0)} frames"
    )
    for low in range(0, 120, 15):
        binned = [f for f in without if low <= f[1] < low + 15]
        if binned:
            print(
                f"   tilt {low:3d}-{low + 15:3d} deg, no white grains: {len(binned):3d} frames, change {sum(f[3] for f in binned):6d} px, most {max(f[3] for f in binned):5d} px"
            )
    ys, xs = numpy.nonzero(mask)
    for rank, f in enumerate(sorted(without, key=lambda f: -f[3])[:3]):
        crop = f[4][
            max(0, ys.min() - 40) : ys.max() + 40, max(0, xs.min() - 40) : xs.max() + 40
        ]
        cv2.imwrite(
            str(
                HERE
                / f"change_without_grains_{bag.name}_{rank}_t{f[0]:.1f}_tilt{f[1]:.0f}.png"
            ),
            crop,
        )
