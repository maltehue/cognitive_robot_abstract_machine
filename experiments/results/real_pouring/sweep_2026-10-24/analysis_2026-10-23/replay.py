import sys, json, numpy
from pathlib import Path
from experiments.real_pouring.wall_fill_validation import (
    WallFillRecording,
    WallFillPartition,
)
from experiments.real_pouring.wall_fill import (
    WallFillCalibration,
    WallColourMeasurement,
)

cal = WallFillCalibration(
    WallColourMeasurement(2.0, 32.0, 2.0),
    15.309292169196675,
    0.007940178974067994,
    0.035201864629384017,
)
for f in sorted(Path(sys.argv[1]).glob("*.npz")):
    with numpy.load(f) as c:
        tilts = c["tilts"]
    import shutil

    shutil.copy(
        Path(sys.argv[2]) / f.stem / "weighed_particles.json", f.with_suffix(".json")
    )
    r = WallFillRecording.read(f, WallFillPartition.DEVELOPMENT)
    rd = r.replay(cal)
    w = r.weighed
    print(f"== {f.stem} weighed {w:.3f}")
    # sample every 1 s over the last part
    t = r.times
    for target in numpy.arange(max(0, t[-1] - 30), t[-1] + 0.01, 1.5):
        i = int(numpy.searchsorted(t, target))
        i = min(i, len(t) - 1)
        print(
            f"  t {t[i]:5.1f} tilt {tilts[i]:6.1f} reading {rd[i] if rd[i] is None else round(rd[i],3)}"
        )
