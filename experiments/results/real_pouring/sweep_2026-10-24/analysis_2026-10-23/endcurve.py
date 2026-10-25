import sys, json, numpy
from pathlib import Path
from experiments.real_pouring.wall_fill import WallColourMeasurement

m = WallColourMeasurement(2.0, 32.0, 2.0)
W = 10
k = 15.309292169196675
for f in sorted(Path(sys.argv[1]).glob(sys.argv[2])):
    c = numpy.load(f)
    d = c["diffs"]
    reg = c["region"]
    tilt = c["tilts"]
    t = c["times"]
    w = json.loads(f.with_suffix(".json").read_text())["received_fill_level"]
    print(
        "==",
        f.stem,
        "weighed",
        round(w, 3),
        "frames",
        len(d),
        "dt",
        round(float(numpy.median(numpy.diff(t))), 4),
    )
    for i in range(W, len(d), 15):
        if t[i] < t[-1] - 14:
            continue
        print(
            f"  t {t[i]:5.1f} tilt {tilt[i]:5.1f} fill {k*m.measure(d[i-W+1:i+1].min(axis=0).astype(float), reg):.3f}"
        )
