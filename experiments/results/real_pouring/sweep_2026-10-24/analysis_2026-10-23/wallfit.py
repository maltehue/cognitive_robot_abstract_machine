import sys, json, numpy
from pathlib import Path
from experiments.real_pouring.wall_fill_validation import (
    WallFillRecording,
    WallFillPartition,
)
from experiments.real_pouring.wall_fill import WallColourMeasurement

d = Path(sys.argv[1])
m = WallColourMeasurement(noise_floor=2.0, colour_span=32.0, height_power=2.0)
rows = []
for f in sorted(d.glob("*.npz")):
    if f.stem == "2026-10-19_16-25-52":
        pass
    o = WallFillRecording.read(f, WallFillPartition.DEVELOPMENT).observe(m)
    rows.append((f.stem, o.sample.signal, o.sample.fill, o.baseline))
s = numpy.array([r[1] for r in rows])
w = numpy.array([r[2] for r in rows])
day = numpy.array([r[0][:10] for r in rows])
cur = 15.309292169196675
for r in rows:
    print(
        f"{r[0]} signal {r[1]:.4f} weighed {r[2]:.3f} current {cur*r[1]:.3f} err {cur*r[1]-r[2]:+.3f}"
    )


def fit(mask, offset):
    A = numpy.c_[s[mask], numpy.ones(mask.sum())] if offset else s[mask][:, None]
    return numpy.linalg.lstsq(A, w[mask], rcond=None)[0]


def pred(p, x):
    return p[0] * x + (p[1] if len(p) > 1 else 0)


supported = s >= 0.0079  # drop the out-of-range 4.99% pour
for name, mask in [
    ("old 18", supported & (day != "2026-10-23")),
    ("all", supported),
    ("today", day == "2026-10-23"),
]:
    for off in (False, True):
        p = fit(mask, off)
        # leave-one-out on mask
        errs = []
        for i in numpy.where(mask)[0]:
            mm = mask.copy()
            mm[i] = False
            errs.append(pred(fit(mm, off), s[i]) - w[i])
        errs = numpy.array(errs)
        today_err = pred(p, s[day == "2026-10-23"]) - w[day == "2026-10-23"]
        print(
            f"{name:7s} offset={off!s:5s} params {numpy.round(p,4)}  LOO mean|e| {numpy.abs(errs).mean()*100:.2f}pp bias {errs.mean()*100:+.2f}pp  max {numpy.abs(errs).max()*100:.2f}  | today in-sample bias {today_err.mean()*100:+.2f}pp |e| {numpy.abs(today_err).mean()*100:.2f}"
        )
