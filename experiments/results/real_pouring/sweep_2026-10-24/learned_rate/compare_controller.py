"""
Score the controller's own 1.5 s prediction against the learned predictors on the same
samples.
"""

import json, sys, glob
from pathlib import Path
import numpy

rows = json.load(open(sys.argv[1]))
learned = numpy.load(sys.argv[2])
pour = numpy.array([r["pour"] for r in rows])
t = numpy.array([r["t"] for r in rows])
y = learned[:, 2]
tilt = numpy.array([r["tilt"] for r in rows])
change = numpy.array([r["tilt_change"] for r in rows])
history, statefree = learned[:, 0], learned[:, 1]
import subprocess

matched = []
for f in sorted(glob.glob(str(Path(sys.argv[3]) / "controller_*.json"))):
    name = Path(f).stem.replace("controller_", "")
    C = numpy.array(json.load(open(f)))
    ct, ctold, cpred = C[:, 0], C[:, 1], C[:, 4]
    idx = numpy.flatnonzero(pour == name)
    for i in idx:
        k = int(numpy.argmin(abs(ct - t[i])))
        if abs(ct[k] - t[i]) < 0.06:
            matched.append((i, cpred[k] - ctold[k]))
I = numpy.array([m[0] for m in matched])
ctrl = numpy.array([m[1] for m in matched])
rmse = lambda e: float(numpy.sqrt(numpy.mean(e**2)))
print(f"{len(I)} matched samples from {len(set(pour[I]))} pours")
for label, sel in (
    ("all", numpy.ones(len(I), bool)),
    ("tilt >= 85", tilt[I] >= 85),
    ("tilt >= 85, not turning back", (tilt[I] >= 85) & (change[I] >= -1.0)),
):
    yy = y[I][sel]
    print(f"== {label}: {sel.sum()} samples, gain sd {yy.std():.3f}")
    for p, est in (
        ("controller (analytic model + bank)", ctrl[sel]),
        ("learned, tilt + fill", statefree[I][sel]),
        ("learned, history (past only)", history[I][sel]),
    ):
        if est is None:
            continue
        e = est - yy
        print(
            f"   {p:36s} rmse {rmse(e):.4f} bias {e.mean():+.4f} explained {1 - numpy.var(e) / numpy.var(yy):+.2f}"
        )
