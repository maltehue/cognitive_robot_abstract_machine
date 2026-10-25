import sys, json, numpy
from pathlib import Path
from experiments.real_pouring.wall_fill import WallColourMeasurement

m = WallColourMeasurement(2.0, 32.0, 2.0)
W = 10
N = 30
rows = []
held = []
for f in sorted(Path(sys.argv[1]).glob("*.npz")):
    c = numpy.load(f)
    d = c["diffs"]
    reg = c["region"]
    tilt = c["tilts"]
    w = json.loads(f.with_suffix(".json").read_text())["received_fill_level"]

    def sig(end):
        return numpy.median(
            [
                m.measure(d[i - W + 1 : i + 1].min(axis=0).astype(float), reg)
                for i in range(end - N + 1, end + 1)
            ]
        )

    low = int(numpy.where(tilt > 30)[0][-1])
    j = low
    while j > 0 and tilt[j - 1] > tilt[j] + 0.02:
        j -= 1
    i_end = j - 3
    held.append((f.stem, float(tilt[i_end]), float(tilt[low]), int(low - j)))
    rows.append((f.stem, sig(i_end), sig(len(d) - 1), w))
st = numpy.array([r[1] for r in rows])
su = numpy.array([r[2] for r in rows])
w = numpy.array([r[3] for r in rows])
day = numpy.array([r[0][:10] for r in rows])
for h in held:
    print("held", h)
for r in rows:
    print(
        f"{r[0]} transfer {r[1]:.4f} upright {r[2]:.4f} ratio {r[1]/r[2]:.3f} weighed {r[3]:.3f}"
    )
ok = su >= 0.0079


def fit(s, mask, off):
    A = numpy.c_[s[mask], numpy.ones(mask.sum())] if off else s[mask][:, None]
    return numpy.linalg.lstsq(A, w[mask], rcond=None)[0]


pr = lambda p, x: p[0] * x + (p[1] if len(p) > 1 else 0)
for label, s in (("upright", su), ("transfer", st)):
    for train_name, train in (("old", ok & (day != "2026-10-23")), ("all", ok)):
        for off in (False, True):
            p = fit(s, train, off)
            loo = []
            for i in numpy.where(ok)[0]:
                mm = train.copy()
                mm[i] = False
                loo.append(pr(fit(s, mm, off), s[i]) - w[i])
            loo = numpy.array(loo)
            td = day[ok] == "2026-10-23"
            print(
                f"{label:8s} train={train_name} offset={off!s:5s} p={numpy.round(p,4)} LOO |e| all {numpy.abs(loo).mean()*100:.2f} bias {loo.mean()*100:+.2f} max {numpy.abs(loo).max()*100:.2f} | today bias {loo[td].mean()*100:+.2f} |e| {numpy.abs(loo[td]).mean()*100:.2f} | old bias {loo[~td].mean()*100:+.2f}"
            )
# current calibration on transfer-end signals
cur = 15.309292169196675
e = cur * st[ok] - w[ok]
td = day[ok] == "2026-10-23"
print(
    f"current calib on transfer-end signal: today bias {e[td].mean()*100:+.2f} old bias {e[~td].mean()*100:+.2f} |e| all {numpy.abs(e).mean()*100:.2f}"
)
print(
    "---- candidate calibrations evaluated on held-tilt (control-time) and upright readings"
)
td = (day == "2026-10-23") & ok
od = (day != "2026-10-23") & ok
offB = float(numpy.mean(w[td] - cur * st[td]))
pA = fit(st, ok, True)
pC = fit(su, ok, True)
for name, p in (
    ("current", [cur]),
    ("A held-tilt all", pA),
    ("B today-only offset", [cur, offB]),
    ("C upright all", pC),
):
    for lab, s in (("held", st), ("upright", su)):
        e = pr(p, s) - w
        print(
            f"{name:20s} {numpy.round(p,4)} {lab:7s} today bias {e[td].mean()*100:+.2f} |e| {numpy.abs(e[td]).mean()*100:.2f} | older bias {e[od].mean()*100:+.2f} |e| {numpy.abs(e[od]).mean()*100:.2f}"
        )
