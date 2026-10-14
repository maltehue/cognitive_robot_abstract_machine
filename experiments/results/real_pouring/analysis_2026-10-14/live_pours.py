"""
Compare the 2026-10-14 pours with the 2026-10-12 demo pours, and refit the 2026-10-12
drain with them added.

Usage: python live_pours.py <2026-10-12 folder holding demo/, ramps/ and onset.json>
<label>::<drain>::<bag folder> ... Each bag folder needs the kin.pkl of
../analysis_2026-10-12/kinematics.py (run extract.py first); a bag with a
live_bank_trace.json also gets its bank trace. Writes live_pours.json next to this
script.
"""

import glob, json, os, pickle, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).parent
MONDAY_ANALYSIS = HERE.parent / "analysis_2026-10-12"
monday = Path(sys.argv[1]).resolve()
pours_of_the_day = [argument.split("::") for argument in sys.argv[2:]]
pours_of_the_day = [
    (label, drain, Path(path).resolve()) for label, drain, path in pours_of_the_day
]
os.chdir(monday)
sys.path.insert(0, str(MONDAY_ANALYSIS))
import fit  # noqa: E402  (reads onset.json and the bags from the working directory)

FITTED = dict(k=2.0281, rho_start=np.radians(27.0), rho_stop=np.radians(27.0))


def load(bag):
    k = pickle.load(open(bag / "kin.pkl", "rb"))
    t = np.array([x["t"] for x in k["rows"]])
    tilt = np.arccos(np.clip([x["tool_R"][2, 1] for x in k["rows"]], -1, 1))
    source = np.array([x.get("source_fill", np.nan) for x in k["rows"]])
    receiver = np.array([x.get("receiver_fill", np.nan) for x in k["rows"]])
    reset = np.argmax(receiver < 0.01)
    start = reset + np.argmax(receiver[reset:] > 0.005)
    weighed = json.load(open(bag / "weighed_particles.json"))
    return t - t[start], tilt, source, receiver, weighed


def bank_trace(bag):
    trace = json.load(open(bag / "live_bank_trace.json"))
    t0 = trace[0]["time"]
    return [
        dict(
            time=round(s["time"] - t0, 2),
            tilt=round(np.degrees(s["tilt"]), 1),
            seen=s["stream_seen"],
            counted=round(s["counted_grains"], 1),
            received=round(s["received_grains"], 1),
            active=s["active"],
            probabilities={h: round(p, 3) for h, p in s["probabilities"].items()},
            repose_correction=round(
                np.degrees(s["corrections"]["repose_correction"]), 1
            ),
            drain_scale=round(float(np.exp(s["corrections"]["log_drain_scale"])), 2),
        )
        for s in trace
    ]


pours = {}
for name, drain, bag in [
    (Path(d).name, "demo", Path(d))
    for d in sorted(glob.glob(str(monday / "demo" / "*")))
] + pours_of_the_day:
    t, tilt, source, receiver, weighed = load(bag)
    keep = (t > -2) & (t < 30)
    left = (weighed["received_mass_grams"] + weighed["spilled_mass_grams"]) / fit.FULL
    pours[name] = dict(
        drain=drain,
        bank=bank_trace(bag) if (bag / "live_bank_trace.json").exists() else None,
        time=np.round(t[keep][::5], 2).tolist(),
        tilt=np.round(np.degrees(tilt[keep][::5]), 1).tolist(),
        believed_receiver=np.round(receiver[keep][::5], 3).tolist(),
        believed_source=np.round(source[keep][::5], 3).tolist(),
        fitted_source=np.round(
            fit.simulate(t, tilt, FITTED["k"], FITTED["rho_start"], FITTED["rho_stop"])[
                keep
            ][::5],
            3,
        ).tolist(),
        peak_tilt=float(np.degrees(tilt.max())),
        received=weighed["received_fill_level"],
        spilled_grams=weighed["spilled_mass_grams"],
        left_the_cup=left,
        fitted_prediction=float(
            1
            - fit.simulate(
                t, tilt, FITTED["k"], FITTED["rho_start"], FITTED["rho_stop"]
            )[-1]
        ),
    )
    print(
        name, {k: round(v, 3) for k, v in pours[name].items() if isinstance(v, float)}
    )

for name, _, bag in pours_of_the_day:
    t, tilt, _, _, weighed = load(bag)
    fit.demos.append(
        (
            name,
            t - t[0],
            tilt,
            (weighed["received_mass_grams"] + weighed["spilled_mass_grams"]) / fit.FULL,
        )
    )
refits = {}
for hysteresis, p0 in (
    (False, [1.0, np.radians(40)]),
    (True, [1.0, np.radians(65), np.radians(30)]),
):
    solution = least_squares(
        fit.residuals,
        p0,
        args=(hysteresis,),
        bounds=([0.01] + [0] * (len(p0) - 1), [50] + [1.6] * (len(p0) - 1)),
        diff_step=1e-3,
    )
    k, start, stop = (
        solution.x if hysteresis else (solution.x[0], solution.x[1], solution.x[1])
    )
    refits["two angles" if hysteresis else "one angle"] = dict(
        k=float(k),
        rho_start=float(np.degrees(start)),
        rho_stop=float(np.degrees(stop)),
        rms=float(np.sqrt(np.mean(solution.fun**2))),
        predictions={
            n: float(1 - fit.simulate(t, th, k, start, stop)[-1])
            for n, t, th, _ in fit.demos
        },
    )
    print(refits)
json.dump(dict(pours=pours, refits=refits), open(HERE / "live_pours.json", "w"))
