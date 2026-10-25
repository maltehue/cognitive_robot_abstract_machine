"""
Per styrofoam bag: goal, weighed, count, told, heap raise, tilt story, restarts.
"""

import json, math, sys
from pathlib import Path
import numpy

for bag in map(Path, sys.argv[1:]):
    s = json.loads((bag / "pour_settings.json").read_text())
    goal = s["goal_fill"]
    w = json.loads((bag / "weighed_particles.json").read_text())
    tr = json.loads((bag / "live_bank_trace.json").read_text())
    t0 = tr[0]["time"]
    t = numpy.array([x["time"] - t0 for x in tr])
    tilt = numpy.degrees([x["tilt"] for x in tr])
    cnt = numpy.array([x["counted_grains"] for x in tr]) / 100
    told = numpy.array([x["told_grains"] for x in tr]) / 100
    fl = numpy.array([x["in_flight_grains"] for x in tr]) / 100
    raise_ = told - cnt - fl
    d = numpy.load(bag / "transfer_control_cycles.npz", allow_pickle=True)
    ct = d["times"]
    pred = d["debug_values"][:, 0]
    rec = w["received_fill_level"]
    spill = w["spilled_mass_grams"] / w["full_cup_mass_grams"]
    first = t[numpy.argmax(cnt > 0.005)]
    # tilt reversals: local maxima of tilt
    dt = numpy.gradient(tilt, t)
    turns = [i for i in range(1, len(t)) if dt[i] < -1 and dt[i - 1] >= -1]
    goal_cross = ct[numpy.argmax(pred > goal)] if (pred > goal).any() else None
    print(
        f"== goal {goal:.2f} | weighed {rec:.3f} ({rec - goal:+.3f}), spilled {spill:.3f}, source gave {rec + spill:.3f} | {bag.name[:19]}"
    )
    print(
        f"   final count {cnt[-1]:.3f} (count/weighed {cnt[-1] / rec:.2f}), told {told[-1]:.3f}, heap raise at end {raise_[-1]:+.3f} (max {raise_.max():+.3f} at {t[numpy.argmax(raise_)]:.1f}s)"
    )
    print(
        f"   first counted {first:.1f}s at {tilt[numpy.argmax(cnt > 0.005)]:.0f} deg, peak tilt {tilt.max():.0f} deg, transfer {t[-1]:.1f}s, pred first > goal at {goal_cross if goal_cross is None else round(float(goal_cross),1)}s"
    )
    for i in turns:
        print(
            f"   turn-back {t[i]:5.1f}s at {tilt[i]:5.1f} deg: count {cnt[i]:.3f}, told {told[i]:.3f}"
        )
