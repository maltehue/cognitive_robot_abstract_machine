"""
Per-pour rows for the 2026-10-25 ablation bags (no bank / fills only), same fields as
the sweep's dataset.py where available.
"""

import json, sys, glob, os
from pathlib import Path
import numpy

C = 722.0 / 361.83
root = Path(sys.argv[1])
rows = []
for d in sorted(glob.glob(str(root / "goalSweep-*" / "*/"))):
    d = Path(d)
    s = json.loads((d / "pour_settings.json").read_text())
    w = json.loads((d / "weighed_particles.json").read_text())
    cond = "nobank" if s["bank"] is None else "fillsonly"
    material = s["material"]["name"].lower()
    translucent = s["receiving_cup"]["name"] == "TRANSLUCENT_CUP"
    per_source = C if translucent else 1.0
    g = 100 * per_source
    goal = s["goal_fill"]
    rec = w["received_fill_level"]
    spill = w["spilled_mass_grams"] / w["full_cup_mass_grams"]
    cc = numpy.load(d / "transfer_control_cycles.npz", allow_pickle=True)
    meta = json.loads(str(cc["metadata"]))
    names = meta["structure"].get("debug_names") or meta["structure"].get(
        "debug_value_names"
    )
    ct = cc["times"] - cc["times"][0]
    dv = cc["debug_values"]
    pred = dv[:, 0]
    wp_names = meta["structure"].get("world_position_names") or meta["structure"].get(
        "position_names"
    )
    row = dict(
        cond=cond,
        material=material,
        goal=goal,
        weighed=round(rec, 3),
        error=round(rec - goal, 3),
        spill=round(spill, 3),
        ending=(d / "transfer_ending.txt").read_text().strip(),
        duration=round(float(ct[-1]), 1),
        max_predicted=round(float(min(pred.max(), 1.0)), 3),
        commit=s["repository"]["commit"][:10],
    )
    if cond == "fillsonly":
        tr = json.loads((d / "live_bank_trace.json").read_text())
        t0 = tr[0]["time"]
        t = numpy.array([x["time"] - t0 for x in tr])
        tilt = numpy.degrees([x["tilt"] for x in tr])
        cnt = numpy.array([x["counted_grains"] for x in tr]) / g
        told = numpy.minimum(numpy.array([x["told_grains"] for x in tr]) / g, 1.0)
        wall = numpy.array(
            [
                (
                    numpy.nan
                    if x["receiver_observation"] is None
                    else x["receiver_observation"] / g
                )
                for x in tr
            ]
        )
        act = [x["active"] for x in tr]
        rep = numpy.degrees([x["corrections"]["repose_correction"] for x in tr])
        dis = numpy.array([x["corrections"]["log_discharge_scale"] for x in tr])
        rise = numpy.r_[0, numpy.diff(cnt)] > 0.002 / per_source
        first_flow = float(t[rise][0]) if rise.any() else None
        first_flow_tilt = float(tilt[rise][0]) if rise.any() else None
        last_rise = float(t[rise][-1]) if rise.any() else None
        eps = []
        last = -10
        for i in numpy.flatnonzero(rise):
            if t[i] - last > 1.0:
                eps.append([cnt[i - 1] if i else 0.0])
            last = t[i]
            eps[-1].append(cnt[i])
        bursts = sum(1 for e in eps if (e[-1] - e[0]) * per_source >= 0.03)
        first_aim = next((float(t[i]) for i, a_ in enumerate(act) if a_ == "H2"), None)
        first_commit = next(
            (float(t[i]) for i, a_ in enumerate(act) if a_ != "H0"), None
        )
        row.update(
            told_end=round(float(told[-1]), 3),
            control_error=round(float(told[-1]) - goal, 3),
            perception_error=round(rec - float(told[-1]), 3),
            count_end=round(float(cnt[-1]), 3),
            wall_end=None if numpy.isnan(wall[-1]) else round(float(wall[-1]), 3),
            peak_tilt=round(float(tilt.max()), 1),
            end_tilt=round(float(tilt[-1]), 1),
            first_flow_s=first_flow,
            first_flow_tilt=(
                None if first_flow_tilt is None else round(first_flow_tilt, 1)
            ),
            stall_s=None if last_rise is None else round(float(t[-1] - last_rise), 1),
            bursts=bursts,
            bank_first_commit=first_commit,
            bank_first_aim=first_aim,
            bank_active_end=act[-1],
            bank_repose_end=round(float(rep[-1]), 1),
            bank_discharge_factor_end=round(float(numpy.exp(dis[-1])), 3),
            bank_repose_at_first_flow=(
                None
                if first_flow is None
                else round(float(rep[numpy.searchsorted(t, first_flow)]), 1)
            ),
        )
        # tilt held in last 5 s
        row["tilt_last5_range"] = round(
            float(tilt[t > t[-1] - 5].max() - tilt[t > t[-1] - 5].min()), 1
        )
    else:
        row["debug_names"] = names
        if wp_names:
            row["wp_names"] = wp_names
    rows.append(row)
json.dump(rows, open(sys.argv[2], "w"), indent=1)
for r in rows:
    print({k: v for k, v in r.items() if k not in ("debug_names", "wp_names")})
print("DEBUG NAMES:", [r for r in rows if r["cond"] == "nobank"][0].get("debug_names"))
print("WP NAMES:", [r for r in rows if r["cond"] == "nobank"][0].get("wp_names"))
print(
    "META KEYS:",
)
