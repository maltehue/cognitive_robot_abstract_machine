"""
One row per pour of both sweeps, plus per-material summaries; writes rows to a json.
"""

import json, math, sys
from pathlib import Path
import numpy

CORN_TRANSLUCENT_CUPS = 722.0 / 361.83
rows = []
for bag in sorted(Path(p) for p in sys.argv[2:]):
    s = json.loads((bag / "pour_settings.json").read_text())
    material = s["material"]["name"].lower()
    cup = s["receiving_cup"]["name"].lower()
    g = 100 * (CORN_TRANSLUCENT_CUPS if cup == "translucent_cup" else 1.0)
    goal = s["goal_fill"]
    w = json.loads((bag / "weighed_particles.json").read_text())
    rec = w["received_fill_level"]
    spill = w["spilled_mass_grams"] / (
        w["full_cup_mass_grams"]
        * (CORN_TRANSLUCENT_CUPS if cup == "translucent_cup" else 1)
    )
    tr = json.loads((bag / "live_bank_trace.json").read_text())
    t0 = tr[0]["time"]
    t = numpy.array([x["time"] - t0 for x in tr])
    tilt = numpy.degrees([x["tilt"] for x in tr])
    cnt = numpy.array([x["counted_grains"] for x in tr]) / g
    told = numpy.array([x["told_grains"] for x in tr]) / g
    wall = [x["receiver_observation"] for x in tr]
    wall_last = next((v / g for v in reversed(wall) if v is not None), None)
    # flow episodes: count rising, separated by >= 1 s without rise
    rise = numpy.r_[0, numpy.diff(cnt)] > 0.002
    eps = []
    last = -10
    for i in numpy.flatnonzero(rise):
        if t[i] - last > 1.0:
            eps.append([t[i], cnt[i - 1] if i else 0.0, tilt[i]])
        last = t[i]
        eps[-1].append(cnt[i])
    episodes = [(e[0], e[2], e[-1] - e[1]) for e in eps]
    big = [e for e in episodes if e[2] >= 0.03]
    d = numpy.load(bag / "transfer_control_cycles.npz", allow_pickle=True)
    ct = d["times"]
    pred = d["debug_values"][:, 0]
    wv = numpy.degrees(d["velocities"][:, 5])
    cross = numpy.flatnonzero(pred > goal)
    first_cross = ct[cross[0]] if len(cross) else None
    # truth after the turn-back: weighed minus told at the moment the prediction first crossed the goal
    told_at_cross = (
        told[min(numpy.searchsorted(t, first_cross), len(t) - 1)]
        if first_cross is not None
        else None
    )
    rows.append(
        dict(
            bag=bag.name[:19],
            material=material,
            goal=goal,
            weighed=rec,
            error=rec - goal,
            spill=spill,
            told=float(told[-1]),
            count=float(cnt[-1]),
            wall=wall_last,
            episodes=len(big),
            episode_list=[(round(a, 1), round(b), round(c, 3)) for a, b, c in big],
            peak_tilt=float(tilt.max()),
            duration=float(t[-1]),
            told_at_cross=None if told_at_cross is None else float(told_at_cross),
            min_wrist_rate=float(wv.min()),
        )
    )
json.dump(rows, open(sys.argv[1], "w"), indent=1)
for r in rows:
    wall = "" if r["wall"] is None else f" wall {r['wall']:.3f}"
    tc = "" if r["told_at_cross"] is None else f" told@cross {r['told_at_cross']:.3f}"
    print(
        f"{r['material'][:5]} {r['bag'][11:]} goal {r['goal']:.1f} weighed {r['weighed']:.3f} err {r['error']:+.3f} spill {r['spill']:.3f} | told {r['told']:.3f} count {r['count']:.3f}{wall} | {r['episodes']} bursts {r['episode_list']} peak {r['peak_tilt']:.0f} | {r['duration']:.0f}s{tc} ret {r['min_wrist_rate']:.1f}"
    )
