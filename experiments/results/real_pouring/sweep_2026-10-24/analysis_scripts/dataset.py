"""
Writes the page's three datasets from both sweeps' bags: pours, turn-backs and time
series.
"""

import json, sys
from pathlib import Path
import numpy

C = 722.0 / 361.83
out = Path(sys.argv[1])
bags = sorted(Path(p) for p in sys.argv[2:])
pours, turns, series = [], [], []
for bag in bags:
    s = json.loads((bag / "pour_settings.json").read_text())
    material = s["material"]["name"].lower()
    translucent = s["receiving_cup"]["name"] == "TRANSLUCENT_CUP"
    per_source = C if translucent else 1.0
    g = 100 * per_source
    pid = bag.name[:19]
    goal = s["goal_fill"]
    w = json.loads((bag / "weighed_particles.json").read_text())
    rec = w["received_fill_level"]
    spill = w["spilled_mass_grams"] / w["full_cup_mass_grams"]
    tr = json.loads((bag / "live_bank_trace.json").read_text())
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
    d = numpy.load(bag / "transfer_control_cycles.npz", allow_pickle=True)
    ct = d["times"]
    pred = d["debug_values"][:, 0]
    rise = numpy.r_[0, numpy.diff(cnt)] > 0.002 / per_source
    eps = []
    last = -10
    for i in numpy.flatnonzero(rise):
        if t[i] - last > 1.0:
            eps.append([cnt[i - 1] if i else 0.0])
        last = t[i]
        eps[-1].append(cnt[i])
    bursts = sum(1 for e in eps if (e[-1] - e[0]) * per_source >= 0.03)
    told_end = float(told[-1])
    dis = numpy.array([x["corrections"]["log_discharge_scale"] for x in tr])
    act = [x["active"] for x in tr]
    rep = numpy.degrees([x["corrections"]["repose_correction"] for x in tr])
    rising = numpy.r_[0, numpy.diff(cnt)] > 0
    seen = numpy.array([x["stream_seen"] for x in tr])
    first_aim = next((float(t[i]) for i, a_ in enumerate(act) if a_ == "H2"), None)
    delivered = rec + spill
    cd0 = s["drain"]["discharge_coefficient"]
    received_belief = numpy.array([x["received_grains"] for x in tr]) / g
    source_belief = numpy.array([x["source_grains"] for x in tr]) / 100
    rate_measured = numpy.gradient(
        numpy.convolve(cnt, numpy.ones(9) / 9, mode="same"), t
    )
    fill_rate = d["debug_values"][:, 1]
    pours.append(
        dict(
            pour=pid,
            material=material,
            receiver="translucent cup" if translucent else "Jeroen cup",
            goal=goal,
            weighed=round(rec, 4),
            error=round(rec - goal, 4),
            told_end=round(told_end, 4),
            control_error=round(told_end - goal, 4),
            perception_error=round(rec - told_end, 4),
            error_source_cups=round((rec - goal) * per_source, 4),
            spill=round(spill, 4),
            count_end=round(float(cnt[-1]), 4),
            peak_tilt=round(float(tilt.max()), 1),
            bursts=bursts,
            duration=round(float(t[-1]), 1),
            in_sweep=goal <= 0.6,
            bank_received_end=round(float(received_belief[-1]), 4),
            max_predicted=round(float(min(pred.max(), 1.0)), 4),
            source_left_belief=round(float(source_belief[-1]), 4),
            source_left_weighed=round(float(1 - delivered * per_source), 4),
            discharge_end=round(float(cd0 * numpy.exp(dis[-1])), 4),
            discharge_factor_end=round(float(numpy.exp(dis[-1])), 3),
            discharge_factor_flowing=round(
                float(numpy.exp(numpy.median(dis[rising]))) if rising.any() else 1.0, 3
            ),
            aim_share=round(sum(1 for a_ in act if a_ == "H2") / len(act), 3),
            first_aim=None if first_aim is None else round(first_aim, 1),
            seen_while_flowing=round(
                float(seen[rising].mean()) if rising.any() else 0.0, 3
            ),
        )
    )
    # turn-backs
    i = 1
    k_turn = 0
    while i < len(t) - 1:
        if tilt[i] >= tilt[i - 1] and tilt[i] > tilt[i + 1]:
            j = i
            while j < len(t) - 1 and tilt[j + 1] <= tilt[j] + 0.05:
                j += 1
            if tilt[i] - tilt[j] >= 3:
                before = cnt[i] - cnt[max(0, numpy.searchsorted(t, t[i] - 2.0))]
                stop = i
                for k in range(i, j + 1):
                    if cnt[k] - cnt[k - 1] > 0.002 / per_source:
                        stop = k
                after = cnt[j] - cnt[i]
                if (before + after) * per_source > 0.02:
                    k_turn += 1
                    turns.append(
                        dict(
                            turn=f"{pid} #{k_turn}",
                            pour=pid,
                            material=material,
                            goal=goal,
                            number=k_turn,
                            peak_tilt=round(float(tilt[i]), 1),
                            before_peak=round(float(before * per_source), 4),
                            after_peak=round(float(after * per_source), 4),
                            tilt_fall=round(float(tilt[i] - tilt[stop]), 1),
                            seconds_to_stop=round(float(t[stop] - t[i]), 2),
                        )
                    )
            i = j
        i += 1
    for x in numpy.arange(0, t[-1], 0.2):
        a = min(numpy.searchsorted(t, x), len(t) - 1)
        b = min(numpy.searchsorted(ct, x), len(ct) - 1)
        series.append(
            dict(
                key=f"{pid} {x:.1f}",
                pour=pid,
                time=round(float(x), 1),
                tilt=round(float(tilt[a]), 1),
                counted=round(float(cnt[a]), 4),
                told=round(float(told[a]), 4),
                wall=None if numpy.isnan(wall[a]) else round(float(wall[a]), 4),
                predicted=round(float(min(pred[b], 1.0)), 4),
                predicted_rate=round(float(fill_rate[b]), 4),
                measured_rate=round(float(rate_measured[a]), 4),
                bank_received=round(float(received_belief[a]), 4),
                source_belief=round(float(source_belief[a]), 4),
                discharge_factor=round(float(numpy.exp(dis[a])), 3),
                repose_correction=round(float(rep[a]), 1),
                hypothesis=act[a],
            )
        )
for name, rows in [("pours", pours), ("turnbacks", turns), ("series", series)]:
    (out / f"{name}.json").write_text(json.dumps(rows))
    print(name, len(rows), (out / f"{name}.json").stat().st_size)
