"""
V-c, step 2: mixture vs commit (COIN's expression rule) on the replayed corn pours.

For every replayed reading the bank's hypotheses have a controller point each and a
probability each. Frozen forecasts along the recorded tilt compare

as_used_recorded  the corrections the robot's bank wrote (recorded trace, nearest
reading): the committed hypothesis's point, blended and                   stall-
weighted, as the controller actually used it active_replayed   the same from the offline
replay (checks that the replay agrees) active_point      the committed hypothesis's
unblended controller point, stall-weighted mixture           the probability-weighted
mix of the hypotheses' controller points                   (repose angle, drain rate;
the discharge coefficient does not                   enter the outflow), stall-weighted,
unblended most_probable     the point of the most probable hypothesis, no commit rule

Targets are the replayed detector's wall readings at every reading (as the 2026-10-19
ablation scored them); the source estimate at the origin is the committed hypothesis's
for every model, so only the parameters differ.

python v_c_mixture_vs_commit.py <replay json ...>
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).parent))
from coin_common import (  # noqa: E402
    BASE,
    DELAYS,
    GEOMETRY,
    HORIZONS,
    OUT,
    STRIDE,
    FrozenForecast,
    ModelPoint,
    TiltTrajectory,
    balanced,
    load_pour,
    SWEEP,
)

replays = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
rows_out = []
agreement = {}

for replay in replays:
    name = replay["bag"][:19]
    rows = replay["rows"]
    stall = replay["stall_probability"]
    grains_per_receiver = replay["grains_per_receiver"]
    recorded = load_pour(SWEEP / "bags_corn" / replay["bag"])
    times = numpy.array([r["time"] for r in rows])
    tilts = numpy.array([r["tilt"] for r in rows])
    keep = numpy.r_[True, numpy.diff(times) > 0]
    rows = [r for r, k in zip(rows, keep) if k]
    times, tilts = times[keep], tilts[keep]
    trajectory = TiltTrajectory(times, tilts)
    heap = numpy.array(
        [math.nan if r["heap_fill"] is None else r["heap_fill"] for r in rows]
    )
    calibrated = numpy.array([r["wall_calibrated"] for r in rows])

    # how well the replay agrees with what the robot's bank did
    rec_index = numpy.searchsorted(recorded.t, times).clip(0, len(recorded.t) - 1)
    active_same = numpy.mean(
        [r["active"] == recorded.active[j] for r, j in zip(rows, rec_index)]
    )
    p2_diff = numpy.array(
        [
            r["hypotheses"]["H2"]["probability"] - recorded.probabilities["H2"][j]
            for r, j in zip(rows, rec_index)
        ]
    )
    rep_diff = numpy.array(
        [
            r["written"]["repose_correction"] - recorded.repose[j]
            for r, j in zip(rows, rec_index)
        ]
    )
    first_commit_replay = next((r["time"] for r in rows if r["active"] == "H2"), None)
    agreement[name] = dict(
        readings=len(rows),
        active_agreement=float(active_same),
        p_h2_mean_abs_diff=float(numpy.mean(numpy.abs(p2_diff))),
        p_h2_max_abs_diff=float(numpy.max(numpy.abs(p2_diff))),
        repose_written_mean_abs_diff_deg=float(
            math.degrees(numpy.mean(numpy.abs(rep_diff)))
        ),
        first_commit_replay=first_commit_replay,
        first_commit_recorded=recorded.first_commit,
        count_end_replay=rows[-1]["counted_grains"],
        count_end_recorded=float(recorded.count[-1]),
    )

    def target(start: float, end: float):
        if start < times[0] or end > times[-1]:
            return None
        first = max(0, int(numpy.searchsorted(times, start, side="right")) - 1)
        last = min(len(times) - 1, int(numpy.searchsorted(times, end)))
        sel = slice(first, last + 1)
        if numpy.any(numpy.isnan(heap[sel])):
            return None
        if not numpy.all(calibrated[sel] == calibrated[first]):
            return None
        return (
            float(
                numpy.interp(end, times[sel], heap[sel])
                - numpy.interp(start, times[sel], heap[sel])
            )
            * 100.0,
            bool(calibrated[first]),
        )

    def stalled(repose: float, drain: float) -> ModelPoint:
        return ModelPoint(
            repose_angle=repose,
            discharge_coefficient=BASE.discharge_coefficient,
            outflow_rate_constant=drain * (1.0 - stall),
        )

    for index in range(0, len(rows), STRIDE):
        row = rows[index]
        origin = float(times[index])
        if origin > recorded.duration:
            continue  # the bag records on after the transfer goal; the trace ends with it
        hyps = row["hypotheses"]
        probabilities = numpy.array([h["probability"] for h in hyps.values()])
        probabilities = probabilities / probabilities.sum()
        reposes = numpy.array([h["repose"] for h in hyps.values()])
        drains = numpy.array([h["drain"] for h in hyps.values()])
        most = max(hyps, key=lambda h: hyps[h]["probability"])
        j = int(rec_index[index])
        models = {
            "as_used_recorded": FrozenForecast.corrected(
                BASE, recorded.corrections_at(j)
            ),
            "active_replayed": FrozenForecast.corrected(
                BASE, type(recorded.corrections_at(j))(**row["written"])
            ),
            "active_point": stalled(
                row["active_point"]["repose"], row["active_point"]["drain"]
            ),
            "mixture": stalled(
                float(probabilities @ reposes), float(probabilities @ drains)
            ),
            "most_probable": stalled(hyps[most]["repose"], hyps[most]["drain"]),
        }
        source = min(100.0, max(0.0, float(row["source_grains"])))
        for horizon in HORIZONS:
            if origin + horizon > times[-1]:
                continue
            predictions = {
                m: FrozenForecast(GEOMETRY, point, source, grains_per_receiver).predict(
                    trajectory, origin, horizon
                )
                * 100.0
                for m, point in models.items()
            }
            for delay in DELAYS:
                t = target(origin + delay, origin + horizon + delay)
                if t is None:
                    continue
                observed, cal = t
                for m, predicted in predictions.items():
                    rows_out.append(
                        dict(
                            pour=name,
                            time=origin,
                            horizon=horizon,
                            delay=delay,
                            model=m,
                            observed=observed,
                            predicted=predicted,
                            error=predicted - observed,
                            flowing=bool(row["stream_seen"]),
                            calibrated=cal,
                            before_commit=bool(
                                first_commit_replay is None
                                or origin < first_commit_replay
                            ),
                            p_h2=float(hyps["H2"]["probability"]),
                            active=row["active"],
                            mixture_repose_deg=math.degrees(
                                float(probabilities @ reposes)
                            ),
                            active_repose_deg=math.degrees(
                                row["active_point"]["repose"]
                            ),
                        )
                    )

with (OUT / "v_c_forecast_rows.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
    writer.writeheader()
    writer.writerows(rows_out)

MODELS = (
    "as_used_recorded",
    "active_replayed",
    "active_point",
    "mixture",
    "most_probable",
)
summary = {}
for horizon in HORIZONS:
    for delay in DELAYS:
        for cut in ("all", "flowing", "before_commit", "after_commit"):
            for m in MODELS:
                sel = [
                    r
                    for r in rows_out
                    if r["horizon"] == horizon
                    and r["delay"] == delay
                    and r["model"] == m
                ]
                if cut == "flowing":
                    sel = [r for r in sel if r["flowing"]]
                elif cut == "before_commit":
                    sel = [r for r in sel if r["before_commit"]]
                elif cut == "after_commit":
                    sel = [r for r in sel if not r["before_commit"]]
                mae = balanced(sel)
                bias = balanced(sel, key=lambda r: r["error"])
                summary[f"h{horizon}|delay{delay}|{cut}|{m}"] = dict(
                    mae=mae["mean"],
                    bias=bias["mean"],
                    origins=mae["origins"],
                    pours=mae["pours"],
                    per_pour=mae["per_pour"],
                )

# paired per-pour comparison mixture vs active at the headline setting
paired = {}
for cut in ("all", "flowing", "before_commit", "after_commit"):
    a = summary[f"h1.5|delay0.33|{cut}|active_point"]["per_pour"]
    b = summary[f"h1.5|delay0.33|{cut}|mixture"]["per_pour"]
    c = summary[f"h1.5|delay0.33|{cut}|most_probable"]["per_pour"]
    common = sorted(set(a) & set(b))
    paired[cut] = dict(
        pours=len(common),
        mixture_better_than_active=sum(b[p] < a[p] for p in common),
        most_probable_better_than_active=sum(c[p] < a[p] for p in common),
        per_pour={
            p: dict(active=a[p], mixture=b[p], most_probable=c[p]) for p in common
        },
    )

json.dump(
    dict(
        agreement=agreement,
        summary=summary,
        paired_h1p5_delay0p33=paired,
        conventions=dict(
            horizons_s=list(HORIZONS),
            delays_s=list(DELAYS),
            origin_stride_readings=STRIDE,
            errors="receiver percentage points, mean within each pour then across pours",
            targets="replayed detector wall readings at every reading, intervals with a missing reading or a calibration transition excluded",
            source="the committed hypothesis's source estimate at the origin for every model",
        ),
    ),
    (OUT / "v_c_results.json").open("w"),
    indent=1,
)

for name, a in agreement.items():
    print(
        f"agreement {name}: active {a['active_agreement']:.3f}, |dP(H2)| mean {a['p_h2_mean_abs_diff']:.3f} max {a['p_h2_max_abs_diff']:.3f}, |d repose| {a['repose_written_mean_abs_diff_deg']:.2f} deg, commit replay {a['first_commit_replay']} recorded {a['first_commit_recorded']}, count end {a['count_end_replay']:.1f} vs {a['count_end_recorded']:.1f}"
    )
for horizon in HORIZONS:
    for delay in DELAYS:
        for cut in ("all", "flowing", "before_commit", "after_commit"):
            print(
                f"h{horizon} delay {delay} {cut}: "
                + " | ".join(
                    f"{m} {summary[f'h{horizon}|delay{delay}|{cut}|{m}']['mae']:.2f} ({summary[f'h{horizon}|delay{delay}|{cut}|{m}']['bias']:+.2f}, n={summary[f'h{horizon}|delay{delay}|{cut}|{m}']['origins']}/{summary[f'h{horizon}|delay{delay}|{cut}|{m}']['pours']})"
                    for m in MODELS
                )
            )
for cut, p in paired.items():
    print(
        f"paired h1.5 delay 0.33 {cut}: mixture better than active in {p['mixture_better_than_active']}/{p['pours']}, most probable better in {p['most_probable_better_than_active']}/{p['pours']}"
    )
