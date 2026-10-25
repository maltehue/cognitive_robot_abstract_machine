"""
V-b: value of a cue-selected memory.

Leave one pour out; the same-substance memory is the median of the other pours' end
values of that substance, the other-substance memory the median over every pour of the
other substance. Frozen forecasts along the held-out pour's recorded tilt under the
uncalibrated default, the same-substance memory, the other-substance memory (and, for
reference, the corrections the controller actually used). Novelty check: the corn
memory's error on each styrofoam pour against the corn leave-one-out errors.

Note: the frozen forecast's outflow depends on the repose angle and the drain rate only
(DrainGeometry.outflow); the discharge coefficient sets where the stream lands, not how
much leaves. A memory therefore differs from another only through its onset here.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).parent))
from coin_common import (  # noqa: E402
    BASE,
    OUT,
    FrozenForecast,
    balanced,
    concept_values,
    load_pours,
    point_from,
    score,
    write_json,
)

HEADLINE_HORIZON = 1.5
WINDOWS = {"whole_pour": None, "first_5s": 5.0, "first_10s": 10.0}


def median_memory(values: list) -> dict:
    reps = [v["repose_end_rad"] for v in values]
    reps_q = [v["repose_quiescent_rad"] for v in values]
    dis = [
        v["log_discharge_last_aim"]
        for v in values
        if v["log_discharge_last_aim"] is not None
    ]
    return dict(
        repose=float(numpy.median(reps)),
        repose_quiescent=float(numpy.median(reps_q)),
        log_discharge=float(numpy.median(dis)) if dis else 0.0,
        n=len(values),
    )


pours = load_pours()
values = {p.name: concept_values(p) for p in pours}
substances = ("corn", "styrofoam")
all_memory = {
    s: median_memory([values[p.name] for p in pours if p.substance == s])
    for s in substances
}

rows = []
loo_memories = {}
for pour in pours:
    same = [
        values[q.name]
        for q in pours
        if q.substance == pour.substance and q.name != pour.name
    ]
    other_substance = "styrofoam" if pour.substance == "corn" else "corn"
    same_memory = median_memory(same)
    other_memory = all_memory[other_substance]
    loo_memories[pour.name] = dict(same=same_memory, other=other_memory)
    models = {
        "default": lambda i, p=pour: point_from(0.0, 0.0, float(p.log_drain[i])),
        "same_memory": lambda i, p=pour, m=same_memory: point_from(
            m["repose"], m["log_discharge"], float(p.log_drain[i])
        ),
        "same_memory_quiescent": lambda i, p=pour, m=same_memory: point_from(
            m["repose_quiescent"], m["log_discharge"], float(p.log_drain[i])
        ),
        "other_memory": lambda i, p=pour, m=other_memory: point_from(
            m["repose"], m["log_discharge"], float(p.log_drain[i])
        ),
        "as_used": lambda i, p=pour: FrozenForecast.corrected(
            BASE, p.corrections_at(i)
        ),
    }
    rows.extend(score(pour, models))

with (OUT / "v_b_forecast_rows.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

# %% recording-balanced summaries

summary = {}
for s in substances:
    for window, limit in WINDOWS.items():
        for horizon in (0.5, 1.5, 3.0):
            for delay in (0.0, 0.33):
                for flowing in (None, True, False):
                    for model in (
                        "default",
                        "same_memory",
                        "same_memory_quiescent",
                        "other_memory",
                        "as_used",
                    ):
                        sel = [
                            r
                            for r in rows
                            if r["substance"] == s
                            and r["horizon"] == horizon
                            and r["delay"] == delay
                            and r["model"] == model
                            and (flowing is None or r["flowing"] == flowing)
                            and (limit is None or r["time"] < limit)
                        ]
                        mae = balanced(sel)
                        bias = balanced(sel, key=lambda r: r["error"])
                        summary[
                            f"{s}|{window}|h{horizon}|delay{delay}|flowing={flowing}|{model}"
                        ] = dict(
                            mae=mae["mean"],
                            bias=bias["mean"],
                            origins=mae["origins"],
                            pours=mae["pours"],
                            per_pour=mae["per_pour"],
                        )

# %% novelty check: corn memory on styrofoam vs corn leave-one-out errors

novelty = {}
for window, limit in WINDOWS.items():
    for delay in (0.0, 0.33):
        corn_loo = summary[
            f"corn|{window}|h{HEADLINE_HORIZON}|delay{delay}|flowing=None|same_memory"
        ]["per_pour"]
        corn_on_styro = summary[
            f"styrofoam|{window}|h{HEADLINE_HORIZON}|delay{delay}|flowing=None|other_memory"
        ]["per_pour"]
        corn_errors = numpy.array(list(corn_loo.values()))
        threshold = float(numpy.percentile(corn_errors, 95))
        maximum = float(corn_errors.max())
        flagged = {
            name: bool(error > threshold) for name, error in corn_on_styro.items()
        }
        first = min(corn_on_styro)
        novelty[f"{window}|delay{delay}"] = dict(
            corn_loo_errors=corn_loo,
            corn_loo_p95=threshold,
            corn_loo_max=maximum,
            corn_memory_on_styrofoam=corn_on_styro,
            flagged=flagged,
            first_styrofoam_pour=first,
            first_flagged=flagged[first],
            flagged_count=sum(flagged.values()),
            styrofoam_pours=len(flagged),
        )

write_json(
    OUT / "v_b_results.json",
    dict(
        memories_all=all_memory,
        memories_leave_one_out=loo_memories,
        summary=summary,
        novelty=novelty,
        conventions=dict(
            horizons_s=[0.5, 1.5, 3.0],
            headline_horizon_s=HEADLINE_HORIZON,
            delays_s=[0.0, 0.33],
            origin_stride_readings=10,
            windows=WINDOWS,
            errors="receiver percentage points, mean within each pour then across pours",
            targets="corn: wall readings admitted to the bank; styrofoam: camera crossing count",
            memories="same_memory: median end-of-pour repose correction and last aim log discharge of the other pours of the same substance; other_memory: median over all pours of the other substance; same_memory_quiescent: hysteresis restored where a pour ended flowing",
            drain_scale="the recorded log_drain_scale at the origin for every model",
        ),
    ),
)

for s in substances:
    print(f"== {s}")
    for window in WINDOWS:
        for delay in (0.0, 0.33):
            for flowing in (None, True):
                line = []
                for model in (
                    "default",
                    "same_memory",
                    "same_memory_quiescent",
                    "other_memory",
                    "as_used",
                ):
                    st = summary[
                        f"{s}|{window}|h1.5|delay{delay}|flowing={flowing}|{model}"
                    ]
                    line.append(
                        f"{model} {st['mae']:.2f} ({st['bias']:+.2f}, n={st['origins']}/{st['pours']})"
                    )
                print(
                    f"  {window} delay {delay} flowing={flowing}: " + " | ".join(line)
                )
for key, n in novelty.items():
    print(
        f"novelty {key}: corn LOO p95 {n['corn_loo_p95']:.2f} max {n['corn_loo_max']:.2f}; corn memory on styrofoam: "
        + ", ".join(
            f"{k[11:]} {v:.2f}{'*' if n['flagged'][k] else ''}"
            for k, v in n["corn_memory_on_styrofoam"].items()
        )
        + f" | first pour flagged: {n['first_flagged']}, {n['flagged_count']}/{n['styrofoam_pours']} flagged"
    )
