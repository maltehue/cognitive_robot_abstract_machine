"""
V-a: would a memory across pours have been justified?

1. End-of-pour concept values per pour (bank trace), per substance in pour order.
2. The three-pours write-back rule replayed offline over each substance's sequence.
3. Frozen 1.5 s forecasts at pour start (origins before the bank's first commit) under
   the uncalibrated default, a memory from the earlier pours of the same substance, and
   the corrections the controller actually used; share of each pour before first commit.

Run from the repository root with /home/malte/venvs/cram2/bin/python.
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
    OUT,
    FrozenForecast,
    balanced,
    concept_values,
    load_pours,
    point_from,
    score,
    write_json,
)
from experiments.pouring_model_verification.hypothesis_bank import (  # noqa: E402
    DrainRateSpread,
    OnsetSpread,
)
from experiments.pouring_model_verification.recovery import (  # noqa: E402
    WRITE_BACK_RUN,
    OnsetWriteBack,
)

REFERENCE = Path(
    "/home/malte/libs/cognitive_robot_abstract_machine/experiments/results/pouring_model_verification/reference_configuration.json"
)
EARLY_WINDOW = 5.0
"""
A fixed early window for the pour-start forecasts, in seconds.
"""

# %% 1. end-of-pour concept values (coin_common.concept_values)


def summarise(values: list, key: str) -> dict:
    data = numpy.array([v[key] for v in values if v[key] is not None], dtype=float)
    if len(data) == 0:
        return dict(n=0)
    return dict(
        n=int(len(data)),
        mean=float(data.mean()),
        median=float(numpy.median(data)),
        std=float(data.std(ddof=1)) if len(data) > 1 else math.nan,
        mad=float(numpy.median(numpy.abs(data - numpy.median(data)))),
        min=float(data.min()),
        max=float(data.max()),
    )


# %% 2. the write-back rule


def generic_write_back(spread, estimates: list) -> dict:
    """
    The rule of recovery.OnsetWriteBack.after, applied to any ParameterSpread (the
    repository class is bound to OnsetSpread; the discharge factor scatters by factors,
    so it is replayed with DrainRateSpread's logarithmic scale).

    Every pour is treated as
    explained: the consistency check is skipped.
    """
    run: list = []
    log = []
    fired = None
    centre, spread_now = spread.centre, spread.spread
    current = type(spread)(centre=centre, spread=spread_now)
    for index, estimate in enumerate(estimates):
        side = current.side_of(estimate)
        if side == 0:
            run = []
        else:
            if run and current.side_of(run[0]) != side:
                run = []
            run.append(estimate)
        log.append(
            dict(pour_index=index + 1, estimate=estimate, side=side, run=len(run))
        )
        if len(run) >= WRITE_BACK_RUN and fired is None:
            scaled = current._scaled(numpy.array(run))
            new_centre = float(current._unscaled(float(numpy.mean(scaled))))
            new_spread = max(spread_now, float(numpy.std(scaled, ddof=1)))
            fired = dict(after_pour=index + 1, centre=new_centre, spread=new_spread)
            current = type(spread)(centre=new_centre, spread=new_spread)
            run = []
    return dict(
        initial=dict(
            centre=spread.centre, spread=spread.spread, interval=spread.interval
        ),
        log=log,
        fired=fired,
    )


def onset_write_back(centre: float, spread: float, estimates: list) -> dict:
    """
    The repository's OnsetWriteBack replayed, every pour taken as explained.
    """
    state = OnsetWriteBack(onset=OnsetSpread(centre=centre, spread=spread))
    log = []
    fired = None
    for index, estimate in enumerate(estimates):
        side = state.onset.side_of(estimate)
        state = state.after(estimate, explained=True)
        log.append(
            dict(
                pour_index=index + 1,
                estimate=estimate,
                side=side,
                run=len(state.run),
                written=state.written,
            )
        )
        if state.written and fired is None:
            fired = dict(
                after_pour=index + 1,
                centre=state.onset.centre,
                spread=state.onset.spread,
            )
    return dict(
        initial=dict(
            centre=centre,
            spread=spread,
            interval=OnsetSpread(centre=centre, spread=spread).interval,
        ),
        log=log,
        fired=fired,
    )


# %% main

pours = load_pours()
values = [concept_values(p) for p in pours]
by_substance = {
    s: [v for v in values if v["substance"] == s] for s in ("corn", "styrofoam")
}

with (OUT / "v_a_concept_values.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(values[0]))
    writer.writeheader()
    writer.writerows(values)

consistency = {
    s: {
        key: summarise(vs, key)
        for key in (
            "repose_end_deg",
            "repose_quiescent_deg",
            "discharge_factor_last_aim",
            "log_discharge_last_aim",
            "first_lead_s",
            "first_commit_s",
            "share_before_commit",
        )
    }
    for s, vs in by_substance.items()
}

reference = json.loads(REFERENCE.read_text())
sim_onset_spread = reference["twin"]["cup"]["repose_angle_dynamic"]["spread"]

write_back = {}
for s, vs in by_substance.items():
    results = {}
    for variant in ("repose_end_rad", "repose_quiescent_rad"):
        estimates = [v[variant] for v in vs]
        scatter = float(numpy.std(estimates, ddof=1))
        results[variant] = dict(
            sim_spread=onset_write_back(estimates[0], sim_onset_spread, estimates),
            scatter_spread=onset_write_back(estimates[0], scatter, estimates),
            scatter_spread_value=scatter,
        )
    factors = [
        v["discharge_factor_last_aim"]
        for v in vs
        if v["discharge_factor_last_aim"] is not None
    ]
    log_scatter = float(numpy.std(numpy.log(factors), ddof=1))
    results["discharge_factor_last_aim"] = dict(
        scatter_spread=generic_write_back(
            DrainRateSpread(centre=factors[0], spread=log_scatter), factors
        ),
        scatter_spread_value=log_scatter,
        pours_with_aim_value=len(factors),
        note="pours that never committed to the aim hypothesis carry no aim value and are skipped",
    )
    write_back[s] = results

# %% 3. forecasts at pour start


def memory_before(index: int, same: list, key: str):
    earlier = [v[key] for v in same[:index] if v[key] is not None]
    return None if not earlier else float(numpy.median(earlier))


def rule_centre(index: int, fired, estimates: list):
    """
    The write-back rule's centre in force before pour ``index`` (1-based list index).
    """
    if fired is not None and index > fired["after_pour"]:
        return fired["centre"]
    return estimates[0]


forecast_rows = []
start_summary = {}
for s, vs in by_substance.items():
    same = [p for p in pours if p.substance == s]
    end_reps = [v["repose_end_rad"] for v in vs]
    quiet_reps = [v["repose_quiescent_rad"] for v in vs]
    fired_end = write_back[s]["repose_end_rad"]["sim_spread"]["fired"]
    for index, (pour, value) in enumerate(zip(same, vs)):
        if index == 0:
            continue
        rep_mem = memory_before(index, vs, "repose_end_rad")
        rep_mem_q = memory_before(index, vs, "repose_quiescent_rad")
        dis_mem = memory_before(index, vs, "log_discharge_last_aim")
        if dis_mem is None:
            dis_mem = 0.0
        rep_rule = rule_centre(index + 1, fired_end, end_reps)
        models = {
            "default": lambda i, p=pour: point_from(0.0, 0.0, float(p.log_drain[i])),
            "memory_end": lambda i, p=pour, r=rep_mem, d=dis_mem: point_from(
                r, d, float(p.log_drain[i])
            ),
            "memory_quiescent": lambda i, p=pour, r=rep_mem_q, d=dis_mem: point_from(
                r, d, float(p.log_drain[i])
            ),
            "memory_rule": lambda i, p=pour, r=rep_rule, d=dis_mem: point_from(
                r, d, float(p.log_drain[i])
            ),
            "as_used": lambda i, p=pour: FrozenForecast.corrected(
                BASE, p.corrections_at(i)
            ),
        }
        commit = pour.first_commit
        limit = pour.duration if commit is None else commit
        rows = score(
            pour,
            models,
            horizons=(1.5,),
            origin_filter=lambda p, i, limit=limit: p.t[i] < limit,
        )
        for r in rows:
            r["window"] = "before_first_commit"
        early = score(
            pour,
            models,
            horizons=(1.5,),
            origin_filter=lambda p, i: p.t[i] < EARLY_WINDOW,
        )
        for r in early:
            r["window"] = "first_5s"
        forecast_rows.extend(rows + early)

for s in by_substance:
    start_summary[s] = {}
    for window in ("before_first_commit", "first_5s"):
        for delay in (0.0, 0.33):
            for flowing in (None, True, False):
                for model in (
                    "default",
                    "memory_end",
                    "memory_quiescent",
                    "memory_rule",
                    "as_used",
                ):
                    sel = [
                        r
                        for r in forecast_rows
                        if r["substance"] == s
                        and r["window"] == window
                        and r["delay"] == delay
                        and r["model"] == model
                        and (flowing is None or r["flowing"] == flowing)
                    ]
                    mae = balanced(sel)
                    bias = balanced(sel, key=lambda r: r["error"])
                    start_summary[s][
                        f"{window}|delay{delay}|flowing={flowing}|{model}"
                    ] = dict(
                        mae=mae["mean"],
                        bias=bias["mean"],
                        origins=mae["origins"],
                        pours=mae["pours"],
                    )

write_json(
    OUT / "v_a_results.json",
    dict(
        concept_values=values,
        consistency=consistency,
        sim_onset_spread_rad=sim_onset_spread,
        write_back=write_back,
        pour_start_forecasts=start_summary,
        conventions=dict(
            horizon_s=1.5,
            delays_s=[0.0, 0.33],
            origin_stride_readings=10,
            windows=dict(
                before_first_commit="origins with t < first H2 commit (whole pour where never committed)",
                first_5s="origins with t < 5 s",
            ),
            errors="receiver percentage points, mean within each pour then across pours",
            memory_end="median of earlier same-substance pours' end-of-pour repose correction and last aim-hypothesis log discharge scale (0 where no earlier pour committed to aim)",
            memory_quiescent="as memory_end but the repose correction has the material's hysteresis restored where the pour ended flowing",
            memory_rule="first pour's end repose until the write-back rule (simulation spread) fires, then its new centre; discharge as memory_end",
            drain_scale="the recorded log_drain_scale at the origin (the 0.8 stall-probability factor) for every model: no drain-rate concept was tracked in the sweep",
        ),
    ),
)
with (OUT / "v_a_forecast_rows.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(forecast_rows[0]))
    writer.writeheader()
    writer.writerows(forecast_rows)

# %% console summary
for s, vs in by_substance.items():
    print(f"== {s}")
    for v in vs:
        print(
            f"  {v['pour'][11:]} goal {v['goal']:.1f} dur {v['duration']:5.1f} end {v['end_active']} "
            f"rep {v['repose_end_deg']:5.1f} (quiescent {v['repose_quiescent_deg']:5.1f}) "
            f"dis {v['discharge_factor_last_aim']} (end {v['discharge_factor_end']}) "
            f"lead {v['first_lead_s']} commit {v['first_commit_s']} share {v['share_before_commit']}"
        )
    for key, stats in consistency[s].items():
        print(f"  {key}: {stats}")
    for variant, res in write_back[s].items():
        if variant.startswith("repose"):
            for spread_name in ("sim_spread", "scatter_spread"):
                r = res[spread_name]
                print(
                    f"  write-back {variant} {spread_name}: interval {tuple(round(math.degrees(x),1) for x in r['initial']['interval'])} deg, sides {[l['side'] for l in r['log']]}, fired {r['fired'] and dict(after_pour=r['fired']['after_pour'], centre_deg=round(math.degrees(r['fired']['centre']),1), spread_deg=round(math.degrees(r['fired']['spread']),2))}"
                )
        else:
            r = res["scatter_spread"]
            print(
                f"  write-back {variant}: interval {tuple(round(x,3) for x in r['initial']['interval'])}, sides {[l['side'] for l in r['log']]}, fired {r['fired']}"
            )
    for key, stats in start_summary[s].items():
        if "flowing=None" in key:
            print(
                f"  {key}: MAE {stats['mae']:.2f} bias {stats['bias']:+.2f} ({stats['origins']} origins, {stats['pours']} pours)"
            )
