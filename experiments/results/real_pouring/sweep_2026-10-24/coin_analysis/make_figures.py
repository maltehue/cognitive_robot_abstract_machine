"""
Figures for the COIN analyses from the result JSON files (v-a, v-b, and v-c where
present).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy  # noqa: E402

OUT = Path(__file__).parent
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#6250d6",
)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)

va = json.loads((OUT / "v_a_results.json").read_text())
vb = json.loads((OUT / "v_b_results.json").read_text())
MODEL_COLORS = {
    "default": MUTED,
    "memory_end": BLUE,
    "memory_quiescent": AQUA,
    "memory_rule": VIOLET,
    "as_used": ORANGE,
    "same_memory": BLUE,
    "same_memory_quiescent": AQUA,
    "other_memory": MAGENTA,
}

# %% v-a: end-of-pour concept values in pour order
fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex="col")
for col, substance in enumerate(("corn", "styrofoam")):
    values = [v for v in va["concept_values"] if v["substance"] == substance]
    x = numpy.arange(1, len(values) + 1)
    ax = axes[0, col]
    wb = va["write_back"][substance]["repose_end_rad"]["sim_spread"]
    lo, hi = (math.degrees(v) for v in wb["initial"]["interval"])
    ax.axhspan(
        lo,
        hi,
        color=BLUE,
        alpha=0.10,
        lw=0,
        label="twin interval (first pour ± 1.96 × sim spread)",
    )
    ax.plot(
        x,
        [v["repose_end_deg"] for v in values],
        "o-",
        color=BLUE,
        ms=5,
        lw=1.5,
        label="repose correction at pour end",
    )
    ax.plot(
        x,
        [v["repose_quiescent_deg"] for v in values],
        "s",
        color=AQUA,
        ms=4,
        mfc="white",
        label="hysteresis restored where ended flowing",
    )
    if wb["fired"]:
        ax.axvline(wb["fired"]["after_pour"] + 0.5, color=VIOLET, lw=1, ls="--")
        ax.text(
            wb["fired"]["after_pour"] + 0.6,
            hi + 1,
            f"rule fires → {math.degrees(wb['fired']['centre']):.1f}°",
            color=VIOLET,
            fontsize=8,
        )
    ax.set_title(f"{substance}: onset (repose) concept per pour", loc="left", color=INK)
    ax.set_ylabel("degrees")
    ax = axes[1, col]
    dis = [
        (
            v["discharge_factor_last_aim"]
            if v["discharge_factor_last_aim"] is not None
            else numpy.nan
        )
        for v in values
    ]
    ax.plot(
        x,
        dis,
        "o-",
        color=ORANGE,
        ms=5,
        lw=1.5,
        label="discharge factor, last while aim active",
    )
    never = [
        i + 1 for i, v in enumerate(values) if v["discharge_factor_last_aim"] is None
    ]
    if never:
        ax.plot(
            never,
            [1.0] * len(never),
            "x",
            color=MUTED,
            ms=6,
            label="never committed to aim (unrevised 1.0)",
        )
    ax.set_yscale("log")
    ax.set_ylabel("factor on C_d = 0.2 (log)")
    ax.set_xlabel("pour in order")
    ax.set_title(f"{substance}: aim concept per pour", loc="left", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([v["pour"][11:16] for v in values], rotation=60, fontsize=7)
axes[0, 0].legend(fontsize=7, loc="lower right", frameon=False)
axes[1, 0].legend(fontsize=7, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_va_concept_values.png", dpi=160)

# %% v-a: forecast error at pour start, and v-b leave-one-out
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, (title, source, models, key_fn) in zip(
    axes,
    [
        (
            "v-a: 1.5 s forecast at pour start (origins before first commit, shift 0.33 s)",
            va["pour_start_forecasts"],
            ["default", "memory_end", "memory_quiescent", "memory_rule", "as_used"],
            lambda s, m: f"before_first_commit|delay0.33|flowing=None|{m}",
        ),
        (
            "v-b: 1.5 s forecast along the whole held-out pour (shift 0.33 s)",
            vb["summary"],
            [
                "default",
                "same_memory",
                "same_memory_quiescent",
                "other_memory",
                "as_used",
            ],
            lambda s, m: f"{s}|whole_pour|h1.5|delay0.33|flowing=None|{m}",
        ),
    ],
):
    width = 0.8 / len(models)
    for k, m in enumerate(models):
        for i, s in enumerate(("corn", "styrofoam")):
            st = source[s][key_fn(s, m)] if s in source else source[key_fn(s, m)]
            ax.bar(
                i + (k - len(models) / 2 + 0.5) * width,
                st["mae"],
                width * 0.92,
                color=MODEL_COLORS[m],
                label=m if i == 0 else None,
            )
            ax.text(
                i + (k - len(models) / 2 + 0.5) * width,
                st["mae"] + 0.15,
                f"{st['mae']:.1f}",
                ha="center",
                fontsize=6.5,
                color=INK,
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["corn (wall target)", "styrofoam (count target)"])
    ax.set_ylabel("MAE, receiver pp (recording-balanced)")
    ax.set_title(title, loc="left", fontsize=8.5, color=INK)
    ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_va_vb_forecast_mae.png", dpi=160)

# %% v-b novelty check
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=False)
for ax, window in zip(axes, ("whole_pour", "first_10s", "first_5s")):
    n = vb["novelty"][f"{window}|delay0.33"]
    corn = list(n["corn_loo_errors"].values())
    styro = n["corn_memory_on_styrofoam"]
    ax.scatter(
        numpy.zeros(len(corn))
        + numpy.random.default_rng(0).uniform(-0.08, 0.08, len(corn)),
        corn,
        color=BLUE,
        s=18,
        label="corn, leave-one-out corn memory",
    )
    names = sorted(styro)
    ax.scatter(
        numpy.ones(len(names)) + numpy.linspace(-0.12, 0.12, len(names)),
        [styro[k] for k in names],
        color=ORANGE,
        s=18,
        label="styrofoam, corn memory",
    )
    first = n["first_styrofoam_pour"]
    ax.scatter(
        [1 - 0.12],
        [styro[first]],
        s=70,
        facecolors="none",
        edgecolors=INK,
        lw=1.2,
        label="first styrofoam pour",
    )
    ax.axhline(n["corn_loo_p95"], color=MUTED, lw=1, ls="--")
    ax.text(
        0.5,
        n["corn_loo_p95"],
        f" p95 of corn = {n['corn_loo_p95']:.2f}",
        fontsize=7,
        color=MUTED,
        va="bottom",
        ha="center",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["corn LOO", "styrofoam"])
    ax.set_title(
        f"{window.replace('_', ' ')}: {n['flagged_count']}/{n['styrofoam_pours']} flagged",
        loc="left",
        fontsize=8.5,
        color=INK,
    )
    ax.set_ylabel("MAE, 1.5 s, receiver pp")
axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
fig.suptitle(
    "Novelty check: corn memory's forecast error on styrofoam vs its leave-one-out error on corn (shift 0.33 s)",
    fontsize=9,
    color=INK,
)
fig.tight_layout()
fig.savefig(OUT / "fig_vb_novelty.png", dpi=160)

# %% v-c
vc_path = OUT / "v_c_results.json"
if vc_path.exists():
    vc = json.loads(vc_path.read_text())
    models = ["as_used_recorded", "active_point", "mixture", "most_probable"]
    colors = {
        "as_used_recorded": ORANGE,
        "active_point": YELLOW,
        "mixture": BLUE,
        "most_probable": AQUA,
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    cuts = ["all", "flowing", "before_commit", "after_commit"]
    width = 0.8 / len(models)
    for k, m in enumerate(models):
        for i, cut in enumerate(cuts):
            st = vc["summary"][f"h1.5|delay0.33|{cut}|{m}"]
            axes[0].bar(
                i + (k - len(models) / 2 + 0.5) * width,
                st["mae"],
                width * 0.92,
                color=colors[m],
                label=m if i == 0 else None,
            )
            axes[0].text(
                i + (k - len(models) / 2 + 0.5) * width,
                st["mae"] + 0.05,
                f"{st['mae']:.2f}",
                ha="center",
                fontsize=6,
                color=INK,
            )
    axes[0].set_xticks(range(len(cuts)))
    axes[0].set_xticklabels(
        [
            f"{c}\n(n={vc['summary'][f'h1.5|delay0.33|{c}|mixture']['origins']} origins)"
            for c in cuts
        ],
        fontsize=7,
    )
    axes[0].set_ylabel("MAE, receiver pp (recording-balanced)")
    axes[0].set_title(
        "v-c corn: committed point vs probability-weighted mix, 1.5 s, shift 0.33 s",
        loc="left",
        fontsize=8.5,
        color=INK,
    )
    axes[0].legend(fontsize=7, frameon=False)
    paired = vc["paired_h1p5_delay0p33"]["all"]["per_pour"]
    names = sorted(paired)
    a = [paired[p]["active"] for p in names]
    b = [paired[p]["mixture"] for p in names]
    axes[1].scatter(a, b, color=BLUE, s=22)
    lim = max(max(a), max(b)) * 1.1
    axes[1].plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
    for p, x, y in zip(names, a, b):
        axes[1].annotate(
            p[11:16],
            (x, y),
            fontsize=6,
            color=MUTED,
            xytext=(3, 2),
            textcoords="offset points",
        )
    axes[1].set_xlabel("committed hypothesis point, MAE pp")
    axes[1].set_ylabel("probability-weighted mix, MAE pp")
    axes[1].set_title("per corn pour, all origins", loc="left", fontsize=8.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_vc_mixture_vs_commit.png", dpi=160)
print("figures written")
