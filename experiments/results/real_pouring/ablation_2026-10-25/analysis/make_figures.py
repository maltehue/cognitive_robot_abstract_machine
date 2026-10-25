"""
Figures: (1) weighed fill against goal for the three feedback conditions; (2) goal-0.3
traces per condition and substance.
"""

import json, glob, sys
from pathlib import Path
import numpy, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
sweep = Path(sys.argv[2])
out = root / "analysis"
COL = {"corn": "#C27A1C", "styrofoam": "#3F6FA8"}
INK = "#333333"
rows = json.load(open(out / "ladder_rows.json"))
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#999999",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)
# --- figure 1: ladder scatter
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
MARK = {
    "nobank": ("v", "none", "no bank (own belief)"),
    "fillsonly": ("s", "none", "fills only (bank silent)"),
    "bank": ("o", "fill", "fills + bank"),
}
for ax, mat in zip(axes, ["corn", "styrofoam"]):
    ax.fill_between(
        [0, 0.7], [-0.05, 0.65], [0.05, 0.75], color="#000000", alpha=0.05, lw=0
    )
    ax.plot([0, 0.7], [0, 0.7], color="#999999", lw=1, ls="--")
    for cond, (m, f, label) in MARK.items():
        rs = [r for r in rows if r["material"] == mat and r["cond"] == cond]
        ax.scatter(
            [r["goal"] for r in rs],
            [r["weighed"] for r in rs],
            marker=m,
            s=46,
            facecolors=COL[mat] if f == "fill" else "white",
            edgecolors=COL[mat],
            linewidths=1.4,
            label=label,
            zorder=3,
        )
        if cond == "fillsonly":
            st = [r for r in rs if r["ending"] != "goal_reached"]
            ax.scatter(
                [r["goal"] for r in st],
                [r["weighed"] for r in st],
                marker="x",
                s=22,
                color=INK,
                linewidths=0.9,
                zorder=4,
                label="ended by operator or time limit" if mat == "corn" else None,
            )
    ax.set_xlim(0.05, 0.65)
    ax.set_ylim(-0.02, 0.9)
    ax.set_xlabel("goal (share of receiver)")
    ax.set_title(mat, color=INK, loc="left", fontsize=10)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("weighed fill")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(out / "fig_ladder_scatter.png", dpi=200)
print("scatter written")


# --- figure 2: traces at goal 0.3
def trace_of(folder):
    tr = json.loads((folder / "live_bank_trace.json").read_text())
    t0 = tr[0]["time"]
    return (
        numpy.array([x["time"] - t0 for x in tr]),
        numpy.degrees([x["tilt"] for x in tr]),
        numpy.array([x["counted_grains"] for x in tr]),
        numpy.minimum(numpy.array([x["told_grains"] for x in tr]), 1e9),
    )


def per_source(folder):
    s = json.loads((folder / "pour_settings.json").read_text())
    return (
        722.0 / 361.83 if s["receiving_cup"]["name"] == "TRANSLUCENT_CUP" else 1.0
    ) * 100


fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), sharey="row")
for i, mat in enumerate(["corn", "styrofoam"]):
    tag = "corn_translucent" if mat == "corn" else "styrofoam_jeroen"
    nb = Path(glob.glob(str(root / f"goalSweep-*noBank/*{tag}*goal30*/"))[0])
    fo = Path(glob.glob(str(root / f"goalSweep-*fillsonly/*{tag}*goal30*/"))[0])
    bk = Path(sorted(glob.glob(str(sweep / f"bags_{mat}/*goal30*/")))[0])
    # no bank: tilt from tf (relative to bag start), belief from control cycles aligned so the goal crossing sits at the tilt peak
    tf = numpy.array(
        json.loads((out / "tilt_series" / (nb.name[:19] + ".json")).read_text())
    )
    tt, tilt = tf[:, 0] - tf[0, 0], tf[:, 1]
    cc = numpy.load(nb / "transfer_control_cycles.npz", allow_pickle=True)
    m = json.loads(str(cc["metadata"]))
    ir = m["world_degree_of_freedom_names"].index(
        "receiving_cup_T_receiving_cup_fill_level_phantom/dof"
    )
    ct, bel, pred = cc["times"], cc["world_positions"][:, ir], cc["debug_values"][:, 0]
    tpeak = tt[numpy.argmax(tilt)]
    cross = ct[numpy.argmax(pred >= 0.3)]
    shift = tpeak - cross
    w = json.loads((nb / "weighed_particles.json").read_text())["received_fill_level"]
    for j, (title, T, TL, series) in enumerate(
        [
            (
                "no bank: the controller's own belief",
                tt,
                tilt,
                [("believed fill", ct + shift, bel, "-")],
            ),
            (
                "fills only: told the camera's fill",
                *trace_of(fo)[:2],
                [
                    (
                        "camera count",
                        trace_of(fo)[0],
                        trace_of(fo)[2] / per_source(fo),
                        "-",
                    )
                ],
            ),
            (
                "fills + bank (sweep 2026-10-24)",
                *trace_of(bk)[:2],
                [
                    (
                        "camera count",
                        trace_of(bk)[0],
                        trace_of(bk)[2] / per_source(bk),
                        "-",
                    )
                ],
            ),
        ]
    ):
        ax = axes[i, j]
        ax2 = ax.twinx() if False else None
        folder = [nb, fo, bk][j]
        wf = json.loads((folder / "weighed_particles.json").read_text())[
            "received_fill_level"
        ]
        ax.plot(T, TL / 100, color="#999999", lw=1.2, label="tilt / 100°")
        for name, x, y, ls in series:
            ax.plot(x, y, color=COL[mat], lw=1.8, ls=ls, label=name)
        ax.axhline(0.3, color=INK, lw=0.8, ls=":", label="goal")
        ax.plot([T[-1]], [wf], marker="D", color=INK, ms=5, ls="none", label="weighed")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, max(T[-1], 45) if j else T[-1] + 1)
        ax.grid(color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        if i == 1:
            ax.set_xlabel("time (s)")
        if j == 0:
            ax.set_ylabel(f"{mat}: fill (share), tilt/100°")
axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
axes[0, 2].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(out / "fig_ladder_traces_goal30.png", dpi=200)
print("traces written")
