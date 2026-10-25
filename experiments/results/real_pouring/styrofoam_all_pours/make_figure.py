"""
All weighed styrofoam pours on Tracy in one figure: share of the source that left the
cup against peak tilt.

Sources: analysis_2026-10-14/live_pours.json (2026-10-12 demo pours, ramps via
analysis_2026-10-12/section_o_data.json, 2026-10-14 open-loop fitted and live-bank
pours), sweep_2026-10-24 (ladder_rows.csv bank rows + the two goal-0.7 bags),
ablation_2026-10-25 (ladder_rows.csv no-bank / fills-only rows; no-bank peak tilt from
analysis/tilt_series).
"""

import json, csv, glob, math, os, sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
lp = json.load(open(R + "/analysis_2026-10-14/live_pours.json"))["pours"]
for n, p in lp.items():
    mode = (
        "open loop, uncalibrated drain"
        if p["drain"] == "demo" and p["bank"] is None
        else (
            "open loop, fitted drain" if p["bank"] is None else "live bank (2026-10-14)"
        )
    )
    rows.append((mode, p["peak_tilt"], p["left_the_cup"], n))
for r in json.load(open(R + "/analysis_2026-10-12/section_o_data.json"))["ramps"]:
    rows.append(("fixed-rate ramp, held tilt", r["final_tilt"], r["weighed"], r["run"]))
lr = list(csv.DictReader(open(R + "/ablation_2026-10-25/analysis/ladder_rows.csv")))
ts = sorted(
    glob.glob(R + "/ablation_2026-10-25/analysis/tilt_series/2026-10-25_13-1*.json")
) + sorted(
    glob.glob(R + "/ablation_2026-10-25/analysis/tilt_series/2026-10-25_13-2*.json")
)
nob = [
    r for r in lr if r["cond"] == "nobank" and r["material"] == "styrofoam"
]  # ordered goal 0.6 .. 0.1, like the bags
assert len(ts) == len(nob) == 6
for r, f in zip(nob, ts):
    peak = max(x[1] for x in json.load(open(f)))
    rows.append(
        (
            "no bank: own belief (ladder)",
            peak,
            float(r["weighed"]) + float(r["spill"]),
            "nobank " + r["goal"],
        )
    )
for r in lr:
    if r["material"] != "styrofoam" or r["cond"] == "nobank":
        continue
    mode = (
        "fills only: bank silent (ladder)"
        if r["cond"] == "fillsonly"
        else "bank (sweep)"
    )
    rows.append(
        (
            mode,
            float(r["peak_tilt"]),
            float(r["weighed"]) + float(r["spill"]),
            r["cond"] + " " + r["goal"],
        )
    )
for d in sorted(glob.glob(R + "/sweep_2026-10-24/bags_styrofoam/*goal70*/")):
    tr = json.load(open(d + "live_bank_trace.json"))
    w = json.load(open(d + "weighed_particles.json"))
    rec = w.get("received_fill_level")
    sp = w.get("spilled_mass_grams", 0) / 2.65
    rows.append(
        (
            "bank (sweep)",
            math.degrees(max(x["tilt"] for x in tr)),
            rec + sp,
            os.path.basename(d.rstrip("/"))[:19],
        )
    )
with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "rows.csv"), "w"
) as f:
    w = csv.writer(f)
    w.writerow(["mode", "peak_tilt_deg", "share_out_of_source", "pour"])
    w.writerows(rows)
style = {
    "bank (sweep)": ("#3F6FA8", "o"),
    "live bank (2026-10-14)": ("#3F6FA8", "P"),
    "no bank: own belief (ladder)": ("#C27A1C", "v"),
    "open loop, uncalibrated drain": ("#C27A1C", "^"),
    "fills only: bank silent (ladder)": ("#3A8F5C", "s"),
    "open loop, fitted drain": ("#7B4FA3", "D"),
    "fixed-rate ramp, held tilt": ("#666666", "x"),
}
fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=200)
for mode, (c, m) in style.items():
    pts = [(r[1], r[2]) for r in rows if r[0] == mode]
    ax.scatter(
        [p[0] for p in pts],
        [p[1] for p in pts],
        s=34,
        c=c,
        marker=m,
        label=f"{mode} (n={len(pts)})",
        linewidths=1.2 if m in "xP" else 0.6,
        edgecolors="white" if m not in "x" else c,
        zorder=3,
    )
ax.axvspan(89, 91, color="#eeeeee", zorder=0)
ax.axvspan(99, 101, color="#eeeeee", zorder=0)
ax.annotate(
    "held 90°: 0.32–0.39 out",
    xy=(90, 0.36),
    xytext=(50, 0.55),
    fontsize=7.5,
    color="#333333",
    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
)
ax.annotate(
    "held 100°: 0.86",
    xy=(100, 0.86),
    xytext=(70, 0.92),
    fontsize=7.5,
    color="#333333",
    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
)
ax.set_xlabel("peak tilt [deg]")
ax.set_ylabel("share of the source that left the cup")
ax.set_xlim(35, 113)
ax.set_ylim(-0.03, 1.05)
ax.grid(True, color="#e6e6e6", lw=0.6, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=6.6, loc="upper left", frameon=False)
fig.tight_layout()
out = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fig_styrofoam_retention.png"
)
fig.savefig(out)
print("wrote", out, len(rows), "pours")
