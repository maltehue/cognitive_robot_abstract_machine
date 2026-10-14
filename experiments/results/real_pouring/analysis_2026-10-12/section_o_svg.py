import json, numpy as np

D = json.load(open("section_o_data.json"))
SER = {
    "video": ("var(--measured)", 2.4, None, "camera (tray pixels)"),
    "one angle": ("var(--s1)", 1.8, None, "one angle (ρ 27°, k 2.03)"),
    "two angles": ("var(--s2)", 1.8, None, "start 75°, stop 26° (k 1.90)"),
    "demo constants": ("var(--s4)", 1.8, "5 4", "demo constants (ρ 0°, k 0.8)"),
}


class Panel:
    def __init__(self, x, y, w, h, xr, yr):
        self.x, self.y, self.w, self.h, self.xr, self.yr = x, y, w, h, xr, yr

    def sx(self, v):
        return self.x + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def sy(self, v):
        return self.y + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def path(self, xs, ys, color, width=1.8, dash=None, opacity=1.0, title=None):
        pts = " ".join(
            f"{self.sx(a):.1f},{self.sy(b):.1f}"
            for a, b in zip(xs, ys)
            if b is not None and not np.isnan(b)
        )
        d = f' stroke-dasharray="{dash}"' if dash else ""
        t = f"<title>{title}</title>" if title else ""
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{d} stroke-opacity="{opacity}" stroke-linejoin="round">{t}</polyline>'

    def frame(
        self,
        xticks,
        yticks,
        xlabel=None,
        ylabel=None,
        yfmt="{:g}",
        xfmt="{:g}",
        show_x=True,
    ):
        s = (
            ['<g class="grid">']
            + [
                f'<line x1="{self.x}" x2="{self.x+self.w}" y1="{self.sy(v):.1f}" y2="{self.sy(v):.1f}"/>'
                for v in yticks
            ]
            + ["</g>"]
        )
        s.append(
            f'<line class="axis" x1="{self.x}" x2="{self.x+self.w}" y1="{self.y+self.h}" y2="{self.y+self.h}"/>'
        )
        s += [
            f'<text class="tick" x="{self.x-6}" y="{self.sy(v)+3.5:.1f}" text-anchor="end">{yfmt.format(v)}</text>'
            for v in yticks
        ]
        if show_x:
            s += [
                f'<text class="tick" x="{self.sx(v):.1f}" y="{self.y+self.h+14}" text-anchor="middle">{xfmt.format(v)}</text>'
                for v in xticks
            ]
        if xlabel:
            s.append(
                f'<text class="label" x="{self.x+self.w/2}" y="{self.y+self.h+30}" text-anchor="middle">{xlabel}</text>'
            )
        if ylabel:
            s.append(f'<text class="label" x="{self.x}" y="{self.y-8}">{ylabel}</text>')
        return "\n".join(s)


def svg(w, h, label, body):
    return f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{label}">\n{body}\n</svg>'


# %% figure 1: the five demo pours
def demo_figure():
    g = D["demo_grid"]
    W = 760
    parts = []
    p1 = Panel(56, 24, 680, 110, (-2, 12), (0, 100))
    p2 = Panel(56, 170, 680, 110, (-2, 12), (0, 0.8))
    p3 = Panel(56, 316, 680, 90, (-2, 12), (0, 1))
    xt = list(range(-2, 13, 2))
    parts.append(p1.frame(xt, [0, 30, 60, 90], ylabel="tilt [°]", show_x=False))
    for r in D["demos"]:
        parts.append(
            p1.path(
                g, r["tilt"], "var(--ink-2)", 1.2, opacity=0.55, title=f"run {r['run']}"
            )
        )
    parts.append(
        p2.frame(
            xt,
            [0, 0.2, 0.4, 0.6, 0.8],
            ylabel="receiver fill, share of a full source cup",
            show_x=False,
            yfmt="{:.1f}",
        )
    )
    for r in D["demos"]:
        parts.append(
            p2.path(
                g,
                r["model"],
                "var(--s4)",
                1.8,
                opacity=0.8,
                title=f"run {r['run']}: model",
            )
        )
    lo, hi = min(r["weighed"] for r in D["demos"]), max(
        r["weighed"] for r in D["demos"]
    )
    parts.append(
        f'<rect x="{p2.x}" y="{p2.sy(hi):.1f}" width="{p2.w}" height="{p2.sy(lo)-p2.sy(hi):.1f}" fill="var(--measured)" fill-opacity="0.25"><title>weighed after the pour: {lo:.2f}–{hi:.2f}</title></rect>'
    )
    parts.append(
        f'<text class="reflabel" x="{p2.x+p2.w-4}" y="{p2.sy(hi)-5:.1f}" text-anchor="end">weighed, received + spilled: {lo:.2f}–{hi:.2f}</text>'
    )
    parts.append(
        f'<text class="reflabel" x="{p2.x+p2.w-4}" y="{p2.sy(0.7)-5:.1f}" text-anchor="end" fill="var(--s4)">the model\'s fill: 0.70</text>'
    )
    parts.append(
        p3.frame(
            xt,
            [0, 0.5, 1],
            xlabel="seconds since the model's flow began",
            ylabel="grains crossing the receiver's opening (camera, normalised)",
            yfmt="{:.1f}",
        )
    )
    for r in D["demos"]:
        parts.append(
            p3.path(
                g,
                r["flow"],
                "var(--s1)",
                1.5,
                opacity=0.75,
                title=f"run {r['run']}: camera",
            )
        )
    return svg(
        W,
        450,
        "Five demo pours. The tilt rises to 91 degrees and settles at 73. The model's receiver fill rises from time zero to 0.70; the weighed result is 0.33 to 0.36. Grains reach the receiver only from about 2.5 to 6 seconds, while the tilt is above about 70 degrees.",
        "\n".join(parts),
    )


# %% figure 2: the five ramps
def ramp_figure():
    cells = []
    for r in D["ramps"]:
        T = r["t"][-1]
        p = Panel(40, 20, 250, 130, (0, T), (0, 1.05))
        q = Panel(40, 186, 250, 40, (0, T), (0, 120))
        s = [
            p.frame(
                [round(v) for v in np.linspace(0, T, 4)[:-1]],
                [0, 0.5, 1],
                ylabel="drained share",
                yfmt="{:.1f}",
                show_x=False,
            )
        ]
        for name in ("demo constants", "one angle", "two angles", "video"):
            c, w, dash, lab = SER[name]
            s.append(p.path(r["t"], r[name], c, w, dash, title=lab))
        s.append(
            f'<rect x="{p.sx(T)-5:.1f}" y="{p.sy(r["weighed"])-5:.1f}" width="10" height="10" fill="var(--ink)"><title>weighed: {r["weighed_g"]} g = {r["weighed"]:.2f}</title></rect>'
        )
        s.append(
            q.frame(
                [round(v) for v in np.linspace(0, T, 4)],
                [0, 90],
                xlabel="s",
                ylabel="tilt [°]",
                show_x=True,
            )
        )
        s.append(q.path(r["t"], r["tilt"], "var(--ink-2)", 1.2))
        fig = svg(
            310,
            262,
            f"Ramp {r['run']}: {r['rate']:g} degrees per second to {r['final_tilt']:g} degrees, hold 5 s. Weighed drained share {r['weighed']:.2f}.",
            "\n".join(s),
        )
        cells.append(
            f'<figure><div class="panel-title">{r["run"]} · {r["rate"]:g}°/s → {r["final_tilt"]:g}°, hold 5 s</div>{fig}</figure>'
        )
    return "\n".join(cells)


# %% figure 3: what stays in after a held tilt
def retention_figure():
    p = Panel(56, 24, 840, 200, (60, 115), (0, 1.05))
    s = [
        p.frame(
            list(range(60, 116, 10)),
            [0, 0.25, 0.5, 0.75, 1],
            xlabel="final tilt, reached at 10°/s and held 5 s [°]",
            ylabel="share left in the source",
            yfmt="{:.2f}",
            xfmt="{:g}°",
        )
    ]
    for name in ("demo constants", "one angle", "two angles"):
        c, w, dash, lab = SER[name]
        pts = D["retention"][name]
        s.append(
            p.path([a for a, b in pts], [b for a, b in pts], c, w, dash, title=lab)
        )
    for r in D["ramps"]:
        s.append(
            f'<circle cx="{p.sx(r["final_tilt"]):.1f}" cy="{p.sy(1-r["weighed"]):.1f}" r="5" fill="var(--ink)" stroke="var(--surface)" stroke-width="2"><title>{r["run"]}, {r["rate"]:g}°/s: {1-r["weighed"]:.2f} left</title></circle>'
        )
    for r in D["demos"]:
        s.append(
            f'<rect x="{p.sx(91.2)-4:.1f}" y="{p.sy(1-r["weighed"])-4:.1f}" width="8" height="8" fill="none" stroke="var(--ink)" stroke-width="1.5"><title>demo {r["run"]}: {1-r["weighed"]:.2f} left after a 91° peak</title></rect>'
        )
    return svg(
        920,
        270,
        "Share left in the source against the held tilt. Ramps: 0.61 to 0.68 at 90 degrees, 0.14 at 100, none at 110. The demo pours, peaking at 91 degrees, leave 0.64 to 0.67. The fitted models leave 0.55 at 90 and 0.26 at 100; the demo constants leave nothing above about 80.",
        "\n".join(s),
    )


def legend(names, extra=""):
    items = []
    for n in names:
        c, w, dash, lab = SER[n]
        items.append(
            f'<span><i class="{"dash" if dash else ""}" style="--c:{c}"></i>{lab}</span>'
        )
    return f'<div class="legend">{"".join(items)}{extra}</div>'


if __name__ == "__main__":
    json.dump(
        {
            "demo": demo_figure(),
            "ramps": ramp_figure(),
            "retention": retention_figure(),
            "legend_ramps": legend(
                ["video", "one angle", "two angles", "demo constants"],
                '<span><i class="dot" style="--c:var(--ink);border-radius:0"></i>weighed</span>',
            ),
            "legend_retention": legend(
                ["one angle", "two angles", "demo constants"],
                '<span><i class="dot" style="--c:var(--ink)"></i>ramp, weighed</span><span><i class="dot" style="--c:transparent;border:1.5px solid var(--ink);border-radius:0"></i>demo pour, at its 91° peak</span>',
            ),
        },
        open("section_o_svg.json", "w"),
    )
    print("ok", [len(v) for v in json.load(open("section_o_svg.json")).values()])
