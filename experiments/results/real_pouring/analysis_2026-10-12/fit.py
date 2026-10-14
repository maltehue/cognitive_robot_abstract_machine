import pickle, numpy as np, json
from scipy.optimize import least_squares
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

H, R = 0.16, 0.035
FULL = 2.65
PX_PER_G = 8664
onset_res = json.load(open("onset.json"))


def tilt_series(kind, n):
    k = pickle.load(open(f"{kind}/{n}/kin.pkl", "rb"))
    T = np.array([x["t"] for x in k["rows"]])
    return T - T[0], np.radians(
        np.degrees(np.arccos(np.clip([x["tool_R"][2, 1] for x in k["rows"]], -1, 1)))
    )


def simulate(t, th, k, rho_start, rho_stop, f0=1.0):
    f = f0
    flowing = False
    out = []
    for i in range(len(t)):
        dt = t[i] - t[i - 1] if i else 0
        geo = np.arctan2(H * (1 - f), R)
        L = np.hypot(H * (1 - f), R)
        if not flowing and th[i] > geo + rho_start:
            flowing = True
        if flowing and th[i] <= geo + rho_stop:
            flowing = False
        head = max(0.0, L * np.sin(th[i] - geo - rho_stop)) if flowing else 0.0
        f = max(0.0, f - k * head / H * dt)
        out.append(f)
    return np.array(out)


ramps = []
for n, r in onset_res.items():
    if r["kind"] != "ramps":
        continue
    t, th = tilt_series("ramps", n)
    it = np.array(r["image_t"]) - 0.15
    m = (
        np.maximum.accumulate(np.convolve(r["counts"], np.ones(5) / 5, "same"))
        / PX_PER_G
        / FULL
    )
    rj = json.load(open(f"ramps/{n}/ramp.json"))
    ramps.append(
        (n, rj, t, th, it, np.minimum(m, 1), rj["weighed"]["drained_mass_grams"] / FULL)
    )
demos = []
for n, r in onset_res.items():
    if r["kind"] != "demo":
        continue
    t, th = tilt_series("demo", n)
    w = json.load(open(f"demo/{n}/weighed_particles.json"))
    demos.append(
        (n, t, th, (w["received_mass_grams"] + w["spilled_mass_grams"]) / FULL)
    )


def residuals(p, hysteresis):
    k, rs, rp = p if hysteresis else (p[0], p[1], p[1])
    res = []
    for n, rj, t, th, it, m, final in ramps:
        f = simulate(t, th, k, rs, rp)
        res += list((1 - np.interp(it, t, f)) - m)[::3]
        res += [5 * ((1 - f[-1]) - final)]
    for n, t, th, final in demos:
        f = simulate(t, th, k, rs, rp)
        res += [5 * ((1 - f[-1]) - final)]
    return np.array(res)


if __name__ == "__main__":
    fits = {}
    for hyst, p0 in [
        (False, [1.0, np.radians(40)]),
        (True, [1.0, np.radians(65), np.radians(30)]),
    ]:
        sol = least_squares(
            residuals,
            p0,
            args=(hyst,),
            bounds=([0.01] + [0] * (len(p0) - 1), [50] + [1.6] * (len(p0) - 1)),
            diff_step=1e-3,
        )
        k, rs, rp = sol.x if hyst else (sol.x[0], sol.x[1], sol.x[1])
        fits["two angles" if hyst else "one angle"] = dict(
            k=k,
            rho_start=np.degrees(rs),
            rho_stop=np.degrees(rp),
            rms=float(np.sqrt(np.mean(sol.fun**2))),
        )
        print(fits)
    fits["demo constants"] = dict(k=0.8, rho_start=0.0, rho_stop=0.0)
    json.dump(fits, open("fits.json", "w"), indent=1)
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for ax, (n, rj, t, th, it, m, final) in zip(axes[0], ramps):
        s = rj["settings"]
        ax.plot(it, m, "k", lw=2, label="video (tray px / 8664 px·g⁻¹)")
        for name, fp in fits.items():
            f = simulate(
                t, th, fp["k"], np.radians(fp["rho_start"]), np.radians(fp["rho_stop"])
            )
            ax.plot(t, 1 - f, label=name)
        ax.plot(t[-1], final, "ks", ms=9, label="weighed")
        ax.set_title(
            f"{s['run_id']} {s['tilt_rate_deg_s']:.0f}°/s → {s['final_tilt_deg']:.0f}°, hold 5 s"
        )
        ax.set_ylim(-0.02, 1.05)
        ax2 = ax.twinx()
        ax2.plot(t, np.degrees(th), color="0.7", ls=":")
        ax2.set_ylim(0, 120)
    axes[0][0].legend(fontsize=8)
    axes[0][0].set_ylabel("drained fraction")
    for ax, (n, t, th, final) in zip(axes[1], demos):
        for name, fp in fits.items():
            f = simulate(
                t, th, fp["k"], np.radians(fp["rho_start"]), np.radians(fp["rho_stop"])
            )
            ax.plot(t, 1 - f, label=name)
        ax.axhline(final, color="k", lw=2, label="weighed received+spilled")
        ax.set_title(f"demo {n[-8:]}")
        ax.set_ylim(-0.02, 1.05)
        ax2 = ax.twinx()
        ax2.plot(t, np.degrees(th), color="0.7", ls=":")
        ax2.set_ylim(0, 120)
    axes[1][0].legend(fontsize=8)
    plt.savefig("fit.png", dpi=70, bbox_inches="tight")
    for name, fp in fits.items():
        print(
            name,
            [
                round(
                    1
                    - simulate(
                        t,
                        th,
                        fp["k"],
                        np.radians(fp["rho_start"]),
                        np.radians(fp["rho_stop"]),
                    )[-1],
                    3,
                )
                for n, t, th, final in demos
            ],
            "weighed",
            [round(d[3], 3) for d in demos],
        )
