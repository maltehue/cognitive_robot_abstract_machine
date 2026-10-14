import pickle, json, numpy as np, sys

sys.argv = ["x"]
from fit import simulate, ramps, demos, H, R, FULL, PX_PER_G  # reuses the fitted series

fits = json.load(open("fits.json"))
flow = json.load(open("demo_flow.json"))
onset = json.load(open("onset.json"))
F = {
    n: (f["k"], np.radians(f["rho_start"]), np.radians(f["rho_stop"]))
    for n, f in fits.items()
}
out = {"fits": fits}
# demo pours: time from the model's flow start
dem = []
for n, t, th, final in demos:
    k = pickle.load(open(f"demo/{n}/kin.pkl", "rb"))
    rows = k["rows"]
    rf = np.array([r.get("receiver_fill", np.nan) for r in rows])
    t0 = flow[n]["model_on_t"]
    c = np.array(onset[n]["counts"], float)
    # receiver-opening signal recomputed in demo_flow.py; re-read from its counts file
    sig = np.load(f"demo/{n}/receiver_signal.npy")
    it = np.array(onset[n]["image_t"])
    grid = np.arange(-2, 12, 0.1)
    reset = np.argmax(rf < 0.01)
    rf2 = rf.copy()
    rf2[:reset] = 0
    dem.append(
        dict(
            run=n[-8:],
            tilt=np.round(np.degrees(np.interp(grid + t0, t, th)), 1).tolist(),
            model=np.round(np.interp(grid + t0, t, rf2), 3).tolist(),
            flow=np.round(np.interp(grid + t0, it, sig / sig.max()), 3).tolist(),
            weighed=round(final, 3),
            received=json.load(open(f"demo/{n}/weighed_particles.json")),
        )
    )
out["demo_grid"] = np.round(np.arange(-2, 12, 0.1), 2).tolist()
out["demos"] = dem
# ramps
rmp = []
for n, rj, t, th, it, m, final in ramps:
    g = np.arange(0, t[-1], 0.1)
    rec = dict(
        run=rj["settings"]["run_id"],
        rate=rj["settings"]["tilt_rate_deg_s"],
        final_tilt=rj["settings"]["final_tilt_deg"],
        t=np.round(g, 2).tolist(),
        tilt=np.round(np.degrees(np.interp(g, t, th)), 1).tolist(),
        video=np.round(np.interp(g, it, m), 3).tolist(),
        weighed=round(final, 3),
        weighed_g=rj["weighed"]["drained_mass_grams"],
        tray_px=float(np.median(np.array(onset[n]["counts"])[-20:])),
        first_ball=round(
            onset[n]["tilt"] - 0.15 * rj["settings"]["tilt_rate_deg_s"], 1
        ),
    )
    for name, (k_, rs, rp) in F.items():
        rec[name] = np.round(
            np.interp(g, t, 1 - simulate(t, th, k_, rs, rp)), 3
        ).tolist()
    rmp.append(rec)
out["ramps"] = sorted(rmp, key=lambda r: r["run"])
# retention after a 10 deg/s ramp and a 5 s hold, per fit
ret = {}
for name, (k_, rs, rp) in F.items():
    pts = []
    for fin in np.arange(60, 116, 2.5):
        tt = np.arange(0, fin / 10 + 5, 0.01)
        th = np.radians(np.minimum(tt * 10, fin))
        pts.append([float(fin), round(float(simulate(tt, th, k_, rs, rp)[-1]), 3)])
    ret[name] = pts
out["retention"] = ret
json.dump(out, open("section_o_data.json", "w"))
print({r["run"]: (r["first_ball"], r["weighed"]) for r in out["ramps"]})
