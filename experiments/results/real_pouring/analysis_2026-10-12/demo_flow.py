import pickle, numpy as np, cv2, glob, json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CX, CY, R = 816, 234, 48
fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
out = {}
for d in sorted(glob.glob("demo/*")):
    n = d.split("/")[1]
    o = pickle.load(open(d + "/extracted.pkl", "rb"))
    k = pickle.load(open(d + "/kin.pkl", "rb"))
    T = np.array([r["t"] for r in k["rows"]])
    it = np.array([im[0] for im in o["images"]])
    tilt = np.degrees(np.arccos(np.clip([r["tool_R"][2, 1] for r in k["rows"]], -1, 1)))
    rf = np.array([r.get("receiver_fill", np.nan) for r in k["rows"]])
    yy, xx = np.mgrid[CY - R : CY + R, CX - R : CX + R]
    disk = (xx - CX) ** 2 + (yy - CY) ** 2 < R * R

    def roi(i):
        return cv2.imdecode(
            np.frombuffer(o["images"][i][2], np.uint8), cv2.IMREAD_COLOR
        )[CY - R : CY + R, CX - R : CX + R]

    t_start = T[np.argmax(tilt > 1.5)]
    ref = np.median(
        np.stack([roi(i) for i in range(len(it)) if it[i] < t_start][-15:]), axis=0
    ).astype(np.int16)
    c = []
    for i in range(len(it)):
        im = roi(i)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        m = (
            (hsv[..., 2] > 150)
            & (hsv[..., 1] < 70)
            & (np.abs(im.astype(np.int16) - ref).sum(2) > 80)
            & disk
        )
        c.append(int(m.sum()))
    c = np.array(c)
    tr = it - T[0]
    # model start: first time the phantom fill rises after its reset
    reset = np.argmax(rf < 0.01)
    model_on = T[reset + np.argmax(rf[reset:] > 0.005)] - T[0]
    model_done = T[reset + np.argmax(rf[reset:] > 0.69)] - T[0]
    on = next(
        i for i in range(len(c) - 5) if tr[i] > model_on - 2 and all(c[i : i + 5] > 30)
    )
    smooth = np.convolve(c, np.ones(15) / 15, "same")
    np.save(d + "/receiver_signal.npy", smooth)
    np.save(d + "/receiver_signal_raw.npy", c.astype(float))
    end = np.where(smooth > 0.97 * smooth.max())[0][0]
    w = json.load(open(d + "/weighed_particles.json"))
    out[n] = dict(
        model_on_t=model_on,
        model_on_tilt=float(np.interp(model_on + T[0], T, tilt)),
        model_done_t=model_done,
        real_on_t=float(tr[on]),
        real_on_tilt=float(np.interp(it[on], T, tilt)),
        real_97_t=float(tr[end]),
        peak_tilt=float(tilt.max()),
        peak_t=float(T[np.argmax(tilt)] - T[0]),
        hold_tilt=float(
            np.median(tilt[(T - T[0] > model_done + 1) & (T - T[0] < model_done + 3)])
        ),
        received_g=w["received_mass_grams"],
        spilled_g=w["spilled_mass_grams"],
        received_fill=w["received_fill_level"],
        tool_x=float(k["rows"][0]["tool_p"][0]),
    )
    t0 = model_on
    ax[0].plot(T - T[0] - t0, tilt, label=n[-8:])
    ax[1].plot(T - T[0] - t0, rf)
    ax[2].plot(tr - t0, smooth / smooth.max())
    print(n, {a: round(b, 2) for a, b in out[n].items()})
ax[1].axhline(0.33, ls="--", c="k")
ax[1].text(0, 0.35, "weighed (mean 0.33)")
ax[0].set_ylabel("tilt [deg]")
ax[1].set_ylabel("model receiver fill")
ax[2].set_ylabel("white px in receiver (norm.)")
ax[2].set_xlabel("s since model flow began")
ax[0].legend()
plt.savefig("demo_flow.png", dpi=80, bbox_inches="tight")
json.dump(out, open("demo_flow.json", "w"), indent=1)
