import pickle, numpy as np, cv2, glob, json, sys

X0, Y0, X1, Y1 = 615, 245, 1015, 485  # black tray, full-res pixels
res = {}
for d in sorted(glob.glob("demo/*") + glob.glob("ramps/*")):
    o = pickle.load(open(d + "/extracted.pkl", "rb"))
    k = pickle.load(open(d + "/kin.pkl", "rb"))
    T = np.array([r["t"] for r in k["rows"]])
    tilt = np.degrees(np.arccos(np.clip([r["tool_R"][2, 1] for r in k["rows"]], -1, 1)))
    it = np.array([im[0] for im in o["images"]])

    def roi(i):
        img = cv2.imdecode(np.frombuffer(o["images"][i][2], np.uint8), cv2.IMREAD_COLOR)
        return img[Y0:Y1, X0:X1]

    first_motion = T[np.argmax(tilt > 1.5)]
    pre = [i for i in range(len(it)) if it[i] < first_motion][-15:]
    ref = np.median(np.stack([roi(i) for i in pre]), axis=0).astype(np.int16)
    counts = []
    for i in range(len(it)):
        c = roi(i)
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        white = (hsv[..., 2] > 160) & (hsv[..., 1] < 60)
        diff = np.abs(c.astype(np.int16) - ref).sum(axis=2) > 80
        counts.append(int((white & diff).sum()))
    counts = np.array(counts)
    hit = next(
        i
        for i in range(len(counts) - 3)
        if it[i] > first_motion and all(counts[i : i + 3] > 40)
    )
    res[d.split("/")[1]] = dict(
        kind=d.split("/")[0],
        t_rel=float(it[hit] - T[0]),
        tilt=float(np.interp(it[hit], T, tilt)),
        counts=counts.tolist(),
        image_t=(it - T[0]).tolist(),
        peak=float(tilt.max()),
        tilt_rate_at_hit=float(np.interp(it[hit], T[1:], np.diff(tilt) / np.diff(T))),
    )
    print(
        d,
        f"landing {res[d.split('/')[1]]['t_rel']:.2f}s tilt {res[d.split('/')[1]]['tilt']:.1f} rate {res[d.split('/')[1]]['tilt_rate_at_hit']:.1f}",
        counts[hit - 3 : hit + 5],
    )
json.dump(res, open("onset.json", "w"))
