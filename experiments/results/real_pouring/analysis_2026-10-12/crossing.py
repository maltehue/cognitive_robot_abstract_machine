"""
What the pole camera reports of the stream crossing the receiver's opening, per 10 Hz
reading, in the five demo bags.
"""

import pickle, json, glob, numpy as np, cv2

CX, CY = 816, 234  # receiver opening centre in the image (from demo_flow.py)
ROI = 90  # px around it: the opening and the rim
OPENING_Z = 0.975 + 0.16  # table top + cup height
flow = json.load(open("demo_flow.json"))
out = {}
for d in sorted(glob.glob("demo/*")):
    n = d.split("/")[1]
    o = pickle.load(open(d + "/extracted.pkl", "rb"))
    k = pickle.load(open(d + "/kin.pkl", "rb"))
    T = np.array([r["t"] for r in k["rows"]])
    it = np.array([im[0] for im in o["images"]])
    K = k["K"]
    r0 = k["rows"][0]

    def project(p):
        c = r0["map_to_cam_R"] @ p + r0["map_to_cam_p"]
        u = K @ c
        return u[:2] / u[2]

    centre = np.array([1.0, 0.099, OPENING_Z])
    m_per_px = 0.035 / np.linalg.norm(project(centre + [0.035, 0, 0]) - project(centre))

    def roi(i):
        return cv2.imdecode(
            np.frombuffer(o["images"][i][2], np.uint8), cv2.IMREAD_COLOR
        )[CY - ROI : CY + ROI, CX - ROI : CX + ROI]

    t_start = T[
        np.argmax(
            np.degrees(
                np.arccos(np.clip([r["tool_R"][2, 1] for r in k["rows"]], -1, 1))
            )
            > 1.5
        )
    ]
    ref = np.median(
        np.stack([roi(i) for i in range(len(it)) if it[i] < t_start][-15:]), axis=0
    ).astype(np.int16)
    f = flow[n]
    t0 = T[0]
    window = [
        i
        for i in range(len(it))
        if f["real_on_t"] <= it[i] - t0 <= f["real_97_t"] + 1.0
    ]
    readings = []
    for i in window[::3]:  # 30 Hz -> 10 Hz, the bank's reading rate
        im = roi(i)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        mask = (
            (hsv[..., 2] > 150)
            & (hsv[..., 1] < 70)
            & (np.abs(im.astype(np.int16) - ref).sum(2) > 80)
        ).astype(np.uint8)
        count, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
        blobs = [cents[j] for j in range(1, count) if stats[j, cv2.CC_STAT_AREA] >= 25]
        if not blobs:
            readings.append(None)
            continue
        c = np.mean(blobs, axis=0) - ROI
        readings.append((c * m_per_px).tolist())
    seen = [r for r in readings if r is not None]
    xy = np.array(seen)
    detrended = (
        xy - np.array([np.convolve(xy[:, j], np.ones(5) / 5, "same") for j in (0, 1)]).T
    )
    out[n] = dict(
        readings=len(readings),
        seen=len(seen),
        dropout=1 - len(seen) / len(readings),
        mm_per_px=1000 * m_per_px,
        spread_mm=(1000 * xy.std(axis=0)).round(1).tolist(),
        reading_to_reading_mm=(1000 * detrended[2:-2].std(axis=0)).round(1).tolist(),
    )
    print(
        n, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in out[n].items()}
    )
json.dump(out, open("crossing.json", "w"), indent=1)
