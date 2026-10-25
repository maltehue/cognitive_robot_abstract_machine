"""
Per burst that the cup turned back from: count before and after the tilt peak, and how
far the tilt fell before the count stopped rising.
"""

import json, sys
from pathlib import Path
import numpy

C = 722.0 / 361.83
out = {}
for bag in sorted(Path(p) for p in sys.argv[1:]):
    s = json.loads((bag / "pour_settings.json").read_text())
    mat = s["material"]["name"].lower()
    g = 100 * (C if s["receiving_cup"]["name"] == "TRANSLUCENT_CUP" else 1)
    src_scale = C if mat == "corn" else 1  # receiver fill -> source-cup share
    tr = json.loads((bag / "live_bank_trace.json").read_text())
    t0 = tr[0]["time"]
    t = numpy.array([x["time"] - t0 for x in tr])
    tilt = numpy.degrees([x["tilt"] for x in tr])
    cnt = numpy.array([x["counted_grains"] for x in tr]) / g
    # tilt peaks: local maxima followed by a fall of >= 3 deg
    i = 1
    while i < len(t) - 1:
        if tilt[i] >= tilt[i - 1] and tilt[i] > tilt[i + 1]:
            j = i
            while j < len(t) - 1 and tilt[j + 1] <= tilt[j] + 0.05:
                j += 1
            if tilt[i] - tilt[j] >= 3:
                before = cnt[i] - cnt[max(0, numpy.searchsorted(t, t[i] - 2.0))]
                stop = i
                for k in range(i, j + 1):
                    if cnt[k] - cnt[k - 1] > 0.002:
                        stop = k
                after = cnt[j] - cnt[i]
                if before + after > 0.02:
                    out.setdefault(mat, []).append(
                        (
                            bag.name[11:19],
                            s["goal_fill"],
                            tilt[i],
                            before * src_scale,
                            after * src_scale,
                            tilt[i] - tilt[stop],
                            t[stop] - t[i],
                        )
                    )
            i = j
        i += 1
for mat, L in out.items():
    print(f"== {mat} (amounts in shares of the source cup, from the count)")
    for b, goal, pk, bf, af, drop, dur in L:
        print(
            f"   {b} goal {goal:.1f}: peak {pk:5.1f} deg, 2 s before peak +{bf:.3f}, after peak +{af:.3f}, flow stopped {drop:4.1f} deg / {dur:.1f} s after the peak"
        )
    A = numpy.array([[x[4], x[5], x[6]] for x in L])
    print(
        f"   median after peak {numpy.median(A[:,0]):.3f}, median fall to stop {numpy.median(A[:,1]):.1f} deg, {numpy.median(A[:,2]):.1f} s"
    )
