import sys, numpy
from pathlib import Path

for f in sorted(Path(sys.argv[1]).glob("*.npz")):
    c = numpy.load(f)
    d = c["diffs"].astype(float)
    reg = c["region"]
    t = c["times"]
    tilt = c["tilts"]
    # end of transfer: last frame with tilt > 70 ; final: last 30 frames
    i_end = numpy.where(tilt > 70)[0][-1]
    a = d[i_end - 9 : i_end + 1].min(axis=0)
    b = d[-10:].min(axis=0)

    def prof(x):
        return numpy.array(
            [
                numpy.median(x[r][reg[r]]) if reg[r].any() else numpy.nan
                for r in range(x.shape[0])
            ]
        )

    pa, pb = prof(a), prof(b)
    rows = numpy.where(reg.any(axis=1))[0]
    print(f"== {f.stem} end-of-transfer t={t[i_end]:.1f} tilt {tilt[i_end]:.0f}")
    for r0 in range(rows[0], rows[-1], 12):
        sl = slice(r0, r0 + 12)
        print(
            f"  rows {r0:3d}-{r0+11:3d}  transfer {numpy.nanmean(pa[sl]):6.1f}  upright {numpy.nanmean(pb[sl]):6.1f}  diff {numpy.nanmean(pa[sl]-pb[sl]):+5.1f}"
        )
