import sys, numpy, shutil
from pathlib import Path

src, bags, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
for f in sorted(src.glob("*.npz")):
    with numpy.load(f) as c:
        d = {k: c[k] for k in c.files}
    d.pop("tilts", None)
    for k in ("diffs", "times"):
        d[k] = d[k][-39:]
    numpy.savez_compressed(out / f.name, **d)
    shutil.copy(bags / f.stem / "weighed_particles.json", out / (f.stem + ".json"))
    print(f.stem, d["diffs"].shape)
