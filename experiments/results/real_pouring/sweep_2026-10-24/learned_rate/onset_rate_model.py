"""Receiver inflow r = A(s) * sigmoid((tilt - onset(s)) / width(s)), each of A, onset, width a small smooth network of the
source fill s; fitted by integrating along each recorded 1.5 s window and scored leave-one-pour-out.
"""

import json, math, sys, time
import numpy, torch

torch.manual_seed(0)
rows = json.load(open(sys.argv[1]))
pour = numpy.array([r["pour"] for r in rows])
tilt_seq = torch.tensor(
    numpy.radians([r["tilt_sequence"] for r in rows]), dtype=torch.float32
)
fill_seq = torch.tensor([r["fill_sequence"] for r in rows], dtype=torch.float32)
source_seq = (1.0 - 722.0 / 361.83 * fill_seq).clamp(0.0, 1.0)
y = torch.tensor([r["target"] for r in rows], dtype=torch.float32)
tilt0 = numpy.array([r["tilt"] for r in rows])
change = numpy.array([r["tilt_change"] for r in rows])


class OnsetRate(torch.nn.Module):
    """
    Smooth in tilt and fill, rising monotonically with tilt above a fill-dependent
    onset.
    """

    def __init__(self, width=12):
        super().__init__()
        self.of_fill = torch.nn.Sequential(
            torch.nn.Linear(1, width), torch.nn.Tanh(), torch.nn.Linear(width, 3)
        )

    def parts(self, source):
        a, o, w = self.of_fill(((source - 0.5) / 0.3).unsqueeze(-1)).unbind(-1)
        maximum = torch.nn.functional.softplus(a - 3.0) * 0.1
        onset = math.radians(80.0) + math.radians(20.0) * torch.tanh(o)
        sharpness = math.radians(3.0) + torch.nn.functional.softplus(w) * math.radians(
            5.0
        )
        return maximum, onset, sharpness

    def forward(self, tilt, source):
        maximum, onset, sharpness = self.parts(source)
        return maximum * torch.sigmoid((tilt - onset) / sharpness)


def gain(m, t, s):
    r = m(t, s)
    return 0.1 * (0.5 * r[:, 0] + r[:, 1:-1].sum(1) + 0.5 * r[:, -1])


def fit(mask, epochs=1500):
    m = OnsetRate()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    mk = torch.tensor(mask)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((gain(m, tilt_seq[mk], source_seq[mk]) - y[mk]) ** 2) * 900.0
        loss.backward()
        opt.step()
        sched.step()
    return m


start = time.time()
along = numpy.zeros(len(rows))
held = numpy.zeros(len(rows))
for p in numpy.unique(pour):
    test = pour == p
    m = fit(~test)
    with torch.no_grad():
        along[test] = gain(m, tilt_seq[test], source_seq[test]).numpy()
        held[test] = 1.5 * m(tilt_seq[test, 0], source_seq[test, 0]).numpy()
yy = y.numpy()
rmse = lambda e: float(numpy.sqrt(numpy.mean(e**2)))
print(f"leave-one-pour-out done in {time.time() - start:.0f}s")
for label, sel in (
    ("all", numpy.ones(len(yy), bool)),
    ("tilt >= 85", tilt0 >= 85),
    ("tilt >= 85, not turning back", (tilt0 >= 85) & (change >= -1.0)),
):
    print(f"== {label}: {sel.sum()} samples, gain sd {yy[sel].std():.3f}")
    for name, est in (
        ("onset model, along the recorded tilt", along),
        ("onset model, tilt held (controller's view)", held),
    ):
        e = est[sel] - yy[sel]
        print(
            f"   {name:44s} rmse {rmse(e):.4f} bias {e.mean():+.4f} explained {1 - numpy.var(e) / numpy.var(yy[sel]):+.2f}"
        )
numpy.save(sys.argv[2], numpy.column_stack([along, held, yy]))
final = fit(numpy.ones(len(rows), bool))
torch.save(final.state_dict(), sys.argv[3])
with torch.no_grad():
    print(
        "fitted on all pours: source fill -> full rate (fill/s), onset (deg), width (deg); inflow at 80/85/90/95/100 deg"
    )
    tilts = torch.tensor(numpy.radians([80, 85, 90, 95, 100]), dtype=torch.float32)
    for s in (0.9, 0.7, 0.5, 0.3):
        mx, on, w = final.parts(torch.tensor([s]))
        r = final(tilts, torch.full_like(tilts, s)).numpy()
        print(
            f"  s={s}: max {mx.item():.3f}, onset {math.degrees(on.item()):.1f}, width {math.degrees(w.item()):.1f} | "
            + " ".join(f"{v:.3f}" for v in r)
        )
