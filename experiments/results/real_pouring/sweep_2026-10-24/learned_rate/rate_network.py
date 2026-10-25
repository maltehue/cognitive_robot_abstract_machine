"""
Fit a smooth receiver-inflow rate r(tilt, source fill) by integrating it along each
recorded 1.5 s window; leave-one-pour-out.
"""

import json, math, sys
import numpy, torch

CAPACITY_RATIO = 722.0 / 361.83
"""
Receiver capacity in source cups: a receiver fill f took 2 f of a full source.
"""

torch.manual_seed(0)
rows = json.load(open(sys.argv[1]))
pour = numpy.array([r["pour"] for r in rows])
tilt_seq = torch.tensor(
    numpy.radians([r["tilt_sequence"] for r in rows]), dtype=torch.float32
)
fill_seq = torch.tensor([r["fill_sequence"] for r in rows], dtype=torch.float32)
source_seq = (1.0 - CAPACITY_RATIO * fill_seq).clamp(0.0, 1.0)
target = torch.tensor([r["target"] for r in rows], dtype=torch.float32)
tilt0 = numpy.array([r["tilt"] for r in rows])
change = numpy.array([r["tilt_change"] for r in rows])


class RateNetwork(torch.nn.Module):
    """
    Smooth 2->1 tanh MLP from raw (tilt[rad], source fill) to receiver inflow in fill
    per second, never negative.
    """

    def __init__(self, width=24):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, 1),
        )

    def forward(self, tilt, source):
        x = torch.stack([tilt / (math.pi / 2), source], dim=-1)
        return torch.nn.functional.softplus(self.net(x).squeeze(-1) * 10.0) / 10.0


def window_gain(model, tilts, sources):
    rates = model(tilts, sources)
    return 0.1 * (0.5 * rates[:, 0] + rates[:, 1:-1].sum(dim=1) + 0.5 * rates[:, -1])


def fit(mask, epochs=1500):
    model = RateNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    m = torch.tensor(mask)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean(
            (window_gain(model, tilt_seq[m], source_seq[m]) - target[m]) ** 2
        )
        loss.backward()
        optimizer.step()
    return model


integrated = numpy.zeros(len(rows))
held = numpy.zeros(len(rows))
for p in numpy.unique(pour):
    test = pour == p
    model = fit(~test)
    with torch.no_grad():
        integrated[test] = window_gain(model, tilt_seq[test], source_seq[test]).numpy()
        held[test] = 1.5 * model(tilt_seq[test, 0], source_seq[test, 0]).numpy()
y = target.numpy()
rmse = lambda e: float(numpy.sqrt(numpy.mean(e**2)))
for label, sel in (
    ("all", numpy.ones(len(y), bool)),
    ("tilt >= 85", tilt0 >= 85),
    ("tilt >= 85, not turning back", (tilt0 >= 85) & (change >= -1.0)),
):
    print(f"== {label}: {sel.sum()} samples, gain sd {y[sel].std():.3f}")
    for name, est in (
        ("network, along the recorded tilt", integrated),
        ("network, tilt held (controller's view)", held),
    ):
        e = est[sel] - y[sel]
        print(
            f"   {name:40s} rmse {rmse(e):.4f} bias {e.mean():+.4f} explained {1 - numpy.var(e) / numpy.var(y[sel]):+.2f}"
        )
numpy.save(sys.argv[2], numpy.column_stack([integrated, held, y]))
final = fit(numpy.ones(len(rows), bool))
torch.save(final.state_dict(), sys.argv[3])
with torch.no_grad():
    print(
        "receiver inflow (fill/s) of the network fitted on all pours, rows source fill, columns tilt 60..110 deg:"
    )
    tilts = numpy.radians(numpy.arange(60, 111, 5))
    print("        " + " ".join(f"{d:6.0f}" for d in range(60, 111, 5)))
    for s in (0.9, 0.7, 0.5, 0.3, 0.1):
        r = final(
            torch.tensor(tilts, dtype=torch.float32), torch.full((len(tilts),), s)
        ).numpy()
        print(f"  s={s:.1f} " + " ".join(f"{v:6.3f}" for v in r))
