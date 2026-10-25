"""
Training-set fit of the rate network under different settings, against a tree on the
same inputs.
"""

import json, math, sys, time
import numpy, torch
from sklearn.ensemble import HistGradientBoostingRegressor

torch.manual_seed(0)
rows = json.load(open(sys.argv[1]))
tilt_seq = torch.tensor(
    numpy.radians([r["tilt_sequence"] for r in rows]), dtype=torch.float32
)
fill_seq = torch.tensor([r["fill_sequence"] for r in rows], dtype=torch.float32)
source_seq = (1.0 - 722.0 / 361.83 * fill_seq).clamp(0.0, 1.0)
y = torch.tensor([r["target"] for r in rows], dtype=torch.float32)
X = numpy.column_stack([[r["tilt"] for r in rows], [r["fill"] for r in rows]])
tree = HistGradientBoostingRegressor(
    max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40
).fit(X, y.numpy())
rmse = lambda e: float(numpy.sqrt(numpy.mean(numpy.asarray(e) ** 2)))
print(
    f"tree on (tilt, fill), training rmse {rmse(tree.predict(X) - y.numpy()):.4f}; sd {y.std():.4f}"
)
SCALE = 30.0


class Net(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, 1),
        )

    def forward(self, tilt, source):
        x = torch.stack([(tilt - 1.4) / 0.3, (source - 0.5) / 0.3], dim=-1)
        return torch.nn.functional.softplus(self.net(x).squeeze(-1)) / SCALE


def gain(m, t, s):
    r = m(t, s)
    return 0.1 * (0.5 * r[:, 0] + r[:, 1:-1].sum(1) + 0.5 * r[:, -1])


for width, lr, epochs in ((24, 3e-3, 1500), (32, 1e-2, 3000), (64, 1e-2, 3000)):
    m = Net(width)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    start = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((gain(m, tilt_seq, source_seq) - y) ** 2) * SCALE**2
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        e = gain(m, tilt_seq, source_seq) - y
        held = 1.5 * m(tilt_seq[:, 0], source_seq[:, 0]) - y
    print(
        f"net width {width} lr {lr} epochs {epochs}: training rmse along tilt {rmse(e):.4f}, held {rmse(held):.4f} ({time.time()-start:.0f}s)"
    )
    with torch.no_grad():
        tilts = torch.tensor(
            numpy.radians(numpy.arange(60, 111, 10)), dtype=torch.float32
        )
        for s in (0.8, 0.5, 0.2):
            print(
                f"     s={s}: "
                + " ".join(
                    f"{v:.3f}" for v in m(tilts, torch.full_like(tilts, s)).numpy()
                )
            )
