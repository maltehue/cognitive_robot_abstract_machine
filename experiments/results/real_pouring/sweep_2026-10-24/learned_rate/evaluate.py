"""
Leave-one-pour-out comparison of predictors of the next 1.5 s fill gain.
"""

import json, sys
import numpy
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

rows = json.load(open(sys.argv[1]))
pour = numpy.array([r["pour"] for r in rows])


def col(k):
    return numpy.array([r[k] for r in rows])


y = col("target")
STATE_FREE = ["tilt", "fill"]
PLANNED = ["tilt", "fill", "tilt_change"]
PAST_ONLY = [
    "tilt",
    "fill",
    "tilt_rate",
    "above_earlier_peak",
    "gain_past_short",
    "gain_past_long",
    "since_burst",
]
HISTORY = [
    "tilt",
    "fill",
    "tilt_change",
    "tilt_rate",
    "above_earlier_peak",
    "gain_past_short",
    "gain_past_long",
    "since_burst",
]
X = {k: col(k) for k in HISTORY}


def matrix(keys):
    return numpy.column_stack([X[k] for k in keys])


def lopo(make, keys):
    est = numpy.zeros_like(y)
    for p in numpy.unique(pour):
        test = pour == p
        if make is None:
            est[test] = 0.0
            continue
        m = make()
        m.fit(matrix(keys)[~test], y[~test])
        est[test] = m.predict(matrix(keys)[test])
    return est


def tree():
    return HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
    )


predictors = {
    "zero": numpy.zeros_like(y),
    "past 1.5 s gain carried on": X["gain_past_long"],
    "linear, tilt + fill + planned tilt": lopo(LinearRegression, PLANNED),
    "trees, tilt + fill (learned-head inputs)": lopo(tree, STATE_FREE),
    "trees, + planned tilt change": lopo(tree, PLANNED),
    "trees, history, no planned tilt": lopo(tree, PAST_ONLY),
    "trees, + history features": lopo(tree, HISTORY),
}
subsets = {
    "all samples": numpy.ones_like(y, dtype=bool),
    "tilt >= 85 deg (where the turn-backs happen)": X["tilt"] >= 85,
    "tilt >= 85, not turning back": (X["tilt"] >= 85) & (X["tilt_change"] >= -1.0),
}
rmse = lambda e: float(numpy.sqrt(numpy.mean(e**2)))
for name, sel in subsets.items():
    print(
        f"== {name}: {sel.sum()} samples, gain mean {y[sel].mean():.3f}, sd {y[sel].std():.3f}"
    )
    for p, est in predictors.items():
        e = est[sel] - y[sel]
        print(
            f"   {p:42s} rmse {rmse(e):.4f}  bias {e.mean():+.4f}  explained {1 - numpy.var(e) / numpy.var(y[sel]):+.2f}"
        )
numpy.save(
    sys.argv[2],
    numpy.column_stack(
        [
            predictors["trees, history, no planned tilt"],
            predictors["trees, tilt + fill (learned-head inputs)"],
            y,
        ]
    ),
)
