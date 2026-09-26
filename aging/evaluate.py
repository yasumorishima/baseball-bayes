"""Score the projections against the pre-registered readings.

Primary metric: PA-weighted mean absolute error (MAE) of relative wOBA over
the evaluation players; co-metric: PA-weighted root mean squared error. Each
difference gets a paired bootstrap over players.

Usage: python evaluate.py <prepared dir> <fit dir> <out dir> [--rehearsal]

--rehearsal prints only the fit diagnostics, the run time and the half-widths
of the bootstrap intervals: it never prints which method did better.
"""
import json
import pathlib
import sys

import numpy as np

SEED = 20260926
BOOT = 10000


def wmae(e, w):
    return np.sum(np.abs(e) * w, axis=-1) / np.sum(w, axis=-1)


def wrmse(e, w):
    return np.sqrt(np.sum(e ** 2 * w, axis=-1) / np.sum(w, axis=-1))


def reading(a, b, lo, hi):
    if hi < 0:
        return f"{a} is better"
    if lo > 0:
        return f"{a} is worse"
    return "indistinguishable"


def main(prep_dir, fit_dir, out_dir, rehearsal=False):
    ev = json.loads((pathlib.Path(prep_dir) / "eval.json").read_text())
    fa = json.loads((pathlib.Path(fit_dir) / "fit_aging.json").read_text())
    f0 = json.loads((pathlib.Path(fit_dir) / "fit_no_aging.json").read_text())
    y = np.array([e["target"] for e in ev])
    w = np.array([e["pa"] for e in ev], dtype=float)
    preds = {"aging": np.array(fa["pred_mean"]),
             "no_aging": np.array(f0["pred_mean"]),
             "marcel": np.array([e["marcel"] for e in ev])}
    err = {k: p - y for k, p in preds.items()}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(BOOT, len(y)))
    comps = {}
    for a, b in (("aging", "marcel"), ("aging", "no_aging")):
        for metric, fn in (("wmae", wmae), ("wrmse", wrmse)):
            d = fn(err[a][idx], w[idx]) - fn(err[b][idx], w[idx])
            lo, hi = (float(v) for v in np.quantile(d, [0.025, 0.975]))
            comps[f"{a} - {b} ({metric})"] = {
                "diff": float(fn(err[a], w) - fn(err[b], w)),
                "ci95": [lo, hi], "half_width": (hi - lo) / 2,
                "reading": reading(a, b, lo, hi)}
    diagnostics = {"aging": fa["diagnostics"], "no_aging": f0["diagnostics"]}
    if rehearsal:
        res = {"n_players": len(y), "diagnostics": diagnostics,
               "ci_half_widths": {k: v["half_width"] for k, v in comps.items()}}
    else:
        res = {"n_players": len(y), "total_pa": float(w.sum()),
               "wmae": {k: float(wmae(e, w)) for k, e in err.items()},
               "wrmse": {k: float(wrmse(e, w)) for k, e in err.items()},
               "comparisons": comps, "diagnostics": diagnostics}
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = "rehearsal.json" if rehearsal else "results.json"
    (out / name).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--rehearsal"]
    if len(args) != 3:
        raise SystemExit(__doc__)
    main(*args, rehearsal="--rehearsal" in sys.argv)
