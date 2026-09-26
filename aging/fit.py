"""Fit aging.stan twice (with and without the aging drift) and save summaries.

Usage: python fit.py <prepared dir> <out dir> [--fallback]

--fallback is the single pre-declared fallback (PREREG.md, "Validity"):
adapt_delta 0.99 and max_treedepth 12, used only if a first run fails the
validity rule, and reported next to it.
"""
import json
import pathlib
import sys
import time

import numpy as np
from cmdstanpy import CmdStanModel

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20260926
CHAINS = 4
WARMUP = 1000
SAMPLES = 1000
SCALARS = ("sd_g", "mu_entry", "b_entry", "sd_entry", "sd_step", "sigma_pa", "tau")


def valid(diag):
    return (diag["divergences"] == 0 and diag["max_treedepth_hits"] == 0
            and diag["max_rhat"] < 1.01 and diag["min_ess_bulk"] >= 400)


def fit(model, data, a_min, use_aging, out, fallback):
    d = dict(data, use_aging=use_aging)
    t0 = time.time()
    f = model.sample(data=d, chains=CHAINS, parallel_chains=CHAINS,
                     iter_warmup=WARMUP, iter_sampling=SAMPLES, seed=SEED,
                     adapt_delta=0.99 if fallback else 0.9,
                     max_treedepth=12 if fallback else 10,
                     show_progress=False)
    seconds = time.time() - t0
    s = f.summary()
    for col in ("R_hat", "ESS_bulk"):
        if col not in s.columns:
            raise SystemExit(f"summary has no {col} column: {list(s.columns)}")
    keep = [c for c in s.index if not c.startswith(("z[", "talent["))]
    diag = {
        "divergences": int(np.sum(f.divergences)),
        "max_treedepth_hits": int(np.sum(f.max_treedepths)),
        "max_rhat": float(s.loc[keep, "R_hat"].max()),
        "min_ess_bulk": float(s.loc[keep, "ESS_bulk"].min()),
        "seconds": round(seconds, 1),
        "fallback": fallback,
    }
    diag["valid"] = valid(diag)
    pred = f.stan_variable("pred")
    res = {"diagnostics": diag,
           "pred_mean": pred.mean(axis=0).tolist(),
           "pred_sd": pred.std(axis=0).tolist()}
    for p in SCALARS:
        v = f.stan_variable(p)
        res[p] = {"mean": float(v.mean()), "q05": float(np.quantile(v, 0.05)),
                  "q95": float(np.quantile(v, 0.95))}
    if use_aging:
        cum = f.stan_variable("cum")
        ages = list(range(a_min, a_min + cum.shape[1]))
        rel = cum - cum.max(axis=1, keepdims=True)
        res["ages"] = ages
        res["curve"] = {str(q): np.quantile(rel, q, axis=0).tolist() for q in (0.05, 0.5, 0.95)}
        peak = np.array(ages)[cum.argmax(axis=1)]
        res["peak_age_counts"] = {str(a): int(np.sum(peak == a)) for a in ages}
    name = "aging" if use_aging else "no_aging"
    (out / f"fit_{name}.json").write_text(json.dumps(res))
    print(name, json.dumps(diag), flush=True)


def main(prep_dir, out_dir, fallback=False):
    prep = pathlib.Path(prep_dir)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((prep / "stan_data.json").read_text())
    a_min = json.loads((prep / "meta.json").read_text())["a_min"]
    model = CmdStanModel(stan_file=str(HERE / "aging.stan"))
    fit(model, data, a_min, 1, out, fallback)
    fit(model, data, a_min, 0, out, fallback)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fallback"]
    if len(args) != 2:
        raise SystemExit(__doc__)
    main(*args, fallback="--fallback" in sys.argv)
