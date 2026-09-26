"""Fit aging.stan twice (with and without the aging drift) and save summaries.

Usage: python fit.py <prepared dir> <out dir>

If either first fit fails the validity rule, both are fitted once more with
the single pre-declared fallback (PREREG.md, "Validity": adapt_delta 0.99,
max_treedepth 12) in the same job, before anything is scored. selection.json
names the pair that is read: the first pair if both of its fits are valid,
otherwise the fallback pair. Both pairs are kept and reported.
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
            and diag["max_rhat"] < 1.01 and diag["min_ess_bulk"] >= 400
            and diag["nan_diagnostic_varying"] == 0)


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
    # Every quantity in the summary counts. R-hat and ESS are NaN only for a
    # quantity that does not vary across draws (e.g. the zero anchor slots of
    # dz); any other NaN makes the fit invalid rather than being skipped.
    if "StdDev" not in s.columns:
        raise SystemExit(f"summary has no StdDev column: {list(s.columns)}")
    nan = s["R_hat"].isna() | s["ESS_bulk"].isna()
    const = s["StdDev"].fillna(0) == 0
    keep = s.index[~nan]
    diag = {
        "divergences": int(np.sum(f.divergences)),
        "divergences_per_chain": [int(x) for x in f.divergences],
        "max_treedepth_hits": int(np.sum(f.max_treedepths)),
        "max_rhat": float(s.loc[keep, "R_hat"].max()),
        "min_ess_bulk": float(s.loc[keep, "ESS_bulk"].min()),
        "nan_diagnostic_constant": int((nan & const).sum()),
        "nan_diagnostic_varying": int((nan & ~const).sum()),
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
        # peak age is read over 21-40 only (PREREG); how often the maximum
        # over all ages falls outside that range is reported next to it
        inside = [i for i, a in enumerate(ages) if 21 <= a <= 40]
        peak = np.array(ages)[inside][cum[:, inside].argmax(axis=1)]
        res["peak_age_counts_21_40"] = {str(ages[i]): int(np.sum(peak == ages[i])) for i in inside}
        res["share_global_peak_outside_21_40"] = float(np.mean(
            [not (21 <= ages[k] <= 40) for k in cum.argmax(axis=1)]))
    name = ("aging" if use_aging else "no_aging") + ("_fallback" if fallback else "")
    (out / f"fit_{name}.json").write_text(json.dumps(res))
    print(name, json.dumps(diag), flush=True)
    return diag["valid"]


def main(prep_dir, out_dir):
    prep = pathlib.Path(prep_dir)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((prep / "stan_data.json").read_text())
    a_min = json.loads((prep / "meta.json").read_text())["a_min"]
    model = CmdStanModel(stan_file=str(HERE / "aging.stan"))
    first = [fit(model, data, a_min, u, out, False) for u in (1, 0)]
    if all(first):
        sel = {"read": "first", "fallback_run": False}
    else:
        [fit(model, data, a_min, u, out, True) for u in (1, 0)]
        sel = {"read": "fallback", "fallback_run": True}
    (out / "selection.json").write_text(json.dumps(sel))
    print("selection", json.dumps(sel), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
