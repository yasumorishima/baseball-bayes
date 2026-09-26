"""Check that aging.stan recovers known parameters from simulated data.

1. The Kalman-filter likelihood used in aging.stan (mirrored here in numpy)
   must equal the dense multivariate-normal likelihood of each player's
   seasons, to 1e-9.
2. Data are simulated from the model with known parameters on the real data's
   structure (same players, ages, gaps and PA as the prepared training set),
   aging.stan is fitted, and the truth must fall inside the 90% posterior
   interval for at least 80% of ages 21-40 of the curve and for every scalar.

Only the structure of the real data is used; its wOBA values are not.

Usage: python simcheck.py <prepared dir> <out dir>
"""
import json
import pathlib
import sys

import numpy as np
from cmdstanpy import CmdStanModel

HERE = pathlib.Path(__file__).resolve().parent
SEED = 7
TRUTH = {"mu_entry": -0.02, "b_entry": -0.002, "sd_entry": 0.03,
         "sd_step": 0.015, "sigma_pa": 0.5, "tau": 0.01}


def true_curve(a_min, K):
    ages = np.arange(a_min, a_min + K)
    # peak at 27, gentle rise before and a steeper decline after
    cum = np.where(ages <= 27, -0.0012 * (27 - ages) ** 2, -0.0009 * (ages - 27) ** 2)
    return ages, cum


def kalman_loglik(d, cum, p, rows):
    ll, m, v = 0.0, 0.0, 1.0
    for n in rows:
        if d["is_first"][n]:
            m = p["mu_entry"] + p["b_entry"] * d["entry_age_c"][n]
            v = p["sd_entry"] ** 2
        else:
            m += cum[d["age_idx"][n] - 1] - cum[d["prev_age_idx"][n] - 1]
            v += p["sd_step"] ** 2 * d["years"][n]
        r = p["sigma_pa"] ** 2 / d["pa"][n] + p["tau"] ** 2
        ll += -0.5 * (np.log(2 * np.pi * (v + r)) + (d["y"][n] - m) ** 2 / (v + r))
        gain = v / (v + r)
        m += gain * (d["y"][n] - m)
        v *= 1 - gain
    return ll


def dense_loglik(d, cum, p, rows):
    # talent_t = entry + drift_t + sum of steps; cov from shared steps
    mean, var_state = [], []
    for i, n in enumerate(rows):
        if i == 0:
            mu = p["mu_entry"] + p["b_entry"] * d["entry_age_c"][n]
            vs = p["sd_entry"] ** 2
        else:
            mu = mean[-1] + cum[d["age_idx"][n] - 1] - cum[d["prev_age_idx"][n] - 1]
            vs = var_state[-1] + p["sd_step"] ** 2 * d["years"][n]
        mean.append(mu)
        var_state.append(vs)
    T = len(rows)
    cov = np.array([[var_state[min(i, j)] for j in range(T)] for i in range(T)])
    cov += np.diag([p["sigma_pa"] ** 2 / d["pa"][n] + p["tau"] ** 2 for n in rows])
    resid = np.array([d["y"][n] for n in rows]) - np.array(mean)
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * (T * np.log(2 * np.pi) + logdet + resid @ np.linalg.solve(cov, resid))


def players(d):
    out, cur = [], []
    for n in range(d["N"]):
        if d["is_first"][n] and cur:
            out.append(cur)
            cur = []
        cur.append(n)
    out.append(cur)
    return out


RAW = ["sd_g_raw", "tau_raw", "sd_step_raw", "sd_entry_raw", "sigma_pa_raw",
       "mu_entry_raw", "b_entry_raw"]


def divergence_report(f):
    """Where in parameter space the divergent transitions are.

    For each unconstrained coordinate the sampler sees, compare the draws
    that ended in a divergence with the rest: medians, and the difference in
    medians in units of the non-divergent sd. Scale parameters are compared
    on the log scale, which is the scale the sampler moves on. Also returns
    per-chain step size, tree-depth hits and divergences, and for the curvature modes (mode_z) the ones
    ranked by that standardized difference.
    """
    div = f.method_variables()["divergent__"].astype(bool)      # (draws, chains)
    depth = f.method_variables()["treedepth__"]
    step = f.method_variables()["stepsize__"]
    flat = div.T.reshape(-1)                                     # chain-major, as draws_pd
    rep = {"n_divergent": int(flat.sum()),
           "per_chain": [{"divergent": int(div[:, c].sum()),
                          "stepsize": float(step[0, c]),
                          "treedepth_max": int(depth[:, c].max()),
                          "at_max_treedepth": int((depth[:, c] >= f.metadata.cmdstan_config.get("max_depth", 10)).sum())}
                         for c in range(div.shape[1])]}
    if flat.sum() == 0:
        return rep

    def cmp(x):
        a, b = x[flat], x[~flat]
        sd = float(np.std(b)) or float("nan")
        return {"median_div": float(np.median(a)), "median_ok": float(np.median(b)),
                "q10_ok": float(np.quantile(b, 0.1)), "q90_ok": float(np.quantile(b, 0.9)),
                "std_diff": (float(np.median(a)) - float(np.median(b))) / sd}

    rep["scalars_log"] = {k: cmp(np.log(f.stan_variable(k).reshape(-1))) if k in RAW[:5]
                          else cmp(f.stan_variable(k).reshape(-1)) for k in RAW}
    # level/slope at the anchor, and the curvature modes; lam is ascending
    # (eigendecompose_sym), so mode j = 0 is the one the data inform least
    ls = f.stan_variable("ls_z")
    rep["ls_z"] = {"level": cmp(ls[:, 0]), "slope": cmp(ls[:, 1])}
    mz = f.stan_variable("mode_z")
    per = [dict(mode=j, **cmp(mz[:, j])) for j in range(mz.shape[1])]
    rep["mode_z_top"] = sorted(per, key=lambda r: -abs(r["std_diff"]))[:8]
    # does the divergence sit where the curvature prior is tight (small sd_g)?
    lg = np.log(f.stan_variable("sd_g_raw").reshape(-1))
    rep["div_rate_by_sd_g_quartile"] = [
        float(flat[(lg >= lo) & (lg <= hi)].mean())
        for lo, hi in zip(np.quantile(lg, [0, .25, .5, .75]), np.quantile(lg, [.25, .5, .75, 1]))]
    lt = np.log(f.stan_variable("tau_raw").reshape(-1))
    rep["div_rate_by_tau_quartile"] = [
        float(flat[(lt >= lo) & (lt <= hi)].mean())
        for lo, hi in zip(np.quantile(lt, [0, .25, .5, .75]), np.quantile(lt, [.25, .5, .75, 1]))]
    return rep


def main(prep_dir, out_dir):
    prep = pathlib.Path(prep_dir)
    d = json.loads((prep / "stan_data.json").read_text())
    a_min = json.loads((prep / "meta.json").read_text())["a_min"]
    ages, cum = true_curve(a_min, d["K"])
    rng = np.random.default_rng(SEED)

    # simulate y from the model on the real structure
    y = np.empty(d["N"])
    talent = 0.0
    for n in range(d["N"]):
        if d["is_first"][n]:
            talent = TRUTH["mu_entry"] + TRUTH["b_entry"] * d["entry_age_c"][n] \
                + TRUTH["sd_entry"] * rng.standard_normal()
        else:
            talent += cum[d["age_idx"][n] - 1] - cum[d["prev_age_idx"][n] - 1] \
                + TRUTH["sd_step"] * np.sqrt(d["years"][n]) * rng.standard_normal()
        r = TRUTH["sigma_pa"] ** 2 / d["pa"][n] + TRUTH["tau"] ** 2
        y[n] = talent + np.sqrt(r) * rng.standard_normal()
    sim = dict(d, y=y.tolist(), use_aging=1)

    # 1. Kalman == dense on multi-season players
    multi = [p for p in players(sim) if len(p) >= 3][:50]
    worst = max(abs(kalman_loglik(sim, cum, TRUTH, p) - dense_loglik(sim, cum, TRUTH, p))
                for p in multi)
    if worst > 1e-9:
        raise SystemExit(f"Kalman and dense likelihoods differ by {worst}")

    # 2. recovery
    model = CmdStanModel(stan_file=str(HERE / "aging.stan"))
    f = model.sample(data=sim, chains=4, parallel_chains=4, iter_warmup=1000,
                     iter_sampling=1000, seed=SEED, adapt_delta=0.9, metric="dense_e",
                     show_progress=False)
    draws = f.stan_variable("cum")
    rel = draws - draws[:, [list(ages).index(27)]]
    true_rel = cum - cum[list(ages).index(27)]
    sel = [i for i, a in enumerate(ages) if 21 <= a <= 40]
    lo, hi = np.quantile(rel, 0.05, axis=0), np.quantile(rel, 0.95, axis=0)
    curve_cov = float(np.mean([lo[i] <= true_rel[i] <= hi[i] for i in sel]))
    scalars = {}
    for k, t in TRUTH.items():
        v = f.stan_variable(k)
        q05, q95 = float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))
        scalars[k] = {"truth": t, "q05": q05, "q95": q95, "covered": q05 <= t <= q95}
    res = {"kalman_vs_dense_max_abs_diff": worst,
           "players_checked": len(multi),
           "divergences": int(np.sum(f.divergences)),
           "divergences_per_chain": [int(x) for x in f.divergences],
           "curve_coverage_21_40": curve_cov,
           "scalars": scalars,
           "divergence_report": divergence_report(f)}
    res["pass"] = (res["divergences"] == 0 and curve_cov >= 0.8
                   and all(s["covered"] for s in scalars.values()))
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "simcheck.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if not res["pass"]:
        raise SystemExit("simulation check failed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
