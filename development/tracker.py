"""Development tracker: where each batter's season landed against the line
the aging model expected for him, in standard deviations.

For every evaluation batter of the pre-registered 2025 test, the Kalman
filter of aging.stan is run through his seasons up to 2024 with the fitted
parameters (posterior means; curve = posterior median), then projected to
his 2025 age. That gives an expected 2025 relative wOBA and its predictive
sd: talent uncertainty after his history, the random steps for the seasons
since, the season-only deviation tau, sampling noise sigma_pa / sqrt(PA), and
the posterior sd of the projection itself. z = (actual - expected) / sd.

The parameters never saw 2025, so the z of 2025 are out of sample, and a
tracker that means what it says must be calibrated there. Declared before
the first run of this script (2026-09-26), on all 410 batters:
  PASS if the share with |z| > 1.645 is within [0.07, 0.13] (nominal 0.10)
  and the sd of z is within [0.90, 1.10].
The same two numbers are also reported for ages <= 25 (where development
tracking matters), with no pass rule.

Plug-in parameters ignore their own posterior uncertainty except through
the projection's posterior sd; the curve enters only through differences.

Usage: python tracker.py <prepared dir (prepare.py, 2024 / 2025)> <final run dir> <out dir>
"""
import csv
import json
import pathlib
import sys

import numpy as np

Z90 = 1.6448536269514722


def main(prep_dir, run_dir, out_dir):
    prep, run, out = pathlib.Path(prep_dir), pathlib.Path(run_dir), pathlib.Path(out_dir)
    d = json.loads((prep / "stan_data.json").read_text())
    ev = json.loads((prep / "eval.json").read_text())
    if (prep / "eval.json").read_bytes() != (run / "eval.json").read_bytes():
        raise SystemExit("prepared eval.json differs from the final run's")
    f = json.loads((run / "fit_aging.json").read_text())
    p = {k: f[k]["mean"] for k in ("mu_entry", "b_entry", "sd_entry", "sd_step", "sigma_pa", "tau")}
    ages = f["ages"]
    cum = np.array(f["curve"]["0.5"])          # relative to peak; only differences used
    a_min = ages[0]

    # Kalman filter through every training row, as in aging.stan
    N = d["N"]
    mf, vf = np.empty(N), np.empty(N)
    m = v = 0.0
    for n in range(N):
        if d["is_first"][n]:
            m = p["mu_entry"] + p["b_entry"] * d["entry_age_c"][n]
            v = p["sd_entry"] ** 2
        else:
            m += cum[d["age_idx"][n] - 1] - cum[d["prev_age_idx"][n] - 1]
            v += p["sd_step"] ** 2 * d["years"][n]
        r = p["sigma_pa"] ** 2 / d["pa"][n] + p["tau"] ** 2
        g = v / (v + r)
        m += g * (d["y"][n] - m)
        v *= 1 - g
        mf[n], vf[n] = m, v

    # ages step by exactly one per season in these data, so the seasons since
    # the last training row are the age difference (checked on every pair)
    for n in range(N):
        if not d["is_first"][n] and d["years"][n] != d["age_idx"][n] - d["prev_age_idx"][n]:
            raise SystemExit(f"row {n}: years != age difference")
    rows = []
    for j, e in enumerate(ev):
        n = e["last_obs"] - 1
        gap = e["target_age_idx"] - d["age_idx"][n]
        exp_ = mf[n] + cum[e["target_age_idx"] - 1] - cum[d["age_idx"][n] - 1]
        var = (vf[n] + p["sd_step"] ** 2 * gap + p["tau"] ** 2
               + p["sigma_pa"] ** 2 / e["pa"] + f["pred_sd"][j] ** 2)
        sd = float(np.sqrt(var))
        z = (e["target"] - exp_) / sd
        rows.append({"player_id": e["player_id"], "player_name": e["player_name"],
                     "age_2025": e["age"], "pa_2025": e["pa"], "seasons_since_last": int(gap),
                     "expected_rel_woba": round(float(exp_), 4), "actual_rel_woba": round(e["target"], 4),
                     "sd": round(sd, 4), "z": round(float(z), 3),
                     "stan_pred_mean": round(f["pred_mean"][j], 4)})

    z = np.array([r["z"] for r in rows])
    young = np.array([r["age_2025"] <= 25 for r in rows])
    # the plug-in expectation must be the Stan projection up to plug-in error
    gap_stan = max(abs(r["expected_rel_woba"] - r["stan_pred_mean"]) for r in rows)

    def cal(x):
        return {"n": int(len(x)), "share_abs_z_gt_1645": float(np.mean(np.abs(x) > Z90)),
                "sd_z": float(np.std(x, ddof=1)), "mean_z": float(np.mean(x))}
    res = {"all": cal(z), "age_le_25": cal(z[young]),
           "max_abs_plugin_minus_stan_mean": gap_stan, "params": p}
    a = res["all"]
    res["pass"] = bool(0.07 <= a["share_abs_z_gt_1645"] <= 0.13 and 0.90 <= a["sd_z"] <= 1.10)
    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: -r["z"])
    with open(out / "tracker_2025.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "calibration_2025.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
