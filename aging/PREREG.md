# Pre-registration: dynamic hierarchical aging curve, 2025 projection test

Committed **before** any 2025 projection was fitted or scored. The code that
runs the test (`prepare.py`, `aging.stan`, `fit.py`, `evaluate.py`,
`.github/workflows/aging.yml`) is frozen at the commit that adds this file
(md5 of each file below). Nothing here changes after the final run starts; a
change goes in an "Amendments" section with its date and reason, and the
first run is still reported.

## What was run before this was written

Ten rehearsals: train on seasons <= 2023, project 2024. A rehearsal that
reaches the real fits reports only fit diagnostics, run time and the
half-widths of the bootstrap intervals, never which method did better
(`evaluate.py --rehearsal`). From the second on, a simulation check
(`simcheck.py`) runs first: data simulated from the model with known
parameters on the real data's structure must be fitted with 0 divergences,
each of the 6 scalars with a known truth inside its 90% interval and the
curve (ages 21-40) inside its 90% band for at least 80% of ages; if it fails
the job stops before the real fits (rehearsals 2-7 stopped there). 2025 values of the evaluation players were
not looked at in any of them.

| # | run | code | change | result |
|---|---|---|---|---|
| 1 | 36207232625 | bed84b8 | latent talent sampled per player-season | both fits invalid: aging 138 divergences, R-hat 1.009, bulk ESS 272; no_aging R-hat 1.023, bulk ESS 319 |
| 2 | 36211714035 | 52f6f31 | talent integrated out by a Kalman filter (matches the dense likelihood to 7e-15) | simcheck fail: 1,122 divergences (90% in one chain) |
| 3 | 36213087656 | 9af8848 | scale parameters sampled on the unit scale of their priors | simcheck fail: 423 divergences, all truths covered |
| 4 | 36215834158 | 0879e52 | diagnostics only: where the divergences sit | at large sd_g and at the walk's anchor (ages 19/20) |
| 5 | 36217746770 | 6076a63 | random walk of the drift anchored at age 27 | simcheck fail: 246; divergence rate 22% in the top quartile of sd_g, 0.5-1.6% below |
| 6 | 36221664387 | 808f63a | curvature steps in the eigenbasis of the data information, each mode scaled by its conditional posterior sd | simcheck fail: 3 |
| 7 | 36222228753 | 907e6be | dense mass matrix | worse, 16; reverted (be5e3c3) |
| 8 | 36223359553 | 744aca9 | level and slope sampled given the curvature steps | simcheck pass (0); aging fit valid; no_aging fit invalid (116) |
| 9 | 36224148867 | 2179e65 | no data information on g in the no-aging fit | simcheck pass (0); both fits valid |
| 10 | 36227691942 | 4026649 | after the audit of this file: fallback decided inside the job, invalid fits never read, NaN rule, peak over 21-40, input and versions pinned | simcheck pass (0); both fits valid; every diagnostic and half-width identical to 9 |

Changes 3, 6, 8 and 9 change only the sampler's coordinates: the priors and
the likelihood are the same (each commit message gives the argument). Change
5 also moves a prior: the level and slope priors of the random walk,
normal(0, 0.02) and normal(0, 0.01), used to sit on g[1] (a drift below the
youngest age that never enters the likelihood) and g[2] - g[1], and now sit
on the drift from 26 to 27 and on the change in drift from there to 27-28;
the second-difference prior is unchanged. That is a model change, made before any 2025 value was fitted.
Change 10 touches no model or sampler setting.

Rehearsal 10, the configuration frozen here: aging fit
0 divergences, 0 tree-depth hits, max R-hat 1.00487, min bulk ESS 1,480,
183 s; no_aging 0, 0, 1.00375, 1,324, 161 s; the only NaN diagnostics are the
2 constant anchor slots in each fit; 416 evaluation players;
bootstrap 95% half-widths aging - marcel 0.00098 (MAE) / 0.00112 (RMSE),
aging - no_aging 0.00069 / 0.00074.

## Question

Does a Bayesian state-space model with a smooth population aging drift
project next-season batter wOBA better than Marcel, and does the aging drift
itself help?

## Data

- `mart_batter_season` from the Hugging Face dataset `yasumorishima/mlb-stats`
  (`marts/`, built and tested by the dbt project in
  `yasumorishima/mlb-data-pipeline`), at dataset commit
  `7746d62815359a23ead7128a3a337a42633401f8`, fixed in the workflow
  (`DATASET_REVISION`) and recorded in the run's `run.json`. The counts below
  are from `prepare.py` at that commit; the 2025 values were not opened.
- Pitchers (`primary_position = 'P'`, the position of that season) excluded;
  bench rows without a regular position (`X`) kept. wOBA is taken relative to
  the PA-weighted league wOBA of the same season over the same population.
- Training: seasons 2015-2024, every player-season with PA >= 1: 6,382
  player-seasons, 1,643 players, ages 19-45.
- Evaluation: every batter with PA >= 100 in 2025 who has at least one
  training season: 410 players.

## Models

1. **aging** (`use_aging = 1`): latent talent per player-season, integrated
   out exactly by a Kalman filter (the model is linear and Gaussian given
   the parameters). Between observed seasons it moves by the population drift for the ages crossed plus
   a random step with variance proportional to the seasons between. Observed
   wOBA = talent + sampling noise (sd sigma_pa / sqrt(PA)) + a season-only
   deviation (sd tau) that does not carry over. Drift per age follows a
   second-order random walk (level normal(0, 0.02) and slope normal(0, 0.01)
   at age 27, second differences normal(0, sd_g)); entry talent depends
   linearly on entry age.
   Projection = filtered talent mean after the last training season plus the
   drift to the 2025 age, averaged over posterior draws.
2. **no_aging**: the same model with the drift switched off.
3. **marcel**: weights 5/4/3 on 2024/2023/2022, 1200 PA of league average,
   age factor 1 + (29 - age) x 0.006 below 29 and x 0.003 from 29, on absolute
   wOBA with the 2022-2024 mean league wOBA as reference. As defined, Marcel
   sees only the last three seasons (a player last seen before 2022 gets the
   league average times the age factor) while models 1-2 see 2015-2024.

Environment: Python 3.11.16, cmdstanpy 1.3.0, numpy 2.4.6, pandas 3.0.6
(`requirements.txt`), GitHub-hosted `ubuntu-latest` (not pinned).
Sampling: CmdStan 2.40.0, 4 chains, 1000 warmup + 1000 draws,
adapt_delta 0.9, max_treedepth 10, diagonal metric, seed 20260926. Priors
as in `aging.stan`.

## Validity

A fit is valid if it has 0 divergent transitions, 0 max-treedepth hits,
max R-hat < 1.01 and bulk ESS >= 400 on every quantity in the CmdStan
summary (parameters, transformed parameters, generated quantities, lp__;
talent is integrated out, so there are no per-season parameters). A NaN
R-hat or ESS is allowed only for a quantity whose posterior sd is 0 (the
zero anchor slots of `dz`); any other NaN makes the fit invalid.

The fallback is decided by `fit.py` inside the same job, before anything is
scored: if either first fit is invalid, both are fitted once more with
adapt_delta 0.99 and max_treedepth 12. The pair that is read is the first
pair if both its fits are valid, otherwise the fallback pair, whole (fits
from the two pairs are never mixed). `selection.json` records which, and the
first pair's diagnostics are reported next to it. In the pair that is read,
a comparison that needs a fit that is not valid is written as "no valid fit"
with no difference or interval (`evaluate.py`): H1 needs aging valid, H2 needs
both. The job has a 300-minute limit; a run that hits it has no result and
counts as invalid.

## Metric and readings

- Primary metric: PA-weighted mean absolute error of relative 2025 wOBA.
  Co-metric: PA-weighted root mean squared error (the posterior mean is the
  optimum for squared error). Both are reported; H1 and H2 are read on the
  primary metric.
- Uncertainty: paired bootstrap over the evaluation players (10,000
  resamples, seed 20260926), 95% interval of the difference.
- **H1 (primary): aging - marcel.** Interval entirely below 0 -> "aging is
  better"; entirely above 0 -> "aging is worse"; otherwise "indistinguishable".
- **H2: aging - no_aging**, same three readings.
- Expected resolution, from rehearsal 10: an interval half-width of about
  0.00098 (MAE) for H1 and 0.00069 for H2. A true difference smaller than that will read
  "indistinguishable", and that reading means only that.
- Descriptive (no test): the aging curve with a 90% band, and the posterior
  of the peak age taken as the maximum over ages 21-40, next to the share of
  draws whose maximum over all ages lies outside 21-40; both for ages 21-40
  only. Outside that range there are
  very few player-seasons (19: 2 rows, 41-45: 9) and the curve is driven by
  the smoothing prior.

## What this does not claim

- One season, one league, one statistic. A win or loss here does not
  generalise to other targets or to NPB.
- Missing seasons are treated as missing at random given the history; players
  who stop playing because they got worse are a known bias of every aging
  curve, including this one.

## Frozen files (md5)

```
e8d138e86fd83e10ebca981162c79f3a  aging/prepare.py
0ec115a2a04a624db4edf1f828c4f397  aging/aging.stan
85b89c4d5ce50266518b23cffb1176e7  aging/fit.py
bbe0bcfdd8ba75d4bfbfad4a2529e944  aging/evaluate.py
7e4c7d5cc256421cd68547821e8cd252  aging/simcheck.py
3b065fdec8334c24c82259a27c7753f2  .github/workflows/aging.yml
69601f79f72424cc121b58a1fde71dac  requirements.txt
```

These are the files at commit 4026649, the code of rehearsal 10. The final
run records the commit it ran (`run.json`); these files must be
byte-identical there.
