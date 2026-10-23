# CLAUDE.md — baseball-bayes

Working rules for Claude Code (cloud or local). This file is the source of truth for this project; the owner's local notes are not visible to cloud sessions.

## Session protocol (cloud)
This project is worked on in Claude Code cloud. The owner starts a session with just 「続き」/「つづきから」.
1. Read this file first, then `aging/PREREG.md`, `aging/RESULTS.md` and `development/README.md`. Start from the first unchecked item in "Next tasks".
2. Before ending, record what you did and what comes next **in this file** (update "Next tasks" and add a dated line to "Progress log"). This file is the only memory that survives between cloud sessions.
3. Cloud can push only its session branch: open a PR to `main` and merge it yourself (own repo) so the next session sees the update. Report the merged commit.
4. Write replies to the owner in Japanese (です/ます). Do not claim anything you did not run in this session; give run ids / commits / numbers only as read from a tool result.

## Progress log
- 2026-09-26: aging test pre-registered (`03ba907`) after 10 rehearsals; final run `36228633245` (train <= 2024, project 2025): H1 aging−marcel MAE −0.00067 [−0.00168, +0.00033], H2 aging−no_aging −0.00071 [−0.00143, +0.00002] — both **indistinguishable**. Development tracker on 2025 (rule written before the run but committed with the script and output): 10.2% outside the 90% interval, sd(z) 0.975, upper tail heavier (6.8% vs 3.4%).
- 2026-10-08: GIFs for the curve and the tracker (PR #1–#3); the GIF gate found two wrong medians in RESULTS.md (33: −0.019, 40: −0.073), corrected with a note. README states only the aging test is pre-registered (PR #4).
- 2026-10-23 (local, migration prep): a fresh clone reproduced the final run's `eval.json` and `meta.json` byte-for-byte from HF `7746d62`; the 7 frozen md5s in PREREG.md match `main`. HF marts `deaeb85b8601c64981cd2082d06941f2d20f5f3f` (2026-10-19) hold a complete 2026 (`is_partial` false, 662 rows, 183,849 PA; at `7746d62` 2026 was partial). Seasons <= 2025 differ between the two revisions in 1 row (season 2016). `prepare.py <deaeb85> … 2025 2026` gives N 7,049 / 1,748 players / M 411; the 2026 values were not opened. Added `.github/workflows/aging-request.yml` (not run yet; fails closed until `aging/PREREG_2026.md` exists).

## What this is
Bayesian models for MLB batters, written in Stan, run on GitHub Actions, inputs from the HF dataset `yasumorishima/mlb-stats` (`marts/mart_batter_season.parquet`, built by `yasumorishima/mlb-data-pipeline`).
- `aging/`: state-space aging curve (talent integrated out by a Kalman filter) vs Marcel vs the same model without aging drift; next-season relative wOBA, PA-weighted MAE. Pre-registered.
- `development/`: per-batter z = (actual − expected) / predictive sd, using the final fit's parameters. Not pre-registered (see its README).

## Non-negotiable discipline
1. **Pre-register, then run once.** Question, data revision, models, metric and the reading of every outcome are committed (merged to `main`) before any fit that is scored. A gap found before the run goes into an "Amendments" section before the run; the first run is always reported.
2. **Frozen = md5.** `aging/PREREG.md` lists 7 files frozen at `4026649`. Never edit them: `aging/prepare.py`, `aging/aging.stan`, `aging/fit.py`, `aging/evaluate.py`, `aging/simcheck.py`, `.github/workflows/aging.yml`, `requirements.txt`. A new test reuses them byte-for-byte or adds new files with their own md5 list. Never regenerate `aging/runs/*` or `development/out/*_2025.*`.
3. **Readings are fixed:** interval entirely below 0 = better, entirely above = worse, otherwise **indistinguishable**. A favourable point estimate is not a win; an interval end of +0.00002 is still "includes 0". Do not reread or re-threshold after seeing numbers.
4. **Pin the input.** Use a 40-hex HF commit, never `main` (the weekly refresh changes it). The DATASET_REVISION goes in the workflow and in the PREREG.
5. **Do not look at the target season's values before the freeze.** Counts (rows, players, M) are fine; wOBA of the evaluation season is not.
6. **Write a rule before the script that it judges, in its own merged commit** (the 2025 tracker rule arrived in the same commit as its output; the repo cannot prove the order — don't repeat).
7. Code that produces a result gets an independent review (Opus subagent) before it runs; verify every review claim against the code/data before accepting it. Numbers in RESULTS/README are checked by a script against the run's JSON, not copied by hand.
8. Report nulls as facts; no "ceiling / limit / nothing left" framing. One season, one league, one statistic: do not generalise to NPB or other targets.
9. Do not suggest writing a blog post as the next step; the next step is the next analysis.

## Next tasks (in order)
1. ~~Request-triggered workflow~~ — done in the migration commit: `.github/workflows/aging-request.yml` runs on a push to `main` that touches `aging/requests/run2026.json`, reuses the frozen scripts with train_end 2025 / eval 2026, reads the input revision from the single line `DATASET_REVISION: <40-hex>` in `aging/PREREG_2026.md`, checks the md5 list there, refuses a second 2026 run, and commits `aging/runs/final2026-<stamp>/`. It fails closed until that PREREG exists. **Do not edit any file under `.github/workflows/`** (it is frozen and cloud pushes there are unverified); everything the run needs comes from `aging/PREREG_2026.md`.
2. **Pre-register the 2026 replication (`aging/PREREG_2026.md`).** Same question, models, priors, sampler settings, validity rule, fallback, metric, bootstrap (10,000, seed 20260926) and readings as `aging/PREREG.md`; only train <= 2025 / evaluate 2026 (PA >= 100, at least one training season) and the data revision change. Write:
   - Data revision: exactly one line `DATASET_REVISION: <40-hex>` at the start of a line (the workflow greps `^DATASET_REVISION: [0-9a-f]{40}$`). Candidate `deaeb85b8601c64981cd2082d06941f2d20f5f3f` (or a newer marts commit — re-check `is_partial` is false for 2026 and record it); counts from `prepare.py` at that revision.
   - Disclosure: 2026 hitter outcomes were already read in aggregate by the owner in `mlb-data-pipeline/analysis/decision_pairs_batters` (2025→2026 pair rules, revision `e006cbe`) and 2026 team totals in the team-wins scoring; no individual 2026 wOBA was opened for this test. Seasons <= 2025 differ from the 2025 test's input revision `7746d62` in 1 row (season 2016).
   - Expected resolution from the 2025 final run: 95% half-width about 0.0010 (H1) and 0.0007 (H2). Decide now (not after) whether any pooled 2025+2026 readout is reported; if not decided, do not add it later.
   - Frozen files (md5): the 7 files above **plus** `.github/workflows/aging-request.yml` — lines in the exact `md5  path` form (two spaces, repo-relative path). The workflow requires each of these 8 paths to appear exactly once and checks them all with `md5sum -c`; it also refuses to run if `aging/runs/final2026-*` exists in its checkout **or on the latest `main`**.
   Opus review of PREREG + workflow before merging; merge.
3. **Run once.** PR adding `aging/requests/run2026.json` (`{"mode": "final", "memo": "..."}`) → merge → read the `Aging curve (request)` run for that merge commit through GitHub MCP (list workflow runs) until it is completed; on success the bot commit `aging: final 2026 run …` is on `main`. A failed run commits nothing — read the failed step's log and report it. The PR that adds `run2026.json` must not touch `aging/PREREG_2026.md` or any frozen file (the workflow refuses a push that does), so the freeze is provably earlier. Never rerun; an invalid or timed-out run is reported as such.
4. **`aging/RESULTS_2026.md` + README row.** Generate the tables from `results.json` / `fit_aging.json` by script; report fit diagnostics, selection (first/fallback), H1/H2 with readings, MAE/RMSE, and the descriptive curve/peak next to the 2025 ones (no test of the difference between years).
5. **Development tracker on 2026.** Keep `development/tracker.py` and its 2025 outputs unchanged. First merge a commit with only the rule (e.g. `development/RULE_2026.md`: same two-sided pass rule; also a declared check of the 2025 tail asymmetry, e.g. tails at PA >= 100 vs a wider set, with the predicted direction). Then a new script parameterised by season that **honours `selection.json`** (the current `tracker.py` always reads `fit_aging.json`, wrong if the fallback pair is read) and asserts the prepared `eval.json` equals the run's. numpy only — can run in the cloud container. Then run once, commit outputs, update `development/README.md`.
6. Later, each with a new PREREG: a target that transfers more (K%, BB%), or AAA batters (Savant `minors=true&hfLevel=AAA|` — without it Savant silently returns MLB data).

## Compute
- Stan fits run on GitHub-hosted `ubuntu-latest` via the workflow (CmdStan 2.40.0 cached; ~3 min per fit on the rehearsals, job limit 300 min). Prefer this: the run id and code commit are recorded in `run.json`, which is the public proof.
- In the cloud container, only light work: `prepare.py` (duckdb only) and numpy scripts. Installing CmdStan here would need `github.com` release downloads and a compile; do not use it for scored runs.

## Data access (cloud network allowlist)
- `huggingface.co` and **`us.aws.cdn.hf.co`** (the parquet `resolve/<rev>/…` redirects there, checked 2026-10-23). Python packages from PyPI (`requirements.txt`: cmdstanpy 1.3.0, duckdb 1.5.5, numpy 2.4.6, pandas 3.0.6; Python 3.11).
- `baseballsavant.mlb.com` only for task 6.
- No secrets are used by this repo (the workflow uses the default `GITHUB_TOKEN`).

## Credits
Data: MLB Advanced Media (Baseball Savant, MLB Stats API) via `yasumorishima/mlb-data-pipeline` → HF `yasumorishima/mlb-stats`. Marcel: Tom Tango.
