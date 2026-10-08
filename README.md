# baseball-bayes

Bayesian models for baseball data, written in Stan and run on GitHub Actions
(free for public repositories). Inputs are the analysis marts published by
[mlb-data-pipeline](https://github.com/yasumorishima/mlb-data-pipeline) to
[yasumorishima/mlb-stats](https://huggingface.co/datasets/yasumorishima/mlb-stats/tree/main/marts).

| Model | What it asks | Status |
| --- | --- | --- |
| [aging/](aging/) | Does a dynamic hierarchical aging curve project next-season wOBA better than Marcel? | [pre-registered](aging/PREREG.md), [result](aging/RESULTS.md): vs Marcel and vs no aging both **indistinguishable** on 2025 (MAE -0.00067 and -0.00071, both 95% intervals include 0); peak age 26-27 ([animated curve](aging/curve.gif); two table medians corrected on 2026-10-08) |
| [development/](development/) | Is each batter ahead of or behind the line the aging model expected for him? | z per batter for 2025; out of sample 10.2% outside the 90% interval and sd of z 0.975, but the upper tail is heavier (6.8% vs 3.4%) |

![The aging curve of the final fit, drawn one age at a time](aging/curve.gif)

![Four batters: the line the aging model drew through 2024, then where 2025 landed](development/out/tracker_2025.gif)

Every test is pre-registered: the question, data, metric and the readings of
each possible outcome are committed before the first fit, and results are
reported as they come out.
