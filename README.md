# baseball-bayes

Bayesian models for baseball data, written in Stan and run on GitHub Actions
(free for public repositories). Inputs are the analysis marts published by
[mlb-data-pipeline](https://github.com/yasumorishima/mlb-data-pipeline) to
[yasumorishima/mlb-stats](https://huggingface.co/datasets/yasumorishima/mlb-stats/tree/main/marts).

| Model | What it asks | Status |
| --- | --- | --- |
| [aging/](aging/) | Does a dynamic hierarchical aging curve project next-season wOBA better than Marcel? | [pre-registered](aging/PREREG.md) after nine rehearsals (2023 -> 2024); 2025 test next |

Every test is pre-registered: the question, data, metric and the readings of
each possible outcome are committed before the first fit, and results are
reported as they come out.
