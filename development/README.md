# Development tracker

Where each batter's 2025 landed against the line the aging model expected
for him: z = (actual - expected) / predictive sd, relative wOBA. Expected is
the Kalman filter of [aging.stan](../aging/aging.stan) run through his seasons
up to 2024 with the parameters of the pre-registered final fit, projected to
his 2025 age. The predictive sd counts what is still unknown about his
talent after his history, the random steps since his last season, the
season-only deviation, sampling noise for his 2025 PA, and the projection's
posterior sd.

## Calibration (2025, out of sample)

The pass rule was written into [tracker.py](tracker.py) before its first run:
share with |z| > 1.645 within [0.07, 0.13] and sd of z within [0.90, 1.10].

| group | n | outside the 90% interval | sd of z | mean z |
|---|---|---|---|---|
| all | 410 | 10.2% | 0.975 | -0.02 |
| age <= 25 | 94 | 9.6% | 0.944 | 0.00 |

**Pass.** So a z of +2 means what it says: about 1 batter in 40 lands that far
above his line by chance. Plug-in parameters (posterior means) are used; the
expected values differ from the Stan projection by at most 0.0009.

## Output

[out/tracker_2025.csv](out/tracker_2025.csv), one row per batter, sorted by z.
Furthest ahead of their line among age <= 25: Perdomo +2.37, Goodman +2.14,
Turang +2.06, Garcia +2.04. Furthest behind: Noel -2.17, Peraza -1.98,
Harris II -1.79.

    python aging/prepare.py <mart_batter_season @ 7746d62> prep 2024 2025
    python development/tracker.py prep aging/runs/final-20260926T081005Z development/out
