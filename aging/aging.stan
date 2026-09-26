// Dynamic hierarchical aging curve for MLB batters, with the latent talent
// integrated out by a Kalman filter.
//
// Each batter has a latent talent (wOBA above the league average of that
// season). From one observed season to the next it moves by the population
// aging drift for the ages crossed, plus a player-specific random step whose
// variance grows with the number of seasons between the two. The observed
// wOBA is talent plus two noises: sampling noise that shrinks with PA, and a
// season-only deviation (tau) that does not carry over to the next season.
//
// The model is linear and Gaussian given the parameters, so the likelihood
// of each player's seasons is computed exactly by a Kalman filter instead of
// sampling one latent talent per player-season (the first rehearsal did that
// and had 138 divergent transitions).
//
// For k >= 2, g[k] is the average change in talent from age A_min + k - 2 to
// A_min + k - 1. cum = cumulative_sum(g) is the aging curve up to a constant;
// only differences of cum enter the likelihood, so g[1] only anchors the
// smoothing prior. g follows a second-order random walk.
//
// Every scale parameter is sampled on the unit scale of its prior and then
// multiplied by that scale (e.g. sd_step = 0.03 * sd_step_raw with
// sd_step_raw ~ half-normal(0, 1), which is sd_step ~ half-normal(0, 0.03)).
// The model is the same; only the sampler's coordinates change. With the raw
// scales (0.005 to 0.05) the default initial values started some chains
// hundreds of times outside the prior, and one chain never recovered.
data {
  int<lower=1> N;                              // player-seasons, sorted by player then season
  int<lower=1> K;                              // number of integer ages
  array[N] int<lower=1, upper=K> age_idx;
  array[N] int<lower=0, upper=1> is_first;     // first season of that player in the data
  array[N] int<lower=0, upper=K> prev_age_idx; // age index of the previous season (0 if first)
  vector<lower=0>[N] years;                    // seasons since the previous one (0 if first)
  vector[N] entry_age_c;                       // age at first season minus 27 (0 if not first)
  vector[N] y;                                 // wOBA minus league wOBA of that season
  vector<lower=1>[N] pa;
  int<lower=0, upper=1> use_aging;             // 0 = ablation with no aging drift

  int<lower=0> M;                              // players to project
  array[M] int<lower=1, upper=N> last_obs;     // their last training season
  array[M] int<lower=1, upper=K> target_age_idx;
}
parameters {
  vector[K] g_z;
  real<lower=0> sd_g_raw;
  real mu_entry_raw;
  real b_entry_raw;
  real<lower=0> sd_entry_raw;
  real<lower=0> sd_step_raw;
  real<lower=0> sigma_pa_raw;
  real<lower=0> tau_raw;
}
transformed parameters {
  real sd_g = 0.005 * sd_g_raw;
  real mu_entry = 0.05 * mu_entry_raw;
  real b_entry = 0.01 * b_entry_raw;
  real sd_entry = 0.05 * sd_entry_raw;
  real sd_step = 0.03 * sd_step_raw;
  real sigma_pa = 0.5 * sigma_pa_raw;
  real tau = 0.02 * tau_raw;
  vector[K] g;
  vector[K] cum;
  g[1] = 0.02 * g_z[1];
  g[2] = g[1] + 0.01 * g_z[2];
  for (k in 3:K) {
    g[k] = 2 * g[k - 1] - g[k - 2] + sd_g * g_z[k];
  }
  cum = cumulative_sum(g);
}
model {
  // Same priors as sd_g ~ half-normal(0, 0.005), mu_entry ~ normal(0, 0.05),
  // b_entry ~ normal(0, 0.01), sd_entry ~ half-normal(0, 0.05),
  // sd_step ~ half-normal(0, 0.03), sigma_pa ~ normal+(0.5, 0.2),
  // tau ~ half-normal(0, 0.02).
  g_z ~ std_normal();
  sd_g_raw ~ std_normal();
  mu_entry_raw ~ std_normal();
  b_entry_raw ~ std_normal();
  sd_entry_raw ~ std_normal();
  sd_step_raw ~ std_normal();
  sigma_pa_raw ~ normal(1, 0.4);
  tau_raw ~ std_normal();
  {
    real m = 0;   // predicted talent mean before seeing y[n]
    real v = 1;   // and its variance
    for (n in 1:N) {
      if (is_first[n]) {
        m = mu_entry + b_entry * entry_age_c[n];
        v = square(sd_entry);
      } else {
        if (use_aging) m += cum[age_idx[n]] - cum[prev_age_idx[n]];
        v += square(sd_step) * years[n];
      }
      real r = square(sigma_pa) / pa[n] + square(tau);
      target += normal_lpdf(y[n] | m, sqrt(v + r));
      real gain = v / (v + r);
      m += gain * (y[n] - m);
      v *= 1 - gain;
    }
  }
}
generated quantities {
  // Projection for a season not in the data: the filtered talent mean after
  // the player's last training season, plus the aging drift to the target age.
  vector[M] pred;
  {
    vector[N] m_filt;
    real m = 0;
    real v = 1;
    for (n in 1:N) {
      if (is_first[n]) {
        m = mu_entry + b_entry * entry_age_c[n];
        v = square(sd_entry);
      } else {
        if (use_aging) m += cum[age_idx[n]] - cum[prev_age_idx[n]];
        v += square(sd_step) * years[n];
      }
      real r = square(sigma_pa) / pa[n] + square(tau);
      real gain = v / (v + r);
      m += gain * (y[n] - m);
      v *= 1 - gain;
      m_filt[n] = m;
    }
    for (j in 1:M) {
      int n = last_obs[j];
      pred[j] = m_filt[n]
                + (use_aging ? cum[target_age_idx[j]] - cum[age_idx[n]] : 0);
    }
  }
}
