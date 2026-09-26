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
// only differences of cum enter the likelihood, so g[1] never does. g follows
// a second-order random walk, anchored at age 27 (anchor_idx): the level
// g[anchor_idx] and slope g[anchor_idx + 1] - g[anchor_idx] have their own
// priors and the walk runs outward from there in both directions. The second
// differences of a random walk are the same forwards and backwards, so only
// where the level and slope priors sit changes. Anchoring at the youngest age
// (the third rehearsal) tied the level and slope the data fix at 23-33 to the
// sum of every curvature step from age 19; the simulation check's divergent
// draws differed most in exactly those two anchor coordinates (standardized
// difference -2.4 and +2.3) and sat at large sd_g.
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
  int<lower=2, upper=K - 1> anchor_idx;        // age index of 27, where the walk is anchored

  int<lower=0> M;                              // players to project
  array[M] int<lower=1, upper=N> last_obs;     // their last training season
  array[M] int<lower=1, upper=K> target_age_idx;
}
transformed data {
  // Sampler coordinates only; the prior on g is unchanged (see model block).
  // T maps (level, slope, second differences) at unit scale to g, by the
  // same recursion as below. Hg is a rough data information on g: each
  // consecutive pair of seasons observes the sum of g over the ages crossed,
  // with precision from prior-typical sigma_pa 0.5, tau 0.016, sd_step 0.024.
  // lam, Q: eigen-decomposition of that information on the K - 2 curvature
  // steps after the level and slope (with their priors) are integrated out.
  array[K - 2] int cidx;
  matrix[K, K] T = rep_matrix(0, K, K);
  matrix[K, K] Hg = rep_matrix(0, K, K);
  vector[K - 2] lam;
  matrix[K - 2, K - 2] Q;
  matrix[2, K - 2] B;      // approx. posterior regression of level/slope on the steps
  matrix[2, 2] L;          // and the Cholesky factor of their conditional covariance
  {
    int j = 1;
    for (k in 1:K) {
      if (k != anchor_idx && k != anchor_idx + 1) {
        cidx[j] = k;
        j += 1;
      }
    }
    for (c in 1:K) {
      vector[K] e = rep_vector(0, K);
      vector[K] t;
      e[c] = 1;
      t[anchor_idx] = e[anchor_idx];
      t[anchor_idx + 1] = t[anchor_idx] + e[anchor_idx + 1];
      for (k in (anchor_idx + 2):K) t[k] = 2 * t[k - 1] - t[k - 2] + e[k];
      for (i in 1:(anchor_idx - 1)) {
        int k = anchor_idx - i;
        t[k] = 2 * t[k + 1] - t[k + 2] + e[k];
      }
      T[:, c] = t;
    }
    for (n in 2:N) {
      if (!is_first[n]) {
        int a = prev_age_idx[n] + 1;
        int b = age_idx[n];
        real w = 1 / (0.25 / pa[n - 1] + 0.25 / pa[n] + 2 * square(0.016)
                      + square(0.024) * years[n]);
        Hg[a:b, a:b] += w;
      }
    }
    matrix[K, K] Hz = T' * Hg * T;
    array[2] int nidx = {anchor_idx, anchor_idx + 1};
    matrix[2, 2] Hnn = Hz[nidx, nidx]
                       + diag_matrix([1 / square(0.02), 1 / square(0.01)]');
    matrix[K - 2, K - 2] S = Hz[cidx, cidx]
                             - Hz[cidx, nidx] * mdivide_left_spd(Hnn, Hz[nidx, cidx]);
    S = 0.5 * (S + S');
    tuple(matrix[K - 2, K - 2], vector[K - 2]) eg = eigendecompose_sym(S);
    lam = fmax(eg.2, 0);
    Q = eg.1;
    B = -mdivide_left_spd(Hnn, Hz[nidx, cidx]);
    L = cholesky_decompose(inverse_spd(Hnn));
  }
}
parameters {
  vector[2] ls_z;          // level and slope given the steps, see below
  vector[K - 2] mode_z;    // curvature modes (columns of Q), see model block
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
  // Curvature steps d (the second differences of g, in the order of cidx)
  // are d = Q * u with u[j] = sd_g * mode_z[j] / sqrt(1 + lam[j] * sd_g^2).
  // With mode_z[j] ~ normal(0, sqrt(1 + lam[j] * sd_g^2)) this is exactly
  // u ~ normal(0, sd_g) iid, and Q is orthogonal, so d ~ normal(0, sd_g) iid
  // as before. The divisor keeps each mode_z near unit posterior scale
  // whether the data or the prior dominates that mode.
  vector[K] dz = rep_vector(0, K);   // zero at the anchor slots
  {
    vector[K - 2] f = sqrt(1 + lam * square(sd_g));
    dz[cidx] = Q * (sd_g * mode_z ./ f);
  }
  // Level and slope (natural units) are the approximate posterior mean given
  // the curvature steps, B * d, plus L * ls_z. The map (ls_z, mode_z) ->
  // (ls, mode_z) is linear with constant Jacobian |L|, so putting the prior
  // on ls itself (model block) keeps the model unchanged.
  vector[2] ls = B * dz[cidx] + L * ls_z;
  g[anchor_idx] = ls[1];
  g[anchor_idx + 1] = g[anchor_idx] + ls[2];
  for (k in (anchor_idx + 2):K) {
    g[k] = 2 * g[k - 1] - g[k - 2] + dz[k];
  }
  for (j in 1:(anchor_idx - 1)) {
    int k = anchor_idx - j;
    g[k] = 2 * g[k + 1] - g[k + 2] + dz[k];
  }
  cum = cumulative_sum(g);
}
model {
  // Same priors as sd_g ~ half-normal(0, 0.005), mu_entry ~ normal(0, 0.05),
  // b_entry ~ normal(0, 0.01), sd_entry ~ half-normal(0, 0.05),
  // sd_step ~ half-normal(0, 0.03), sigma_pa ~ normal+(0.5, 0.2),
  // tau ~ half-normal(0, 0.02).
  ls ~ normal(0, [0.02, 0.01]');   // level ~ N(0, 0.02), slope ~ N(0, 0.01), as before
  mode_z ~ normal(0, sqrt(1 + lam * square(sd_g)));
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
