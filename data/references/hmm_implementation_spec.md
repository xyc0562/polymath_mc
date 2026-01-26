# 2‑State Negative‑Binomial HMM for Musk Tweet Daily Counts (Implementation Spec)

This spec describes how to implement a **2‑state Hidden Markov Model (HMM)** with **Negative Binomial** emissions for daily tweet counts, and how to integrate it into the existing forecasting/backtest pipeline (Monte Carlo → bin probabilities → Kelly executor).

It is written to be *implementation‑ready* for Claude: exact formulas, boundary conditions, and integration points.

---

## Why HMM (what problem it fixes)

Your current model needs large “dispersion inflation” (e.g. ×4) to achieve interval coverage. That almost always indicates **missing shared uncertainty/correlation** across days (regime shifts / mania weeks).

A 2‑state HMM fixes this by:
1. **Mixture tails**: the predictive distribution is a mixture of “low” and “high” states → fatter tails.
2. **Persistence**: regimes persist (high stays high) → **correlated** high counts across multiple days → multi‑day totals widen naturally.

This typically reduces the need for blunt dispersion inflation and improves calibration of multi‑day totals (4d/7d horizons).

---

## High‑level deliverable

Implement module(s):

- `hmm_nb.py` (core HMM):
  - 2 states: `LOW (0)` and `HIGH (1)`
  - emission: **Negative Binomial** with state‑specific mean & dispersion
  - EM training (Baum–Welch)
  - forward filtering and backward smoothing in **log space**
- `interday_hmm.py` (adapter):
  - fit HMM on daily totals
  - given a forecast anchor day, return:
    - state posterior at anchor (`alpha_last`)
    - forecast distribution for days ahead via Monte Carlo state paths
- Integration into `monte_carlo.py`:
  - replace / optionally toggle old interday forecaster with HMM‑driven future day sampling
- Add config flags and backtest ablation:
  - old interday
  - HMM interday
  - HMM + intraday updating (optional; can be Phase 2)

---

## Data assumptions / preprocessing

### Daily series
You need a daily count series `y[t]` for contract days, aligned to Polymarket definition (12pm ET windows). Use the same “contract day” transform you already have.

- Training window: typically 30–90 days (you used 45).
- Remove incomplete final days (already done in your pipeline).

### Weekend indicator (optional)
You may include a weekend effect later. MVP can ignore covariates (simpler & robust). Start with no covariates.

---

## Model definition

### Hidden states
`S_t ∈ {0,1}` follows Markov transitions:

- Initial distribution: `π[s] = P(S_1 = s)`
- Transition matrix `A` (2×2):

```
A[i,j] = P(S_t=j | S_{t-1}=i)
Rows sum to 1.
```
Interpretation: `A[1,1]` high→high persistence; `A[0,0]` low→low persistence.

### Emission: Negative Binomial by state
Daily count `Y_t` given state `s`:

```
Y_t | (S_t=s) ~ NegBin(mean=μ_s, dispersion=k_s)
```
- `μ_s > 0` mean daily tweets in state s
- `k_s > 0` dispersion/shape (larger k → closer to Poisson; smaller k → fatter tails)

#### NB parameterization (must be consistent)
Use NB in `(r, p)` form (common in SciPy / many implementations):

- `r = k`
- `p = r / (r + μ)`
- Mean: `E[Y] = r*(1-p)/p = μ`
- Variance: `Var[Y] = μ + μ^2 / r = μ + μ^2/k`

PMF:
```
P(Y=y) = C(y+r-1, y) * (1-p)^y * p^r
```
Log PMF (stable; use gammaln):
```
logP = gammaln(y+r) - gammaln(r) - gammaln(y+1) + r*log(p) + y*log(1-p)
```

**Boundary conditions**
- Enforce `μ_s >= μ_min` (e.g. 1e-3 or 1)
- Enforce `k_s >= k_min` (e.g. 0.5 or 1.0)
- Use `log1p` tricks to avoid `log(0)`.

---

## Training: EM (Baum–Welch)

You fit parameters θ = {π, A, μ0, μ1, k0, k1}.

### Core computations in log space
Because probabilities get tiny, implement forward/backward using `logsumexp`.

Let emission likelihood:
```
B_t(s) = P(Y_t=y_t | S_t=s)
logB_t(s) = log NB_PMF(y_t; μ_s, k_s)
```

### Forward pass (log α)
Define `log_alpha[t,s] = log P(y_1..y_t, S_t=s)`.

Initialization:
```
log_alpha[0,s] = log(π[s]) + logB_0(s)
```

Recurrence:
```
log_alpha[t,s] = logB_t(s) + logsumexp_i( log_alpha[t-1,i] + log(A[i,s]) )
```

Log-likelihood:
```
loglik = logsumexp_s(log_alpha[T-1,s])
```

### Backward pass (log β)
Define `log_beta[t,s] = log P(y_{t+1}..y_T | S_t=s)`.

Initialization:
```
log_beta[T-1,s] = 0
```

Recurrence:
```
log_beta[t,i] = logsumexp_j( log(A[i,j]) + logB_{t+1}(j) + log_beta[t+1,j] )
```

### Posterior state probabilities γ and transition posteriors ξ
State posterior:
```
log_gamma[t,s] = log_alpha[t,s] + log_beta[t,s] - loglik
gamma[t,s] = exp(log_gamma[t,s])
```

Transition posterior (for t=1..T-1):
```
log_xi[t,i,j] = log_alpha[t-1,i] + log(A[i,j]) + logB_t(j) + log_beta[t,j] - loglik
xi[t,i,j] = exp(log_xi[t,i,j])
```

### M-step updates
#### π
```
π[s] = gamma[0,s]
```

#### A
```
A[i,j] = sum_{t=1..T-1} xi[t,i,j] / sum_{t=1..T-1} gamma[t-1,i]
```
Add smoothing to avoid zeros:
- numerator += `eps_A`
- denominator += `2*eps_A` (since 2 states)

#### μ_s
Weighted mean:
```
μ_s = (sum_t gamma[t,s] * y_t) / (sum_t gamma[t,s])
```
Enforce μ_min.

#### k_s (dispersion)
You have two options.

**Option 1 (MVP / robust): weighted method-of-moments**
Compute weighted variance:
```
var_s = (sum_t gamma[t,s]*(y_t - μ_s)^2) / (sum_t gamma[t,s])
k_s = μ_s^2 / max(var_s - μ_s, eps)
```
Then clamp to k_min.

This is usually good enough for regime separation and calibration.

**Option 2 (refine later): weighted NB MLE**
Solve 1D equation for k (Newton/bisection). Only do this after MVP works.

### Convergence criteria
Run `n_iter` (e.g. 30–50), stop early if:
```
|loglik_new - loglik_old| < tol  (e.g. 1e-4)
```
Also cap iterations to avoid endless loops.

---

## Initialization (important)

Bad initialization → label switching or collapse. Use one of these:

### Recommended init (simple & stable)
1. Compute `z_t = log1p(y_t)`.
2. Run 2-means clustering on `z_t` (1D kmeans is trivial):
   - init centers at 30th and 80th percentiles of z
   - iterate 10 times
3. Assign clusters → compute μ0, μ1 from cluster means (in original y space)
4. Set k0=k1=10 initially (or from MoM)
5. Estimate A from cluster labels (with smoothing):
   - count transitions low→low, low→high, etc.
   - add +1 pseudo-count to each cell
6. π from first label frequency (or [0.5,0.5])

Alternative: fixed sticky A:
```
A = [[0.90, 0.10],
     [0.10, 0.90]]
```
and μ0/μ1 from quantiles.

---

## Forecasting with the HMM

You need predictive simulations for **future days (1..6)** for the 7‑day contract window.

### Step 1: filtering to get current state posterior
Given training daily series up to day `t-1`, compute forward messages and take:
```
alpha_last[s] = P(S_{t-1}=s | y_1..y_{t-1})
```
In log space:
```
log_alpha_last = log_alpha[T-1] - logsumexp(log_alpha[T-1])
alpha_last = exp(log_alpha_last)
```

### Step 2: simulate future state path and counts
For each Monte Carlo path:

1. Sample `S_{t}` from `alpha_last @ A` (i.e. one-step forecast state distribution).
2. For h=0..H-1 (H=6 for future days):
   - sample `Y_{t+h} ~ NB(μ_{S_{t+h}}, k_{S_{t+h}})`
   - sample next state from `A[S_{t+h}, :]`

This naturally induces **multi-day correlation** because state persists.

### Step 3: integrate with today’s nowcast
Keep your existing **today** model (lognormal nowcast floored at cum_so_far).

Then each simulation path total is:
```
total_7d = today_sample + sum_{h=1..6} future_day_samples[h]
```
(Use your contract day indexing carefully; “future days” means the next 6 *contract* days.)

---

## Optional Phase 2: using intraday data safely (recommended later)

Instead of scaling future means, use intraday evidence to update the *state probability* for today or tomorrow.

At time `now` on day t, you can produce an implied full-day estimate `implied_today` (or distribution) from nowcast/progress curve.

Compute approximate likelihood of that observation under each state:
```
ℓ_s ∝ NB_PMF(round(implied_today); μ_s, k_s)
```
Then update state posterior for today:
```
posterior_s ∝ prior_s * ℓ_s
normalize posterior
```
Use this posterior as the start of your state path simulation.

MVP can skip this; the HMM alone already helps coverage a lot.

---

## Integration points in your repo

### New config flags
Add to config (example names):
- `use_hmm_interday: bool = True`
- `hmm_states: int = 2` (fixed at 2 for now)
- `hmm_em_iters: int = 40`
- `hmm_em_tol: float = 1e-4`
- `hmm_mu_min: float = 1.0`
- `hmm_k_min: float = 1.0`
- `hmm_A_smoothing: float = 1e-3`
- `hmm_init_method: str = "kmeans_log1p"`
- `hmm_seed: int = 12345`

### Replace / wrap the existing interday forecaster
Where you currently call something like:
- `interday.forecast_day(h, date)`
or compute `future_days_estimate`

Add an alternate path:
- fit HMM on training daily counts
- produce simulations for day+1..day+6 from HMM
- use those in Monte Carlo paths

### Keep the old model for ablation
Ablations you must run in backtest:
- old interday only
- HMM interday only
- full model with today nowcast + HMM future days
Compare MAE + coverage + log score.

---

## Metrics expectations / sanity checks

After implementing HMM:
- Coverage_90 should move **up** materially without giant dispersion inflation.
- The distribution of 6/7-day totals should have heavier right tail (more realistic).
- Interday-only MAE may stay similar; main win is calibration/tails.

Sanity checks to log:
1. Learned μ0 < μ1 (if not, swap labels)
2. A[0,0] and A[1,1] reasonably high (e.g. >0.7) unless data suggests frequent switching
3. State occupancy not degenerate (not all mass in one state)

---

## Implementation checklist (Claude)

1. Implement NB log PMF (stable, uses gammaln).
2. Implement `logsumexp` utilities.
3. Implement HMM forward-backward in log space.
4. Implement EM loop with:
   - A smoothing
   - μ update
   - k update via weighted MoM
   - label swap if μ0 > μ1
5. Unit test on synthetic HMM data:
   - generate known 2-state NB HMM, fit, confirm rough recovery.
6. Integrate into Monte Carlo:
   - for each sim path, sample state path and future day counts
   - combine with today nowcast sample
7. Run backtest ablations and compare coverage/MAE/logscore.
8. Only after MVP works:
   - optional: weighted MLE for k
   - optional: intraday likelihood update of state posterior

---

## Minimal pseudo-code (structure only)

```python
class NB2StateHMM:
    def __init__(self, mu, k, A, pi, eps=1e-12, seed=123):
        self.mu = np.array(mu)   # shape (2,)
        self.k  = np.array(k)    # shape (2,)
        self.A  = np.array(A)    # shape (2,2)
        self.pi = np.array(pi)   # shape (2,)

    def log_emission(self, y):  # returns shape (T,2)
        ... # NB log pmf for each state

    def forward_backward(self, y):
        logB = self.log_emission(y)
        # log_alpha, log_beta, loglik, gamma, xi
        ...

    def fit_em(self, y, n_iter=40, tol=1e-4):
        init if needed
        for it in range(n_iter):
            gamma, xi, loglik = self.forward_backward(y)
            update pi, A (with smoothing), mu, k
            enforce bounds, swap labels if mu0 > mu1
            if converged: break
        return self

    def filter_last(self, y):
        # forward only
        ... # return alpha_last (2,)

    def sample_future_counts(self, alpha_last, n_days, rng):
        # simulate states and NB counts
        ...
```

---

## Notes on speed

2-state HMM is tiny. EM on ~45–90 days converges quickly.
This will not be your bottleneck vs Monte Carlo sims.

---

## What to do if you still see undercoverage after HMM

If coverage_90 is still low:
- add **week-level shock** Z per simulation path (optional)
- or allow mild dispersion inflation (≤1.5)
But do not jump back to ×4; that usually means missing correlation remains.
