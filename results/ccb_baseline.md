# CCB evaluation on BCI-IV-2b — headline + gap-closure

Reads the 2a benchmark from `results/fbcsp_baseline.csv` (filter `dataset == 'BCI-IV-2a'` — no-leakage enforced).

## Summary (mean ± std κ over 9 subjects)

| Dataset / policy | κ within | κ official |
|---|---|---|
| BCI-IV-2a FBCSP (22-ch benchmark) | 0.532 ± 0.263 | 0.468 ± 0.261 |
| BCI-IV-2b CCB (3-ch, default hyperparams) | 0.124 ± 0.156 | 0.010 ± 0.141 |
| BCI-IV-2b CCB (3-ch, tuned: budget_frac=1.0, alpha=0.5, arm_pool=full, calibration_frac=0.2) | 0.103 ± 0.183 | 0.081 ± 0.105 |
| BCI-IV-2b CCB (3-ch, best-factorial: alpha=0.5, calibration_frac=0.3, arm_pool=pruned, include_recent_rewards=False, per_round_cap=inf, budget_frac=1.0, policy=oplb) | 0.184 ± 0.170 | 0.158 ± 0.153 |

Best-factorial seed-std (across seeds {0, 1, 2, 3, 42}, subject-averaged κ per seed):
- **within**: σ = **0.065**
- **official**: σ = **0.047**

## Gap-to-benchmark Δκ = κ<sub>2a FBCSP</sub> − κ<sub>2b CCB</sub>

**Default hyperparameters** (budget_frac=1.0, alpha=1.0, arm_pool=pruned, calibration_frac=0.3):
- **within**: Δκ = **+0.409** (per-subject mean across 9 subjects)
- **official**: Δκ = **+0.458** (per-subject mean across 9 subjects)

**Tuned** (budget_frac=1.0, alpha=0.5, arm_pool=full, calibration_frac=0.2):
- **within**: Δκ = **+0.430** (per-subject mean across 9 subjects)
- **official**: Δκ = **+0.387** (per-subject mean across 9 subjects)

**Best-factorial** (alpha=0.5, calibration_frac=0.3, arm_pool=pruned, include_recent_rewards=False, per_round_cap=inf, budget_frac=1.0, policy=oplb):
- **within**: Δκ = **+0.349** (per-subject mean across 9 subjects)
- **official**: Δκ = **+0.310** (per-subject mean across 9 subjects)

## Per-subject Δκ

| Subject | Protocol | κ 2a FBCSP | κ 2b CCB (default) | Δκ |
|---|---|---|---|---|
| 1 | official | 0.611 | -0.100 | +0.711 |
| 2 | official | -0.042 | 0.033 | -0.075 |
| 3 | official | 0.750 | -0.100 | +0.850 |
| 4 | official | 0.167 | 0.071 | +0.095 |
| 5 | official | 0.625 | 0.014 | +0.611 |
| 6 | official | 0.347 | -0.083 | +0.431 |
| 7 | official | 0.500 | 0.217 | +0.283 |
| 8 | official | 0.667 | -0.183 | +0.850 |
| 9 | official | 0.583 | 0.217 | +0.367 |
| 1 | within | 0.618 | 0.100 | +0.518 |
| 2 | within | 0.133 | -0.142 | +0.275 |
| 3 | within | 0.771 | 0.042 | +0.729 |
| 4 | within | 0.307 | 0.438 | -0.132 |
| 5 | within | 0.624 | 0.192 | +0.432 |
| 6 | within | 0.172 | 0.133 | +0.039 |
| 7 | within | 0.764 | 0.175 | +0.589 |
| 8 | within | 0.819 | 0.150 | +0.669 |
| 9 | within | 0.583 | 0.025 | +0.558 |

## Sensitivity sweeps (mean ± std κ per protocol × axis value)

### alpha

| value | protocol | κ (mean ± std) | n_subjects |
|---|---|---|---|
| 0.1 | official | 0.159 ± 0.216 | 9 |
| 0.1 | within | 0.091 ± 0.181 | 9 |
| 0.5 | official | 0.181 ± 0.211 | 9 |
| 0.5 | within | 0.184 ± 0.226 | 9 |
| 1.0 | official | 0.010 ± 0.141 | 9 |
| 1.0 | within | 0.109 ± 0.195 | 9 |
| 2.0 | official | 0.071 ± 0.108 | 9 |
| 2.0 | within | 0.089 ± 0.186 | 9 |
| 5.0 | official | 0.006 ± 0.156 | 9 |
| 5.0 | within | 0.050 ± 0.105 | 9 |

### arm_pool

| value | protocol | κ (mean ± std) | n_subjects |
|---|---|---|---|
| full | official | 0.162 ± 0.226 | 9 |
| full | within | 0.111 ± 0.154 | 9 |
| pruned | official | 0.010 ± 0.141 | 9 |
| pruned | within | 0.109 ± 0.195 | 9 |

### budget_frac

| value | protocol | κ (mean ± std) | n_subjects |
|---|---|---|---|
| 0.25 | official | 0.092 ± 0.243 | 9 |
| 0.25 | within | 0.072 ± 0.211 | 9 |
| 0.5 | official | 0.010 ± 0.141 | 9 |
| 0.5 | within | 0.072 ± 0.211 | 9 |
| 1.0 | official | 0.010 ± 0.141 | 9 |
| 1.0 | within | 0.109 ± 0.195 | 9 |
| 2.0 | official | 0.010 ± 0.141 | 9 |
| 2.0 | within | 0.109 ± 0.195 | 9 |

### calibration_frac

| value | protocol | κ (mean ± std) | n_subjects |
|---|---|---|---|
| 0.2 | official | 0.130 ± 0.127 | 9 |
| 0.2 | within | 0.177 ± 0.175 | 9 |
| 0.3 | official | 0.010 ± 0.141 | 9 |
| 0.3 | within | 0.109 ± 0.195 | 9 |
| 0.5 | official | 0.080 ± 0.136 | 9 |
| 0.5 | within | 0.154 ± 0.187 | 9 |

### include_recent_rewards

| value | protocol | κ (mean ± std) | n_subjects |
|---|---|---|---|
| False | official | 0.134 ± 0.208 | 9 |
| False | within | 0.171 ± 0.173 | 9 |
| True | official | 0.081 ± 0.148 | 9 |
| True | within | 0.109 ± 0.201 | 9 |

### per_round_cap

| value | protocol | κ (mean ± std) | n_subjects |
|---|---|---|---|
| 2.0 | official | 0.046 ± 0.098 | 9 |
| 2.0 | within | 0.071 ± 0.147 | 9 |
| 3.0 | official | 0.081 ± 0.148 | 9 |
| 3.0 | within | 0.109 ± 0.201 | 9 |
| 4.0 | official | 0.081 ± 0.148 | 9 |
| 4.0 | within | 0.109 ± 0.201 | 9 |
| inf | official | 0.081 ± 0.148 | 9 |
| inf | within | 0.109 ± 0.201 | 9 |

## Policy ablation — design-doc §8.4

Compares OPLB (default) against three drop-in alternatives: **fixed** (top-κ calibration arm, no exploration), **eps_greedy** (random ε-exploration instead of UCB), and **unconstrained** (OPLB with knapsack stripped). κ is the mean ± std across 9 subjects × seeds at the default hyperparameter cell.

| policy | protocol | κ (mean ± std) | n rows | n_subjects |
|---|---|---|---|---|
| oplb | official | 0.081 ± 0.148 | 45 | 9 |
| oplb | within | 0.109 ± 0.201 | 45 | 9 |
| fixed | official | 0.165 ± 0.209 | 45 | 9 |
| fixed | within | 0.122 ± 0.233 | 45 | 9 |
| eps_greedy | official | 0.135 ± 0.172 | 45 | 9 |
| eps_greedy | within | 0.155 ± 0.203 | 45 | 9 |
| unconstrained | official | 0.081 ± 0.148 | 45 | 9 |
| unconstrained | within | 0.109 ± 0.201 | 45 | 9 |

---
The CCB thesis contribution is to shrink Δκ on 2b while obeying the no-leakage constraint (never touching 2a at training time). Values reported here correspond to the default hyperparameter cell `(budget_frac=1.0, alpha=1.0, arm_pool=pruned, calibration_frac=0.3)`; other cells are tabulated in the sensitivity sections above.
