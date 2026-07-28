# Technical Primer: CCB, EEG Baselines, and Metrics

Self-contained background for the design document. Read this first if bandits or EEG signal processing are new. Math is included where it clarifies, but every section opens with intuition.

> **Scope note (2026-05-19).** The thesis has been re-anchored to a CL-primary multi-paradigm scope (`ccb-formulation.md` §1.4). This primer's worked examples in §4–§7 use motor imagery because that is where the CCB literature was originally validated and where our own validated baselines live — it is the **classical entry point**. The mapping in §4 and the metrics in §6 generalize to cognitive load (STEW) and fatigue (SEED-VIG) per `ccb-formulation.md` §2.6–§2.7; paradigm-specific feature extractors replace the MI-specific CSP/FBCSP stack with band-power, asymmetry, and engagement-style features, but the bandit machinery is unchanged.

---

## 1. Multi-Armed Bandits (MAB) — the base problem

**Setup.** You face $K$ "arms" (imagine slot machines). At each round $t = 1, \dots, T$ you pull one arm $a_t$ and observe a stochastic reward $r_t$. Each arm $a$ has an unknown mean reward $\mu_a$. The goal is to maximize $\sum_t r_t$.

**The tension.** You don't know the means up front, so you face exploration vs exploitation:
- **Exploit**: pull the arm that looks best so far.
- **Explore**: pull under-sampled arms to reduce uncertainty.

**Regret.** The standard yardstick:

$$
R_T = \underbrace{T \cdot \max_a \mu_a}_{\text{best fixed arm in hindsight}} - \mathbb{E}\!\left[\sum_{t=1}^T r_t\right].
$$

If $R_T$ is *sublinear* in $T$ (e.g., $O(\sqrt{T})$ or $O(\log T)$), the algorithm eventually matches the best arm. Linear regret means you never learn.

**Canonical algorithm — UCB1.** At round $t$, for each arm $a$ pulled $n_a$ times with empirical mean $\hat{\mu}_a$, play

$$
a_t = \arg\max_a \left(\hat{\mu}_a + \sqrt{\tfrac{2 \log t}{n_a}}\right).
$$

The second term is an "optimism bonus" that shrinks as $n_a$ grows. You end up pulling each arm enough times to estimate it, then locking onto the best.

---

## 2. Contextual Bandits (CB) — arms that depend on state

A plain MAB has no notion of "situation." Real problems do: which ad to show *this* user, which treatment for *this* patient, which EEG pipeline for *this* trial.

**Setup.** At round $t$:
1. Observe **context** $x_t \in \mathbb{R}^d$ (features describing the current situation).
2. Choose arm $a_t$ using $x_t$ and the history.
3. Observe reward $r_t$ that depends on $(x_t, a_t)$.

**Linear assumption.** The workhorse model: there is an unknown $\theta^* \in \mathbb{R}^d$ such that
$$
\mathbb{E}[r_t \mid x_t, a_t] = \langle \theta^*, \phi(x_t, a_t) \rangle,
$$
for some context-arm feature map $\phi$.

**Three algorithms you need to know.**

- **LinUCB** [Li et al. 2010] — UCB applied to the linear model. Maintain $\hat{\theta}_t$ (ridge-regression estimate) and a design matrix $A_t$; play
  $$
  a_t = \arg\max_a \Big[ \langle \hat{\theta}_t, \phi(x_t,a) \rangle + \alpha \sqrt{\phi(x_t,a)^\top A_t^{-1} \phi(x_t,a)} \Big].
  $$
  Regret $\tilde{O}(d\sqrt{T})$.

- **OFUL** [Abbasi-Yadkori et al. 2011] — Same idea but with tighter "self-normalized" confidence sets, tightening the regret constant. Every modern linear-bandit paper uses OFUL-style confidence sets.

- **Thompson Sampling (TS)** [Agrawal & Goyal 2013] — Bayesian alternative. Maintain a posterior over $\theta^*$ (Gaussian, with closed-form updates under Gaussian noise). Each round, *sample* $\tilde{\theta}_t$ from the posterior and play $\arg\max_a \langle \tilde{\theta}_t, \phi(x_t,a)\rangle$. Often better empirically; worst-case regret $\tilde{O}(d^{3/2}\sqrt{T})$.

---

## 3. Constrained Contextual Bandits (CCB) — adding a budget

Real deployments have resource limits: compute, latency, calibration trials, battery. CCB formalizes these.

**Setup.** Same as CB, but each pull also incurs a **cost** $c_t(a)$. Common constraint types:

| Constraint type | Formal | Example |
|---|---|---|
| Cumulative budget (knapsack) | $\sum_t c_t(a_t) \le B$ | Total compute budget per session |
| Per-round sparsity | $\lvert\{\text{active arms}\}\rvert \le K$ | At most $K$ feature extractors run per trial |
| Expected-cost threshold | $\mathbb{E}[c(a_t)] \le \tau$ | Average latency below $\tau$ ms |

**Key algorithms.**

- **Bandits with Knapsacks (BwK)** [Badanidiyuru, Kleinberg, Slivkins 2013/JACM 2018] — Non-contextual base case. The game ends when *any* budget is exhausted.
- **Linear CB with Knapsacks (linCBwK)** [Agrawal & Devanur 2016] — Contextual linear version. Both reward and cost are linear in the context.
- **Constrained Contextual Bandits** [Wu, Srikant, Liu, Jiang 2015] — The direct namesake of this thesis. Gives logarithmic or $\sqrt{T}$ regret under an expected-cost threshold.
- **OPLB** [Pacchiano, Ghavamzadeh, Bartlett, Jiang 2021] — The concrete algorithm we chose. UCB-style with a per-round feasibility filter (only play arms whose UCB-cost is below the remaining budget). Regret
  $$
  R_T \le C \cdot \frac{d \sqrt{T \log T}}{\zeta} + O(1),
  $$
  where $\zeta > 0$ is the slack between the best feasible action's cost and the constraint threshold. Tighter slack = harder problem.

**Mental model.** CCB lets you say: "be adaptive, but don't exceed this resource." In our case, the resource is *feature-extraction cost*, which matters because low-cost BCI hardware has limited compute.

---

## 4. How CCB Maps to Our EEG Problem (Formulation A)

The full spec is in [ccb-formulation.md](ccb-formulation.md) §6. The intuition:

| Bandit concept | What it is in our problem |
|---|---|
| Round $t$ | One motor-imagery trial (~4 s of 3-channel EEG) |
| Context $x_t$ | Per-trial signal stats: band-powers ($\mu$, $\beta$), spectral entropy, variance, artifact flag, recent arm-wise rewards |
| Arm $a = (b, s, f, w)$ | A feature-extraction pipeline: sub-band $b$ × spatial filter $s$ × feature type $f$ × time window $w$ |
| Feature map $\phi_a(E_t)$ | Process raw epoch $E_t$ through pipeline $a$ → feature vector |
| Reward $r_t$ | $\mathbf{1}\{\hat{y}_t = y_t\}$ — classification correctness (labels come from the visual cue) |
| Cost $c(a)$ | Feature-vector length or compute cost of pipeline $a$ |
| Constraint | $\sum_t c(a_t) \le B$ (compute budget per session) |

**The CCB is a meta-learner** — it doesn't classify directly; it decides which pipeline to use on each trial. Each arm's "head" (the final linear classifier $w_a$) is pre-trained offline. What the CCB learns online is which arm is best for the current subject/trial/context, subject to budget.

**Why this is not just channel selection.** The 3 channels (C3, Cz, C4) are *fixed hardware inputs* — they're bipolar recordings, not a choice. The bandit adapts *how to process* those 3 channels, not which to use.

---

## 5. EEG Feature Extraction — the Building Blocks

### 5.1 Motor-imagery neurophysiology (1 paragraph)

Imagining hand movement desynchronizes the mu (8–12 Hz) and beta (13–30 Hz) rhythms over the *contralateral* motor cortex: imagine your right hand → power drops at C3 (left hemisphere), and vice versa for C4. This **event-related desynchronization (ERD)** is the signal we classify. Cz is the midline reference/ground.

### 5.2 Common Spatial Patterns (CSP) [Koles 1990, Ramoser 2000]

**Intuition.** Find linear combinations of channels whose variance separates the two classes as much as possible. For MI, variance ≈ band power, so this is really finding spatial filters that maximize the left-right ERD contrast.

**Math.** Given class-conditional covariance matrices $\Sigma_1, \Sigma_2$ of the (band-passed) epochs, solve the generalized eigenvalue problem
$$
\Sigma_1 w = \lambda (\Sigma_1 + \Sigma_2) w.
$$
Eigenvectors with $\lambda$ near 1 maximize class-1 variance; eigenvectors with $\lambda$ near 0 maximize class-2. Keep $m$ from each end ($m=2$ or $m=3$ is typical, giving $2m$ components).

**Feature.** Log-variance of each component's time series.

**Limitation.** Assumes a fixed frequency band — usually $\mu$ (8–12 Hz). Different subjects peak at different bands, which motivates FBCSP.

### 5.3 Filter-Bank CSP (FBCSP) [Ang et al. 2008, 2012]

**The fix for CSP's band problem.** Run CSP independently on each of several band-pass filtered copies of the signal; concatenate features; classify.

**Our implementation** follows Ang 2012 on the filter bank and matches their 2a CSP setting, with one deliberate deviation on the classifier:
- 9 Chebyshev Type II band-pass filters: 4–8, 8–12, 12–16, ..., 36–40 Hz (Ang 2012 §2.1, verbatim).
- CSP with `n_components = 4` per band (Ang 2012 §3.1.1 uses `m = 2` on 2a; he uses `m = 1` on 2b but our default is held at 4 across datasets, with MNE silently capping to 3 on 2b — see the closed "CSP component count" justification item).
- **Classifier deviation:** shrinkage Linear Discriminant Analysis (LDA), not Ang 2012's Naïve Bayesian Parzen Window (NBPW). ref: `lotte2018review` §III.B reports the two classifiers within ~1–2 κ-points on BCI-IV; LDA gives us one fewer hyperparameter and convex optimization.
- Reported numbers from Ang 2012 under the official protocol: mean $\kappa = 0.569$ on BCI-IV-2a (4-class, Table 2 OVR); mean $\kappa = 0.599$ on BCI-IV-2b (2-class, Table 4 MIRSR variant + NBPW).

This is our **benchmark classifier** for both datasets — the gap against this is what the CCB has to close.

### 5.4 Why this matters for CCB

Each arm in Formulation A is essentially a specific (sub-band, spatial filter, feature, window) tuple — i.e., a *fragment* of an FBCSP pipeline. FBCSP runs all of them and concatenates; CCB selects a subset per trial based on context and budget. If CCB picks the wrong arms uniformly, it underperforms FBCSP. If CCB picks adaptively better than the best *fixed* arm set, it can in principle match or exceed FBCSP under the budget.

---

## 6. Metrics

- **Accuracy** — fraction correct. Simple but bad for class imbalance.
- **Cohen's $\kappa$** — the BCI-IV standard:
  $$\kappa = \frac{p_o - p_e}{1 - p_e},$$
  where $p_o$ is observed accuracy and $p_e$ is expected-by-chance accuracy. $\kappa = 0$ is chance, $\kappa = 1$ is perfect. In MI-BCI, $\kappa > 0.4$ is considered "usable," $\kappa > 0.6$ is "good."
- **Cumulative regret** $R_t = \sum_{s \le t}[r^\star(x_s) - r_s]$ — bandit-specific. Plot against $t$: sublinear curve = learning happening.
- **Per-paradigm $\kappa$** — the **headline metric** under the re-anchored framing. Reported separately for cognitive load (STEW + additional CL datasets), motor imagery (BCI-IV-2a/2b, Cho2017), and fatigue (SEED-VIG). The thesis characterizes CCB *per paradigm*; no single number "wins."
- **Gap-to-benchmark for the MI entry case** $\Delta\kappa = \kappa_{2a,\pi_{22}} - \kappa_{2b,\pi_\theta}$ — a **per-paradigm diagnostic** for the MI entry-case study, not the thesis's headline. Always $\ge 0$ for honest comparisons. Small $\Delta\kappa$ = the CCB closes the MI gap on its classical entry case; the thesis explicitly does not require this to happen for the multi-paradigm contribution to stand.

Our smoke run (subject 1 only) gave $\Delta\kappa_{\text{within}} = +0.283$, $\Delta\kappa_{\text{official}} = +0.433$ for the MI entry case. These are placeholder numbers from Phase 4; the CL primary investigation will report its own per-paradigm $\kappa$ baselines once data lands.

---

## 7. Evaluation Protocols (why three)

- **Within-subject** — split each subject's trials into train/test independently. The standard in MI-BCI because of huge inter-subject variance. Primary protocol here.
- **Official BCI Competition** — 2a: session 1 train / session 2 test (matches Ang 2012 exactly). 2b: **our adaptation** restricts to sessions 1 and 2 (screening) and uses session 1 train / session 2 test, dropping Ang's feedback sessions 3–5 — see `ccb-formulation.md` §2.2–§2.3 for the no-feedback rationale. So the 2a "official" protocol matches Ang 2012; the 2b one is a thesis-specific screening-only restriction of it.
- **LOSO (leave-one-subject-out)** — train on 8 subjects, test on the 9th; hardest. Probes cross-subject generalization — MI-BCI typically fails this because each subject's $\mu/\beta$ topography differs.

All three get reported in the thesis, with sensitivity sweeps over CCB hyperparameters ($K$, $M$, $\alpha$, $d$).

---

## 8. Common Confusions & Sharp Intuitions

1. **"CCB selects channels" — NO.** It selects feature-extraction pipelines. The 3 channels are fixed.
2. **"Why not just grid-search the best pipeline per subject?"** — Because (a) the best pipeline varies per *trial cluster* (signal drifts, artifact bursts), not just per subject; and (b) grid search over-fits the calibration set, whereas bandit regret bounds give a principled confidence story.
3. **"Why a constraint?"** — Without one, the CCB would pick the most expensive arm always. The constraint is the whole point: low-cost BCI deployment.
4. **"Why linear reward?"** — Best worst-case regret bounds, tractable updates, matches our per-arm linear heads. Non-linear CCBs exist (neural contextual bandits, kernel CB) but add complexity without obvious gain for this problem.
5. **"Where does the reward come from online?"** — From the visual cue. Every MI trial starts with a known label (left/right arrow), so the reward is observable in real time. In a true closed-loop deployment without labels, you'd need a different (e.g., self-supervised) reward.
6. **"Why is the regret bound's $\zeta$ important?"** — If the best feasible action is *exactly at* the budget boundary, the bandit has very little slack to explore nearby arms → regret blows up. Choose $B$ generously relative to the cheapest-good-enough arm.
7. **"How does CCB differ from reinforcement learning?"** — Bandits have no *state transitions*. The action doesn't change the future context. RL handles sequential state dependence; bandits are "single-step RL." For MI trials that are i.i.d.-ish given subject, bandits are the right framing.

---

## 9. Suggested Reading Order (from `references.bib`)

For someone new to bandits, I'd read these in this order:

1. **[Li et al. 2010]** — LinUCB; easiest start, clear problem motivation.
2. **[Agrawal & Goyal 2013]** — Thompson Sampling; builds intuition for posterior-based methods.
3. **[Badanidiyuru et al. 2018]** — BwK; conceptually important for constraints.
4. **[Agrawal & Devanur 2016]** — linCBwK; the contextual extension of BwK.
5. **[Pacchiano et al. 2021]** — OPLB; the concrete algorithm used here.

For EEG side:

1. **[Pfurtscheller & Neuper 2001]** — MI neurophysiology foundations.
2. **[Ramoser et al. 2000]** — CSP for MI-BCI.
3. **[Ang et al. 2012]** — FBCSP on BCI-IV-2a/2b; our benchmark numbers.
4. **[Lotte et al. 2018]** — comprehensive 10-year review of classifiers.

For the bandits-in-BCI intersection (sparse field):

1. **[Fruitet et al. 2013]** — UCB-classif, first MI-bandit paper.
2. **[Heskebeck et al. 2022]** — recent review; good positioning.

For the cognitive-load primary paradigm:

1. **[Lim et al. 2018]** — STEW dataset paper; the 1–9 subjective workload scale, 3-class binning convention, and reference $\kappa = 0.46$ baseline. Required reading before running anything CL-related.
2. *(Additional CL references to be added in the next research wave as new datasets and bandit-on-CL precedents are surveyed; tracked under `open-justifications.md`.)*

For fatigue / vigilance:

1. **[Zheng & Lu 2017]** — SEED-VIG dataset; PERCLOS-based vigilance score and the alert/drowsy operationalization the thesis adopts.

Beyond that, Lattimore & Szepesvári's free textbook *Bandit Algorithms* (Cambridge 2020) is the canonical reference — chapters 1–6 for MAB/CB, chapters 19–21 for linear bandits, chapter 28+ for adversarial.

---

*Last updated: 2026-05-19 (scope note + CL/fatigue reading lists; metric headline corrected to per-paradigm $\kappa$).*
