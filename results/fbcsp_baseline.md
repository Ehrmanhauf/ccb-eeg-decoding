# FBCSP+LDA baseline — per-subject κ and accuracy

## Per-subject

| Dataset | Subject | κ within | acc within | κ official | acc official |
|---|---|---|---|---|---|
| BCI-IV-2a | 1 | 0.618 | 0.809 | 0.611 | 0.806 |
| BCI-IV-2a | 2 | 0.133 | 0.566 | -0.042 | 0.479 |
| BCI-IV-2a | 3 | 0.771 | 0.886 | 0.750 | 0.875 |
| BCI-IV-2a | 4 | 0.307 | 0.653 | 0.167 | 0.583 |
| BCI-IV-2a | 5 | 0.624 | 0.812 | 0.625 | 0.812 |
| BCI-IV-2a | 6 | 0.172 | 0.587 | 0.347 | 0.674 |
| BCI-IV-2a | 7 | 0.764 | 0.882 | 0.500 | 0.750 |
| BCI-IV-2a | 8 | 0.819 | 0.910 | 0.667 | 0.833 |
| BCI-IV-2a | 9 | 0.583 | 0.792 | 0.583 | 0.792 |
| BCI-IV-2b | 1 | 0.333 | 0.667 | 0.083 | 0.542 |
| BCI-IV-2b | 2 | 0.067 | 0.533 | 0.117 | 0.558 |
| BCI-IV-2b | 3 | 0.150 | 0.575 | 0.250 | 0.625 |
| BCI-IV-2b | 4 | 0.523 | 0.762 | 0.443 | 0.721 |
| BCI-IV-2b | 5 | 0.523 | 0.762 | 0.343 | 0.671 |
| BCI-IV-2b | 6 | 0.325 | 0.662 | 0.367 | 0.683 |
| BCI-IV-2b | 7 | 0.333 | 0.667 | 0.450 | 0.725 |
| BCI-IV-2b | 8 | 0.150 | 0.575 | 0.150 | 0.575 |
| BCI-IV-2b | 9 | 0.225 | 0.613 | 0.200 | 0.600 |

## Per-dataset summary (mean ± std over 9 subjects)

| Dataset | κ within | κ official |
|---|---|---|
| BCI-IV-2a | 0.532 ± 0.263 | 0.468 ± 0.261 |
| BCI-IV-2b | 0.292 ± 0.161 | 0.267 ± 0.139 |

## Gap-to-benchmark Δκ = κ<sub>2a</sub> − κ<sub>2b</sub>

- Within-subject Δκ = **+0.240**
- Official protocol Δκ = **+0.201**

The CCB thesis contribution is to shrink these gap numbers while the 3-channel CCB policy keeps the no-leakage constraint.
