r"""Classical-classifier comparators (B3--B5) on the fixed-pipeline features.

Provides off-the-shelf scikit-learn classifier heads — an RBF-kernel Support
Vector Machine (B3), a single CART Decision Tree (B4), and a Random Forest (B5)
— to be fitted on the **same** engineered features as the fixed-pipeline
baselines B1/B2 and the CCB arm-heads (see ``thesis.baselines.feature_transformers``).
Holding the feature representation fixed and varying only the classifier head
isolates the classifier-choice effect: it answers the recurring reviewer
question "how does the CCB compare to a decision tree (or an SVM, or a random
forest)?" by testing whether the CCB-vs-fixed-pipeline gap is specific to the
shrinkage-LDA head or general to fixed classical classifiers.

**Choice / alternatives / reason (CLAUDE.md justification discipline):**

- *Choice:* SVM-RBF (B3), CART Decision Tree (B4), Random Forest (B5), each on
  the B1 (FBCSP) and/or B2 (band-power) engineered features.
- *Alternatives considered:* k-nearest-neighbours, logistic regression, naive
  Bayes, gradient boosting; and raw-epoch (no-feature-engineering) inputs.
- *Reason:* the SVM and tree-family classifiers are the standard non-LDA
  comparators in the canonical EEG-BCI classifier reviews of Lotte et al. 2007
  (``lotte2007review``) and 2018 (``lotte2018review``). The SVM is the
  most-cited non-LDA BCI classifier; the single Decision Tree is the
  interpretable baseline a reviewer names explicitly; the Random Forest is its
  ensemble strengthening. k-NN / logistic regression / naive Bayes add little
  beyond LDA+SVM in those reviews, and raw-epoch inputs (e.g. 22 ch × 1000
  samples) are over-parameterised near-chance straw men, so they are excluded.
  Heads are held at scikit-learn **library defaults** to avoid a per-head
  tuning protocol that would itself demand leakage-safe nested CV; this
  untuned-defaults decision is tracked in
  ``design-doc/open-justifications.md``.

**No-leakage:** features are fitted train-only (the supervised FBCSP CSP is
re-fit per fold; the SVM's ``StandardScaler`` is fitted train-only inside the
head pipeline); each dataset / configuration is fitted only on its own trials,
honouring the generalised no-leakage rule of ``CLAUDE.md`` §2.
"""

from __future__ import annotations

from typing import Mapping

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from thesis.baselines.feature_transformers import BandPowerTransformer, FBCSPTransformer

# JUSTIFY (open-justifications.md: "classical-baseline classifier heads at
# library defaults"): 200 trees is a common library-scale Random-Forest default
# trading variance reduction against runtime; it is not separately
# literature-anchored for BCI. Held fixed across all datasets.
_RF_N_ESTIMATORS = 200

#: The three classical comparator heads reported in the thesis (B3--B5).
CLASSICAL_CLASSIFIERS: tuple[str, ...] = ("svm", "decision_tree", "random_forest")

#: Engineered feature families a classifier head can be paired with.
FEATURE_FAMILIES: tuple[str, ...] = ("fbcsp", "bandpower")

#: Human-readable labels for tables / summaries.
CLASSIFIER_LABELS: dict[str, str] = {
    "lda": "shrinkage LDA",
    "svm": "SVM (RBF)",
    "decision_tree": "Decision Tree (CART)",
    "random_forest": "Random Forest",
}


def make_classifier(name: str, *, random_state: int = 42) -> BaseEstimator:
    """Return a classifier head at locked scikit-learn defaults.

    The SVM is wrapped with a ``StandardScaler`` (RBF kernels are scale
    sensitive); the tree-family heads are scale invariant and need no scaler.
    ``probability=False`` keeps the SVM fast — only hard predictions are needed
    for accuracy / Cohen's :math:`\\kappa`. The shrinkage-LDA head is exposed too
    as a consistency anchor that reproduces the B1/B2 baselines on the same
    features.
    """
    if name == "svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=1.0,
                        kernel="rbf",
                        gamma="scale",
                        probability=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=_RF_N_ESTIMATORS, random_state=random_state)
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    raise ValueError(f"unknown classifier name: {name!r}")


def make_feature_transformer(
    family: str,
    *,
    sfreq: float = 250.0,
    channel_roles: Mapping[str, list[int]] | None = None,
) -> BaseEstimator:
    """Return the leakage-safe feature transformer for the requested family.

    ``channel_roles`` is consumed only by the band-power family (it appends the
    cognitive-load derived aggregates); it is ignored by the FBCSP family.
    """
    if family == "fbcsp":
        return FBCSPTransformer(sfreq=sfreq)
    if family == "bandpower":
        return BandPowerTransformer(sfreq=sfreq, channel_roles=channel_roles)
    raise ValueError(f"unknown feature family: {family!r}")


def make_classical_pipeline(
    family: str,
    classifier: str,
    *,
    sfreq: float = 250.0,
    channel_roles: Mapping[str, list[int]] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Full leakage-safe pipeline: feature transformer -> (scaler ->) classifier.

    This is the canonical single-object form (used by the unit tests). The
    experiment runner extracts features once per fold and reuses them across the
    several heads for efficiency, but the per-fold semantics are identical to
    this pipeline's: the transformer (and the SVM scaler) are fitted on the
    training split only.
    """
    transformer = make_feature_transformer(family, sfreq=sfreq, channel_roles=channel_roles)
    head = make_classifier(classifier, random_state=random_state)
    return Pipeline([("feat", transformer), ("clf", head)])
