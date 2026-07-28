# Design documents

The written specification behind the code. Under this project's
[justification discipline](../CLAUDE.md), no non-trivial choice — a filter cutoff, an arm-pool
composition, a protocol, a reward definition — ships without a recorded reason, and these
documents are where those reasons live.

| Document | What it is |
|---|---|
| [`ccb-formulation.md`](ccb-formulation.md) | The full spec. Scope and framing (§1.4), datasets and their provenance (§2), the no-leakage rule (§2.4), the CCB formulation, arm pool, reward and constraint (§3–§5), evaluation protocols, and the recorded results. Start here. |
| [`primer.md`](primer.md) | Self-contained background: multi-armed and contextual bandits, the constrained variant, EEG decoding for motor imagery / cognitive load / vigilance, and the metrics. Read this first if either half of the topic is new. |
| [`open-justifications.md`](open-justifications.md) | Every `JUSTIFY:` marker in the codebase, tracked to resolution — open items at the top, closed ones with their evidence (a BibTeX key, an experiment CSV, or a derivation from a locked principle) below. |
| [`references.bib`](references.bib) | The bibliography. Every entry was verified against DOI/Crossref, DBLP, or the publisher's canonical page — the audit is in [`../results/citation_audit.md`](../results/citation_audit.md). |

## A note on cross-references

The private working repository also held internal planning documents — an advisor progress
briefing, the near-ear reframe work plan, a benchmark-comparison plan, a cleanup manifest, and a
finalization checklist. Those are **not part of this public mirror**: they are process artifacts,
and some quote supervisor correspondence.

A handful of code comments and design-doc passages still cite them by name (for example
`design-doc/near-ear-reframe-workplan.md §3.1` in the dataset loaders). Those citations are kept
as **historical provenance markers** rather than rewritten, so the audit trail stays honest —
treat them as "this choice was recorded in the project's planning notes on that date", not as a
link you can follow. The substantive content they anchor is reproduced in
`ccb-formulation.md` and `open-justifications.md`.

Likewise, `desc_2a.pdf` and `desc_2b.pdf` are cited throughout as the authority on the BCI-IV
montage order and trial timing. They are the competition organisers' own documents and are not
redistributed here — download them from <https://www.bbci.de/competition/iv/>.
