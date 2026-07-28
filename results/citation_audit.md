# Citation audit — every cited reference verified against authoritative sources

**Date:** 2026-06-11
**Scope:** all `\cite`d entries in `design-doc/references.bib` (identical to the manuscript bibliography).
**Question answered:** are all references real, existing papers with correct metadata, no hallucinations?

## Method

`design-doc/references.bib` holds 64 entries; 58 are actually cited in the thesis text. Each cited
entry was checked against at least one **authoritative** source (not memory):

- **DOI resolution** via `doi.org` and the **Crossref** REST API (`api.crossref.org/works/<doi>`) — the canonical DOI metadata registry;
- **DBLP** for CS venues; **PMLR** (`proceedings.mlr.press`), **NeurIPS proceedings**, **JMLR**, **AAAI OJS**, **MLSys proceedings**, **ACL/OpenReview** for conference/journal papers;
- **PubMed / PubMed Central** for the neuro/physiology references;
- publisher pages (IEEE Xplore, IOPscience, Frontiers, Nature/Scientific Data, MDPI, Springer, Elsevier, Wiley, de Gruyter, OUP);
- **arXiv** for the `@misc` preprints (confirming the entry claims no false peer-reviewed venue).

Each entry was cross-checked on title, first author (usually full author list), venue
(journal vs. conference), and year. The edited `.bib` was then re-validated with
`biber --tool --validate-datamodel` (clean parse, no datamodel warnings).

## Result

**58 / 58 cited references are real, existing papers.** No fabricated or hallucinated entries.
**2** entries had a metadata defect (not a fabrication); both corrected against the canonical record.

### Corrections applied (both bib files)

1. **`hinss2022cogbci`** — the `@article` (DOI `10.1038/s41597-022-01898-y`) carried the *Zenodo
   dataset* title/year/volume rather than the *published Scientific Data article*. The earlier
   fields were read from the Zenodo deposit because the Nature page sits behind an auth wall.
   Verified against **Crossref** and **PMC9918545**:
   - title `COG-BCI Database: A Multi-Session and Multi-Task EEG Cognitive Dataset for Passive Brain–Computer Interfaces` → `Open Multi-Session and Multi-Task EEG Cognitive Dataset for Passive Brain–Computer Interface Applications`
   - volume `9` → `10`; year `2022` → `2023`; added article number `pages = {85}`.
   - author list and DOI were already correct. Citation key left unchanged (renaming would touch ~8 `\cite` sites for no correctness gain); APA author-year now renders "(Hinss et al., 2023)".

2. **`li2020asha`** — author list was missing the 5th author. Verified against the **MLSys 2020
   proceedings** page (8 authors): inserted `Ben-tzur, Jonathan` between Gonina and Hardt.

### Bibliography hygiene note (no action needed)

Many entries carry long internal `note` fields (provenance, dataset descriptions). Inspection of
the `biblatex-apa` style (`apa.bbx`) confirms `\printfield{note}` is reached only by the
`legadminmaterial`/`legmaterial` legal drivers; the `article`, `inproceedings`, and `misc`
drivers print `addendum`/`annotation`, **not** `note`. So these internal notes do **not** appear
in the rendered References — consistent with the clean prior compile.

## Per-entry verdicts (58 cited)

All VERIFIED except the two CORRECTED above. Representative authoritative source per entry:

| key | verdict | source |
|---|---|---|
| li2010linucb | VERIFIED | DBLP / ACM DL (WWW 2010) |
| abbasi2011oful | VERIFIED | NeurIPS proceedings (NIPS 2011) |
| agrawal2013ts | VERIFIED | PMLR v28 (ICML 2013) |
| badanidiyuru2018bwk | VERIFIED | Crossref 10.1145/3164539 (JACM) |
| wu2015cccb | VERIFIED | DBLP / NeurIPS (NIPS 2015) |
| agrawal2016linbwk | VERIFIED | NeurIPS proceedings (NIPS 2016) |
| pacchiano2021linconstr | VERIFIED | PMLR v130 (AISTATS 2021) |
| fruitet2013ucbclassif | VERIFIED | IOPscience 10.1088/1741-2560/10/1/016012 |
| ma2021tsp300 | VERIFIED | DBLP / IEEE BIBM 2021 |
| heskebeck2022mabbci | VERIFIED | Frontiers Hum. Neurosci. 10.3389/fnhum.2022.931085 |
| abdullah2022channelsel | VERIFIED | PMC9774545 / MDPI Bioengineering |
| koles1990csp | VERIFIED | Crossref 10.1007/BF01129656 (Brain Topography) |
| ramoser2000csp | VERIFIED | PubMed 11204034 / IEEE 10.1109/86.895946 |
| pfurtscheller2001mi | VERIFIED | TU Graz Pure / IEEE 10.1109/5.939829 |
| ang2008fbcsp | VERIFIED | IEEE IJCNN 2008 10.1109/IJCNN.2008.4634130 |
| ang2012fbcsp | VERIFIED | Frontiers Neurosci. 10.3389/fnins.2012.00039 |
| lotte2018review | VERIFIED | IOPscience 10.1088/1741-2552/aab2f2 |
| cho2017mieeg | VERIFIED | OUP GigaScience 10.1093/gigascience/gix034 |
| zheng2017vigilance | VERIFIED | IOPscience 10.1088/1741-2552/aa5a98 |
| lim2018stew | **CORRECTED** (2026-07-19) | IEEE Xplore / OpenAlex — third author "Wang, Lipo P." had a spurious middle initial; corrected to "Wang, Lipo" |
| kocanogullari2018query | VERIFIED | PMC6777547 / IEEE SP Letters |
| zhou2023seqbai | VERIFIED | arXiv 2305.11908 (@misc) |
| fidencio2022errprl | **CORRECTED** (2026-07-19) | Crossref structured family-name field — surname is the compound "Xavier Fidêncio"; author field re-braced so BibTeX parses it correctly |
| lei2017actorcriticmhealth | VERIFIED | arXiv 1706.09090 (@misc) |
| zhu2018robustmhealth | VERIFIED | arXiv 1802.09714 (@misc) |
| tomkins2021intelligentpooling | VERIFIED | Springer Mach. Learn. 110 / PMC8494236 |
| yang2020hatch | VERIFIED | ACM DL 10.1145/3366423.3380115 (WWW 2020) |
| jagerman2020safeexploration | VERIFIED | Crossref 10.1145/3385670 (ACM TOIS) |
| belfer2022adaptivecurriculum | **CORRECTED** (2026-07-19) | Crossref / DBLP / OpenAlex — pages 617–629 → 724–730 |
| pacchiano2025stagewise | VERIFIED | JMLR v26 (2025) |
| rangi2018unifyingbwk | **CORRECTED** (2026-07-19) | Version of record found: IJCAI 2019, pp. 3311–3317, doi 10.24963/ijcai.2019/459 — @misc preprint replaced |
| li2018hyperband | VERIFIED | JMLR v18 (2018) |
| jamieson2016nonstochastic | VERIFIED | PMLR v51 (AISTATS 2016) |
| falkner2018bohb | VERIFIED | PMLR v80 (ICML 2018) |
| li2020asha | **CORRECTED** | MLSys 2020 proceedings — added missing author Ben-tzur |
| ding2013mabbv | VERIFIED | AAAI OJS (AAAI 2013) |
| wei2024latencyaware | VERIFIED | arXiv 2410.13109 (@misc) |
| banerjee2024askquery | **CORRECTED** (2026-07-19) | Version of record found: IEEE ICRA 2025, pp. 1378–1384, doi 10.1109/ICRA55743.2025.11127795 — @misc preprint replaced |
| albuquerque2020wauc | VERIFIED | Frontiers Neurosci. 10.3389/fnins.2020.549524 |
| hinss2022cogbci | **CORRECTED** | Crossref / PMC9918545 — title/volume/year fixed to the published article |
| hernandez2022pilots | VERIFIED | MDPI Appl. Sci. 10.3390/app12052298 |
| dinges1985pvt | VERIFIED | Crossref 10.3758/BF03200977 |
| basner2011pvt | VERIFIED | OUP Sleep 10.1093/sleep/34.5.581 |
| klimesch1999alphareview | VERIFIED | Crossref 10.1016/s0165-0173(98)00056-3 |
| pope1995engagement | VERIFIED | PubMed 7647180 / Elsevier Biol. Psychol. |
| mullen2015asr | VERIFIED | PubMed 26415149 / IEEE TBME |
| guo2021p300rl | VERIFIED | IEEE CISP-BMEI 2021 (doc 9624451) |
| foster2019modelselection | VERIFIED | NeurIPS proceedings (NeurIPS 2019) |
| lin2018adaptivefeature | VERIFIED | DBLP / IEEE ICDMW 2018 |
| hsu2024neuralts | VERIFIED | IEEE ICCPS 2024 10.1109/ICCPS61052.2024.00027 |
| wilson2024omsmab | VERIFIED | Wiley Expert Systems 10.1111/exsy.13626 |
| abdullah2023dqnchannel | VERIFIED | Crossref 10.1109/REEDCON57544.2023.10151281 |
| lotte2007review | VERIFIED | IOPscience 10.1088/1741-2560/4/2/R01 |
| steyrl2016randomforest | VERIFIED | PubMed 25830903 / de Gruyter Biomed. Eng. |
| welch1967 | VERIFIED | NASA ADS / IEEE 10.1109/TAU.1967.1161901 |
| blankertz2008csp | VERIFIED | NASA ADS / IEEE 10.1109/MSP.2008.4408441 |
| blankertz2011shrinkage | VERIFIED | Elsevier NeuroImage 10.1016/j.neuroimage.2010.06.048 |
| ledoit2004shrinkage | VERIFIED | Crossref 10.1016/S0047-259X(03)00096-4 |

(6 of the 64 bib entries are uncited and not part of this audit.)


---

## Re-audit, 2026-07-19 (pre-submission pass)

The full bibliography was re-verified from scratch ahead of final submission. Every one of the
**72** entries then in `design-doc/references.bib` was checked independently against DOI resolution
(doi.org / Crossref), DBLP, and the publisher's canonical page, covering author list and order,
title, venue, entry type, year, volume and pages. Any entry flagged on the first pass was then
re-checked by a **second, independent** verification attempt before any edit was made, so that a
single failed search could not condemn a sound reference.

**Result: no fabricated reference, and no unverifiable reference.** All 72 entries describe real,
locatable works. Six required metadata corrections, all applied and listed above and below:

| Key | Defect | Correction |
|---|---|---|
| `lim2018stew` | spurious middle initial in third author | `Wang, Lipo P.` → `Wang, Lipo` |
| `fidencio2022errprl` | compound surname mis-parsed by BibTeX | author re-braced as `{Xavier Fid{\^e}ncio}, Aline` |
| `belfer2022adaptivecurriculum` | wrong page range | `617--629` → `724--730` |
| `rangi2018unifyingbwk` | cited the preprint; a version of record exists | `@misc` arXiv → `@inproceedings` IJCAI 2019 |
| `banerjee2024askquery` | cited the preprint; a version of record exists | `@misc` arXiv → `@inproceedings` IEEE ICRA 2025 |
| `aristimunha2023moabb` | version/DOI described v1.0.0, not the version used | pinned to the MOABB **1.5.0** actually used (`uv.lock`), with the Zenodo concept DOI explicitly labelled as all-versions |

Five entries were subsequently **added** to support the operational-interpretation section
(§`sec:results-meaning`), each verified the same way and each cited for a claim the source
demonstrably makes: `landis1977observer`, `sim2005kappa`, `combrisson2015exceeding`,
`kubler2008paralysis`, `vidaurre2010cure`. Candidate sources whose *content* could not be read
directly — only their metadata — were deliberately **not** added, so that no threshold is
attributed to a paper without confirming the paper states it.

Final state: **77 entries, 77 cited, zero uncited entries and zero undefined citations.**
