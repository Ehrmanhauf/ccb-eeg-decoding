# Notice on data and third-party material

The [MIT licence](LICENSE) covers the **source code, documentation and experiment result files**
in this repository only.

**No EEG recordings, dataset label files, or dataset description documents are distributed here.**
Each dataset used in this work — BCI Competition IV 2a/2b, Cho2017, STEW, WAUC, UAB Flight-Deck,
COG-BCI — remains under the licence and terms of use set by its original providers, and must be
obtained directly from them. See [README.md → Getting the data](README.md#getting-the-data) for
the source of each, and the `data/*.README.md` notes for the expected on-disk layout.

The BCI Competition IV dataset description documents (`desc_2a.pdf`, `desc_2b.pdf`), cited
throughout the code comments as the authority on montage order and trial timing, are the
competition organisers' own documents; download them from <https://www.bbci.de/competition/iv/>.

If you use one of these datasets, cite its source paper. BibTeX entries for all of them are in
[`design-doc/references.bib`](design-doc/references.bib).
