# Manuscript source

`main.tex` is the supplied current manuscript, with asset-path repairs for the repository layout. `references.bib` is the supplied `tmlr.bib`, renamed to match the manuscript's bibliography command. Figures retain their existing values; Figure 4 is the supplied illustrative Pareto PDF, not a new throughput measurement.

Build from this directory with a TeX distribution containing the packages used by the manuscript:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Or run `make paper` at the repository root. The official TMLR style source and license are recorded alongside the two style files. Repository validation checks asset references and bibliography/style-file presence; a full LaTeX build is a separate check.

Author and publication metadata remain anonymized/placeholders from the supplied source. This copy does not assert a final publication date, DOI, or author list. Empirical tables can be regenerated with `python tools/reproduce_paper.py` from the repository root. Updating the manuscript with regenerated tables is explicit: `python tools/sync_paper_tables.py --tables artifacts/reproduction/tables`; the previous source is saved in ignored `artifacts/manuscript_backups/`.
