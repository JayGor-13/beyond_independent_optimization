# Repository preparation notes

Prepared 26 November 2026 from the paper's research workspace.

- Runnable code is grouped by experiment under `experiments/`.
- Shared analysis and CPU test dependencies live in `requirements/`. Archived prototypes are nested under `experiments/prototypes/`.
- Historical measurements and the original output notebook are separate from code and are retained byte-for-byte.
- The full-sweep InternVL source is selected as the canonical multimodel package; stale duplicate package copies are omitted.
- The unresolved Gemma prototype is retained separately and is not presented as a working experiment.
- Extraction, plotting, notebook builders, and TextVQA helper discovery use the new repository paths.
- The manuscript's missing `.jpg` references now point to the supplied vector PDFs. Its missing Figure 4 reference uses the supplied Pareto PDF. The bibliography is available as the expected `references.bib`.
- Official TMLR style files are included with source attribution and their upstream license.
- Local caches, downloaded environments, third-party paper PDFs, duplicate ZIPs, internal reviewer/revision notes, old manuscript snapshots and generated outputs are outside this release.

The research workspace remains the source archive. This repository is a separate upload-ready copy; code changes made here should be developed here to avoid diverging duplicate implementations. [The source manifest](source_manifest.json) records the initial mapping.

## Publication metadata

The manuscript still uses the supplied anonymized author block and placeholder publication metadata. Before a public paper release, fill those fields and add the final paper citation. No author identities, DOI, acceptance date, GitHub owner or repository URL have been guessed.

A repository-wide software license has not been selected. Add the authors' chosen license before granting reuse rights; the bundled TMLR style license applies only to those upstream files. Preparing the repository does not assign a license to the research code or manuscript.

## Upload

Upload the contents of this directory as the repository root, or use `dist/tmlr-survey-code.zip`. Do not upload the enclosing research workspace. The archive contains a `SHA256SUMS` file and can be rebuilt with `python tools/build_release.py`.

Repository checks and CPU workflows are configured in [GitHub Actions](../.github/workflows/checks.yml). They do not run full GPU experiments or certify scientific conclusions. See [the validation record](VALIDATION.md) for what was actually checked during preparation.
