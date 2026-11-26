# Release validation

Validation performed on 26 November 2026, on the local Apple Silicon Mac using temporary Python 3.12 environments. Full-checkpoint GPU inference and a full LaTeX build are outside these checks.

## Completed checks

| Check | Outcome |
|---|---|
| Multimodel retention-sweep/adapters suite (Transformers 4.57.6) | 36 passed, 2 skipped: CUDA and optional pinned-source fixture unavailable |
| Original compression metric/runtime fixtures | 5 passed |
| TextVQA scoring/analysis fixtures | 3 passed |
| Graded temporal-perturbation fixtures | 8 passed |
| Control-accuracy matching/resume fixtures | 5 passed |
| Minor-revision adapter suite (Transformers 5.5.0) | 12 passed, 1 skipped: CUDA unavailable |
| Empirical paper extraction/plotting | Rebuilt both empirical comparison tables and their PDF/SVG/PNG figures from the included evidence |
| Original historical evidence | Source hashes retained in `source_manifest.json`; repository checker verifies unchanged bytes |
| Numerical reproduction | All nine included comparison/paper-value CSVs matched regenerated values within 1e-12 |

Total: **69 tests passed, 3 skipped** across separate suite processes. Static repository checks resolve all manuscript assets and canonical documentation links. Reproduced compression and temporal-routing figures were visually inspected. Kaggle bundle builders and both embedded-notebook builders executed locally.

The tests use synthetic/tiny model or metric fixtures and do not establish real-data model quality, full-checkpoint fit, GPU runtime or causal effects. GPU-only tests remain skipped locally. The Gemma prototype is excluded from supported test workflows and has an unresolved historical full-checkpoint failure.

The GitHub Actions workflow is supplied but has not been executed on GitHub during this preparation. Source and notebook syntax, declared asset paths, public-release file selection, and the release archive are checked locally before packaging.

## Portable source archive

The initial release ZIP was extracted into a fresh temporary directory. All 196 bundled file checksums verified; repository checks, empirical reproduction and all six CPU suites passed from that extraction. The nine reproduced result CSVs again matched the included values within 1e-12, independently of the original research-folder paths.

## Dependency and prototype folder cleanup

Shared dependency profiles now live in `requirements/`; the archived Gemma prototype is under `experiments/prototypes/gemma4/`. Both moved test profiles resolved offline against their existing environments without dependency changes. CI profile paths, documentation links, and source-manifest destinations were validated. The reorganized 198-file source archive was extracted and every checksum verified; obsolete top-level dependency/prototype folders are absent. Active experiment implementations and published measurements are unchanged by this layout cleanup.
