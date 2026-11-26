# Development and testing

The analysis/reproduction workflow needs no CUDA. Python 3.12 is the tested interpreter for release preparation.

## Static checks and existing paper results

```bash
python -m pip install -r requirements/analysis.txt
python tools/check_repository.py
python tools/reproduce_paper.py
```

Checks cover Python/shell syntax, notebook output hygiene, unchanged historical evidence, manuscript asset paths, source credential-like strings, ordinary Git file-size limits, and canonical documentation links. Reproduction reads the original evidence and writes to ignored `artifacts/`.

## CPU model and metric suites

Create separate environments for these two commands; their Transformers versions conflict:

```bash
# Environment A: legacy metric/runtime fixtures plus Transformers 4.57.6 adapters
python -m pip install -r requirements/tests-multimodel.txt
python tools/run_tests.py multimodel legacy

# Environment B: Transformers 5.5.0 revision adapter
python -m pip install -r requirements/tests-minor.txt
python tools/run_tests.py minor_revision
```

The original compression suite uses synthetic router/attention fixtures and can also be run separately with `python tools/run_tests.py compression`. Actual DeepSeek/FastMMoE inference still requires that experiment's external code and legacy dependencies. Optional InternVL-source tests skip when their separately obtained pinned fixture is absent. CUDA/NF4 tests skip on a CPU host.

Keep the per-experiment test files with their code. A root-level `pytest` run can mix identically named modules from different suites; use the isolated wrapper. New scientific changes should be tested within the affected environment, including a full-checkpoint smoke test for loading/adapter changes.

## Bundles

`python tools/build_kaggle_bundle.py all` creates self-contained upload ZIPs under `dist/`; the TextVQA ZIP includes its shared compression helper. Notebook builders preserve historical upload-folder aliases while using canonical repository paths for source/result inputs. No ZIP or generated environment needs to be committed.
