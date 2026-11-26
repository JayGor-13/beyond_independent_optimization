# Beyond Independent Optimization

Code, retained experimental evidence, and manuscript assets for **Compression, MoE Routing, and Quantization Interactions in Multimodal Edge Intelligence**.

This repository studies how visual-token compression changes downstream expert routing and how temporal routing consistency responds to video-frame controls. Published measurements and additional revision experiments are kept separate. The scripts do not establish that routing drift causes accuracy loss.

## Start here

- **Rebuild the existing empirical figures/tables on a CPU or Mac:** follow the commands below.
- **Run the additional 122B-model experiments on an H200:** use [the minor-revision guide](experiments/minor_revision/README.md).
- **Reproduce the completed InternVL retention sweep:** use [the multimodel guide](experiments/multimodel/README.md).
- **Understand which measurements are included:** read [results provenance](docs/RESULTS.md) and [experiment status](docs/EXPERIMENTS.md).

## Repository layout

```text
experiments/             Standalone runners, analysis, tests, and Kaggle launchers
  compression_routing/  DeepSeek ScienceQA compression/routing experiment
  temporal_routing/     DeepSeek NExT-QA routing controls
  multimodel/           InternVL3.5 and Qwen3-VL adapters; full InternVL sweep
  textvqa/              TextVQA extension
  temporal_accuracy/    Paired temporal-perturbation extension
  control_accuracy/     Follow-up generation for missing control accuracies
  minor_revision/       Qwen3.5 122B/35B and optional Qwen3.6 revision experiments
results/
  published/            Unmodified supplied measurements and metadata
  evidence/             Original executed InternVL notebook, with evidence outputs
  paper_tables/         Consolidated CSVs and empirical LaTeX tables
paper/                  Manuscript, bibliography, figure assets, TMLR style files
tools/                  Reproduction, validation, notebook bundling, release packaging
environments/           Separate CPU test dependency profiles
experimental/gemma4/    Unresolved prototype; outside the supported runner workflows
docs/                   Reproducibility, provenance, development, release notes
```

Generated runs go in `artifacts/` or the experiment's ignored output directory. Downloaded upstream code belongs in `external/`; weights, dataset caches, environments and generated ZIPs are excluded from Git.

## Rebuild existing tables and figures without a GPU

Use Python 3.12 or newer:

```bash
python3.12 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
python tools/check_repository.py
python tools/reproduce_paper.py
```

Outputs appear under `artifacts/reproduction/tables/` and `artifacts/reproduction/figures/`. This reconstructs the empirical figures and table values from the supplied CSVs and executed notebook; it does not rerun model inference or recover unavailable raw routes. Retained published evidence is not overwritten.

## Run model experiments

Different model families require different Transformers/runtime versions. Install each in a **separate environment**; use the experiment guides rather than installing all requirements into one environment.

| Workflow | Guide | Hardware / runtime |
|---|---|---|
| Additional larger-model validation | [Minor revision](experiments/minor_revision/README.md) | H200; Transformers 5.5.0 |
| InternVL full sweep / Qwen3-VL replication | [Multimodel](experiments/multimodel/README.md) | CUDA; Transformers 4.57.6 |
| Original compression/routing | [Compression](experiments/compression_routing/README.md) | Legacy DeepSeek/FastMMoE stack |
| Temporal routing controls | [Temporal routing](experiments/temporal_routing/README.md) | Legacy DeepSeek stack |
| TextVQA extension | [TextVQA](experiments/textvqa/README.md) | CUDA; shared compression helper included |
| Graded temporal perturbations | [Temporal accuracy](experiments/temporal_accuracy/README.md) | CUDA; separate legacy environment |
| Missing control accuracies | [Control accuracy](experiments/control_accuracy/README.md) | Self-contained Kaggle launcher; Python 3.11 |

For the H200 workflow, begin with the guide's smoke command. The smoke test still loads the entire model. Full-size GPU execution is not part of repository-preparation validation.

## Kaggle and source archives

```bash
python tools/build_kaggle_bundle.py all
python tools/build_release.py
```

These write archives to ignored `dist/`. Kaggle bundle folder names match the existing notebook lookup patterns even though the repository uses cleaner names. For the self-contained control-accuracy notebook, rebuild its embedded code with `python experiments/control_accuracy/build_notebook.py`.

## Development and manuscript

See [development/testing](docs/DEVELOPMENT.md), [the manuscript build guide](paper/README.md), and [the release notes](docs/RELEASE.md). CPU tests check code and synthetic model contracts; paper claims must use completed real-data runs with recorded provenance.

The supplied manuscript retains anonymized author and publication metadata. A project-wide software license and final paper citation have not been assigned by this repository preparation; those decisions belong to the paper's authors. External models, data and bundled TMLR style files retain their own terms; see [third-party sources](docs/THIRD_PARTY.md).
