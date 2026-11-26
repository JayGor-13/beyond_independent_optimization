# Experiment inventory

| Directory | Question | Release status |
|---|---|---|
| [compression_routing](../experiments/compression_routing) | Does visual-token pruning change retained-token routing and ScienceQA accuracy? | Legacy runner; saved DeepSeek summary included |
| [temporal_routing](../experiments/temporal_routing) | Does TRC respond to temporal/coherence controls? | Legacy runner; saved DeepSeek CSVs included |
| [multimodel](../experiments/multimodel) | Replication across InternVL3.5/Qwen3-VL | Full InternVL retention-sweep implementation; historical InternVL evidence included |
| [textvqa](../experiments/textvqa) | Compression/routing with OCR-oriented VQA scoring | Implementation; no completed measurements supplied |
| [temporal_accuracy](../experiments/temporal_accuracy) | Do paired TRC changes track answer-margin/correctness changes? | Implementation; no completed measurements supplied |
| [control_accuracy](../experiments/control_accuracy) | Recover missing answer-generating control accuracies on the original sample | Follow-up implementation; sample/provenance matching required |
| [minor_revision](../experiments/minor_revision) | Does the compression/routing finding extend to 122B-class models and harder benchmarks? | Additional revision implementation; full GPU validation pending |
| [Gemma prototype](../experimental/gemma4) | Additional architectural replication | Unresolved prototype; excluded from supported runners and CPU CI |

These are separate scripts with local modules such as `core.py`, `run.py`, and `analyze.py`. Run each suite in its own process and compatible environment. [The test wrapper](../tools/run_tests.py) handles process/import isolation.

The original compression, temporal, and control-accuracy packages have different historical dependency stacks. Use their requirements/setup guides for actual model inference; CPU tests in a newer environment are not evidence that a legacy full-checkpoint GPU run is compatible with that environment.

For optional FastMMoE comparisons, the setup downloads upstream source into ignored `external/` on a normal server or `/kaggle/working/` in Kaggle. Large weights, raw videos and dataset caches are downloaded separately from their public sources. They are not committed here.
