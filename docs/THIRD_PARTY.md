# External code, models, data and style files

Model weights and benchmark data are obtained from their original publishers and are not redistributed in this source repository. Preserve their model-card, dataset and upstream license requirements when downloading or sharing them.

| Dependency/source | Role |
|---|---|
| [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2) | Original image MoE-VLM implementation |
| [FastMMoE](https://github.com/MindVLA-Team/FastMMoE) | Optional upstream comparison/source; pinned setup commit in the compression scripts |
| [Transformers](https://github.com/huggingface/transformers) | Qwen/InternVL model interfaces and architecture implementations |
| [Qwen](https://huggingface.co/Qwen) | Qwen3-VL, Qwen3.5 and Qwen3.6 checkpoints |
| [OpenGVLab](https://huggingface.co/OpenGVLab) | InternVL3.5 checkpoints |
| [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) | Image-question compression evaluation |
| [NExT-QA annotations](https://huggingface.co/datasets/lmms-eval/NExTQA) | Temporal-control evaluation |
| [TextVQA](https://huggingface.co/datasets/facebook/textvqa) | OCR-oriented extension |
| [MMMU-Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) | Harder multimodal revision benchmark |
| [MathVista](https://huggingface.co/datasets/AI4Math/MathVista) | Visual mathematical reasoning revision benchmark |
| [Official TMLR style](https://github.com/JmlrOrg/tmlr-style-file) | `paper/tmlr.sty` and `paper/tmlr.bst`; source commit in `paper/style_source.json`, upstream license in `paper/TMLR_STYLE_LICENSE` |

The code uses PyTorch, Accelerate, bitsandbytes and other dependencies declared in each environment. Upstream model source snapshots and reference-paper PDFs from the local workspace are intentionally not vendored. The experimental Gemma files are local research adapters/prototypes; obtain the model implementation from the cited Transformers release and the checkpoint from its publisher.
