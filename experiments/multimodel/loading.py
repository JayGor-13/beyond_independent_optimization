"""Shard-at-a-time loading with explicit NF4 quantization of routed experts.

Qwen3-VL 4.57.6 stores experts as 3D Parameters, which ordinary BNB Linear
replacement misses. Split those tensors into equivalent per-expert Linear
modules before loading. Routers, attention, vision, embeddings and heads stay
in the requested floating-point dtype for both model families.
"""
import gc
import json
import math
import tempfile
from pathlib import Path

import torch
from torch import nn
from core import MODELS


class SplitExperts(nn.Module):
    def __init__(self, count, hidden, intermediate, act_fn):
        super().__init__()
        self.num_experts = count
        self.hidden_size = hidden
        self.act_fn = act_fn
        self.gate_up = nn.ModuleList([nn.Linear(hidden, 2 * intermediate, bias=False) for _ in range(count)])
        self.down = nn.ModuleList([nn.Linear(intermediate, hidden, bias=False) for _ in range(count)])

    def forward(self, hidden_states, routing_weights, router_indices):
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        output = torch.zeros_like(flat)
        for expert in router_indices.unique().tolist():
            tokens = torch.where((router_indices == expert).any(dim=-1))[0]
            gate, up = self.gate_up[expert](flat[tokens]).chunk(2, dim=-1)
            values = self.down[expert](self.act_fn(gate) * up)
            output.index_add_(0, tokens, (values * routing_weights[tokens, expert, None]).to(output.dtype))
        return output.reshape(shape)


def unpack_experts(model):
    names = []
    for name, module in list(model.named_modules()):
        if hasattr(module, "gate_up_proj") and isinstance(module.gate_up_proj, nn.Parameter):
            count, hidden, twice_intermediate = module.gate_up_proj.shape
            with torch.device("meta"):
                replacement = SplitExperts(count, hidden, twice_intermediate // 2, module.act_fn)
            parent, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attr, replacement)
            names.append(name)
    return names


def placement(model, family, devices):
    if devices < 1:
        raise ValueError("At least one CUDA device is required.")
    prefix = "model.language_model" if family == "qwen3vl" else "language_model.model"
    layers = model.get_submodule(prefix).layers
    mapping = {f"{prefix}.embed_tokens": 0, f"{prefix}.norm": devices - 1,
               f"{prefix}.rotary_emb": 0}
    for i in range(len(layers)):
        mapping[f"{prefix}.layers.{i}"] = min(devices - 1, i * devices // len(layers))
    if family == "qwen3vl":
        mapping.update({"model.visual": 0, "lm_head": devices - 1})
    else:
        mapping.update({"vision_model": 0, "mlp1": 0, "language_model.lm_head": devices - 1})
    return mapping


def device_for(name, mapping):
    matches = [key for key in mapping if name == key or name.startswith(key + ".")]
    if not matches:
        raise ValueError(f"No device placement for {name}.")
    return mapping[max(matches, key=len)]


def put_weight(model, name, value, mapping, dtype, quantize):
    from accelerate.utils import set_module_tensor_to_device
    device = device_for(name, mapping)
    if quantize and ".experts." in name and name.endswith(".weight"):
        import bitsandbytes as bnb
        module_name = name.removesuffix(".weight")
        old = model.get_submodule(module_name)
        if not isinstance(old, nn.Linear) or old.bias is not None:
            raise ValueError(f"Unsupported quantized expert layer {module_name}.")
        with torch.device("meta"):
            layer = bnb.nn.Linear4bit(old.in_features, old.out_features, bias=False,
                                     compute_dtype=dtype, compress_statistics=True, quant_type="nf4")
        layer.weight = bnb.nn.Params4bit(value.to(dtype).contiguous(), requires_grad=False,
                                        compress_statistics=True, quant_type="nf4").to(f"cuda:{device}")
        parent, attr = module_name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr, layer)
    else:
        set_module_tensor_to_device(model, name, f"cuda:{device}", value=value, dtype=dtype)


def expanded_weights(name, tensor, packed_names):
    for packed in packed_names:
        if name == packed + ".gate_up_proj" or name == packed + ".down_proj":
            projection = "gate_up" if name.endswith("gate_up_proj") else "down"
            for expert in range(tensor.shape[0]):
                yield f"{packed}.{projection}.{expert}.weight", tensor[expert].T.contiguous()
            return
    yield name, tensor


def load_model(family, dtype, quantize=True, scratch=None):
    from accelerate import dispatch_model, init_empty_weights
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModel, Qwen3VLMoeForConditionalGeneration
    from transformers.modeling_utils import no_init_weights
    import transformers

    if transformers.__version__ != "4.57.6":
        raise RuntimeError("Run setup_kaggle.sh and restart the kernel: transformers==4.57.6 is required.")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU in Kaggle; T4 x2 is recommended for expert-NF4 runs.")
    repo, revision = MODELS[family]
    config = AutoConfig.from_pretrained(repo, revision=revision, trust_remote_code=(family == "internvl35"))
    with init_empty_weights(include_buffers=False), no_init_weights():
        if family == "qwen3vl":
            config._attn_implementation = "sdpa"
            model = Qwen3VLMoeForConditionalGeneration(config)
        else:
            model = AutoModel.from_config(config, trust_remote_code=True, code_revision=revision, use_flash_attn=False)
    packed = unpack_experts(model)
    mapping = placement(model, family, torch.cuda.device_count())
    expected = set(model.state_dict())
    estimates = [0] * torch.cuda.device_count()
    for name, parameter in model.named_parameters():
        # NF4 plus double-quantization metadata; reserve runtime space separately.
        estimates[device_for(name, mapping)] += parameter.numel() * (
            0.6 if quantize and ".experts." in name else 2)
    for device, size in enumerate(estimates):
        free, _ = torch.cuda.mem_get_info(device)
        required = size + 2 * 1024**3
        if free < required:
            raise RuntimeError(f"GPU {device}: estimated weights plus 2 GiB reserve need "
                               f"{required / 1024**3:.1f} GiB, but only {free / 1024**3:.1f} GiB is free. "
                               "Use T4 x2 with --precision expert-nf4, or larger GPUs.")
    index = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=revision)).read_text())
    loaded = set()
    scratch = Path(scratch or "/kaggle/working/model_shards")
    scratch.mkdir(parents=True, exist_ok=True)
    shards = sorted(set(index["weight_map"].values()))
    for i, shard in enumerate(shards):
        print(f"Loading {family} shard {i + 1}/{len(shards)}: {shard}", flush=True)
        # Only this newly-created temporary directory is removed on exit.
        with tempfile.TemporaryDirectory(prefix="shard_", dir=scratch) as directory:
            path = hf_hub_download(repo, shard, revision=revision, local_dir=directory)
            with safe_open(path, framework="pt", device="cpu") as source:
                for key in source.keys():
                    tensor = source.get_tensor(key)
                    for name, value in expanded_weights(key, tensor, packed):
                        if name not in expected or name in loaded:
                            raise ValueError(f"Unexpected or duplicate checkpoint weight: {name}")
                        put_weight(model, name, value, mapping, dtype, quantize)
                        loaded.add(name)
                    del tensor, value
            gc.collect()
        torch.cuda.empty_cache()
    missing = expected - loaded
    if missing:
        raise ValueError(f"Checkpoint did not populate model state: {sorted(missing)[:10]}")
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Uninitialized model parameters remain after loading.")
    model = dispatch_model(model, device_map=mapping, force_hooks=True)
    model.eval()
    model.requires_grad_(False)
    return model, {"device_map": mapping, "estimated_weight_gib": [round(x / 1024**3, 2) for x in estimates],
                   "packed_expert_modules_split": len(packed), "loaded_tensors": len(loaded)}
