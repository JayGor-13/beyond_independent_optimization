"""Shard-wise BF16 loading or explicit NF4 quantization of packed routed experts.

Only routed expert matrices are quantized. Routers, shared experts, attention,
vision, embeddings and output heads retain BF16 weights. No CPU offloading.
"""
from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path

import torch
from torch import nn

TRANSFORMERS_VERSION = "5.5.0"


class SplitExperts(nn.Module):
    """Same Top-k dispatch as Qwen3.5, with quantizable per-expert linears."""
    def __init__(self, count, hidden, intermediate, activation):
        super().__init__()
        self.num_experts, self.hidden_dim, self.intermediate_dim = count, hidden, intermediate
        self.act_fn = activation
        self.gate_up = nn.ModuleList([nn.Linear(hidden, 2 * intermediate, bias=False) for _ in range(count)])
        self.down = nn.ModuleList([nn.Linear(intermediate, hidden, bias=False) for _ in range(count)])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        for expert in torch.unique(top_k_index).tolist():
            token, slot = torch.where(top_k_index == expert)
            gate, up = self.gate_up[expert](hidden_states[token]).chunk(2, dim=-1)
            value = self.down[expert](self.act_fn(gate) * up)
            output.index_add_(0, token, (value * top_k_weights[token, slot, None]).to(output.dtype))
        return output


def split_experts(model):
    names = []
    for name, module in list(model.named_modules()):
        if name.endswith(".mlp.experts") and hasattr(module, "gate_up_proj"):
            count, twice_intermediate, hidden = module.gate_up_proj.shape
            if tuple(module.down_proj.shape) != (count, hidden, twice_intermediate // 2):
                raise ValueError("Checkpoint expert layout differs from pinned Qwen3.5 implementation.")
            with torch.device("meta"):
                replacement = SplitExperts(count, hidden, twice_intermediate // 2, module.act_fn)
            parent, attribute = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attribute, replacement)
            names.append(name)
    if len(names) != model.config.text_config.num_hidden_layers:
        raise ValueError("Expected one routed expert collection per decoder layer.")
    return set(names)


def make_placement(model, device_count):
    prefix = "model.language_model"
    layers = len(model.model.language_model.layers)
    mapping = {"model.visual": 0, f"{prefix}.embed_tokens": 0,
               f"{prefix}.rotary_emb": 0, f"{prefix}.norm": device_count - 1, "lm_head": device_count - 1}
    for index in range(layers):
        mapping[f"{prefix}.layers.{index}"] = min(device_count - 1, index * device_count // layers)
    return mapping


def device_for(name, mapping):
    matches = [prefix for prefix in mapping if name == prefix or name.startswith(prefix + ".")]
    if not matches:
        raise ValueError(f"No explicit device placement for {name}.")
    return mapping[max(matches, key=len)]


def expanded_weights(source, name, packed):
    parent, _, attribute = name.rpartition(".")
    if parent in packed and attribute in {"gate_up_proj", "down_proj"}:
        tensor = source.get_slice(name)
        projection = "gate_up" if attribute == "gate_up_proj" else "down"
        for expert in range(tensor.get_shape()[0]):
            # Qwen3.5 stores [expert, output, input]; no transpose is needed.
            yield f"{parent}.{projection}.{expert}.weight", tensor[expert].contiguous()
    else:
        yield name, source.get_tensor(name)


def put_weight(model, name, value, mapping, quantize):
    from accelerate.utils import set_module_tensor_to_device
    device = f"cuda:{device_for(name, mapping)}"
    if quantize and ".mlp.experts." in name:
        import bitsandbytes as bnb
        module_name = name.removesuffix(".weight")
        old = model.get_submodule(module_name)
        if not isinstance(old, nn.Linear) or old.bias is not None:
            raise ValueError(f"Unexpected expert projection: {name}.")
        with torch.device("meta"):
            layer = bnb.nn.Linear4bit(old.in_features, old.out_features, bias=False,
                                     compute_dtype=torch.bfloat16, compress_statistics=True, quant_type="nf4")
        layer.weight = bnb.nn.Params4bit(value.to(torch.bfloat16).contiguous(), requires_grad=False,
                                        compress_statistics=True, quant_type="nf4").to(device)
        parent, attribute = module_name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attribute, layer)
    else:
        set_module_tensor_to_device(model, name, device, value=value, dtype=torch.bfloat16)


def load_model(repo, revision, precision, scratch, reserve_gib=12):
    import transformers
    from accelerate import dispatch_model, init_empty_weights
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3_5MoeForConditionalGeneration
    from transformers.initialization import no_init_weights

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Use the separate supplied environment: transformers=={TRANSFORMERS_VERSION}.")
    if not torch.cuda.is_available() or not all(torch.cuda.get_device_capability(i)[0] >= 8
                                              for i in range(torch.cuda.device_count())):
        raise RuntimeError("This BF16 package requires CUDA GPUs with compute capability >= 8.0.")
    if precision not in {"bf16", "expert-nf4"}:
        raise ValueError("Unsupported precision.")
    config = AutoConfig.from_pretrained(repo, revision=revision)
    if config.model_type != "qwen3_5_moe":
        raise ValueError("Only the declared Qwen3.5/Qwen3.6 multimodal MoE architecture is supported.")
    config._attn_implementation = "sdpa"
    config._experts_implementation = "eager"
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with init_empty_weights(include_buffers=False), no_init_weights():
            model = Qwen3_5MoeForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    quantize = precision == "expert-nf4"
    packed = split_experts(model) if quantize else set()
    mapping = make_placement(model, torch.cuda.device_count())
    expected = set(model.state_dict())
    estimates = [0.0] * torch.cuda.device_count()
    for name, parameter in model.named_parameters():
        estimates[device_for(name, mapping)] += parameter.numel() * (0.60 if quantize and ".mlp.experts." in name else 2)
    for device, size in enumerate(estimates):
        free, _ = torch.cuda.mem_get_info(device)
        if free < size + reserve_gib * 1024**3:
            raise RuntimeError(f"GPU {device}: estimated weights {size / 1024**3:.1f} GiB + "
                               f"{reserve_gib:g} GiB reserve exceed free {free / 1024**3:.1f} GiB. "
                               "Use expert-nf4 or more suitable GPUs; hide the RTX 4090 for the H200 run.")
    index = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=revision)).read_text())
    ignored = sorted(name for name in index["weight_map"] if name.startswith("mtp."))
    selected = {name: shard for name, shard in index["weight_map"].items() if name not in ignored}
    loaded = set()
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    for number, shard in enumerate(sorted(set(selected.values())), 1):
        print(f"Loading shard {number}/{len(set(selected.values()))}: {shard}", flush=True)
        with tempfile.TemporaryDirectory(prefix="tmlr_shard_", dir=scratch) as directory:
            path = hf_hub_download(repo, shard, revision=revision, local_dir=directory)
            with safe_open(path, framework="pt", device="cpu") as source:
                for key in source.keys():
                    if key in ignored:
                        continue
                    for name, value in expanded_weights(source, key, packed):
                        if name not in expected or name in loaded:
                            raise ValueError(f"Unexpected or duplicate checkpoint weight: {name}.")
                        put_weight(model, name, value, mapping, quantize)
                        loaded.add(name)
                    del value
            gc.collect()
        torch.cuda.empty_cache()
    if expected != loaded or any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError(f"Uninitialized model state: {sorted(expected - loaded)[:10]}.")
    model = dispatch_model(model, device_map=mapping, force_hooks=True)
    model.eval().requires_grad_(False)
    metadata = {"repo": repo, "revision": revision, "precision": precision, "compute_dtype": "bfloat16",
                "estimated_weight_gib": [size / 1024**3 for size in estimates], "device_map": mapping,
                "quantized_expert_collections": len(packed), "ignored_mtp_keys": ignored,
                "gpus": [{"name": torch.cuda.get_device_name(i),
                          "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                         for i in range(torch.cuda.device_count())]}
    return model, metadata
