"""Gemma 4: bounded HTTP range reads and per-expert NF4 loading.

The official first shard is ~50 GB, too large for Kaggle's working disk.
Read authenticated, pinned-revision byte ranges instead of downloading shards.
"""
import json
import math
import struct
import time
from pathlib import Path

import torch
from torch import nn

from core import MODELS
from loading import device_for, log_memory, put_weight

VERSION = "2026-09-30-gemma4-vision-fp32"


def weight_dtype(name, text_dtype):
    # Gemma's vision residual stream can exceed FP16's range before the final
    # standardization. Keep the entire vision path wide, not just the final norm.
    if text_dtype == torch.float16 and name.startswith(("model.vision_tower.", "model.embed_vision.")):
        return torch.float32
    return text_dtype


def serial_vision(model):
    """Bound FP32 vision activations to one image; keep all frame tokens/order."""
    tower = model.model.vision_tower
    original = tower.forward

    def forward(pixel_values, pixel_position_ids, **kwargs):
        if len(pixel_values) <= 1:
            return original(pixel_values=pixel_values, pixel_position_ids=pixel_position_ids, **kwargs)
        if kwargs.get("output_hidden_states") or kwargs.get("output_attentions") or kwargs.get("return_dict") is False:
            raise ValueError("Serial Gemma vision requires dictionary outputs without auxiliary hidden states/attentions.")
        from transformers.modeling_outputs import BaseModelOutputWithPast
        features = []
        for index in range(len(pixel_values)):
            output = original(pixel_values=pixel_values[index:index+1],
                              pixel_position_ids=pixel_position_ids[index:index+1], **kwargs)
            features.append(output.last_hidden_state)
        return BaseModelOutputWithPast(last_hidden_state=torch.cat(features, dim=0))

    tower.forward = forward


class HubRangeReader:
    def __init__(self, repo, filename, revision, chunk_bytes=32 * 1024**2):
        import requests
        self.repo, self.filename, self.revision = repo, filename, revision
        self.chunk_bytes = chunk_bytes
        self.session = requests.Session()
        self.position, self.cache_start, self.cache = 0, 0, b""
        self.downloaded, self.reported = 0, 0
        self.refresh()

    def refresh(self):
        from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
        self.url = hf_hub_url(self.repo, self.filename, revision=self.revision)
        metadata = get_hf_file_metadata(self.url, token=get_token())
        if metadata.commit_hash and metadata.commit_hash != self.revision:
            raise ValueError("Hub resolved an unexpected model revision.")
        self.size, self.location = metadata.size, metadata.location
        if self.size is None:
            raise ValueError("Checkpoint size is missing from Hub metadata.")
        # Signed storage URLs carry their own authorization; never send the Hub
        # bearer token to a storage host.
        self.headers = {}
        if self.location == self.url and (token := get_token()):
            self.headers["Authorization"] = f"Bearer {token}"

    def seek(self, position):
        if not 0 <= position <= self.size:
            raise ValueError("Checkpoint seek is outside the file.")
        self.position = position

    def read(self, size):
        result = bytearray(size)
        if self.readinto(result) != size:
            raise ValueError("Truncated checkpoint read.")
        return result

    def readinto(self, target):
        target = memoryview(target)
        remaining = min(len(target), self.size - self.position)
        total = remaining
        while remaining:
            if not self.cache_start <= self.position < self.cache_start + len(self.cache):
                start = self.position
                end = min(self.size, start + self.chunk_bytes) - 1
                self.cache = b""
                for attempt in range(3):
                    try:
                        with self.session.get(self.location, headers={**self.headers,
                                              "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                                              stream=True, timeout=(30, 120)) as response:
                            if response.status_code in (401, 403):
                                self.refresh()
                                raise IOError("Refreshing an expired download URL.")
                            response.raise_for_status()
                            expected = f"bytes {start}-{end}/{self.size}"
                            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                                raise ValueError("Server did not honor a bounded byte range; refusing a full-shard download.")
                            data = response.raw.read(end - start + 2)
                            if len(data) != end - start + 1:
                                raise IOError("Incomplete checkpoint byte range.")
                            self.cache, self.cache_start = data, start
                            self.downloaded = getattr(self, "downloaded", 0) + len(data)
                            if self.downloaded - getattr(self, "reported", 0) >= 512 * 1024**2:
                                print(f"[Gemma download] {self.filename}: {self.downloaded / 1024**3:.1f} / "
                                      f"{self.size / 1024**3:.1f} GiB received", flush=True)
                                self.reported = self.downloaded
                        break
                    except ValueError:
                        raise
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
            offset = self.position - self.cache_start
            count = min(remaining, len(self.cache) - offset)
            target[total - remaining:total - remaining + count] = memoryview(self.cache)[offset:offset + count]
            self.position += count
            remaining -= count
        return total

    def close(self):
        self.cache = b""
        self.session.close()


def read_header(reader):
    length = struct.unpack("<Q", reader.read(8))[0]
    if not 2 <= length <= min(100 * 1024**2, reader.size - 8):
        raise ValueError("Invalid safetensors header length.")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate safetensors header key: {key}")
            result[key] = value
        return result
    header = json.loads(reader.read(length), object_pairs_hook=unique_pairs)
    header.pop("__metadata__", None)
    cursor = 0
    for key, item in sorted(header.items(), key=lambda pair: pair[1]["data_offsets"][0]):
        if item["dtype"] not in {"F16", "BF16", "F32"}:
            raise ValueError(f"Unsupported checkpoint dtype for {key}")
        shape, (start, end) = item["shape"], item["data_offsets"]
        if not all(isinstance(n, int) and n > 0 for n in shape):
            raise ValueError(f"Invalid checkpoint shape: {key}")
        size = math.prod(shape) * (4 if item["dtype"] == "F32" else 2)
        if start != cursor or end - start != size:
            raise ValueError(f"Invalid checkpoint offsets: {key}")
        cursor = end
    if 8 + length + cursor != reader.size:
        raise ValueError("Checkpoint header and file size disagree.")
    return header, 8 + length


def range_weights(reader, header, data_start, packed):
    dtypes = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
    for key, item in sorted(header.items(), key=lambda pair: pair[1]["data_offsets"][0]):
        start, end = item["data_offsets"]
        shape = item["shape"]
        module_name, _, projection = key.rpartition(".")
        split = module_name in packed and projection in {"gate_up_proj", "down_proj"}
        count = shape[0] if split else 1
        if end - start > 2 * 1024**3 and not split:
            raise ValueError(f"Non-expert tensor exceeds the 2 GiB read bound: {key}")
        for expert in range(count):
            size = (end - start) // count
            reader.seek(data_start + start + expert * size)
            buffer = reader.read(size)
            value = torch.frombuffer(buffer, dtype=dtypes[item["dtype"]]).reshape(shape[1:] if split else shape)
            name = (f"{module_name}.{'gate_up' if projection == 'gate_up_proj' else 'down'}.{expert}.weight"
                    if split else key)
            yield name, value
            del buffer, value


class GemmaSplitExperts(nn.Module):
    def __init__(self, original):
        super().__init__()
        count, twice_intermediate, hidden = original.gate_up_proj.shape
        self.gate_up = nn.ModuleList([nn.Linear(hidden, twice_intermediate, bias=False) for _ in range(count)])
        self.down = nn.ModuleList([nn.Linear(twice_intermediate // 2, hidden, bias=False) for _ in range(count)])
        self.act_fn = original.act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states)
        for expert in top_k_index.unique().tolist():
            token, slot = torch.where(top_k_index == expert)
            gate, up = self.gate_up[expert](hidden_states[token]).chunk(2, dim=-1)
            value = self.down[expert](self.act_fn(gate) * up)
            result.index_add_(0, token, (value * top_k_weights[token, slot, None]).to(result.dtype))
        return result


def split_experts(model):
    packed = []
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "Gemma4TextExperts":
            with torch.device("meta"):
                replacement = GemmaSplitExperts(module)
            parent, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attr, replacement)
            packed.append(name)
    return packed


def gemma_placement(model, devices):
    prefix = "model.language_model"
    mapping = {f"{prefix}.embed_tokens": 0, f"{prefix}.rotary_emb": 0,
               f"{prefix}.norm": devices - 1, "model.vision_tower": 0,
               "model.embed_vision": 0, "lm_head": 0}
    layers = model.model.language_model.layers
    # Put fewer decoder layers beside the vision tower and tied embedding/head.
    first = max(1, len(layers) // 2 - 2) if devices == 2 else len(layers)
    for index in range(len(layers)):
        mapping[f"{prefix}.layers.{index}"] = (int(index >= first) if devices == 2
                                               else min(devices - 1, index * devices // len(layers)))
    return mapping


def load_gemma(dtype, quantize=True, scratch=None):
    import transformers
    from accelerate import dispatch_model, init_empty_weights
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, Gemma4ForConditionalGeneration
    from transformers.initialization import no_init_weights
    if transformers.__version__ != "5.5.0":
        raise RuntimeError("Gemma requires its own setup: transformers==5.5.0.")
    if not torch.cuda.is_available():
        raise RuntimeError("Select T4 x2 in Kaggle before loading Gemma.")
    repo, revision = MODELS["gemma4"]
    config = AutoConfig.from_pretrained(repo, revision=revision)
    config._attn_implementation = "sdpa"
    with init_empty_weights(include_buffers=False), no_init_weights():
        model = Gemma4ForConditionalGeneration(config)
    original_shapes = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    packed = split_experts(model)
    mapping = gemma_placement(model, torch.cuda.device_count())
    expected = set(model.state_dict()) - {"lm_head.weight"}
    estimates = [0.0] * torch.cuda.device_count()
    for name, parameter in model.named_parameters():
        if name != "lm_head.weight":
            size = (0.6 if quantize and ".experts." in name else
                    torch.empty((), dtype=weight_dtype(name, dtype)).element_size())
            estimates[device_for(name, mapping)] += parameter.numel() * size
    for device, size in enumerate(estimates):
        free, _ = torch.cuda.mem_get_info(device)
        if free < size + 2 * 1024**3:
            raise RuntimeError(f"GPU {device}: weights plus 2 GiB reserve require {(size / 1024**3 + 2):.1f} GiB; "
                               f"only {free / 1024**3:.1f} GiB free. Select T4 x2 and expert-nf4.")
    scratch = Path(scratch or "/kaggle/working/model_shards")
    scratch.mkdir(parents=True, exist_ok=True)
    log_path = scratch / "gemma4_loading_memory.jsonl"
    index = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=revision)).read_text())
    shards = sorted(set(index["weight_map"].values()))
    print(f"Loader {VERSION}: 32 MiB HTTP windows; no full shard download.", flush=True)
    log_memory("Gemma structure ready", log_path)
    loaded = set()
    for number, shard in enumerate(shards, 1):
        print(f"Streaming Gemma shard {number}/{len(shards)}: {shard}", flush=True)
        reader = HubRangeReader(repo, shard, revision)
        try:
            header, data_start = read_header(reader)
            if set(header) != {key for key, filename in index["weight_map"].items() if filename == shard}:
                raise ValueError("Shard keys differ from the pinned index.")
            for key, value in header.items():
                if tuple(value["shape"]) != original_shapes.get(key):
                    raise ValueError(f"Checkpoint/model shape mismatch: {key}")
            with torch.no_grad():
                for i, (name, value) in enumerate(range_weights(reader, header, data_start, packed), 1):
                    if name not in expected or name in loaded:
                        raise ValueError(f"Unexpected or duplicate Gemma weight: {name}")
                    put_weight(model, name, value, mapping, weight_dtype(name, dtype), quantize)
                    loaded.add(name)
                    del value
                    if i % 128 == 0:
                        log_memory(f"Gemma shard {number}, {i} tensors loaded", log_path)
        finally:
            reader.close()
        torch.cuda.empty_cache()
        log_memory(f"Gemma shard {number} complete", log_path)
    if expected != loaded:
        raise ValueError(f"Missing Gemma weights: {sorted(expected - loaded)[:10]}")
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Gemma has uninitialized parameters.")
    model = dispatch_model(model, device_map=mapping, force_hooks=True)
    serial_vision(model)
    model.eval().requires_grad_(False)
    return model, {"loader_version": VERSION, "device_map": mapping, "loaded_tensors": len(loaded),
                   "estimated_weight_gib": [round(n / 1024**3, 2) for n in estimates],
                   "packed_expert_modules_split": len(packed), "memory_log": str(log_path),
                   "vision_dtype": str(weight_dtype("model.vision_tower.weight", dtype)),
                   "vision_images_per_forward": 1}
