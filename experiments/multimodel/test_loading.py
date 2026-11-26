"""Small checkpoint loading check; runs on Kaggle CUDA without large downloads."""
import copy
import json
import shutil
from pathlib import Path

import pytest
import torch

from loading import load_model
from test_adapters import tiny_qwen


def test_streamed_meta_loading_without_network_or_gpu(tmp_path, monkeypatch):
    import accelerate
    import huggingface_hub
    import loading
    from accelerate.utils import set_module_tensor_to_device
    from safetensors.torch import save_file
    from transformers import AutoConfig
    original = tiny_qwen()
    config = copy.deepcopy(original.config)
    source = tmp_path / "source"
    source.mkdir()
    save_file(original.state_dict(), source / "weights.safetensors")
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "weights.safetensors" for key in original.state_dict()}}))

    def download(repo, filename, revision=None, local_dir=None):
        if local_dir:
            destination = Path(local_dir) / filename
            shutil.copyfile(source / filename, destination)
            return str(destination)
        return str(source / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (100 * 1024**3, 100 * 1024**3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(accelerate, "dispatch_model", lambda model, **kwargs: model)
    monkeypatch.setattr(loading, "put_weight", lambda model, name, value, mapping, dtype, quantize:
                        set_module_tensor_to_device(model, name, "cpu", value=value, dtype=dtype))
    loaded, info = load_model("qwen3vl", torch.float32, quantize=False, scratch=tmp_path / "scratch")
    assert not any(parameter.is_meta for parameter in loaded.parameters())
    assert not list((tmp_path / "scratch").iterdir())
    with torch.inference_mode():
        ids = torch.tensor([[1, 2, 3]])
        expected = original(input_ids=ids, logits_to_keep=1).logits
        actual = loaded(input_ids=ids, logits_to_keep=1).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed for NF4 loading/dispatch validation")
def test_streamed_nf4_load_quantizes_packed_experts(tmp_path, monkeypatch):
    import bitsandbytes as bnb
    import huggingface_hub
    from safetensors.torch import save_file
    from transformers import AutoConfig
    model = tiny_qwen()
    config = copy.deepcopy(model.config)
    state = model.state_dict()
    keys = sorted(state)
    source = tmp_path / "source"
    source.mkdir()
    weight_map = {}
    for number, subset in enumerate((keys[::2], keys[1::2])):
        filename = f"shard{number}.safetensors"
        save_file({key: state[key].contiguous() for key in subset}, source / filename)
        weight_map.update({key: filename for key in subset})
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    def download(repo, filename, revision=None, local_dir=None):
        if local_dir:
            path = Path(local_dir) / filename
            shutil.copyfile(source / filename, path)
            return str(path)
        return str(source / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    loaded, info = load_model("qwen3vl", torch.float16, quantize=True, scratch=tmp_path / "scratch")
    expert_linears = [module for name, module in loaded.named_modules() if ".experts." in name
                      and isinstance(module, bnb.nn.Linear4bit)]
    assert len(expert_linears) == 2 * 4 * 2
    assert all(module.weight.quant_state is not None for module in expert_linears)
    assert info["packed_expert_modules_split"] == 2
    assert not list((tmp_path / "scratch").iterdir())
    ids = torch.tensor([[1, 2, 3]], device="cuda:0")
    with torch.inference_mode():
        logits = loaded(input_ids=ids, logits_to_keep=1).logits.float().cpu()
        reference = model(input_ids=ids.cpu(), logits_to_keep=1).logits.float()
    assert torch.isfinite(logits).all()
    assert float((logits - reference).abs().mean()) < 0.05
