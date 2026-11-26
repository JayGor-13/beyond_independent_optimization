"""Optional integration test using a local copy of the pinned official source.

Set INTERNVL_TEST_SOURCE to the parent directory of the internvl_source package.
No downloads occur in this test.
"""
import os
import sys

import pytest
import torch
from PIL import Image

from adapters import Adapter
from core import compress
from loading import device_for, placement


@pytest.mark.skipif(not os.environ.get("INTERNVL_TEST_SOURCE"), reason="Optional pinned InternVL source fixture not supplied")
def test_real_internvl_native_preparation_routing_and_decode():
    sys.path.insert(0, os.environ["INTERNVL_TEST_SOURCE"])
    from internvl_source.configuration_internvl_chat import InternVLChatConfig
    from internvl_source.modeling_internvl_chat import InternVLChatModel
    from transformers import BatchEncoding
    config = InternVLChatConfig(
        force_image_size=56, template="internvl2_5", ps_version="v2",
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                       "num_attention_heads": 2, "image_size": 56, "patch_size": 14, "use_flash_attn": False},
        llm_config={"architectures": ["Qwen3MoeForCausalLM"], "hidden_size": 16,
                    "intermediate_size": 32, "moe_intermediate_size": 8, "vocab_size": 100,
                    "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                    "head_dim": 8, "num_experts": 4, "num_experts_per_tok": 2})
    model = InternVLChatModel(config, use_flash_attn=False).eval()
    class Tokenizer:
        unk_token_id = 0
        def convert_tokens_to_ids(self, token):
            return 91 if token == "<IMG_CONTEXT>" else 99
        def __call__(self, text, **kwargs):
            count = text.count("<IMG_CONTEXT>") // 4
            ids = torch.tensor([[1] + [v for _ in range(count) for v in [90, 91, 91, 91, 91, 92]] + [3]])
            return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        def decode(self, tokens, **kwargs):
            return ",".join(map(str, tokens))
    adapter = Adapter.__new__(Adapter)
    adapter.model, adapter.family, adapter.dtype = model, "internvl35", torch.float32
    adapter.decoder, adapter.head = model.language_model.model, model.language_model.lm_head
    adapter.embedding = adapter.decoder.get_input_embeddings()
    adapter.device, adapter.tokenizer = torch.device("cpu"), Tokenizer()
    adapter.blocks = [(name, b) for name, b in adapter.decoder.named_modules() if hasattr(b, "top_k")]
    adapter.n_experts, adapter.top_k = 4, 2
    images = [Image.new("RGB", (56, 56), color=c) for c in ("red", "blue")]
    batch = adapter.prepare(images, "q", temporal=True)
    assert adapter.check_native(batch)["next_token_matches"]
    batch["eos"] = [-1]
    topk, _, _, generated = adapter.evaluate(batch, 3)
    assert topk.shape == (2, 8, 2)
    with torch.inference_mode():
        native = model.language_model.generate(inputs_embeds=batch["embeddings"],
                                               attention_mask=batch["attention_mask"], max_new_tokens=3,
                                               do_sample=False, eos_token_id=None, pad_token_id=0)
    assert generated == ",".join(map(str, native[0].tolist()))
    pruned = compress(batch, "l2", 0.5, 0)
    assert adapter.evaluate(pruned, 2)[0].shape == (2, 4, 2)
    mapping = placement(model, "internvl35", 2)
    assert all(device_for(name, mapping) in (0, 1) for name, _ in model.named_parameters())
