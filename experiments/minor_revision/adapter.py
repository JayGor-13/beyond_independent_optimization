"""Native visual preparation followed by identity-preserving decoder pruning."""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from data import prompt_content
from protocol import example_seed
from routing import select_visual_ids


class PreparedInputs(Exception):
    pass


def prune_batch(batch, condition, example_id):
    keep_ids = select_visual_ids(batch["norms"], batch["image_ids"], condition["method"],
                                 condition["retention"], example_seed(example_id, condition["seed"] or 0))
    mask = batch["visual_mask"]
    positions = torch.where(mask)[0]
    keep = ~mask.clone()
    keep[positions[torch.as_tensor(keep_ids, device=positions.device)]] = True
    return {**batch, "embeddings": batch["embeddings"][:, keep],
            "attention_mask": batch["attention_mask"][:, keep],
            "position_ids": batch["position_ids"][..., keep], "visual_mask": mask[keep],
            "visual_ids": keep_ids}


class Adapter:
    def __init__(self, model, processor, max_input_tokens=16384):
        self.model, self.processor = model, processor
        self.decoder, self.head = model.model.language_model, model.lm_head
        self.embedding = self.decoder.get_input_embeddings()
        self.device = self.embedding.weight.device
        self.max_input_tokens = max_input_tokens
        self.routers = [(f"layers.{i}.mlp.gate", layer.mlp.gate) for i, layer in enumerate(self.decoder.layers)]
        self.layer_names = [name for name, _ in self.routers]
        self.n_experts = model.config.text_config.num_experts
        self.top_k = model.config.text_config.num_experts_per_tok
        eos = model.config.text_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.eos.add(processor.tokenizer.eos_token_id)

    @torch.inference_mode()
    def prepare(self, item, images):
        content = prompt_content(item, images)
        messages = [{"role": "user", "content": content}]
        query = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)
        ordered_images = [part["image"] for part in content if part["type"] == "image"]
        raw = self.processor(text=[query], images=ordered_images, return_tensors="pt").to(self.device)
        if raw["input_ids"].shape[1] > self.max_input_tokens:
            raise RuntimeError(f"{item['id']} exceeds --max-input-tokens; use a new run with a suitable budget. "
                               "The example was not silently skipped or truncated.")
        captured = {}

        def intercept(module, args, kwargs):
            captured.update(kwargs)
            raise PreparedInputs()

        handle = self.decoder.register_forward_pre_hook(intercept, with_kwargs=True)
        try:
            try:
                self.model.model(**raw, use_cache=False, return_dict=True)
            except PreparedInputs:
                pass
        finally:
            handle.remove()
        if captured.get("inputs_embeds") is None or captured.get("position_ids") is None:
            raise RuntimeError("Native multimodal preparation did not provide embeddings and position IDs.")
        mask = raw["input_ids"][0].eq(self.model.config.image_token_id)
        grid = raw["image_grid_thw"].detach().cpu().numpy()
        merge = self.model.config.vision_config.spatial_merge_size
        sizes = np.prod(grid, axis=1) // (merge * merge)
        if len(sizes) != len(ordered_images) or int(sizes.sum()) != int(mask.sum()):
            raise RuntimeError("Native image grids and visual-token identities do not align.")
        embeddings = captured["inputs_embeds"]
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("Non-finite prepared embeddings.")
        return {"embeddings": embeddings, "position_ids": captured["position_ids"],
                "attention_mask": raw["attention_mask"], "visual_mask": mask,
                "norms": embeddings[0, mask].float().norm(dim=-1).cpu().numpy(),
                "image_ids": np.repeat(np.arange(len(sizes)), sizes), "native": dict(raw)}

    def decoder_kwargs(self, batch, use_cache=False):
        return {"inputs_embeds": batch["embeddings"], "attention_mask": batch["attention_mask"],
                "position_ids": batch["position_ids"], "use_cache": use_cache, "return_dict": True}

    @contextlib.contextmanager
    def capture_routes(self, batch):
        records, handles = {}, []
        try:
            for name, router in self.routers:
                def capture(module, args, output, name=name):
                    if not isinstance(output, tuple) or len(output) != 3:
                        raise RuntimeError("The pinned native router return contract has changed.")
                    _, weights, ids = output
                    mask = batch["visual_mask"].to(ids.device)
                    if ids.shape != (len(mask), self.top_k) or not torch.isfinite(weights).all():
                        raise RuntimeError("Router output is non-finite or not aligned to the input sequence.")
                    if name in records:
                        raise RuntimeError("A router was called twice in one prefill.")
                    records[name] = ids[mask].to(torch.int16).cpu().numpy()
                handles.append(router.register_forward_hook(capture))
            yield records
            if set(records) != set(self.layer_names):
                raise RuntimeError("Some decoder routes were not captured.")
        finally:
            for handle in handles:
                handle.remove()

    def route_array(self, records):
        return np.stack([records[name] for name in self.layer_names])

    @torch.inference_mode()
    def check_native(self, batch):
        with self.capture_routes(batch) as native_records:
            native = self.model(**batch["native"], use_cache=False, return_dict=True, logits_to_keep=1).logits[:, -1].float()
        with self.capture_routes(batch) as adapted_records:
            out = self.decoder(**self.decoder_kwargs(batch))
        adapted = self.head(out.last_hidden_state[:, -1:]).squeeze(1).float().to(native.device)
        if not torch.isfinite(native).all() or not torch.isfinite(adapted).all():
            raise RuntimeError("Native-equivalence logits are non-finite.")
        torch.testing.assert_close(adapted, native, atol=0.05, rtol=0.01)
        if not torch.equal(native.argmax(-1), adapted.argmax(-1)):
            raise RuntimeError("Native and adapted next-token predictions differ.")
        if not np.array_equal(self.route_array(native_records), self.route_array(adapted_records)):
            raise RuntimeError("Native and adapted unpruned routes differ.")
        return {"max_absolute_logit_error": float((native - adapted).abs().max()),
                "next_token_matches": True, "all_visual_routes_match": True}

    @torch.inference_mode()
    def evaluate(self, batch, max_new_tokens):
        with self.capture_routes(batch) as records:
            out = self.decoder(**self.decoder_kwargs(batch, use_cache=True))
        routes = self.route_array(records)
        generated, stop = [], "token_limit"
        attention = batch["attention_mask"]
        next_position = int(batch["position_ids"].max()) + 1
        for step in range(max_new_tokens):
            logits = self.head(out.last_hidden_state[:, -1:]).float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite answer-generation logits.")
            token = int(logits.argmax(-1).item())
            generated.append(token)
            if token in self.eos:
                stop = "eos"
                break
            if step + 1 == max_new_tokens:
                break
            attention = torch.cat([attention, torch.ones_like(attention[:, :1])], dim=1)
            position = torch.full((3, 1, 1), next_position + step, dtype=torch.long, device=self.device)
            out = self.decoder(input_ids=torch.tensor([[token]], device=self.device), attention_mask=attention,
                               position_ids=position, past_key_values=out.past_key_values,
                               use_cache=True, return_dict=True)
        answer = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {"routes": routes, "raw_answer": answer, "generated_token_ids": generated,
                "generated_tokens": len(generated), "stop_reason": stop}
