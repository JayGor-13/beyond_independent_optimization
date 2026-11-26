"""Gemma-specific visual pruning, native masking and routed-expert capture."""
import numpy as np
import torch
from contextlib import nullcontext

from adapters import PreparedInputs
from core import MODELS, frame_groups
from gemma_numerics import require_finite, trace_finite


class GemmaAdapter:
    def __init__(self, model, dtype, max_soft_tokens=280, diagnostic_path=None):
        from transformers import AutoProcessor
        self.model, self.dtype, self.family = model, dtype, "gemma4"
        self.diagnostic_path = diagnostic_path
        self._native_validated = False
        repo, revision = MODELS[self.family]
        self.processor = AutoProcessor.from_pretrained(repo, revision=revision)
        self.processor.image_processor.max_soft_tokens = max_soft_tokens
        self.tokenizer = self.processor.tokenizer
        self.decoder = model.model.language_model
        self.embedding = self.decoder.get_input_embeddings()
        self.device = self.embedding.weight.device
        self.blocks = [(name, module) for name, module in self.decoder.named_modules()
                       if type(module).__name__ == "Gemma4TextRouter"]
        config = model.config.text_config
        self.n_experts, self.top_k = config.num_experts, config.top_k_experts
        if len(self.blocks) != config.num_hidden_layers or config.hidden_size_per_layer_input:
            raise ValueError("Unsupported Gemma topology; expected the pinned 26B-A4B configuration.")

    @torch.inference_mode()
    def prepare(self, images, text, temporal=False):
        content = []
        if temporal:
            content.append({"type": "text", "text": "The following images are chronological frames from one video."})
        for index, image in enumerate(images):
            if temporal:
                content.append({"type": "text", "text": f"Frame {index + 1}:"})
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": ("Question:\n" if temporal else "") + text})
        query = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
        raw = self.processor(text=[query], images=[images], return_tensors="pt").to(self.device)
        if "mm_token_type_ids" not in raw:
            raise ValueError("Gemma processor did not supply modality IDs for vision attention.")
        captured = {}
        def intercept(module, args, kwargs):
            captured.update(kwargs)
            raise PreparedInputs()
        hook = self.decoder.register_forward_pre_hook(intercept, with_kwargs=True)
        try:
            try:
                with (nullcontext() if getattr(self, "_native_validated", False) else
                      trace_finite(self.model, "prepare", getattr(self, "diagnostic_path", None))):
                    self.model.model(**raw, use_cache=False)
            except PreparedInputs:
                pass
        finally:
            hook.remove()
        if "inputs_embeds" not in captured:
            raise RuntimeError("Gemma native visual preparation was not intercepted.")
        require_finite(captured["inputs_embeds"], "prepare", "combined_embeddings",
                       getattr(self, "diagnostic_path", None))
        mask = raw["input_ids"][0].eq(self.model.config.image_token_id)
        eos = self.model.config.eos_token_id
        return {"embeddings": captured["inputs_embeds"], "attention_mask": raw["attention_mask"],
                "input_ids": raw["input_ids"], "mm_token_type_ids": raw["mm_token_type_ids"],
                "visual_mask": mask, "frame_ids": frame_groups(mask, len(images)),
                "native": dict(raw), "eos": eos if isinstance(eos, list) else [eos]}

    def model_kwargs(self, batch, use_cache):
        # Use Gemma's multimodal forward to reconstruct full/sliding masks after
        # pruning, preserve bidirectional vision attention and apply logit softcap.
        return {"inputs_embeds": batch["embeddings"], "attention_mask": batch["attention_mask"],
                "mm_token_type_ids": batch["mm_token_type_ids"], "use_cache": use_cache,
                "logits_to_keep": 1, "return_dict": True}

    @torch.inference_mode()
    def check_native(self, batch):
        path = getattr(self, "diagnostic_path", None)
        with trace_finite(self.model, "native", path):
            native = self.model(**batch["native"], use_cache=False, logits_to_keep=1).logits[:, -1].float()
        with trace_finite(self.model, "adapted", path):
            adapted = self.model(**self.model_kwargs(batch, False)).logits[:, -1].float().to(native.device)
        torch.testing.assert_close(adapted, native, atol=0.03, rtol=0.01)
        if not torch.equal(adapted.argmax(-1), native.argmax(-1)):
            raise RuntimeError("Gemma native and adapted next tokens differ.")
        self._native_validated = True
        return {"max_absolute_logit_error": float((adapted - native).abs().max()), "next_token_matches": True}

    @torch.inference_mode()
    def evaluate(self, batch, max_new_tokens=8):
        records, hooks = {}, []
        try:
            for name, router in self.blocks:
                def capture(module, args, output, name=name):
                    probabilities, _, ids = output
                    mask = batch["visual_mask"].to(ids.device)
                    if len(ids) != len(mask):
                        raise ValueError("Gemma routing tokens do not match the prepared sequence.")
                    # Probabilistic TRC: normalized selected probabilities before
                    # Gemma's learned per-expert output scales. The native forward
                    # still applies those scales to expert contributions.
                    weights = probabilities.gather(-1, ids).float()
                    weights = weights / weights.sum(-1, keepdim=True)
                    require_finite(weights, "evaluate", name + ".routing_probabilities",
                                   getattr(self, "diagnostic_path", None))
                    records[name] = (ids[mask].short().cpu().numpy(), weights[mask].cpu().numpy())
                hooks.append(router.register_forward_hook(capture))
            out = self.model(**self.model_kwargs(batch, max_new_tokens > 0))
        finally:
            for hook in hooks:
                hook.remove()
        names = [name for name, _ in self.blocks]
        if set(records) != set(names):
            raise RuntimeError("Not all Gemma routers were captured.")
        attention = batch["attention_mask"]
        generated = []
        for step in range(max_new_tokens):
            require_finite(out.logits[:, -1], "evaluate", "generation_logits",
                           getattr(self, "diagnostic_path", None))
            token = int(out.logits[:, -1].argmax(-1).item())
            generated.append(token)
            if token in batch["eos"] or step + 1 == max_new_tokens:
                break
            attention = torch.cat([attention, torch.ones_like(attention[:, :1])], dim=1)
            out = self.model(input_ids=torch.tensor([[token]], device=self.device), attention_mask=attention,
                             past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1,
                             return_dict=True)
        return (np.stack([records[name][0] for name in names]), np.stack([records[name][1] for name in names]),
                np.asarray(names), self.tokenizer.decode(generated, skip_special_tokens=True).strip())
