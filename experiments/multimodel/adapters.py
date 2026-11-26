"""Model adapters for image-token pruning and sampled-frame TRC."""
import copy

import numpy as np
import torch

from core import MODELS, frame_groups


class PreparedInputs(Exception):
    """Stop the native multimodal forward just before the text decoder."""


class Adapter:
    def __init__(self, model, family, dtype, max_pixels=262144):
        from transformers import AutoProcessor, AutoTokenizer
        self.model, self.family, self.dtype = model, family, dtype
        repo, revision = MODELS[family]
        if family == "qwen3vl":
            self.processor = AutoProcessor.from_pretrained(repo, revision=revision,
                                                           min_pixels=65536, max_pixels=max_pixels)
            self.tokenizer = self.processor.tokenizer
            self.decoder = model.model.language_model
            self.head = model.lm_head
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision,
                                                           trust_remote_code=True, use_fast=False)
            self.decoder = model.language_model.model
            self.head = model.language_model.lm_head
        self.embedding = self.decoder.get_input_embeddings()
        self.device = self.embedding.weight.device
        self.blocks = [(name, block) for name, block in self.decoder.named_modules()
                       if hasattr(block, "gate") and hasattr(block, "experts") and hasattr(block, "top_k")]
        if not self.blocks:
            raise RuntimeError("No supported MoE routers found.")
        self.n_experts = int(self.blocks[0][1].num_experts)
        self.top_k = int(self.blocks[0][1].top_k)
        if any(int(b.num_experts) != self.n_experts or int(b.top_k) != self.top_k for _, b in self.blocks):
            raise RuntimeError("Heterogeneous routing requires a different route schema.")

    @torch.inference_mode()
    def prepare(self, images, text, temporal=False):
        if self.family == "qwen3vl":
            content = []
            if temporal:
                content.append({"type": "text", "text": "The following images are chronological frames from one video."})
            for i, image in enumerate(images):
                if temporal:
                    content.append({"type": "text", "text": f"Frame {i + 1}:"})
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": ("Question:\n" if temporal else "") + text})
            query = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                       tokenize=False, add_generation_prompt=True)
            raw = self.processor(text=[query], images=images, return_tensors="pt").to(self.device)
            captured = {}

            def intercept(module, args, kwargs):
                captured.update(kwargs)
                raise PreparedInputs()

            handle = self.decoder.register_forward_pre_hook(intercept, with_kwargs=True)
            try:
                try:
                    self.model.model(**raw, use_cache=False)
                except PreparedInputs:
                    pass
            finally:
                handle.remove()
            if "inputs_embeds" not in captured or captured.get("deepstack_visual_embeds") is None:
                raise RuntimeError("Qwen native visual preparation did not provide DeepStack features.")
            mask = captured["visual_pos_masks"][0].bool()
            batch = {"embeddings": captured["inputs_embeds"], "attention_mask": raw["attention_mask"],
                     "input_ids": raw["input_ids"], "visual_mask": mask,
                     "position_ids": captured["position_ids"],
                     "deepstack": captured["deepstack_visual_embeds"], "native": dict(raw)}
            eos = self.model.config.text_config.eos_token_id
        else:
            from torchvision import transforms
            from torchvision.transforms.functional import InterpolationMode
            size = self.model.config.force_image_size or self.model.config.vision_config.image_size
            transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            pixels = torch.stack([transform(img.convert("RGB")) for img in images]).to(self.device, self.dtype)
            # One tile per frame/image; fixed across conditions to bound memory.
            features = self.model.extract_feature(pixels)
            image_token = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            if image_token is None or image_token == self.tokenizer.unk_token_id:
                raise RuntimeError("Missing InternVL image context token.")
            self.model.img_context_token_id = image_token
            placeholder = "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token + "</img>"
            prefix = ("The following images are chronological frames from one video.\n" if temporal else "")
            prefix += "\n".join((f"Frame {i + 1}:\n" if temporal else "") + placeholder for i in range(len(images)))
            prompt = prefix + "\n" + ("Question:\n" if temporal else "") + text
            template = copy.deepcopy(self.model.conv_template)
            template.system_message = self.model.system_message
            template.append_message(template.roles[0], prompt)
            template.append_message(template.roles[1], None)
            raw = self.tokenizer(template.get_prompt(), return_tensors="pt").to(self.device)
            mask = raw.input_ids[0].eq(image_token)
            embeddings = self.embedding(raw.input_ids).clone()
            features = features.reshape(-1, features.shape[-1]).to(embeddings)
            if int(mask.sum()) != len(features):
                raise ValueError("Visual features and placeholders differ in length.")
            embeddings[0, mask] = features
            batch = {"embeddings": embeddings, "attention_mask": raw.attention_mask,
                     "input_ids": raw.input_ids, "visual_mask": mask,
                     "native": {**dict(raw), "pixel_values": pixels,
                                "image_flags": torch.ones((len(images), 1), device=self.device, dtype=torch.long)}}
            eos = self.tokenizer.convert_tokens_to_ids(template.sep.strip())
        batch["frame_ids"] = frame_groups(batch["visual_mask"], len(images))
        batch["eos"] = eos if isinstance(eos, list) else [eos]
        return batch

    def decoder_kwargs(self, batch, use_cache):
        kwargs = {"inputs_embeds": batch["embeddings"], "attention_mask": batch["attention_mask"],
                  "use_cache": use_cache, "return_dict": True}
        if "position_ids" in batch:
            kwargs.update(position_ids=batch["position_ids"], visual_pos_masks=batch["visual_mask"][None],
                          deepstack_visual_embeds=batch["deepstack"])
        return kwargs

    @torch.inference_mode()
    def check_native(self, batch):
        """Fail before collecting data if the adapter changes the unpruned forward."""
        def last_token(module, args):
            return (args[0][:, -1:, :],)
        hook = self.head.register_forward_pre_hook(last_token)
        try:
            native = self.model(**batch["native"], use_cache=False, return_dict=True).logits[:, -1].float()
        finally:
            hook.remove()
        out = self.decoder(**self.decoder_kwargs(batch, False))
        adapted = self.head(out.last_hidden_state[:, -1:]).squeeze(1).float().to(native.device)
        error = float((adapted - native).abs().max())
        torch.testing.assert_close(adapted, native, atol=0.03, rtol=0.01)
        if not torch.equal(adapted.argmax(-1), native.argmax(-1)):
            raise RuntimeError("Unpruned native and adapter next-token predictions differ.")
        return {"max_absolute_logit_error": error, "next_token_matches": True}

    @torch.inference_mode()
    def evaluate(self, batch, max_new_tokens=8):
        records, handles = {}, []
        try:
            for name, block in self.blocks:
                def capture(module, args, output, name=name, block=block):
                    logits = output.reshape(-1, block.num_experts)
                    mask = batch["visual_mask"].to(logits.device)
                    if len(logits) != len(mask):
                        raise ValueError("Router token count differs from prepared sequence.")
                    # Match the native router's softmax -> Top-k -> renormalization.
                    weights, ids = logits.float().softmax(-1).topk(block.top_k, dim=-1)
                    weights = weights / weights.sum(-1, keepdim=True)
                    weights = weights.to(args[0].dtype)
                    records[name] = (ids[mask].short().cpu().numpy(), weights[mask].float().cpu().numpy())
                handles.append(block.gate.register_forward_hook(capture))
            out = self.decoder(**self.decoder_kwargs(batch, max_new_tokens > 0))
        finally:
            for handle in handles:
                handle.remove()
        if set(records) != {name for name, _ in self.blocks}:
            raise RuntimeError("Some routed layers were not captured.")
        names = np.asarray([name for name, _ in self.blocks])
        ids = np.stack([records[name][0] for name in names])
        weights = np.stack([records[name][1] for name in names])
        generated = []
        attention = batch["attention_mask"]
        next_position = (int(batch["position_ids"].max()) + 1 if "position_ids" in batch
                         else batch["embeddings"].shape[1])
        for step in range(max_new_tokens):
            token = int(self.head(out.last_hidden_state[:, -1:]).argmax(-1).item())
            generated.append(token)
            if token in batch["eos"] or step + 1 == max_new_tokens:
                break
            attention = torch.cat((attention, torch.ones_like(attention[:, :1])), dim=1)
            position = torch.tensor([[next_position + step]], device=self.device)
            if self.family == "qwen3vl":
                position = position[None].expand(3, -1, -1)
            kwargs = {"input_ids": torch.tensor([[token]], device=self.device),
                      "attention_mask": attention, "position_ids": position,
                      "past_key_values": out.past_key_values, "use_cache": True, "return_dict": True}
            out = self.decoder(**kwargs)
        return ids, weights, names, self.tokenizer.decode(generated, skip_special_tokens=True).strip()
