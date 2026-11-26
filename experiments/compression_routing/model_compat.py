import types


def patch_eager_attention(model):
    """Make the FastMMoE fork's eager attention match its decoder interface."""
    for layer in model.language.model.layers:
        attention = layer.self_attn
        if getattr(attention, "_fastmmoe_eager_compatible", False):
            continue

        original = attention.forward

        def eager_forward(module, *args, original_method=original, **kwargs):
            return_logits = bool(kwargs.get("return_logits", False))
            if return_logits:
                kwargs["output_attentions"] = True

            result = original_method(*args, **kwargs)
            if len(result) == 4:
                return result

            attention_output, attention_weights, cache = result
            relation = attention_weights if return_logits else None
            return attention_output, attention_weights, cache, relation

        attention.forward = types.MethodType(eager_forward, attention)
        attention._fastmmoe_eager_compatible = True
