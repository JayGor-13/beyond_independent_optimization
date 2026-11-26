"""Fail at the first non-finite Gemma activation, without saving input content."""
import json
from contextlib import contextmanager
from pathlib import Path

import torch


def require_finite(value, stage, module, path=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        return
    finite = torch.isfinite(value)
    if bool(finite.all()):
        return
    record = {"stage": stage, "module": module, "dtype": str(value.dtype),
              "shape": list(value.shape), "nan_count": int(torch.isnan(value).sum()),
              "inf_count": int(torch.isinf(value).sum())}
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    print("[Gemma non-finite] " + json.dumps(record), flush=True)
    raise FloatingPointError(f"Gemma {stage}: first non-finite output at {module}; "
                             f"{record['nan_count']} NaN, {record['inf_count']} Inf. "
                             "No result may be used from this forward pass.")


def tensor_outputs(output):
    if isinstance(output, torch.Tensor):
        yield output
    elif isinstance(output, dict):
        for value in output.values():
            yield from tensor_outputs(value)
    elif isinstance(output, (list, tuple)):
        for value in output:
            yield from tensor_outputs(value)


@contextmanager
def trace_finite(model, stage, path=None):
    # Inspect outputs, not inputs: attention masks legitimately contain -Inf.
    hooks = []
    try:
        for name, module in model.named_modules():
            def check(module, args, output, name=name):
                for index, value in enumerate(tensor_outputs(output)):
                    require_finite(value, stage, f"{name or '<model>'}.output[{index}]", path)
            hooks.append(module.register_forward_hook(check))
        yield
    finally:
        for hook in hooks:
            hook.remove()
