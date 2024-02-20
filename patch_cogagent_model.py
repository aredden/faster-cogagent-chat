from torch_bnb_fp4 import LinearHijack
from torch import nn
import torch
from cogagent_model.modeling_cogagent import (
    VisionExpertMLP,
    VisionExpertAttention,
    CrossAttention,
)


def set_linears_in_submodule(module, as_dtype=torch.bfloat16):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                LinearHijack(child).to(device=child.weight.device, dtype=as_dtype),
            )
        elif isinstance(child, nn.Module):
            set_linears_in_submodule(child, as_dtype=as_dtype)
    if isinstance(module, nn.Linear):
        setattr(
            module,
            name,
            LinearHijack(module).to(device=module.weight.device, dtype=as_dtype),
        )


def hijack_cogagent_linears(module: nn.Module, as_dtype=torch.bfloat16):
    for child_layer in module.modules():
        if isinstance(child_layer, VisionExpertMLP):
            set_linears_in_submodule(child_layer, as_dtype=as_dtype)
        elif isinstance(child_layer, VisionExpertAttention):
            set_linears_in_submodule(child_layer, as_dtype=as_dtype)
        elif isinstance(child_layer, CrossAttention):
            setattr(
                child_layer,
                "dense",
                LinearHijack(child_layer.dense).to(
                    device=child_layer.dense.weight.device, dtype=as_dtype
                ),
            )
            setattr(
                child_layer,
                "key_value",
                LinearHijack(child_layer.key_value).to(
                    device=child_layer.key_value.weight.device, dtype=as_dtype
                ),
            )
