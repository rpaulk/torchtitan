# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Nemotron 3.5 Lightning model flavors.

Only flavors backed by a published Nemotron 3.5 ``config.json`` live here
(plus a tiny CPU/CI debug model). The ``lightning`` flavor is
NVIDIA-Nemotron-3.5-Lightning-30B-A3B: 52 layers = 23 Mamba / 23 MoE / 6
attention, 128 routed experts top-6, one shared expert, NoPE attention. The
checkpoint's multi-token-prediction head (``mtp.*``) is not built; the HF load
reads only trunk keys, so it is ignored.
"""

from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn

from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.common import Embedding, Linear, Sigmoid
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_gqa_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    GroupedLinear,
    RowParallelLinear,
    SharedExpertRowParallelLinear,
)
from torchtitan.models.common.nn_modules import Conv1d, RMSNorm
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher

from .mamba import a_log_init, Mamba2Mixer, MambaRMSNormGated
from .model import Nemotron35Model, NemotronBlock, NemotronMLP
from .moe import NemotronRoutedExperts

__all__ = ["MODEL_FLAVORS", "build_model_config", "parse_hybrid_pattern"]

_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_MAMBA_STATE_INIT = {
    "A_log": a_log_init,
    "D": nn.init.ones_,
    "dt_bias": nn.init.ones_,
}

# M = Mamba-2, * = attention, - = ungated MLP, E = MoE.
_PATTERN_TO_BLOCK_TYPE = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(
            nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)
        ),
        "bias": nn.init.zeros_,
    }


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {"weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s)}


def parse_hybrid_pattern(pattern: str) -> list[str]:
    """Expand a ``hybrid_override_pattern`` string into per-layer block types."""
    unknown = sorted(set(pattern) - set(_PATTERN_TO_BLOCK_TYPE))
    if unknown:
        raise ValueError(
            f"Unknown hybrid_override_pattern character(s): {unknown}. "
            f"Valid characters are {sorted(_PATTERN_TO_BLOCK_TYPE)}."
        )
    return [_PATTERN_TO_BLOCK_TYPE[c] for c in pattern]


def _norm(dim: int, eps: float) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=eps, param_init=_NORM_INIT)


def _mamba_config(
    *,
    dim: int,
    layer_id: int,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    state_dim: int,
    conv_kernel: int,
    chunk_size: int,
    eps: float,
) -> Mamba2Mixer.Config:
    inter = num_heads * head_dim
    conv_dim = inter + 2 * n_groups * state_dim
    return Mamba2Mixer.Config(
        num_heads=num_heads,
        head_dim=head_dim,
        n_groups=n_groups,
        state_dim=state_dim,
        chunk_size=chunk_size,
        in_proj=Linear.Config(
            in_features=dim,
            out_features=inter + conv_dim + num_heads,
            param_init=_LINEAR_INIT,
        ),
        conv1d=Conv1d.Config(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=conv_kernel,
            groups=conv_dim,
            padding=conv_kernel - 1,
            param_init=_LINEAR_INIT,
        ),
        norm=MambaRMSNormGated.Config(
            dim=inter, group_size=inter // n_groups, eps=eps, param_init=_NORM_INIT
        ),
        out_proj=Linear.Config(
            in_features=inter, out_features=dim, param_init=_depth_init(layer_id)
        ),
        param_init=_MAMBA_STATE_INIT,
    )


def _ungated_mlp(
    *, dim: int, hidden_dim: int, layer_id: int, shared: bool
) -> NemotronMLP.Config:
    down_cls = SharedExpertRowParallelLinear if shared else RowParallelLinear
    return NemotronMLP.Config(
        up_proj=ColumnParallelLinear.Config(
            in_features=dim, out_features=hidden_dim, param_init=_LINEAR_INIT
        ),
        down_proj=down_cls.Config(
            in_features=hidden_dim, out_features=dim, param_init=_depth_init(layer_id)
        ),
    )


def _build_layers(
    *,
    block_types: list[str],
    dim: int,
    eps: float,
    attn_backend: str,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    mlp_hidden_dim: int,
    num_experts: int,
    top_k: int,
    moe_hidden_dim: int,
    shared_expert_dim: int | None,
    route_scale: float,
    mamba_num_heads: int,
    mamba_head_dim: int,
    mamba_n_groups: int,
    mamba_state_dim: int,
    mamba_conv_kernel: int = 4,
    mamba_chunk_size: int = 128,
) -> list[NemotronBlock.Config]:
    inner_attention = get_attention_config(attn_backend)
    layers = []
    for layer_id, block_type in enumerate(block_types):
        kwargs = {}
        if block_type == "mamba":
            kwargs["mamba"] = _mamba_config(
                dim=dim,
                layer_id=layer_id,
                num_heads=mamba_num_heads,
                head_dim=mamba_head_dim,
                n_groups=mamba_n_groups,
                state_dim=mamba_state_dim,
                conv_kernel=mamba_conv_kernel,
                chunk_size=mamba_chunk_size,
                eps=eps,
            )
        elif block_type == "attention":
            # Nemotron-H attention applies no positional embedding (NoPE).
            kwargs["attention"] = make_gqa_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_depth_init(layer_id),
                inner_attention=inner_attention,
                rope=None,
            )
        elif block_type == "mlp":
            kwargs["feed_forward"] = _ungated_mlp(
                dim=dim, hidden_dim=mlp_hidden_dim, layer_id=layer_id, shared=False
            )
        else:
            kwargs["moe"] = make_moe_config(
                num_experts=num_experts,
                # Sigmoid scores, renormalized top-k, scaled by
                # routed_scaling_factor; n_group = topk_group = 1 (no
                # group-limited routing).
                router=make_router_config(
                    dim=dim,
                    num_experts=num_experts,
                    gate_param_init=_LINEAR_INIT,
                    score_func=Sigmoid.Config(),
                    top_k=top_k,
                    route_norm=True,
                    route_scale=route_scale,
                ),
                routed_experts=NemotronRoutedExperts.Config(
                    w13=GroupedLinear.Config(
                        group_size=num_experts,
                        in_features=dim,
                        out_features=moe_hidden_dim,
                        param_init={"weight": _LINEAR_INIT["weight"]},
                    ),
                    w2=GroupedLinear.Config(
                        group_size=num_experts,
                        in_features=moe_hidden_dim,
                        out_features=dim,
                        param_init={"weight": _depth_init(layer_id)["weight"]},
                    ),
                    token_dispatcher=AllToAllTokenDispatcher.Config(
                        num_experts=num_experts, top_k=top_k
                    ),
                ),
                shared_experts=_ungated_mlp(
                    dim=dim,
                    hidden_dim=shared_expert_dim,
                    layer_id=layer_id,
                    shared=True,
                )
                if shared_expert_dim
                else None,
            )
        layers.append(
            NemotronBlock.Config(
                block_type=block_type, norm=_norm(dim, eps), **kwargs
            )
        )
    return layers


def _model_config(
    *, dim: int, vocab_size: int, seq_len: int, eps: float, layers: list
) -> Nemotron35Model.Config:
    return Nemotron35Model.Config(
        max_context_length=seq_len,
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=_norm(dim, eps),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def _debugmodel(attn_backend: str, *, seq_len: int) -> Nemotron35Model.Config:
    dim = 256
    return _model_config(
        dim=dim,
        vocab_size=2048,
        seq_len=seq_len,
        eps=1e-5,
        layers=_build_layers(
            block_types=parse_hybrid_pattern("ME*EM-"),
            dim=dim,
            eps=1e-5,
            attn_backend=attn_backend,
            n_heads=8,
            n_kv_heads=2,
            head_dim=32,
            mlp_hidden_dim=512,
            num_experts=8,
            top_k=2,
            moe_hidden_dim=128,
            shared_expert_dim=256,
            route_scale=2.5,
            mamba_num_heads=8,
            mamba_head_dim=32,
            mamba_n_groups=2,
            mamba_state_dim=16,
            mamba_chunk_size=16,
        ),
    )


def _lightning(attn_backend: str, *, seq_len: int) -> Nemotron35Model.Config:
    # NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/config.json
    dim = 2688
    eps = 1e-5
    return _model_config(
        dim=dim,
        vocab_size=131072,
        seq_len=seq_len,
        eps=eps,
        layers=_build_layers(
            block_types=parse_hybrid_pattern(
                "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
            ),
            dim=dim,
            eps=eps,
            attn_backend=attn_backend,
            n_heads=32,
            n_kv_heads=2,
            head_dim=128,
            mlp_hidden_dim=1856,
            num_experts=128,
            top_k=6,
            moe_hidden_dim=1856,
            shared_expert_dim=3712,
            route_scale=2.5,
            mamba_num_heads=64,
            mamba_head_dim=64,
            mamba_n_groups=8,
            mamba_state_dim=128,
        ),
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 4096),
    "lightning": (_lightning, 1048576),
}


def build_model_config(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Nemotron35Model.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(attn_backend=attn_backend, seq_len=context_len)
    if converters is not None:
        validate_converter_compatibility(converters)
        for c in converters:
            config = c.build().convert(config)
    return config
