# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Nemotron 3.5 (hybrid Mamba-2 / GQA / MoE).

- Attention: standard GQA TP (fused wqkv colwise, wo rowwise).
- Ungated MLP / shared expert: up_proj colwise, down_proj rowwise.
- MoE routed experts: upstream EP/TP helpers (w13 holds only up_proj).
- Mamba-2: NOT tensor-parallel. The mixer gathers the full sequence, runs
  replicated across TP in a local SPMD region, and returns to the layer layout.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_spmd,
)
from torchtitan.models.common.moe_sharding import (
    set_routed_moe_sharding_config,
    shared_expert_rowwise_config,
)
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from .model import Nemotron35Model, NemotronBlock


def _layer_layout(enable_sp: bool):
    return (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )


def _set_mamba_sharding(mamba_cfg, *, enable_sp: bool) -> None:
    layer_layout = _layer_layout(enable_sp)
    # Under SP the scan needs the whole sequence: gather to Replicate. Without
    # SP the input is already Invariant over TP; params follow the same type.
    tp_type = spmd.R if enable_sp else spmd.I
    full_layout = dense_activation_placement(tp=tp_type, cp=spmd.S(0))
    param = dense_param_placement(tp=tp_type)
    mamba_cfg.sharding_config = ShardingConfig(
        state_shardings={"A_log": param, "D": param, "dt_bias": param},
        in_src_shardings={"x": layer_layout},
        in_dst_shardings={"x": full_layout},
        out_src_shardings=full_layout,
        out_dst_shardings=layer_layout,
        local_spmd=True,
    )
    for child in (mamba_cfg.in_proj, mamba_cfg.out_proj, mamba_cfg.conv1d):
        child.sharding_config = ShardingConfig(
            state_shardings={"weight": param, "bias": param}
        )
    mamba_cfg.norm.sharding_config = ShardingConfig(
        state_shardings={"weight": param}
    )


def _set_mlp_sharding(mlp_cfg, *, input_layout, output_layout, down_rowwise) -> None:
    mlp_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x": input_layout},
        out_src_shardings=output_layout,
    )
    mlp_cfg.up_proj.sharding_config = colwise_config(input_layout=input_layout)
    mlp_cfg.down_proj.sharding_config = down_rowwise(output_layout=output_layout)


def set_nemotron_sharding_config(
    config: "Nemotron35Model.Config",
    *,
    enable_sp: bool,
    enable_ep: bool = False,
) -> None:
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    layout = _layer_layout(enable_sp)
    for layer_cfg in config.layers:
        layer_cfg.sharding_config = ShardingConfig(
            in_src_shardings={"x": layout},
            out_src_shardings=layout,
        )
        _set_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)


def _set_layer_sharding(
    layer_cfg: "NemotronBlock.Config", *, enable_sp: bool, enable_ep: bool
) -> None:
    layout = _layer_layout(enable_sp)
    layer_cfg.norm.sharding_config = norm_config(enable_sp=enable_sp)

    if layer_cfg.mamba is not None:
        _set_mamba_sharding(layer_cfg.mamba, enable_sp=enable_sp)
    elif layer_cfg.attention is not None:
        set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
        set_gqa_inner_attention_local_spmd(layer_cfg.attention.inner_attention)
    elif layer_cfg.feed_forward is not None:
        _set_mlp_sharding(
            layer_cfg.feed_forward,
            input_layout=layout,
            output_layout=layout,
            down_rowwise=rowwise_config,
        )
    elif layer_cfg.moe is not None:
        moe_cfg = layer_cfg.moe
        set_routed_moe_sharding_config(
            moe_cfg, enable_ep=enable_ep, enable_sp=enable_sp
        )
        if moe_cfg.shared_experts is not None:
            # Without SP the shared output stays Partial so routed + shared
            # partials combine before the MoE's single all-reduce.
            shared_out = (
                dense_sequence_parallel_placement()
                if enable_sp
                else dense_activation_placement(tp=spmd.P, cp=spmd.S(0))
            )
            _set_mlp_sharding(
                moe_cfg.shared_experts,
                input_layout=layout,
                output_layout=shared_out,
                down_rowwise=shared_expert_rowwise_config,
            )
