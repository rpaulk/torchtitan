# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for common vision encoder components."""

from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import set_gqa_inner_attention_local_map
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.common.vision_encoder import VisionTransformerBlock


DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def _vision_placement(
    *,
    dp: spmd.PerMeshAxisSpmdType,
    tp: spmd.PerMeshAxisSpmdType,
    cp: spmd.PerMeshAxisSpmdType | None = None,
) -> SpmdType:
    axis_types = {DP: dp}
    if cp is not None:
        axis_types[CP] = cp
    axis_types[TP] = tp
    return SpmdType(axis_types)


def multimodal_input_sharding(
    *, cp: spmd.PerMeshAxisSpmdType | None = None
) -> dict[str, SpmdType]:
    """SPMD layouts for VLM vision inputs (folded into a model's input_sharding).

    The vision tensors are DP-local (``V@DP``) -- each DP rank owns its own
    images -- and TP-invariant (``I@TP``). Callers that prepare a complete
    multimodal embedding sequence before CP sharding pass ``cp=R``.
    """
    layout = _vision_placement(dp=spmd.V, cp=cp, tp=spmd.I)
    return {
        "pixel_values": layout,
        "pixel_values_videos": layout,
        "grid_thw": layout,
        "grid_thw_videos": layout,
    }


def invariant_norm_config(
    *, cp: spmd.PerMeshAxisSpmdType | None = None
) -> ShardingConfig:
    """Norm whose state and activations are invariant across TP ranks."""
    return ShardingConfig(
        state_shardings={
            "weight": _vision_placement(dp=spmd.R, cp=cp, tp=spmd.I),
            "bias": _vision_placement(dp=spmd.R, cp=cp, tp=spmd.I),
        },
        in_src_shardings={
            "input": _vision_placement(dp=spmd.V, cp=cp, tp=spmd.I),
        },
        in_dst_shardings={
            "input": _vision_placement(dp=spmd.V, cp=cp, tp=spmd.I),
        },
        out_src_shardings=_vision_placement(dp=spmd.V, cp=cp, tp=spmd.I),
        out_dst_shardings=_vision_placement(dp=spmd.V, cp=cp, tp=spmd.I),
    )


def vision_invariant_linear_config(
    *, cp: spmd.PerMeshAxisSpmdType | None = None
) -> ShardingConfig:
    """Unsharded linear whose state and activations are invariant at TP."""
    return ShardingConfig(
        state_shardings={
            "weight": _vision_placement(dp=spmd.R, cp=cp, tp=spmd.I),
            "bias": _vision_placement(dp=spmd.R, cp=cp, tp=spmd.I),
        },
        in_src_shardings={
            "input": _vision_placement(dp=spmd.V, cp=cp, tp=spmd.I),
        },
        in_dst_shardings={
            "input": _vision_placement(dp=spmd.V, cp=cp, tp=spmd.I),
        },
        out_src_shardings=_vision_placement(dp=spmd.V, cp=cp, tp=spmd.I),
        out_dst_shardings=_vision_placement(dp=spmd.V, cp=cp, tp=spmd.I),
    )


def vision_colwise_config(
    *,
    input_tp: spmd.PerMeshAxisSpmdType = spmd.I,
    cp: spmd.PerMeshAxisSpmdType | None = None,
) -> ShardingConfig:
    """Colwise vision linear with a TP-replicated local matmul input."""
    return ShardingConfig(
        state_shardings={
            "weight": _vision_placement(dp=spmd.R, cp=cp, tp=spmd.S(0)),
            "bias": _vision_placement(dp=spmd.R, cp=cp, tp=spmd.S(0)),
        },
        in_src_shardings={
            "input": _vision_placement(dp=spmd.V, cp=cp, tp=input_tp),
        },
        in_dst_shardings={
            "input": _vision_placement(dp=spmd.V, cp=cp, tp=spmd.R),
        },
        out_src_shardings=_vision_placement(dp=spmd.V, cp=cp, tp=spmd.S(-1)),
    )


def vision_scaled_bias_rowwise_config(
    *, cp: spmd.PerMeshAxisSpmdType | None = None
) -> ShardingConfig:
    """Scaled-bias rowwise vision linear returning a TP-invariant activation."""
    input_layout = _vision_placement(dp=spmd.V, cp=cp, tp=spmd.S(1))
    return ShardingConfig(
        state_shardings={
            "weight": _vision_placement(dp=spmd.R, cp=cp, tp=spmd.S(1)),
            "bias": _vision_placement(dp=spmd.R, cp=cp, tp=spmd.R),
        },
        in_src_shardings={
            "input": input_layout,
        },
        in_dst_shardings={
            "input": input_layout,
        },
        out_src_shardings=_vision_placement(dp=spmd.V, cp=cp, tp=spmd.P),
        out_dst_shardings=_vision_placement(dp=spmd.V, cp=cp, tp=spmd.I),
        local_map=LocalMapConfig(in_grad_placements=(input_layout,)),
    )


def set_vision_transformer_block_sharding_config(
    block: "VisionTransformerBlock.Config",
    *,
    rope_cache_dp: spmd.PerMeshAxisSpmdType,
    cp: spmd.PerMeshAxisSpmdType | None = None,
) -> None:
    """Set TP sharding for the common vision transformer block."""
    block.norm1.sharding_config = invariant_norm_config(cp=cp)
    block.norm2.sharding_config = invariant_norm_config(cp=cp)

    block.attn.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": _vision_placement(dp=spmd.V, cp=cp, tp=spmd.I),
            "rope_cache": _vision_placement(dp=rope_cache_dp, cp=cp, tp=spmd.I),
        },
        in_dst_shardings={
            "x": _vision_placement(dp=spmd.V, cp=cp, tp=spmd.R),
            "rope_cache": _vision_placement(dp=rope_cache_dp, cp=cp, tp=spmd.R),
        },
    )
    block.attn.wq.sharding_config = vision_colwise_config(input_tp=spmd.R, cp=cp)
    block.attn.wk.sharding_config = vision_colwise_config(input_tp=spmd.R, cp=cp)
    block.attn.wv.sharding_config = vision_colwise_config(input_tp=spmd.R, cp=cp)
    block.attn.proj.sharding_config = vision_scaled_bias_rowwise_config(cp=cp)
    if cp is None:
        set_gqa_inner_attention_local_map(block.attn.inner_attention)
    else:
        set_gqa_inner_attention_local_map(block.attn.inner_attention, cp=cp)

    block.mlp.fc1.sharding_config = vision_colwise_config(cp=cp)
    block.mlp.fc2.sharding_config = vision_scaled_bias_rowwise_config(cp=cp)
