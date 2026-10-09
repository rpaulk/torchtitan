# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Nemotron 3.5 Lightning: hybrid Mamba-2 / attention / MoE decoder.

Nemotron-H layers hold a SINGLE mixer each (HF ``NemotronHBlock``):
``x + mixer(norm(x))`` where the mixer is one of Mamba-2 (``M``),
self-attention (``*``), ungated MLP (``-``) or MoE (``E``). Each block
therefore owns exactly one ``norm`` and one mixer, mirroring HF's
``backbone.layers.{N}.norm`` / ``backbone.layers.{N}.mixer`` layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import (
    AttentionMetadataMap,
    FlexAttentionMetadata,
    GQAttention,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module

from .mamba import Mamba2Mixer
from .state_dict_adapter import NemotronStateDictAdapter

BLOCK_TYPES = ("mamba", "attention", "mlp", "moe")


class NemotronMLP(Module):
    """Ungated feed-forward: ``down_proj(relu(up_proj(x))**2)`` (no gate)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        up_proj: Linear.Config
        down_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.up_proj = config.up_proj.build()
        self.down_proj = config.down_proj.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up_proj(x)
        remat.recompute_needs_tensor(h)
        return self.down_proj(torch.square(F.relu(h)))


class NemotronBlock(Module):
    """One Nemotron-H layer: ``x + mixer(norm(x))`` with exactly one mixer."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        block_type: str
        norm: RMSNorm.Config
        # Exactly one of these is set, selected by block_type. The field names
        # ``attention`` / ``feed_forward`` / ``moe`` are what Decoder.Config's
        # first_* helpers and the TP/EP checks look up.
        mamba: Mamba2Mixer.Config | None = None
        attention: GQAttention.Config | None = None
        feed_forward: NemotronMLP.Config | None = None
        moe: MoE.Config | None = None

        def __post_init__(self) -> None:
            if self.block_type not in BLOCK_TYPES:
                raise ValueError(f"unknown block_type {self.block_type!r}")
            mixers = {
                "mamba": self.mamba,
                "attention": self.attention,
                "mlp": self.feed_forward,
                "moe": self.moe,
            }
            for kind, cfg in mixers.items():
                if (cfg is not None) != (kind == self.block_type):
                    raise ValueError(
                        f"block_type={self.block_type!r} requires exactly the "
                        f"{self.block_type!r} mixer config to be set"
                    )

    def __init__(self, config: Config):
        super().__init__()
        self.block_type = config.block_type
        self.moe_enabled = config.block_type == "moe"
        self.norm = config.norm.build()
        # Only the active mixer is registered, so FQNs stay one-mixer-per-layer.
        if config.mamba is not None:
            self.mamba = config.mamba.build()
        if config.attention is not None:
            self.attention = config.attention.build()
        if config.feed_forward is not None:
            self.feed_forward = config.feed_forward.build()
        if config.moe is not None:
            self.moe = config.moe.build()
        self.attention_metadata_key = (
            self.attention.attention_metadata_key
            if config.attention is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm(x)
        if self.block_type == "mamba":
            out = self.mamba(h, positions)
        elif self.block_type == "attention":
            out = self.attention(h, attention_metadata, positions)
        elif self.block_type == "mlp":
            out = self.feed_forward(h)
        else:
            out = self.moe(
                h,
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        return remat.region(
            torch.add, self.remat_region_name("residual"), recompute=False
        )(x, out)


class Nemotron35Model(Decoder):
    state_dict_adapter_cls = NemotronStateDictAdapter

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        local_compile_regions: list[str] = field(default_factory=lambda: ["loss"])

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            # Routed-expert params are already weighted by the active ratio.
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_op_flops = 0
            for layer in self.layers:
                attention = layer.attention
                if attention is None:
                    continue
                head_dim = attention.head_dim or attention.dim // attention.n_heads
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=head_dim,
                    v_head_dim=head_dim,
                    seq_len=seq_len,
                )
            return nparams, 6 * active_nparams + attention_op_flops

        def set_sharding_(self, parallelism: ParallelismConfig) -> None:
            from .sharding import set_nemotron_sharding_config

            set_nemotron_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ):
        # Same as Decoder.forward except the metadata key is read per block:
        # mamba/mlp/moe blocks have no attention module and take None.
        import spmd_types as spmd

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        with spmd.no_typecheck():
            aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        for layer in self.layers.values():
            key = cast(NemotronBlock, layer).attention_metadata_key
            layer_metadata = (
                None
                if attention_metadata is None or key is None
                else attention_metadata.get(key)
            )
            h = layer(
                h,
                layer_metadata,
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        h = self.norm(h) if self.norm is not None else h
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h
