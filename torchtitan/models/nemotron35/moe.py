# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ungated (non-gated) routed experts for Nemotron-H / Nemotron 3.5.

Nemotron-H's experts are *not* SwiGLU. Where a Llama/Mixtral expert computes
``w2(silu(w1(x)) * w3(x))``, a Nemotron-H expert computes
``down(relu(up(x))**2)``: two matrices, no gate branch (``has_gate: false`` in
NVIDIA's config). Upstream ``RoutedExperts`` requires a fused gate/up ``w13``
(``num_linears == 2``) and a binary activation, so this subclass relaxes that
to a single up projection (``num_linears == 1``) and applies relu^2 inline.

Parameter layout (matches upstream naming, so the common MoE sharding helpers
apply unchanged):
  ``w13.weight``  [E, F, D]  -- the up projection (HF ``experts.{i}.up_proj``)
  ``w2.weight``   [E, D, F]  -- the down projection (HF ``experts.{i}.down_proj``)
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.models.common.moe import RoutedExperts

# Shape suffixes:
# T = tokens, D = model dim, E = experts, F = expert hidden dim,
# R = routed tokens, K = experts per token.


class NemotronRoutedExperts(RoutedExperts):
    """``RoutedExperts`` with ungated relu^2 experts (no gate projection)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        def __post_init__(self) -> None:
            # Same invariants as the base class, except w13 holds ONLY the up
            # projection. activation_fn is ignored (relu^2 is fixed).
            if self.w13.group_size != self.w2.group_size:
                raise ValueError("w13 and w2 must contain the same number of experts")
            if self.token_dispatcher.num_experts != self.w13.group_size:
                raise ValueError(
                    "token dispatcher and grouped linears must contain the same "
                    "number of experts"
                )
            if self.w13.in_features != self.w2.out_features:
                raise ValueError("w13 input and w2 output dimensions must match")
            if self.w13.num_linears != 1:
                raise ValueError(
                    "Nemotron experts are ungated: w13 must hold one up projection"
                )
            if self.w13.out_features != self.w2.in_features:
                raise ValueError("w13 output and w2 input dimensions must match")
            if self.w2.num_linears != 1:
                raise ValueError("w2 must contain one down projection")
            if self.output_postprocess is not None:
                raise ValueError("Nemotron experts take no output_postprocess")

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        offsets_E = torch.cumsum(
            num_global_tokens_per_local_expert_e, dim=0, dtype=torch.int32
        )

        with maybe_set_sparse_mesh():
            remat.recompute_needs_tensor(routed_input_RD)
            up_RF = self.w13(routed_input_RD.bfloat16(), offsets_E)
            remat.recompute_needs_tensor(up_RF)
            # relu^2, matching NemotronHExperts' `relu2` activation.
            hidden_RF = torch.square(F.relu(up_RF))
            routed_output_RD = self.w2(hidden_RF, offsets_E)
            if routed_output_RD.dtype != routed_input_RD.dtype:
                remat.recompute_needs_tensor(routed_output_RD)
            routed_output_RD = routed_output_RD.type_as(routed_input_RD)
        return self.token_dispatcher.combine(routed_output_RD, metadata, x_TD)
