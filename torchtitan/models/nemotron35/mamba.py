# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mamba-2 mixer for Nemotron 3.5 (NemotronHMamba2Mixer), on the Module protocol.

Kernel helpers (fused mamba-ssm scan, fused causal-conv1d, PyTorch reference
scan) are carried over verbatim from the pre-merge port; only the module
wrapper changed so upstream ``Module._parallelize`` / ``init_states`` own
every parameter.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d
from torchtitan.protocols.module import Module

# --- Fused Mamba-2 scan (optional) ------------------------------------------
# mamba-ssm's Mamba-2 chunked scan is pure Triton, so it runs on both CUDA and
# ROCm (verified on MI355X / gfx950). It is dramatically cheaper than the
# PyTorch reference below -- the reference materializes the full
# (B, H, n_chunks, chunk, chunk) segment-sum matrices in fp32, which costs
# ~67 GiB and ~174 ms fwd+bwd for one 31B mamba layer at T=8192, versus
# ~1.1 GiB and ~3.3 ms for the kernel.
#
# The import is guarded because mamba-ssm is an optional dependency and its
# top-level __init__ eagerly imports `selective_scan_cuda`, the compiled
# Mamba-*1* extension, which is absent when the package is installed with
# MAMBA_SKIP_CUDA_BUILD=TRUE. We only need the Mamba-2 Triton path, so we
# stub that module out. The stub raises on attribute access rather than
# returning something usable, so if any Mamba-1 code path is ever reached it
# fails loudly instead of silently computing the wrong thing.
def _load_fused_mamba_scan():
    import os
    import sys
    import types

    # Escape hatch: forces the PyTorch reference path. Useful for A/B timing
    # and for bisecting a suspected kernel numerics problem.
    if os.environ.get("NEMOTRON_DISABLE_FUSED_MAMBA", "") not in ("", "0"):
        return None

    if "selective_scan_cuda" not in sys.modules:
        stub = types.ModuleType("selective_scan_cuda")

        # Dunder lookups must raise AttributeError, not RuntimeError: torch and
        # inspect walk sys.modules probing things like `__file__`, and a stub
        # that explodes on those breaks unrelated machinery (torch.library's
        # fake-kernel registration, specifically).
        stub.__file__ = "<nemotron35 selective_scan_cuda stub>"

        def _missing(name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            raise RuntimeError(
                "selective_scan_cuda (the Mamba-1 CUDA extension) is not built; "
                f"attribute {name!r} was requested. Nemotron-3 only uses the "
                "Mamba-2 Triton path, so reaching this is a bug."
            )

        stub.__getattr__ = _missing
        sys.modules["selective_scan_cuda"] = stub
        _installed_stub = True
    else:
        _installed_stub = False

    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        return mamba_chunk_scan_combined
    except Exception:
        # Roll the stub back so we do not mask a real mamba-ssm install later.
        if _installed_stub:
            sys.modules.pop("selective_scan_cuda", None)
        return None


_fused_mamba_chunk_scan = _load_fused_mamba_scan()
FUSED_MAMBA_SCAN_AVAILABLE = _fused_mamba_chunk_scan is not None


# --- Fused causal depthwise conv (optional) ----------------------------------
# causal-conv1d fuses the depthwise causal convolution and its SiLU into one
# kernel, replacing nn.Conv1d + slice + SiLU. It is the same kernel HF's
# NemotronH uses on its fully-fused path.
#
# Unlike mamba-ssm's Mamba-2 scan this is a compiled HIP/CUDA extension, not
# Triton, so it must be built against the installed torch. See this model's
# README for the ROCm build (upstream pins -std=c++17, which no longer compiles
# against torch >= 2.14).
def _load_fused_causal_conv():
    import os

    # Escape hatch, mirroring NEMOTRON_DISABLE_FUSED_MAMBA: forces the
    # nn.Conv1d reference path for A/B timing or bisecting numerics.
    if os.environ.get("NEMOTRON_DISABLE_FUSED_CONV", "") not in ("", "0"):
        return None

    try:
        from causal_conv1d import causal_conv1d_fn
    except ImportError:
        return None

    return causal_conv1d_fn


_fused_causal_conv = _load_fused_causal_conv()
FUSED_CAUSAL_CONV_AVAILABLE = _fused_causal_conv is not None


# --- Mamba-2 Pure PyTorch Chunk Scan Helpers ---

def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)
    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)

def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )

def segment_sum(input_tensor):
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum

def mamba2_chunk_scan(
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor | None = None,
):
    batch_size, sequence_length, num_heads, head_dim = hidden_states.shape
    num_groups = B.shape[2]

    hidden_states = hidden_states.float()
    B = B.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)
    C = C.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    # The D skip connection is a true residual on the *undiscretized* input, so it
    # must be captured before x is scaled by dt below (reference:
    # transformers Mamba2Mixer.torch_forward computes D_residual first).
    D_residual = None
    if D is not None:
        D_residual = D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

    # Discretize x and A
    hidden_states = hidden_states * dt[..., None].float()
    A = A.to(hidden_states.dtype) * dt.float()

    # Rearrange into blocks/chunks
    hidden_states, A, B, C = [reshape_into_chunks(tensor, pad_size, chunk_size) for tensor in (hidden_states, A, B, C)]

    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk
    L = torch.exp(segment_sum(A))
    G = (C[:, :, :, None, :, :] * B[:, :, None, :, :, :]).sum(dim=-1)
    M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
    Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

    # 2. Compute the state for each intra-chunk
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

    import torch.nn.functional as F
    
    # 3. Compute the inter-chunk SSM recurrence
    previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states = new_states[:, :-1]

    # 4. Compute output for inter-chunk
    state_decay_out = torch.exp(A_cumsum)
    C_times_states = C[..., None, :] * states[:, :, None, ...]
    Y_off = C_times_states.sum(-1) * state_decay_out.permute(0, 2, 3, 1)[..., None]

    # Add output of intra-chunk and inter-chunk
    Y = Y_diag + Y_off

    Y = Y.reshape(batch_size, -1, num_heads, head_dim)

    # Add the D residual (computed pre-discretization) while still padded, then trim.
    if D_residual is not None:
        Y = Y + D_residual.reshape(batch_size, -1, num_heads, head_dim)

    if pad_size > 0:
        Y = Y[:, :-pad_size, :, :]

    return Y


def mamba2_scan(
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor | None = None,
):
    """Mamba-2 chunked scan: fused Triton kernel when possible, else PyTorch.

    Both paths take the same arguments and return the same [B, T, H, P] tensor.
    ``dt`` must already have its bias, softplus, and clamp applied by the
    caller, which is why the kernel is invoked with ``dt_softplus=False`` and
    no ``dt_bias``/``dt_limit`` -- passing those here would apply them twice.

    The Triton kernel consumes the ``G`` group dimension of B/C natively, so
    the fused path skips the ``repeat_interleave`` group expansion that the
    PyTorch reference has to do.
    """
    if _fused_mamba_chunk_scan is not None and hidden_states.is_cuda:
        return _fused_mamba_chunk_scan(
            hidden_states,
            dt,
            A,
            B,
            C,
            chunk_size=chunk_size,
            D=D,
            dt_softplus=False,
        )
    return mamba2_chunk_scan(hidden_states, dt, A, B, C, chunk_size, D=D)



# --- Module-protocol wrappers -------------------------------------------------


def a_log_init(param: torch.Tensor) -> None:
    """A = -exp(A_log) = -(1..H). Position-dependent, so build the full vector
    and let distribute_tensor cut the matching shard under FSDP/DTensor."""
    with torch.no_grad():
        full = torch.log(
            torch.arange(1, param.shape[0] + 1, dtype=param.dtype, device=param.device)
        )
        if isinstance(param, DTensor):
            param.copy_(distribute_tensor(full, param.device_mesh, param.placements))
        else:
            param.copy_(full)


class MambaRMSNormGated(Module):
    """Group-wise gated RMSNorm (Zamba2RMSNormGated as used by Nemotron-H):
    gate first, then RMS-normalize per group of channels."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        group_size: int
        eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__()
        self.group_size = config.group_size
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        h = x.float() * F.silu(gate.float())
        *prefix, channels = h.shape
        h = h.reshape(*prefix, channels // self.group_size, self.group_size)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = h.reshape(*prefix, channels)
        return self.weight * h.to(input_dtype)


class Mamba2Mixer(Module):
    """NemotronHMamba2Mixer. Owns in_proj / conv1d / norm / out_proj as Module
    children and A_log / D / dt_bias as its own parameters (initialized from
    ``param_init`` and placed via ``sharding_config.state_shardings``)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int
        n_groups: int
        state_dim: int
        chunk_size: int = 128
        # Reference clamps dt to (time_step_min, inf); released configs use 0.001.
        dt_min: float = 0.001
        dt_max: float = float("inf")
        in_proj: Linear.Config
        conv1d: Conv1d.Config
        norm: MambaRMSNormGated.Config
        out_proj: Linear.Config

        def __post_init__(self) -> None:
            if self.num_heads % self.n_groups != 0:
                raise ValueError(
                    f"num_heads ({self.num_heads}) must be divisible by "
                    f"n_groups ({self.n_groups})"
                )
            inter = self.num_heads * self.head_dim
            conv_dim = inter + 2 * self.n_groups * self.state_dim
            if self.in_proj.out_features != inter + conv_dim + self.num_heads:
                raise ValueError("in_proj out_features != z + xBC + dt widths")
            if self.conv1d.in_channels != conv_dim:
                raise ValueError("conv1d channels must equal conv_dim")

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.n_groups = config.n_groups
        self.state_dim = config.state_dim
        self.chunk_size = config.chunk_size
        self.dt_min = config.dt_min
        self.dt_max = config.dt_max
        self.intermediate_size = config.num_heads * config.head_dim
        self.conv_dim = self.intermediate_size + 2 * config.n_groups * config.state_dim

        self.in_proj = config.in_proj.build()
        self.conv1d = config.conv1d.build()
        # Follow the model dtype (checkpoints are BF16); precision is recovered
        # at compute time via A_log.float(). Non-uniform dtypes break FSDP.
        self.dt_bias = nn.Parameter(torch.empty(config.num_heads))
        self.A_log = nn.Parameter(torch.empty(config.num_heads))
        self.D = nn.Parameter(torch.empty(config.num_heads))
        self.norm = config.norm.build()
        self.out_proj = config.out_proj.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flattened = x.dim() == 2
        if flattened:
            x = x.unsqueeze(0)
        B, L, _ = x.shape

        z, x_bc, dt = torch.split(
            self.in_proj(x),
            [self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )

        x_bc = x_bc.transpose(1, 2)
        if _fused_causal_conv is not None and x_bc.is_cuda:
            x_bc = _fused_causal_conv(
                x_bc, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation="silu"
            ).transpose(1, 2)
        else:
            x_bc = F.silu(self.conv1d(x_bc)[:, :, :L].transpose(1, 2))

        gs = self.n_groups * self.state_dim
        x_m, B_p, C_p = torch.split(x_bc, [self.intermediate_size, gs, gs], dim=-1)
        B_p = B_p.reshape(B, L, self.n_groups, self.state_dim)
        C_p = C_p.reshape(B, L, self.n_groups, self.state_dim)

        dt = F.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, min=self.dt_min, max=self.dt_max)
        A = -torch.exp(self.A_log.float())

        y = mamba2_scan(
            x_m.reshape(B, L, self.num_heads, self.head_dim),
            dt, A, B_p, C_p, self.chunk_size, D=self.D,
        )
        y = y.reshape(B, L, -1).to(x_bc.dtype)
        out = self.out_proj(self.norm(y, z))
        return out.squeeze(0) if flattened else out
