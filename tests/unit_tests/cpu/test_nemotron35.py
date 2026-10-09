# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter

import pytest
import torch

pytest.importorskip("attn_gym")

from torchtitan.models.nemotron35 import (
    build_model_config,
    MODEL_FLAVORS,
    Nemotron35Model,
)
from torchtitan.models.nemotron35.flavors import parse_hybrid_pattern
from torchtitan.models.nemotron35.model import NemotronBlock
from torchtitan.models.nemotron35.state_dict_adapter import NemotronStateDictAdapter


def _debug_model(seq_len: int = 64):
    config = build_model_config("debugmodel", seq_len=seq_len)
    torch.manual_seed(0)
    model = config.build()
    model.init_states()
    return config, model


def _forward(model, config, seq_len: int = 64) -> torch.Tensor:
    tokens = torch.randint(0, config.vocab_size, (seq_len,))
    positions = torch.arange(seq_len)
    metadata = model._get_attention_metadata(
        positions=positions,
        padding_mask=None,
        max_num_documents=None,
        max_context_length=seq_len,
    )
    with torch.no_grad():
        return model(tokens, positions, metadata)


def test_registry_exposes_flavors() -> None:
    assert set(MODEL_FLAVORS) == {"debugmodel", "lightning"}
    assert Nemotron35Model.state_dict_adapter_cls is NemotronStateDictAdapter


def test_lightning_matches_hugging_face_config() -> None:
    config = build_model_config("lightning", seq_len=4096)
    assert config.dim == 2688
    assert config.vocab_size == 131072
    assert len(config.layers) == 52
    kinds = Counter(layer.block_type for layer in config.layers)
    assert kinds == {"mamba": 23, "moe": 23, "attention": 6}

    attn = next(layer.attention for layer in config.layers if layer.attention)
    assert (attn.n_heads, attn.n_kv_heads, attn.head_dim) == (32, 2, 128)
    assert attn.rope is None  # NoPE

    moe = next(layer.moe for layer in config.layers if layer.moe)
    assert moe.num_experts == 128
    assert moe.router.top_k == 6
    assert moe.routed_experts.w13.num_linears == 1  # ungated relu^2 experts


def test_lightning_to_hf_key_count_matches_checkpoint_trunk() -> None:
    # The released checkpoint has 6243 trunk tensors (+270 mtp.* not built).
    config = build_model_config("lightning", seq_len=4096)
    with torch.device("meta"):
        model = config.build()
    hf = NemotronStateDictAdapter(config, None).to_hf(dict(model.state_dict()))
    assert len(hf) == 6243
    assert not any(k.startswith("mtp.") for k in hf)


def test_parse_hybrid_pattern_rejects_unknown_chars() -> None:
    assert parse_hybrid_pattern("ME*-") == ["mamba", "moe", "attention", "mlp"]
    with pytest.raises(ValueError, match="Unknown hybrid_override_pattern"):
        parse_hybrid_pattern("MX")


def test_block_config_requires_exactly_matching_mixer() -> None:
    config = build_model_config("debugmodel", seq_len=64)
    mamba_layer = config.layers[0]
    assert mamba_layer.block_type == "mamba"
    with pytest.raises(ValueError, match="requires exactly"):
        NemotronBlock.Config(
            block_type="attention", norm=mamba_layer.norm, mamba=mamba_layer.mamba
        )
    with pytest.raises(ValueError, match="unknown block_type"):
        NemotronBlock.Config(block_type="ssm", norm=mamba_layer.norm)


def test_one_norm_and_one_mixer_per_layer() -> None:
    config, model = _debug_model()
    mixers = {"mamba", "attention", "feed_forward", "moe"}
    for layer_id, layer in model.layers.items():
        present = mixers & {name for name, _ in layer.named_children()}
        assert len(present) == 1, (layer_id, present)
        assert hasattr(layer, "norm")


def test_debug_forward_is_finite() -> None:
    config, model = _debug_model()
    out = _forward(model, config)
    assert out.shape == (64, config.vocab_size)
    assert torch.isfinite(out).all()


def test_state_dict_adapter_round_trip_is_exact() -> None:
    config, model = _debug_model()
    state_dict = dict(model.state_dict())
    hf = NemotronStateDictAdapter(config, None).to_hf(state_dict)
    assert all(k.startswith(("backbone.", "lm_head.")) for k in hf)
    back = NemotronStateDictAdapter(config, None).from_hf(hf)
    assert set(back) == set(state_dict)
    for key, value in state_dict.items():
        assert torch.equal(back[key], value), key

    reloaded = config.build()
    reloaded.load_state_dict(back, strict=True)
    torch.manual_seed(1)
    expected = _forward(model, config)
    torch.manual_seed(1)
    assert torch.equal(_forward(reloaded, config), expected)


def test_sharded_fused_qkv_round_trip_keeps_values_and_layout(tmp_path) -> None:
    # Regression: to_hf used to reshape a sharded wqkv DTensor to
    # (n_kv, R, head_dim, dim). n_kv=2 is not divisible by the shard count, so
    # under TP/FSDP that reshape raised or redistributed rows wrongly (GPU
    # step-1 loss 3.96 instead of ~2.9). The adapter must split the full
    # tensor and re-shard the fused result with the original layout.
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, DTensor, Shard

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,))
        config, model = _debug_model()
        state_dict = dict(model.state_dict())
        qkv_keys = [k for k in state_dict if k.endswith("qkv_linear.wqkv.weight")]
        assert qkv_keys
        sharded = {
            k: distribute_tensor(v, mesh, [Shard(0)]) if k in qkv_keys else v
            for k, v in state_dict.items()
        }
        adapter = NemotronStateDictAdapter(config, None)
        hf = adapter.to_hf(sharded)
        assert not any(isinstance(hf[k], DTensor) for k in hf if "_proj" in k)
        back = adapter.from_hf(hf)
        for key in qkv_keys:
            assert isinstance(back[key], DTensor), key
            assert back[key].placements == (Shard(0),), key
            assert torch.equal(back[key].full_tensor(), state_dict[key]), key
    finally:
        dist.destroy_process_group()


def test_from_hf_refuses_unmapped_and_incomplete_experts() -> None:
    config, model = _debug_model()
    hf = NemotronStateDictAdapter(config, None).to_hf(dict(model.state_dict()))

    with pytest.raises(ValueError, match="no torchtitan mapping"):
        NemotronStateDictAdapter(config, None).from_hf(
            {**hf, "backbone.layers.0.mixer.bogus.weight": torch.zeros(1)}
        )

    missing_expert = {
        k: v for k, v in hf.items() if k != "backbone.layers.1.mixer.experts.3.up_proj.weight"
    }
    with pytest.raises(ValueError):
        NemotronStateDictAdapter(config, None).from_hf(missing_expert)


def test_mamba_scan_resets_state_at_packed_document_boundaries():
    """A packed row must equal its documents scanned separately (no state leak)."""
    import torch
    from torchtitan.models.nemotron35 import mamba as M

    torch.manual_seed(0)
    T1, T2, H, P, N, chunk = 37, 50, 4, 8, 16, 16
    T = T1 + T2
    x = torch.randn(1, T, H, P)
    dt = torch.rand(1, T, H) * 0.1 + 0.01
    A = -torch.rand(H)
    B, C = torch.randn(1, T, 1, N), torch.randn(1, T, 1, N)
    D = torch.randn(H)
    positions = torch.cat([torch.arange(T1), torch.arange(T2)])
    seq_idx = M.seq_idx_from_positions(positions, 1, T)
    assert seq_idx.tolist() == [[0] * T1 + [1] * T2]

    packed = M.mamba2_scan(x, dt, A, B, C, chunk, D=D, seq_idx=seq_idx)
    separate = torch.cat(
        [
            M.mamba2_scan(x[:, s], dt[:, s], A, B[:, s], C[:, s], chunk, D=D)
            for s in (slice(0, T1), slice(T1, T))
        ],
        dim=1,
    )
    torch.testing.assert_close(packed, separate)
    leaked = M.mamba2_scan(x, dt, A, B, C, chunk, D=D)
    assert not torch.allclose(leaked[:, T1:], separate[:, T1:])
