# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.context_parallel import cp_shard, validate_cp_backend
from torchtitan.distributed.pipeline_parallel import pipeline_vlm


class TestValidateCpBackend(unittest.TestCase):
    """``validate_cp_backend`` gates CP on the backend that implements it."""

    @staticmethod
    def _parallelism(*, spmd_backend: str, cp: int) -> ParallelismConfig:
        return ParallelismConfig(spmd_backend=spmd_backend, context_parallel_degree=cp)

    def test_rejects_cp_on_partial_dtensor(self):
        with self.assertRaisesRegex(ValueError, "spmd_backend='spmd_types'"):
            validate_cp_backend(self._parallelism(spmd_backend="partial_dtensor", cp=2))

    def test_allows_cp_on_spmd_types(self):
        validate_cp_backend(self._parallelism(spmd_backend="spmd_types", cp=2))

    def test_allows_partial_dtensor_without_cp(self):
        # Only CP is gated; partial_dtensor stays valid for FSDP/TP/EP runs.
        validate_cp_backend(self._parallelism(spmd_backend="partial_dtensor", cp=1))


class TestContextParallelMaskSharding(unittest.TestCase):
    def test_mixed_mask_mapping_preserves_non_block_mask_metadata(self):
        input_T = torch.arange(8)
        input_shard_T = input_T[:4]
        block_mask = mock.Mock(spec=BlockMask)
        sharded_block_mask = mock.Mock(spec=BlockMask)
        varlen_metadata = mock.sentinel.varlen_metadata
        cp_context = mock.sentinel.cp_context
        attention_masks = {
            "quadratic_attention": block_mask,
            "deltanet": varlen_metadata,
            "deltanet_cp_context": cp_context,
        }
        cp_mesh = mock.Mock()
        cp_mesh.size.return_value = 2

        with mock.patch(
            "torchtitan.distributed.context_parallel.api._context_parallel_shard",
            side_effect=[[input_shard_T], [sharded_block_mask]],
        ):
            sharded_inputs, sharded_masks = cp_shard(
                cp_mesh,
                (input_T,),
                attention_masks,
                load_balancer_type=None,
            )

        self.assertIs(sharded_inputs[0], input_shard_T)
        assert isinstance(sharded_masks, dict)
        self.assertIs(sharded_masks["quadratic_attention"], sharded_block_mask)
        self.assertIs(sharded_masks["deltanet"], varlen_metadata)
        self.assertIs(sharded_masks["deltanet_cp_context"], cp_context)


class TestVlmPipelineInputModules(unittest.TestCase):
    def test_post_scatter_reshard_stays_with_token_embeddings(self):
        model = mock.Mock()
        model.decoder_input_reshard = mock.Mock()
        parallelism = ParallelismConfig(
            module_fqns_per_model_part=[
                ["vision_encoder", "tok_embeddings", "layers.0"],
                ["layers.1", "norm", "lm_head"],
            ]
        )
        expected = mock.sentinel.pipeline_result

        with mock.patch(
            "torchtitan.distributed.pipeline_parallel.pipeline_llm",
            return_value=expected,
        ) as pipeline_llm:
            result = pipeline_vlm(
                model,
                parallel_dims=mock.sentinel.parallel_dims,
                parallelism=parallelism,
                model_config=mock.sentinel.model_config,
            )

        self.assertIs(result, expected)
        stage_fqns = pipeline_llm.call_args.kwargs[
            "parallelism"
        ].module_fqns_per_model_part
        self.assertEqual(
            stage_fqns[0],
            [
                "vision_encoder",
                "tok_embeddings",
                "decoder_input_reshard",
                "layers.0",
            ],
        )


class TestDecoderConfigCpValidation(unittest.TestCase):
    """``Decoder.Config.update_from_config`` applies the CP gates at config time."""

    @staticmethod
    def _config(*, spmd_backend: str, cp: int, varlen: bool = False):
        from torchtitan.models.llama3.config_registry import (
            llama3_debugmodel,
            llama3_debugmodel_varlen_attn,
        )

        config = (llama3_debugmodel_varlen_attn if varlen else llama3_debugmodel)()
        config.parallelism.spmd_backend = spmd_backend
        config.parallelism.context_parallel_degree = cp
        config.training.max_context_length = 512
        return config

    def test_rejects_cp_on_partial_dtensor(self):
        config = self._config(spmd_backend="partial_dtensor", cp=2)
        with self.assertRaisesRegex(ValueError, "spmd_backend='spmd_types'"):
            config.model_spec.model.update_from_config(config=config)

    def test_allows_partial_dtensor_without_cp(self):
        config = self._config(spmd_backend="partial_dtensor", cp=1)
        config.model_spec.model.update_from_config(config=config)

    def test_allows_flex_cp_on_spmd_types(self):
        config = self._config(spmd_backend="spmd_types", cp=2)
        config.model_spec.model.update_from_config(config=config)

    def test_rejects_varlen_cp_on_spmd_types(self):
        # Only FlexAttention's BlockMask represents global key positions for CP.
        config = self._config(spmd_backend="spmd_types", cp=2, varlen=True)
        with self.assertRaisesRegex(NotImplementedError, "VarlenAttention"):
            config.model_spec.model.update_from_config(config=config)


class TestFluxConfigCpValidation(unittest.TestCase):
    """Flux is not a ``Decoder`` but applies the same backend gate."""

    @staticmethod
    def _config(*, spmd_backend: str, cp: int):
        pytest.importorskip(
            "torchtitan.models.flux.config_registry",
            reason="Flux requires optional image dependencies",
        )
        from torchtitan.models.flux.config_registry import flux_debugmodel

        config = flux_debugmodel()
        config.parallelism.spmd_backend = spmd_backend
        config.parallelism.context_parallel_degree = cp
        return config

    def test_rejects_cp_on_partial_dtensor(self):
        config = self._config(spmd_backend="partial_dtensor", cp=2)
        with self.assertRaisesRegex(ValueError, "spmd_backend='spmd_types'"):
            config.model_spec.model.update_from_config(config=config)

    def test_allows_partial_dtensor_without_cp(self):
        # Flux on partial_dtensor is FSDP-only but still a valid configuration.
        config = self._config(spmd_backend="partial_dtensor", cp=1)
        config.model_spec.model.update_from_config(config=config)


class TestQwen35ConfigCpValidation(unittest.TestCase):
    @staticmethod
    def _config():
        try:
            from torchtitan.models.qwen3_5.config_registry import qwen35_debugmodel
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(
                f"Qwen3.5 optional dependency unavailable: {exc.name}"
            ) from exc

        config = qwen35_debugmodel()
        config.parallelism.spmd_backend = "spmd_types"
        config.parallelism.context_parallel_degree = 2
        config.training.max_context_length = 512
        return config

    def test_rejects_context_parallel_load_balancing(self):
        config = self._config()
        config.parallelism.context_parallel_load_balancer = "headtail"
        with self.assertRaisesRegex(ValueError, "contiguous sequence shards"):
            config.model_spec.model.update_from_config(  # pyrefly: ignore[missing-attribute]
                config=config
            )

    def test_allows_contiguous_context_parallel_sharding(self):
        import spmd_types as spmd

        from torchtitan.distributed.parallel_dims import MeshAxisName
        from torchtitan.distributed.spmd_types import (
            _per_axis_types,
            spmd_validate_redistributions,
        )

        config = self._config()
        config.parallelism.context_parallel_load_balancer = None
        config.model_spec.model.update_from_config(  # pyrefly: ignore[missing-attribute]
            config=config
        )

        model_config = config.model_spec.model
        reshard_config = model_config.decoder_input_reshard.sharding_config
        assert reshard_config is not None
        assert reshard_config.in_src_shardings is not None
        assert reshard_config.in_dst_shardings is not None
        self.assertEqual(
            _per_axis_types(reshard_config.in_src_shardings["input"])[MeshAxisName.CP],
            spmd.R,
        )
        self.assertEqual(
            _per_axis_types(reshard_config.in_dst_shardings["input"])[MeshAxisName.CP],
            spmd.S(0),
        )
        spmd_validate_redistributions(reshard_config)
        first_layer_config = model_config.layers[0].sharding_config
        assert first_layer_config is not None
        spmd_validate_redistributions(first_layer_config)


if __name__ == "__main__":
    unittest.main()
