# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified Nemotron 3.5 Lightning model recipes."""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.nemotron35 import build_model_config
from torchtitan.trainer import Trainer


def nemotron35_lightning(seq_len: int | None = None) -> Trainer.Config:
    """NVIDIA-Nemotron-3.5-Lightning-30B-A3B on one 8-GPU node.

    TP=2 x EP=4 with DP replicate=2 x shard=2: EP must divide
    dp_shard * cp * tp, and the product of all degrees must equal 8.
    """
    model_config = build_model_config("lightning", seq_len=seq_len or 4096)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=3e-5)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=50,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=2,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            expert_parallel_degree=4,
        ),
        checkpointer=CheckpointManager.Config(
            initial_load_in_hf=True,
            interval=500,
            last_save_model_only=True,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def nemotron35_lightning_sft_smoke(seq_len: int | None = None) -> Trainer.Config:
    """Short SFT smoke run on real agentic rows through the per-user-turn split.

    Uses one converted dataset and the Nemotron 3.5 renderer; the full mixed
    recipe comes later. Packing keeps document boundaries (positions == 0),
    which attention and Mamba (seq_idx) both respect.
    """
    from renderers.configs import Nemotron35RendererConfig

    from torchtitan.components.data import FirstFitPackingConfig
    from torchtitan.components.data.dataset import SingleDatasetConfig
    from torchtitan.components.data.sources import IndexedJsonlSource
    from torchtitan.components.renderer import from_renderers
    from torchtitan.hf_datasets.text_datasets import ChatProcessor

    config = nemotron35_lightning(seq_len=seq_len or 32768)
    config.dataloader = GrainDataLoader.Config(
        dataset=FirstFitPackingConfig(
            dataset=SingleDatasetConfig(
                source=IndexedJsonlSource.Config(
                    patterns=(
                        "/mnt/powerscale/data/datasets/nemotron35-lightning-sft/"
                        "converted/agentic_search.jsonl",
                    )
                ),
                processor=ChatProcessor.Config(
                    messages_fn=lambda row: row["messages"],
                    tools_fn=lambda row: row.get("tools") or None,
                    renderer=from_renderers(Nemotron35RendererConfig()),
                ),
                post_filters=(lambda sample: sample is not None,),
            )
        ),
    )
    config.optim.optimizer.optimizers[0].lr = 1e-5
    config.optim.lr_scheduler.warmup_steps = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = (
        config.model.max_context_length
    )
    config.training.steps = 10
    config.checkpointer.interval = 10_000
    config.dump_folder = "/data/nemotron35_sft_smoke"
    return config
