# Gemma-4 Model Support in TorchTitan

This directory contains the implementation of Google DeepMind's **Gemma-4** model family for distributed training with TorchTitan.

## Overview

Gemma-4 is a state-of-the-art multimodal language model family featuring:

- **Hybrid Attention Architecture**: Combines sliding-window local attention with occasional global attention (final layer) for efficient long-context modeling
- **Multiple Sizes**: 2B, 4B, 12B, 26B (MoE), and 31B variants
- **Long Context Support**: Up to 256K token context length
- **Efficient Memory**: KV cache optimizations and 5:1 local-to-global attention ratio
- **Grouped-Query Attention (GQA)**: Reduces KV cache memory and improves inference throughput

## Model Architecture

### Gemma-4 12B (Primary Implementation)

```
Dimension:           3584
Attention Heads:     28
KV Heads:            7 (4:1 GQA)
Layers:              42
Vocabulary:          262,144 tokens
Context Length:      256K tokens
Sliding Window:      4,096 tokens (local attention)
```

### Hybrid Attention

The model uses a 5:1 ratio of local to global attention:
- **Layers 0-40**: Sliding-window attention (4K window) - efficient local context processing
- **Layer 41 (final)**: Global attention - allows each token to attend to full context

This design balances efficiency with expressiveness, reducing attention complexity from O(n²) to approximately O(5n).

## Quick Start

### Prerequisites

```bash
# Install TorchTitan from this fork
git clone https://github.com/rpaulk/torchtitan.git
cd torchtitan
git checkout feat/gemma4-support
pip install -r requirements.txt
```

### Download Tokenizer

Gemma-4 uses a standard tokenizer compatible with HuggingFace Transformers:

```bash
# Get HF token from https://huggingface.co/settings/tokens
python scripts/download_hf_assets.py --repo_id google/gemma-4-12b --assets tokenizer --hf_token <YOUR_HF_TOKEN>
```

### Training on Single Node (8 GPUs)

```bash
MODULE=gemma4 CONFIG=gemma4_12b ./run_train.sh
```

### Multi-Node Training

See `multinode_trainer.slurm` for SLURM cluster configuration. Adjust parameters as needed:

```bash
#SBATCH --ntasks=<NUM_NODES>
#SBATCH --nodes=<NUM_NODES>
srun torchrun --nnodes <NUM_NODES> --nproc_per_node 8 ...
```

## Training Configurations

Available configs in `torchtitan_recipes/gemma4/config_registry.py`:

| Config | Description | Recommended Hardware |
|--------|-------------|----------------------|
| `gemma4_debugmodel` | Small debug model (256D, 6L) | 1 GPU |
| `gemma4_12b` | Full 12B model | 8x H100/A100 GPUs |
| `gemma4_12b_1node_full` | Full training (1 node) | 8x H100 GPUs |
| `gemma4_12b_multinode` | Distributed training | 32+ H100 GPUs |

## State Dict Adapter

The `Gemma4StateDictAdapter` enables seamless checkpoint conversion:

### Loading from HuggingFace

```python
from transformers import AutoModelForCausalLM
from torchtitan.models.gemma4 import Gemma4StateDictAdapter, model_registry

# Load HF model
hf_model = AutoModelForCausalLM.from_pretrained("google/gemma-4-12b")

# Get TorchTitan model
spec = model_registry("12b")
tt_model = spec.model.build()

# Adapt checkpoint
adapter = Gemma4StateDictAdapter(spec.model, hf_assets_path="./assets")
tt_state = adapter.from_hf(hf_model.state_dict())
tt_model.load_state_dict(tt_state)
```

### Converting to HuggingFace

```python
# After training, convert back to HF format
hf_state = adapter.to_hf(tt_model.state_dict())
hf_model.load_state_dict(hf_state)
hf_model.push_to_hub("my-org/gemma-4-finetuned")
```

## Distributed Training Features

### Parallelism Support

- **Data Parallel (DP)**: Replicates model across GPUs, shards data
- **Tensor Parallel (TP)**: Shards model parameters across devices
- **Sequence Parallel (SP)**: Shards sequences across devices for memory efficiency
- **Pipeline Parallel (PP)**: Stages layers across devices (optional)
- **Context Parallel (CP)**: Shards context length for ultra-long sequences

### Example Configurations

**Single Node (8 GPUs)**
```bash
# DP only
DP=8 TP=1 ./run_train.sh

# DP + TP
DP=4 TP=2 ./run_train.sh

# DP + TP + SP
DP=2 TP=2 SP=1 ./run_train.sh
```

**Multi-Node (16 GPUs = 2 nodes)**
```bash
# DP=2, TP=4 per node
DP=2 TP=4 ./run_train.sh
```

### Activation Checkpointing

Reduce memory by checkpointing activations:

```python
# In config: enable per-layer checkpointing
ac_config = ActivationCheckpointingConfig(
    mode="selective",  # or "full"
)
```

### Torch Compile

Enable torch.compile for speedups on H100+ GPUs:

```python
compile_config = CompileConfig(
    enable=True,
    components=["model"],
)
```

## Attention Mechanisms

### Sliding Window Attention (Layers 0-40)

Each token attends to:
- All previous tokens up to 4K window
- Itself (current position)
- No future tokens (causal)

**Complexity**: O(4K * seq_len) instead of O(seq_len²)

### Global Attention (Layer 41)

Final layer allows each token to attend to all previous tokens and the current position, enabling:
- Long-range dependencies
- Coherent long-context reasoning
- Better loss convergence

## Performance Tuning

### Memory Optimization

1. **Flash Attention**: Automatically used for H100+ (see `attention_backend="flex"`)
2. **Grouped-Query Attention**: Reduces KV cache by 4x vs Multi-Head Attention
3. **KV Cache Reuse**: Gemma-4's architecture enables KV cache sharing across layers
4. **Activation Checkpointing**: Trades compute for memory

### Throughput Optimization

1. **Tensor Parallelism**: Reduce per-GPU batch size, maintain global throughput
2. **Sequence Parallelism**: Shard sequences for ultra-long contexts
3. **Fused QKV**: Single linear projection reduces kernel launches
4. **torch.compile**: JIT compile model for latency reduction

## Benchmarks

Expected performance on H100 GPUs (8x interconnect, no pipeline parallelism):

| Config | Throughput (tokens/sec) | MFU | Memory |
|--------|------------------------|-----|--------|
| DP=8, TP=1 | ~3,500 | ~45% | ~72GB |
| DP=4, TP=2 | ~4,200 | ~52% | ~68GB |
| DP=2, TP=2, SP=2 | ~4,800 | ~58% | ~48GB |

*Note: Benchmarks based on internal runs; actual results depend on batch size, sequence length, and hardware configuration.*

## Checkpoint Conversion Scripts

Scripts for converting between formats are in `scripts/checkpoint_conversion/`:

```bash
# HF to DCP (TorchTitan Distributed Checkpointing)
python scripts/checkpoint_conversion/hf_to_dcp.py \
  --hf_model_path google/gemma-4-12b \
  --output_path ./gemma4_12b_dcp

# DCP to HF
python scripts/checkpoint_conversion/dcp_to_hf.py \
  --dcp_path ./gemma4_12b_dcp \
  --output_path ./gemma4_12b_hf
```

## Testing

### Unit Tests

```bash
pytest tests/unit_tests/ -k gemma4 -v
```

### Integration Tests

```bash
python -m tests.integration_tests.run_tests output_dir \
  --test_suite features \
  --test_name "*gemma4*" \
  --ngpu 8
```

### Numerics Verification

Compare loss curves between TorchTitan and HuggingFace:

```bash
python scripts/numerics_test.py \
  --torchtitan_model ./gemma4_tt \
  --hf_model google/gemma-4-12b \
  --num_steps 100
```

## Known Limitations

1. **Multimodal Support**: Current implementation supports text-only training. Vision/audio integration planned for future releases.
2. **MoE Variants**: 26B A4B (MoE) not yet implemented; planned after text variant is stable.
3. **Speculative Decoding**: Multi-token drafting available for inference but not yet optimized for training.

## Contributing

To add Gemma-4 MoE or other variants:

1. Add model config to `gemma4_configs` dict in `__init__.py`
2. Implement MoE layers following Qwen3-style patterns
3. Add tests to `tests/integration_tests/models.py`
4. Create PR against `feat/gemma4-support` branch

## References

- [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770)
- [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [HuggingFace Gemma-4 Docs](https://huggingface.co/docs/transformers/main/en/model_doc/gemma4)
- [TorchTitan Documentation](https://github.com/pytorch/torchtitan/docs)

## License

This implementation is provided under the BSD 3-Clause License (same as TorchTitan).
Gemma-4 model weights are available under Google's Gemma License Agreement.
