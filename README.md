# nanoGPT-MLA

## Last Update: March 21, 2024
- Added induction head detection:
  - Created `find_induction_heads.py` for analyzing attention patterns
  - Implemented hooks to capture attention patterns in both GPT-2 and MLA models
  - Found induction heads in both architectures, with MLA showing more layer-wise specialization?

![GPT-2 vs MLA Induction Heads](assets/induction_heads.png)

The heatmaps above show the induction scores for each attention head in both architectures. While GPT-2's induction heads are distributed across 3 layers, MLA shows a more concentrated pattern with stronger induction behavior in layer 8: a different organization of capabilities? Using the common 0.4 threshold, we detect:

```
GPT-2 Induction Heads:
Layer 5, Head 0: 0.453
Layer 5, Head 1: 0.891
Layer 5, Head 5: 0.922
Layer 6, Head 9: 0.910
Layer 7, Head 2: 0.834
Layer 7, Head 10: 0.910
Layer 9, Head 6: 0.484
Layer 9, Head 9: 0.490
Layer 10, Head 0: 0.432
Layer 10, Head 1: 0.490

MLA Induction Heads:
Layer 6, Head 10: 0.849
Layer 8, Head 1: 0.927
Layer 8, Head 2: 0.482
Layer 8, Head 5: 0.746
Layer 8, Head 6: 0.853
Layer 8, Head 9: 0.491
Layer 8, Head 10: 0.927
Layer 11, Head 2: 0.483
```

---

- Fixed sampling scripts to work with DDP-trained models:
  - Updated both `sample.py` and `sample_mla.py` to properly handle state dictionaries from models trained with DistributedDataParallel
  - Added automatic removal of 'module.' prefix from DDP-trained model weights



![Full Run](assets/full_run.png)

---

![Multi-Latent Attention](assets/mla_diagram.jpg)

This project is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT) that implements and explores Multi-Latent Attention (MLA) as an alternative to standard Multi-Head Attention (MHA) in transformer architectures.

The primary objective of this project is to explore and understand the mechanistic differences between Multi-Head Attention and Multi-Latent Attention in transformer models.

## What to know

- Drop-in replacement for standard attention using Multi-Latent Attention
- Maintains compatibility with the original nanoGPT training pipeline
- Supports both character-level (Shakespeare) and BPE token (GPT-2) training
- Adding testing infrastructure for comparing MHA and MLA models

## Installation

Same dependencies as the original nanoGPT:

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## Quick Start: Shakespeare Character-Level Model

![Multi-Latent Attention](assets/comparison.png)

To quickly test the implementation and compare MHA vs MLA models:

```bash
# First, prepare the Shakespeare dataset
python data/shakespeare_char/prepare.py

# Run the comparison test (trains both models and generates samples)
python test_shakespeare_quick.py
```

This will train both a standard GPT model and an MLA variant on the Shakespeare dataset, allowing you to compare:
- Training dynamics
- Parameter counts
- Generated text samples
- Final validation loss

You should see a train loss of ~1.22 after 2000 steps.

## Training GPT-2 Scale Models

### 1. Prepare the OpenWebText Dataset

```bash
python data/openwebtext/prepare.py
```

This downloads and tokenizes the OpenWebText dataset using GPT-2's BPE tokenizer.

### 2. Train the Model

For single GPU:
```bash
python train_mla.py config/train_gpt2_mla.py
```

For multiple GPUs on one node:
```bash
torchrun --standalone --nproc_per_node=8 train_mla.py config/train_gpt2_mla.py
```

For multiple nodes:
```bash
# On master node
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train_mla.py

# On worker node
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train_mla.py
```

I **have not** tested this yet.

---

Useful context:


The Multi-Latent Attention mechanism modifies the standard transformer attention by:
1. Introducing a shared latent encoder across attention heads (for keys and values ONLY)
2. Reducing the dimensionality of key/query computations
3. Maintaining value projections in the original embedding space

Key differences from standard GPT-2 (code changes):
- Addition of `n_latent` parameter (default: half of `n_embd`)
- Modified attention computation using latent representations
- Shared latent encoder across heads

As for usage:

1. The training script (`train_mla.py`) mirrors the original nanoGPT's `train.py` exactly, just using our MLA model instead.
2. All original nanoGPT features are supported (I THINK?):
   - Distributed training
   - Mixed precision training
   - Gradient accumulation
   - Learning rate scheduling
   - Checkpointing
   - WandB logging


## License

TODO

---

### Changelog

#### March 20, 2024
- Added sampling infrastructure:
  - Created `sample_mla.py`