"""
Quick test script to compare standard GPT and Multi-Latent Attention (MLA) GPT 
on the Shakespeare character dataset, with proper decoding and more training iterations.
"""

import os
import time
import pickle
import numpy as np
import torch
from contextlib import nullcontext

from model import GPTConfig, GPT
from model_mla import MLAConfig, GPT_MLA

# This got annoying
import warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")


# -----------------------------------------------------------------------------
# Hyperparameters for both models - match the README configuration
# -----------------------------------------------------------------------------
batch_size = 64
block_size = 256
max_iters = 5000 #1500 # Reduced to 1500 iterations to finish faster
eval_interval = 250
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# MLA specific
n_latent = 192  # latent dimension (half of embedding dimension)
# -----------------------------------------------------------------------------

torch.manual_seed(1337)

def prepare_data():
    """Load the Shakespeare character dataset and its metadata"""
    data_dir = os.path.join('data', 'shakespeare_char')
    train_data_path = os.path.join(data_dir, 'train.bin')
    val_data_path = os.path.join(data_dir, 'val.bin')
    meta_path = os.path.join(data_dir, 'meta.pkl')
    
    # Check if the data exists
    if not (os.path.exists(train_data_path) and os.path.exists(val_data_path)):
        print("Shakespeare dataset not found, preparing it now...")
        os.system("python data/shakespeare_char/prepare.py")
    
    # Load the meta data (character mappings)
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        vocab_size = meta['vocab_size']
        print(f"Loaded meta with vocab_size = {vocab_size}")
    else:
        print("Meta file not found! This will lead to incorrect text generation.")
        vocab_size = 65  # Fallback
        itos = None
    
    # Load the data
    train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')  # Use uint16 to match prepare.py
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    print(f"Train data: {len(train_data)} tokens")
    print(f"Val data: {len(val_data)} tokens")
    
    # Convert to torch Tensors
    train_data = torch.from_numpy(train_data.astype(np.int64))
    val_data = torch.from_numpy(val_data.astype(np.int64))
    
    return train_data, val_data, vocab_size, itos

def get_batch(split, train_data, val_data):
    """Get a batch of data"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Estimate the loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(model, train_data, val_data, model_type="GPT", itos=None):
    """Train the model and report progress and loss"""
    print(f"Training {model_type} model for {max_iters} iterations...")
    
    # Set up the optimizer
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, 
                                          betas=(0.9, 0.99), device_type=device.split(':')[0])
    
    # Training loop
    model.train()
    t0 = time.time()
    
    for iter_num in range(max_iters + 1):
        # Every once in a while evaluate the loss on train and val sets
        if iter_num % eval_interval == 0 or iter_num == max_iters:
            losses = estimate_loss(model, train_data, val_data)
            print(f"{model_type} step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Generate sample text every 500 iterations
        if iter_num % 1000 == 0 and iter_num > 0:
            print(f"\n--- {model_type} sample at iteration {iter_num} ---")
            # Temporarily switch to evaluation mode
            model.eval()
            with torch.no_grad():
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated_tokens = model.generate(context, max_new_tokens=100)[0].tolist()
                
                # Decode using the proper mapping from the meta file
                if itos is not None:
                    generated_text = ''.join([itos[i] for i in generated_tokens])
                else:
                    # Fallback mapping (may not be correct)
                    chars = " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                    generated_text = ''.join([chars[i % len(chars)] for i in generated_tokens])
                
                print(generated_text)
                print("---")
            # Switch back to training mode
            model.train()
        
        # Sample a batch of data
        xb, yb = get_batch('train', train_data, val_data)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass and optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Final evaluation
    losses = estimate_loss(model, train_data, val_data)
    elapsed = time.time() - t0
    print(f"{model_type} final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time: {elapsed:.2f}s")
    
    return losses

def sample_text(model, itos, model_type="GPT"):
    """Generate sample text from the model using the correct character mapping"""
    print(f"\n{model_type} model generates:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
    
    # Decode using the proper mapping from the meta file
    if itos is not None:
        generated_text = ''.join([itos[i] for i in generated_tokens])
    else:
        # Fallback mapping (may not be correct)
        chars = " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        generated_text = ''.join([chars[i % len(chars)] for i in generated_tokens])
    
    print(generated_text)
    print("-" * 80)


def main():
    """Main function to test both models"""
    print("Testing Standard GPT vs Multi-Latent Attention GPT on Shakespeare character dataset")
    print("-" * 80)
    
    # Prepare data
    train_data, val_data, vocab_size, itos = prepare_data()
    
    # Create standard GPT model
    gpt_config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=True
    )
    gpt_model = GPT(gpt_config)
    gpt_model.to(device)
    
    # Create MLA model
    mla_config = MLAConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_latent=n_latent,
        dropout=dropout,
        bias=True
    )
    mla_model = GPT_MLA(mla_config)
    mla_model.to(device)
    
    # Report model sizes
    gpt_params = sum(p.numel() for p in gpt_model.parameters())
    mla_params = sum(p.numel() for p in mla_model.parameters())
    
    print(f"GPT model size: {gpt_params/1e6:.2f}M parameters")
    print(f"MLA model size: {mla_params/1e6:.2f}M parameters")
    print(f"Parameter reduction with MLA: {100 * (1 - mla_params/gpt_params):.2f}%")
    print("-" * 80)
    
    # Train both models
    gpt_losses = train_model(gpt_model, train_data, val_data, "GPT", itos)
    mla_losses = train_model(mla_model, train_data, val_data, "MLA", itos)
    
    # Generate sample text
    sample_text(gpt_model, itos, "GPT")
    sample_text(mla_model, itos, "MLA")
    
    # Compare results
    print("\nFinal comparison:")
    print(f"GPT validation loss: {gpt_losses['val']:.4f}")
    print(f"MLA validation loss: {mla_losses['val']:.4f}")
    
    if mla_losses['val'] < gpt_losses['val']:
        print("MLA model performed better!")
    elif mla_losses['val'] > gpt_losses['val']:
        print("Standard GPT model performed better.")
    else:
        print("Both models performed equally.")

if __name__ == "__main__":
    main() 