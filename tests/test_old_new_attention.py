import torch
import numpy as np
import math
from model import GPTConfig, GPT
from model_mla import MLAConfig, GPT_MLA

# Set a fixed seed for reproducibility
torch.manual_seed(1337)
np.random.seed(1337)

def test_models():
    """
    Test both the original GPT and the Multi-Latent Attention GPT models
    with random inputs of the same shape.
    """
    print("Testing GPT vs GPT-MLA models...")
    
    # Define small test configuration
    batch_size = 4
    seq_length = 16
    vocab_size = 1000
    n_layer = 2
    n_head = 4
    n_embd = 128
    n_latent = 32  # latent dimension for MLA
    
    # Configure and instantiate both models
    gpt_config = GPTConfig(
        block_size=seq_length,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=True
    )
    
    mla_config = MLAConfig(
        block_size=seq_length,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_latent=n_latent,
        dropout=0.0,
        bias=True
    )
    
    print("Initializing standard GPT model...")
    gpt_model = GPT(gpt_config)
    
    print("Initializing Multi-Latent Attention GPT model...")
    mla_model = GPT_MLA(mla_config)
    
    # Check parameter counts
    gpt_params = sum(p.numel() for p in gpt_model.parameters())
    mla_params = sum(p.numel() for p in mla_model.parameters())
    
    print(f"GPT parameter count: {gpt_params/1e6:.2f}M")
    print(f"MLA parameter count: {mla_params/1e6:.2f}M")
    print(f"Parameter reduction: {100 * (1 - mla_params/gpt_params):.2f}%")
    
    # Generate random inputs of the same shape
    x = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)
    
    # Forward pass with both models in evaluation mode
    gpt_model.eval()
    mla_model.eval()
    
    print("\nRunning forward pass on both models...")
    with torch.no_grad():
        # Forward with regular GPT
        gpt_logits, gpt_loss = gpt_model(x, y)
        
        # Forward with MLA GPT
        mla_logits, mla_loss = mla_model(x, y)
    
    print("\nOutput shapes:")
    print(f"GPT logits shape: {gpt_logits.shape}")
    print(f"MLA logits shape: {mla_logits.shape}")
    
    print("\nLoss values:")
    print(f"GPT loss: {gpt_loss.item():.4f}")
    print(f"MLA loss: {mla_loss.item():.4f}")
    
    # Test generation
    context = torch.zeros((1, 1), dtype=torch.long)
    
    print("\nGenerating from both models...")
    with torch.no_grad():
        gpt_output = gpt_model.generate(context, max_new_tokens=10)
        mla_output = mla_model.generate(context, max_new_tokens=10)
    
    print(f"GPT generation shape: {gpt_output.shape}")
    print(f"MLA generation shape: {mla_output.shape}")
    
    print("\nBoth models work as expected!")

def test_training():
    """
    Test that both models can be trained with gradient updates.
    """
    print("\n" + "-"*50)
    print("Testing training capability of both models...")
    print("-"*50)
    
    # Define small test configuration
    batch_size = 4
    seq_length = 16
    vocab_size = 1000
    n_layer = 2
    n_head = 4
    n_embd = 128
    n_latent = 32  # latent dimension for MLA
    learning_rate = 3e-4
    
    # Configure and instantiate both models
    gpt_config = GPTConfig(
        block_size=seq_length,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.1,  # add some dropout for training
        bias=True
    )
    
    mla_config = MLAConfig(
        block_size=seq_length,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_latent=n_latent,
        dropout=0.1,  # add some dropout for training
        bias=True
    )
    
    # Create models in training mode
    gpt_model = GPT(gpt_config)
    mla_model = GPT_MLA(mla_config)
    
    gpt_model.train()
    mla_model.train()
    
    # Set up optimizers
    gpt_optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=learning_rate)
    mla_optimizer = torch.optim.AdamW(mla_model.parameters(), lr=learning_rate)
    
    # Generate fixed random inputs for consistent testing
    x = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)
    
    print("\nTraining standard GPT model for 5 steps...")
    gpt_losses = []
    for step in range(5):
        # Forward pass
        gpt_logits, gpt_loss = gpt_model(x, y)
        
        # Backward pass and optimization
        gpt_optimizer.zero_grad()
        gpt_loss.backward()
        gpt_optimizer.step()
        
        gpt_losses.append(gpt_loss.item())
        print(f"Step {step+1}, Loss: {gpt_loss.item():.4f}")
    
    print("\nTraining Multi-Latent Attention GPT model for 5 steps...")
    mla_losses = []
    for step in range(5):
        # Forward pass
        mla_logits, mla_loss = mla_model(x, y)
        
        # Backward pass and optimization
        mla_optimizer.zero_grad()
        mla_loss.backward()
        mla_optimizer.step()
        
        mla_losses.append(mla_loss.item())
        print(f"Step {step+1}, Loss: {mla_loss.item():.4f}")
    
    # Verify that loss decreased for both models
    print("\nVerifying loss reduction:")
    gpt_loss_decreased = gpt_losses[0] > gpt_losses[-1]
    mla_loss_decreased = mla_losses[0] > mla_losses[-1]
    
    print(f"GPT loss decreased: {gpt_loss_decreased} (from {gpt_losses[0]:.4f} to {gpt_losses[-1]:.4f})")
    print(f"MLA loss decreased: {mla_loss_decreased} (from {mla_losses[0]:.4f} to {mla_losses[-1]:.4f})")
    
    if gpt_loss_decreased and mla_loss_decreased:
        print("\nBoth models successfully trained with gradient updates!")
    else:
        print("\nWarning: Loss did not decrease for at least one model.")

if __name__ == "__main__":
    test_models()
    test_training() 