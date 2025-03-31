"""
Test script for the k-ahead GPT model.
This script validates the model's behavior by:
1. Creating a model instance
2. Running a forward pass with random data
3. Analyzing the output logits and losses
"""

import math
import torch
from model_k_ahead import GPTConfig, GPT, K_AHEAD

def print_tensor_stats(tensor, name):
    """Helper function to print statistics about a tensor"""
    print(f"\n{name} statistics:")
    print(f"Shape: {tensor.shape}")
    print(f"Mean: {tensor.mean().item():.4f}")
    print(f"Std: {tensor.std().item():.4f}")
    print(f"Min: {tensor.min().item():.4f}")
    print(f"Max: {tensor.max().item():.4f}")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(1337)
    
    # Model configuration
    config = GPTConfig(
        block_size=128,  # Smaller block size for testing
        vocab_size=50257,  # Standard GPT-2 vocab size
        n_layer=2,       # Smaller model for testing
        n_head=4,
        n_embd=128,
        dropout=0.0      # No dropout for testing
    )
    
    # Create model
    print(f"\nInitializing model with {K_AHEAD} ahead prediction...")
    model = GPT(config)
    
    # Generate random batch
    batch_size = 4
    seq_length = 64
    
    # Random input indices
    x = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    # Random target indices - need seq_length targets for each k-ahead prediction
    y = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    print(f"\nInput shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()  # Set to eval mode to disable dropout
    with torch.no_grad():
        logits, loss = model(x, y)
    
    # Analyze logits
    print("\nAnalyzing output logits...")
    print(f"Logits shape: {logits.shape}")  # Should be (batch_size, K_AHEAD, seq_length, vocab_size)
    
    # Print statistics for each head's logits
    for k in range(K_AHEAD):
        print_tensor_stats(logits[:, k], f"Head {k} logits")
        
        # Compute softmax of logits
        probs = torch.nn.functional.softmax(logits[:, k], dim=-1)
        print(f"Head {k} probability statistics:")
        print(f"Mean probability: {probs.mean().item():.6f}")  # Should be close to 1/vocab_size
        print(f"Expected for uniform: {1/config.vocab_size:.6f}")
    
    # Compute and analyze individual losses
    print("\nAnalyzing losses...")
    individual_losses = []
    for k in range(K_AHEAD):
        # Get logits and targets for k-step ahead prediction
        k_logits = logits[:, k]  # Shape: (batch_size, seq_length, vocab_size)
        
        # Compute loss
        k_loss = torch.nn.functional.cross_entropy(
            k_logits.reshape(-1, k_logits.size(-1)),
            y.reshape(-1)
        )
        individual_losses.append(k_loss.item())
        print(f"Loss for {k}-ahead prediction: {k_loss.item():.4f}")
    
    print(f"\nFinal (average) loss: {loss.item():.4f}")
    
    # Theoretical analysis
    theoretical_uniform_loss = -math.log(1/config.vocab_size)
    print(f"\nTheoretical loss for uniform distribution: {theoretical_uniform_loss:.4f}")
    print("Note: Initial losses should be close to this value for a randomly initialized model")
    
    # Additional validation
    print("\nValidation checks:")
    print(f"1. Are losses close to theoretical? " +
          f"{'Yes' if abs(loss.item() - theoretical_uniform_loss) < 1.0 else 'No'}")
    print(f"2. Are individual losses similar? " +
          f"{'Yes' if max(individual_losses) - min(individual_losses) < 1.0 else 'No'}")
    print(f"3. Is final loss average of individual? " +
          f"{'Yes' if abs(loss.item() - sum(individual_losses)/len(individual_losses)) < 1e-5 else 'No'}")

if __name__ == '__main__':
    main() 