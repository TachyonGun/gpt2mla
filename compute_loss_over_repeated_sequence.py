import os
import math
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from model import GPT, GPTConfig
from model_mla import GPT_MLA, MLAConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_repeated_tokens(model, seq_len: int, batch_size: int = 1):
    """
    Generates a sequence of repeated random tokens in the format:
    [BOS token] [random sequence] [same random sequence]
    
    Args:
        model: GPT or MLA model (used to get vocab size)
        seq_len: length of the random sequence to generate (will be repeated)
        batch_size: number of sequences to generate in parallel
    
    Returns:
        tensor of shape [batch_size, 1 + 2*seq_len] containing the token sequences
    """
    vocab_size = model.transformer.wte.weight.shape[0]  # Get vocab size from embedding layer
    
    # Step 1: Create BOS token (token ID 50256 in GPT-2)
    # Shape: [batch_size, 1]
    bos_token = (torch.ones(batch_size, 1) * (vocab_size - 1)).long().to(device)
    
    # Step 2: Generate random sequence
    # Shape: [batch_size, seq_len]
    random_sequence = torch.randint(0, vocab_size - 1, (batch_size, seq_len), device=device)
    
    # Step 3: Concatenate [BOS, random_sequence, random_sequence]
    # Shape: [batch_size, 1 + 2*seq_len]
    full_sequence = torch.cat([bos_token, random_sequence, random_sequence], dim=1)
    
    return full_sequence

def get_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute log probabilities of the actual next tokens.
    
    Args:
        logits: shape [batch_size, sequence_length, vocab_size] - raw model outputs
        tokens: shape [batch_size, sequence_length] - input token IDs
    
    Returns:
        tensor of shape [batch_size, sequence_length-1] containing log probs
        of correctly predicting each next token
    """
    print(f"Logits shape: {logits.shape}")
    print(f"Tokens shape: {tokens.shape}")
    
    # Step 1: Convert logits to log probabilities
    # Shape remains [batch_size, sequence_length, vocab_size]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    print(f"Log probs shape: {log_probs.shape}")
    
    # Step 2: Get the target tokens (what we're trying to predict)
    # Shape: [batch_size, sequence_length-1]
    target_tokens = tokens[:, 1:]  # All tokens except the first
    print(f"Target tokens shape: {target_tokens.shape}")
    
    # Step 3: Get the predictions (all except last position, as it has no target)
    # Shape: [batch_size, sequence_length-1, vocab_size]
    predictions = log_probs[:, :-1, :]  # Remove last position
    print(f"Predictions shape: {predictions.shape}")
    
    # Step 4: For each position, get the log prob of the actual next token
    # First unsqueeze adds a dimension for gathering
    # Shape: [batch_size, sequence_length-1, 1]
    target_tokens_unsqueezed = target_tokens.unsqueeze(-1)
    print(f"Target tokens unsqueezed shape: {target_tokens_unsqueezed.shape}")
    
    # Gather the log probs of the actual next tokens
    # Shape: [batch_size, sequence_length-1]
    correct_log_probs = torch.gather(predictions, dim=-1, index=target_tokens_unsqueezed).squeeze(-1)
    print(f"Correct log probs shape: {correct_log_probs.shape}")
    
    return correct_log_probs

def compute_loss_over_sequence(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute loss for each position in the sequence.
    
    Args:
        model: the language model (GPT-2 or MLA)
        tokens: shape [batch_size, sequence_length] input token IDs
    
    Returns:
        tensor of shape [sequence_length-1] containing the loss at each position
    """
    model.eval()  # Ensure model is in eval mode
    with torch.no_grad():
        # Step 1: Get model predictions
        # Pass dummy targets to force computation of all logits
        dummy_targets = tokens.clone()
        logits, _ = model(tokens, targets=dummy_targets)
        print(f"\nIn compute_loss_over_sequence:")
        print(f"Input tokens shape: {tokens.shape}")
        print(f"Output logits shape: {logits.shape}")
        
        # Step 2: Compute log probs for each position
        # Shape: [batch_size, sequence_length-1]
        log_probs = get_log_probs(logits, tokens)
        
        # Step 3: Convert to loss (negative log prob) and take first batch item
        # Shape: [sequence_length-1]
        loss = -log_probs[0]
        print(f"Final loss shape: {loss.shape}")
        
        return loss

def plot_loss_comparison(gpt2_loss: torch.Tensor, mla_loss: torch.Tensor, seq_len: int):
    """
    Create a plotly figure comparing the loss of both models over the sequence.
    
    Args:
        gpt2_loss: shape [2*seq_len] losses for GPT-2
        mla_loss: shape [2*seq_len] losses for MLA
        seq_len: length of each half of the sequence
    """
    positions = list(range(len(gpt2_loss)))
    
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=positions,
        y=gpt2_loss.cpu(),
        name="GPT-2",
        mode='lines',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=positions,
        y=mla_loss.cpu(),
        name="MLA",
        mode='lines',
        line=dict(color='red')
    ))
    
    # Add vertical separator
    fig.add_vline(x=seq_len-0.5, line_dash="dash", line_color="gray")
    
    # Update layout
    fig.update_layout(
        title="Loss Over Repeated Sequence: GPT-2 vs MLA",
        xaxis_title="Position in Sequence",
        yaxis_title="Loss (negative log prob)",
        showlegend=True,
        hovermode='x unified'
    )
    
    # Add background colors
    fig.add_vrect(
        x0=0, x1=seq_len-0.5,
        fillcolor="lightgray", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="First Half",
        annotation_position="top"
    )
    fig.add_vrect(
        x0=seq_len-0.5, x1=2*seq_len-1,
        fillcolor="lightblue", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Second Half (Repeated)",
        annotation_position="top"
    )
    
    fig.show()

def main():
    # Configuration
    seq_len = 50  # Length of sequence to repeat
    batch_size = 1  # Keep it simple with batch size 1
    
    # Step 1: Load GPT-2 model
    print("Loading GPT-2...")
    gpt2_model = GPT.from_pretrained('gpt2', override_args=dict(dropout=0.0))
    gpt2_model.to(device)
    
    # Step 2: Load MLA model
    print("Loading MLA model...")
    ckpt_path = os.path.join('out_mla', 'ckpt_mla.pt')  # Keep the underscore path
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # Create and configure MLA model
    mla_config = MLAConfig(**checkpoint['model_args'])
    mla_model = GPT_MLA(mla_config)
    
    # Fix DDP state dict
    state_dict = checkpoint['model']
    new_state_dict = {(k[7:] if k.startswith('module.') else k): v 
                     for k, v in state_dict.items()}
    
    mla_model.load_state_dict(new_state_dict)
    mla_model.to(device)
    
    # Step 3: Generate test sequence
    print("\nGenerating test sequence...")
    tokens = generate_repeated_tokens(gpt2_model, seq_len, batch_size)
    print(f"Sequence shape: {tokens.shape}")  # Should be [1, 1 + 2*seq_len]
    
    # Step 4: Compute losses
    print("\nComputing GPT-2 losses...")
    gpt2_loss = compute_loss_over_sequence(gpt2_model, tokens)
    print("\nComputing MLA losses...")
    mla_loss = compute_loss_over_sequence(mla_model, tokens)
    
    # Step 5: Plot comparison
    print("\nPlotting results...")
    plot_loss_comparison(gpt2_loss, mla_loss, seq_len)
    
    # Step 6: Print statistics
    def print_stats(name, loss):
        first_half = loss[:seq_len]
        second_half = loss[seq_len:]
        print(f"\n{name} Average Losses:")
        print(f"First half: {first_half.mean():.3f}")
        print(f"Second half: {second_half.mean():.3f}")
        print(f"Difference: {first_half.mean() - second_half.mean():.3f}")
    
    print_stats("GPT-2", gpt2_loss)
    print_stats("MLA", mla_loss)

if __name__ == "__main__":
    main() 