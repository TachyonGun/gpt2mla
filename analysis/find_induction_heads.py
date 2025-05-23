import os
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import GPT, GPTConfig
from model_mla import GPT_MLA, MLAConfig
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_repeated_tokens(model, seq_len: int, batch_size: int = 1):
    """
    Generates a sequence of repeated random tokens for testing induction heads.
    Format: [BOS, random_tokens, same_random_tokens]
    """
    prefix = (torch.ones(batch_size, 1) * model.transformer.wte.weight.shape[0] - 1).long()
    rep_tokens_half = torch.randint(0, model.transformer.wte.weight.shape[0], (batch_size, seq_len))
    rep_tokens = torch.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens

def run_with_cache(model, tokens):
    """
    Run the model and store intermediate attention patterns
    """
    B, T = tokens.size()
    assert T <= model.config.block_size

    # Storage for attention patterns
    attention_patterns = []
    
    def attention_hook(module, input, output):
        if isinstance(model, GPT_MLA):
            # For MLA, reconstruct attention pattern from q, k, v
            x = input[0]
            B, T, C = x.shape
            
            # Get latent representation
            z = module.latent_enc(x)
            
            # Get q, k, v
            q = module.query(x)
            k = module.key(z)
            v = module.value(z)
            
            # Split into heads
            head_size = C // module.n_head
            q = q.view(B, T, module.n_head, head_size).transpose(1, 2)
            k = k.view(B, T, module.n_head, head_size).transpose(1, 2)
            v = v.view(B, T, module.n_head, head_size).transpose(1, 2)
            
            # Compute attention scores
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
            if hasattr(module, 'bias'):
                att = att.masked_fill(module.bias[:,:,:T,:T] == 0, float('-inf'))
            else:
                # Create causal mask
                mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
                att = att.masked_fill(mask, float('-inf'))
            
            pattern = torch.nn.functional.softmax(att, dim=-1)
        else:
            # For regular GPT-2
            x = input[0]
            B, T, C = x.shape
            
            # Project input to q, k, v using c_attn
            qkv = module.c_attn(x)
            q, k, v = qkv.split(module.n_embd, dim=2)
            
            # Split into heads
            head_size = C // module.n_head
            q = q.view(B, T, module.n_head, head_size).transpose(1, 2)
            k = k.view(B, T, module.n_head, head_size).transpose(1, 2)
            v = v.view(B, T, module.n_head, head_size).transpose(1, 2)
            
            # Compute attention scores
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
            if hasattr(module, 'bias'):
                att = att.masked_fill(module.bias[:,:,:T,:T] == 0, float('-inf'))
            else:
                # Create causal mask
                mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
                att = att.masked_fill(mask, float('-inf'))
            
            pattern = torch.nn.functional.softmax(att, dim=-1)
            
        attention_patterns.append(pattern.detach())

    # Register hooks
    hooks = []
    if isinstance(model, GPT_MLA):
        for block in model.transformer.h:
            hooks.append(block.attn.register_forward_hook(attention_hook))
    else:  # Regular GPT
        for block in model.transformer.h:
            hooks.append(block.attn.register_forward_hook(attention_hook))

    # Run model
    with torch.no_grad():
        logits, _ = model(tokens, targets=None)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return logits, attention_patterns

def compute_induction_scores(attention_patterns, seq_len):
    """
    Compute induction scores for each head in each layer.
    An induction score measures how much a head attends to tokens that appeared seq_len positions ago.
    """
    n_layers = len(attention_patterns)
    n_heads = attention_patterns[0].shape[1]
    scores = torch.zeros(n_layers, n_heads)
    
    for layer in range(n_layers):
        pattern = attention_patterns[layer][0]  # Take first batch item
        for head in range(n_heads):
            # Look at attention to tokens that appeared seq_len positions ago
            # by looking at the diagonal offset by seq_len
            induction_stripe = pattern[head].diagonal(-seq_len+1)
            scores[layer, head] = induction_stripe.mean()
    
    return scores

def plot_induction_scores(scores, model_name):
    """
    Create a heatmap of induction scores with an additional barplot showing layer averages
    """
    # Convert scores to numpy for calculations
    scores_np = scores.cpu().numpy()
    
    # Calculate layer averages (mean across heads for each layer)
    layer_averages = scores_np.mean(axis=1)
    
    # Create subplot layout with custom widths
    fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2],
                        horizontal_spacing=0.15)
    
    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=scores_np,
            colorscale="RdBu",
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(
                title="Induction Score",
                x=-0.15,  # Moved colorbar to the left
                len=0.7,
                thickness=15
            )
        ),
        row=1, col=1
    )
    
    # Add vertical barplot
    fig.add_trace(
        go.Bar(
            x=layer_averages,
            y=list(range(len(layer_averages))),
            orientation='h',
            marker=dict(color=layer_averages, colorscale="RdBu"),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Induction Scores for {model_name}",
        height=600,
        yaxis=dict(title="Layer", autorange="reversed"),
        yaxis2=dict(title="", showticklabels=False, autorange="reversed"),
        xaxis=dict(title="Head"),
        xaxis2=dict(title="Average Score", range=[0, 1]),
        margin=dict(l=120, r=50)  # Added left margin for colorbar
    )
    
    fig.show()

def load_gpt_model(use_openai_weights=False):
    """
    Load GPT model either from local checkpoint or OpenAI weights
    """
    if use_openai_weights:
        print("Loading GPT-2 from OpenAI weights...")
        model = GPT.from_pretrained('gpt2', override_args=dict(dropout=0.0))
    else:
        print("Loading GPT-2 from local checkpoint...")
        ckpt_path = os.path.join('out', 'ckpt.pt')
        print(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        
        # Fix the state dict if it was saved from a DDP model
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
    
    return model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze induction heads in GPT models')
    parser.add_argument('--openai_weights', action='store_true',
                      help='Use OpenAI pretrained weights instead of local checkpoint')
    args = parser.parse_args()

    # Test sequence parameters
    seq_len = 50
    batch_size = 1

    # Load GPT-2
    gpt2_model = load_gpt_model(args.openai_weights)
    gpt2_model.to(device)
    gpt2_model.eval()

    # Generate test sequence and run GPT-2
    gpt2_tokens = generate_repeated_tokens(gpt2_model, seq_len, batch_size)
    _, gpt2_patterns = run_with_cache(gpt2_model, gpt2_tokens)
    gpt2_scores = compute_induction_scores(gpt2_patterns, seq_len)
    plot_induction_scores(gpt2_scores, "GPT-2")

    # Load MLA model from checkpoint
    print("\nTesting MLA model...")
    ckpt_path = os.path.join('out_mla', 'ckpt_mla.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    mla_config = MLAConfig(**checkpoint['model_args'])
    mla_model = GPT_MLA(mla_config)
    
    # Fix the state dict if it was saved from a DDP model
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    
    mla_model.load_state_dict(new_state_dict)
    mla_model.to(device)
    mla_model.eval()

    # Generate test sequence and run MLA
    mla_tokens = generate_repeated_tokens(mla_model, seq_len, batch_size)
    _, mla_patterns = run_with_cache(mla_model, mla_tokens)
    mla_scores = compute_induction_scores(mla_patterns, seq_len)
    plot_induction_scores(mla_scores, "MLA")

    # Print heads with high induction scores (threshold = 0.4)
    threshold = 0.4
    print("\nGPT-2 Induction Heads:")
    for layer in range(len(gpt2_scores)):
        for head in range(len(gpt2_scores[layer])):
            if gpt2_scores[layer, head] > threshold:
                print(f"Layer {layer}, Head {head}: {gpt2_scores[layer, head]:.3f}")

    print("\nMLA Induction Heads:")
    for layer in range(len(mla_scores)):
        for head in range(len(mla_scores[layer])):
            if mla_scores[layer, head] > threshold:
                print(f"Layer {layer}, Head {head}: {mla_scores[layer, head]:.3f}")

if __name__ == "__main__":
    main() 