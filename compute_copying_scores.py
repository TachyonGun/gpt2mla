import os
import math
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import GPT, GPTConfig
from model_mla import GPT_MLA, MLAConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of names for the IOI experiment (copied from the original IOI dataset implementation)
NAMES = [
    "Aaron", "Adam", "Alan", "Alex", "Alice", "Amy", "Anderson", "Andre", "Andrew", 
    "Andy", "Anna", "Anthony", "Arthur", "Austin", "Blake", "Brandon", "Brian", 
    "Carter", "Charles", "Charlie", "Christian", "Christopher", "Clark", "Cole", 
    "Collins", "Connor", "Crew", "Crystal", "Daniel", "David", "Dean", "Edward", 
    "Elizabeth", "Emily", "Eric", "Eva", "Ford", "Frank", "George", "Georgia", 
    "Graham", "Grant", "Henry", "Ian", "Jack", "Jacob", "Jake", "James", "Jamie", 
    "Jane", "Jason", "Jay", "Jennifer", "Jeremy", "Jessica", "John", "Jonathan", 
    "Jordan", "Joseph", "Joshua", "Justin", "Kate", "Kelly", "Kevin", "Kyle", 
    "Laura", "Leon", "Lewis", "Lisa", "Louis", "Luke", "Madison", "Marco", "Marcus", 
    "Maria", "Mark", "Martin", "Mary", "Matthew", "Max", "Michael", "Michelle", 
    "Morgan", "Patrick", "Paul", "Peter", "Prince", "Rachel", "Richard", "River", 
    "Robert", "Roman", "Rose", "Ruby", "Russell", "Ryan", "Sarah", "Scott", "Sean", 
    "Simon", "Stephen", "Steven", "Sullivan", "Taylor", "Thomas", "Tyler", "Victoria", 
    "Warren", "William"
]

def extract_attention_weights(model, layer, head):
    """
    Extract the weight matrices for a specific attention head.
    
    Args:
        model: GPT or MLA model
        layer: Layer index
        head: Head index
        
    Returns:
        W_V: Value projection weights for this head
        W_O: Output projection weights for this head
    """
    if isinstance(model, GPT_MLA):
        # MLA model has separate value and output matrices
        attn = model.transformer.h[layer].attn
        
        # For MLA, value matrix projects from latent space
        W_V = attn.value.weight  # [n_embd, n_latent]
        
        # Split W_O for the specific head
        head_size = model.config.n_embd // model.config.n_head
        head_start = head * head_size
        head_end = (head + 1) * head_size
        W_O_full = attn.c_proj.weight  # [n_embd, n_embd]
        W_O = W_O_full[:, head_start:head_end]  # [n_embd, head_size]
        
    else:  # GPT-2 model
        # In GPT-2, c_attn is a single linear layer that projects to q, k, v together
        # We need to be careful about extracting the weights correctly
        attn = model.transformer.h[layer].attn
        head_size = model.config.n_embd // model.config.n_head
        n_embd = model.config.n_embd
        
        # The c_attn weight matrix shape is [n_embd, 3*n_embd]
        # The bias shape is [3*n_embd]
        W_qkv = attn.c_attn.weight  # [n_embd, 3*n_embd]
        
        # Extract just the value portion (last third) for this head
        # First compute starting index for values (after q and k)
        v_start_idx = 2 * n_embd  # Skip past the query and key parts
        
        # For this specific head, extract its portion of V
        head_v_start = v_start_idx + (head * head_size)
        head_v_end = head_v_start + head_size
        
        # Extract the value matrix for this head
        # W_V has shape [n_embd, head_size]
        W_V = W_qkv[:, head_v_start:head_v_end]
        
        # Extract the output matrix from c_proj
        # c_proj has shape [n_embd, n_embd]
        W_O_full = attn.c_proj.weight  # [n_embd, n_embd]
        
        # For this head, extract its portion from the output 
        # W_O has shape [head_size, n_embd]
        head_o_start = head * head_size
        head_o_end = head_o_start + head_size
        W_O = W_O_full[head_o_start:head_o_end, :].t()  # Transpose to match expected dims
    
    return W_V, W_O

def get_copying_scores(model, k=5, names=NAMES):
    """
    Gets copying scores (both positive and negative) for each attention head.
    
    The copying score measures how well the OV circuit of a head copies token embeddings.
    For each head, we:
    1. Apply the OV circuit (W_V @ W_O) to name embeddings (after MLP0)
    2. Check if the original name appears in the top-k/bottom-k tokens predicted
    
    Based on the IOI paper methodology described on page 6.
    
    Args:
        model: GPT or MLA model
        k: Number of top tokens to consider
        names: List of names to test
        
    Returns:
        Tensor of shape [2, n_layers-1, n_heads] where:
            - First dimension is [positive_score, negative_score]
            - Second dimension is layer (excluding layer 0)
            - Third dimension is head
    """
    # Get model configuration
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    
    # Initialize results tensor - omitting layer 0 as per the IOI paper
    results = torch.zeros((2, n_layer-1, n_head), device=device)
    
    # Step 1: Tokenize the names
    # Since we don't have direct access to the tokenizer, we'll use the embedding
    # to assign unique token IDs
    names_tokens = []
    for name in names:
        # Simplified tokenization - in reality should use the model's tokenizer
        # Here we're just ensuring each name gets a unique token ID
        token_id = hash(name) % (model.transformer.wte.weight.shape[0] - 100)
        names_tokens.append(token_id)
    
    name_tokens = torch.tensor(names_tokens, device=device).unsqueeze(1)  # [batch, 1]
    
    # Step 2: Get embeddings
    name_embeddings = model.transformer.wte(name_tokens)  # [batch, 1, n_embd]
    
    # Step 3: Apply MLP0 (the first transformer block's MLP)
    # First apply layer norm
    ln0 = model.transformer.h[0].ln_2
    mlp0 = model.transformer.h[0].mlp
    
    ln_output = ln0(name_embeddings)
    mlp_output = mlp0(ln_output)
    
    # Add to residual stream (embeddings + MLP output)
    resid_after_mlp0 = name_embeddings + mlp_output  # [batch, 1, n_embd]
    
    # Print debug info
    print(f"Shape of residual after MLP0: {resid_after_mlp0.shape}")
    
    # Step 4: Loop through all layers (except 0) and heads
    for layer in range(1, n_layer):
        for head in range(n_head):
            # Extract weights for this head
            W_V, W_O = extract_attention_weights(model, layer, head)
            
            # Print debug info for first head
            if layer == 1 and head == 0:
                print(f"W_V shape: {W_V.shape}, W_O shape: {W_O.shape}")
            
            # Compute OV circuit
            # For MLA, we need special handling since W_V projects from latent space
            if isinstance(model, GPT_MLA):
                # First get the latent representation
                attn = model.transformer.h[layer].attn
                latent = attn.latent_enc(resid_after_mlp0)  # [batch, 1, n_latent]
                
                # Then apply V projection (latent -> head output)
                head_size = model.config.n_embd // model.config.n_head
                v_output = torch.matmul(latent, W_V.t())  # [batch, 1, n_embd]
                
                # Reshape to extract just this head's output
                head_start = head * head_size
                head_end = (head + 1) * head_size
                v_output_head = v_output[:, :, head_start:head_end]  # [batch, 1, head_size]
                
                # Apply O projection for this head
                o_output = torch.matmul(v_output_head, W_O.t())  # [batch, 1, n_embd]
            else:
                # For GPT-2, apply V and O projections
                # First apply V projection (input -> head representation)
                v_output = torch.matmul(resid_after_mlp0, W_V)  # [batch, 1, head_size]
                
                # Then apply O projection (head representation -> output)
                o_output = torch.matmul(v_output, W_O)  # [batch, 1, n_embd]
            
            # Apply positive and negative OV circuits
            resid_after_OV_pos = o_output
            resid_after_OV_neg = -o_output
            
            # Apply final layer norm
            ln_final = model.transformer.ln_f
            ln_pos_output = ln_final(resid_after_OV_pos)  # [batch, 1, n_embd]
            ln_neg_output = ln_final(resid_after_OV_neg)  # [batch, 1, n_embd]
            
            # Get logits by projecting through the embedding matrix
            logits_pos = torch.matmul(ln_pos_output, model.transformer.wte.weight.t())  # [batch, 1, vocab]
            logits_neg = torch.matmul(ln_neg_output, model.transformer.wte.weight.t())  # [batch, 1, vocab]
            
            # Remove sequence dimension
            logits_pos = logits_pos.squeeze(1)  # [batch, vocab]
            logits_neg = logits_neg.squeeze(1)  # [batch, vocab]
            
            # Get top-k predictions
            topk_pos = torch.topk(logits_pos, k=k, dim=1).indices  # [batch, k]
            topk_neg = torch.topk(logits_neg, k=k, dim=1).indices  # [batch, k]
            
            # Check if original token is in top-k
            in_topk_pos = torch.any(topk_pos == name_tokens, dim=1)  # [batch]
            in_topk_neg = torch.any(topk_neg == name_tokens, dim=1)  # [batch]
            
            # Calculate scores (percentage of names appearing in top-k)
            pos_score = in_topk_pos.float().mean().item()
            neg_score = in_topk_neg.float().mean().item()
            
            # Store results - layer-1 because we're omitting layer 0
            results[0, layer-1, head] = pos_score
            results[1, layer-1, head] = neg_score
    
    return results

def plot_copying_scores(gpt2_scores, mla_scores):
    """
    Plot copying scores for both models side by side.
    
    Args:
        gpt2_scores: Copying scores for GPT-2 [2, n_layers-1, n_heads]
        mla_scores: Copying scores for MLA [2, n_layers-1, n_heads]
    """
    # Create subplots for positive and negative scores
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("GPT-2 Positive Copying Scores", "MLA Positive Copying Scores",
                        "GPT-2 Negative Copying Scores", "MLA Negative Copying Scores")
    )
    
    # Convert tensors to numpy arrays
    gpt2_pos = gpt2_scores[0].cpu().numpy() * 100  # Convert to percentage
    gpt2_neg = gpt2_scores[1].cpu().numpy() * 100
    mla_pos = mla_scores[0].cpu().numpy() * 100
    mla_neg = mla_scores[1].cpu().numpy() * 100
    
    # Create heatmaps
    fig.add_trace(
        go.Heatmap(
            z=gpt2_pos,
            x=[f"Head {h}" for h in range(gpt2_pos.shape[1])],
            y=[f"Layer {l+1}" for l in range(gpt2_pos.shape[0])],  # +1 since layer 0 is omitted
            colorscale='Viridis',
            colorbar=dict(title='Score %', x=0.46),
            zmin=0, zmax=100
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=mla_pos,
            x=[f"Head {h}" for h in range(mla_pos.shape[1])],
            y=[f"Layer {l+1}" for l in range(mla_pos.shape[0])],
            colorscale='Viridis',
            colorbar=dict(title='Score %', x=1.0),
            zmin=0, zmax=100
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Heatmap(
            z=gpt2_neg,
            x=[f"Head {h}" for h in range(gpt2_neg.shape[1])],
            y=[f"Layer {l+1}" for l in range(gpt2_neg.shape[0])],
            colorscale='Viridis',
            colorbar=dict(title='Score %', x=0.46),
            zmin=0, zmax=100
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=mla_neg,
            x=[f"Head {h}" for h in range(mla_neg.shape[1])],
            y=[f"Layer {l+1}" for l in range(mla_neg.shape[0])],
            colorscale='Viridis',
            colorbar=dict(title='Score %', x=1.0),
            zmin=0, zmax=100
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Copying Scores: GPT-2 vs MLA",
        height=800,
        width=1200
    )
    
    fig.show()

def main():
    # Load GPT-2 model
    print("Loading GPT-2 model...")
    gpt2_model = GPT.from_pretrained('gpt2', override_args=dict(dropout=0.0))
    gpt2_model.to(device)
    gpt2_model.eval()
    
    # Load MLA model from checkpoint
    print("Loading MLA model...")
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
    
    # Compute copying scores for both models
    print("Computing copying scores for GPT-2...")
    gpt2_scores = get_copying_scores(gpt2_model)
    
    print("Computing copying scores for MLA...")
    mla_scores = get_copying_scores(mla_model)
    
    # Plot results
    plot_copying_scores(gpt2_scores, mla_scores)
    
    # Print summary statistics
    print("\nGPT-2 Statistics:")
    print(f"Average positive copying score: {gpt2_scores[0].mean().item():.2%}")
    print(f"Average negative copying score: {gpt2_scores[1].mean().item():.2%}")
    
    # Find max positive and negative scores with their positions
    max_pos_val, max_pos_idx = gpt2_scores[0].max(), gpt2_scores[0].argmax()
    max_pos_layer = max_pos_idx // gpt2_scores.shape[2] + 1  # +1 because layer 0 is omitted
    max_pos_head = max_pos_idx % gpt2_scores.shape[2]
    print(f"Max positive score: {max_pos_val.item():.2%} at Layer {max_pos_layer}, Head {max_pos_head}")
    
    max_neg_val, max_neg_idx = gpt2_scores[1].max(), gpt2_scores[1].argmax()
    max_neg_layer = max_neg_idx // gpt2_scores.shape[2] + 1
    max_neg_head = max_neg_idx % gpt2_scores.shape[2]
    print(f"Max negative score: {max_neg_val.item():.2%} at Layer {max_neg_layer}, Head {max_neg_head}")
    
    print("\nMLA Statistics:")
    print(f"Average positive copying score: {mla_scores[0].mean().item():.2%}")
    print(f"Average negative copying score: {mla_scores[1].mean().item():.2%}")
    
    # Find max positive and negative scores with their positions
    max_pos_val, max_pos_idx = mla_scores[0].max(), mla_scores[0].argmax()
    max_pos_layer = max_pos_idx // mla_scores.shape[2] + 1
    max_pos_head = max_pos_idx % mla_scores.shape[2]
    print(f"Max positive score: {max_pos_val.item():.2%} at Layer {max_pos_layer}, Head {max_pos_head}")
    
    max_neg_val, max_neg_idx = mla_scores[1].max(), mla_scores[1].argmax()
    max_neg_layer = max_neg_idx // mla_scores.shape[2] + 1
    max_neg_head = max_neg_idx % mla_scores.shape[2]
    print(f"Max negative score: {max_neg_val.item():.2%} at Layer {max_neg_layer}, Head {max_neg_head}")

if __name__ == "__main__":
    main() 