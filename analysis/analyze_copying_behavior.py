import os
os.chdir('..')
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from model import GPT, GPTConfig
from model_mla import GPT_MLA, MLAConfig

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of names for testing copying behavior (same as in the original IOI dataset implementation)
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
        # For MLA model
        attn = model.transformer.h[layer].attn
        
        # Get dimensions
        head_size = model.config.n_embd // model.config.n_head
        n_embd = model.config.n_embd
        
        # Get the value weights - in MLA, these project from latent space
        # Extract just this head's portion
        W_V_full = attn.value.weight  # [n_embd, n_latent]
        
        # Get the output weights for this head
        head_start = head * head_size
        head_end = (head + 1) * head_size
        W_O_full = attn.c_proj.weight  # [n_embd, n_embd]
        W_O = W_O_full[:, head_start:head_end]  # [n_embd, head_size]
        
        return W_V_full, W_O
        
    else:  # Regular GPT-2 model
        attn = model.transformer.h[layer].attn
        head_size = model.config.n_embd // model.config.n_head
        n_embd = model.config.n_embd
        
        # For GPT-2, we need a different approach
        # Instead of directly using extract_attention_weights for value computation,
        # we'll handle this in the main function for GPT-2
        
        # Return None for W_V as we'll handle the value computation differently
        return None, None

def get_copying_scores(model, k=5, names=NAMES):
    """
    Gets copying scores (both positive and negative) for each attention head.
    
    The copying score measures how well the OV circuit of a head copies token embeddings.
    For each head, we:
    1. Apply the OV circuit (W_V @ W_O) to name embeddings (after MLP0)
    2. Check if the original name appears in the top-k/bottom-k tokens predicted
    
    Based on the methodology described in the IOI paper.
    
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
    # Force k to be an integer (in case it's somehow getting overridden as a tensor)
    k = int(5) if not isinstance(k, int) else k
    
    # Get model configuration
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    
    # Initialize results tensor - omitting layer 0 as per the IOI paper
    results = torch.zeros((2, n_layer-1, n_head), device=device)
    
    # Step 1: Tokenize the names
    # Since we don't have direct access to the tokenizer, we'll assign unique token IDs
    # This is a simplified approach for demonstration
    name_tokens = []
    for name in names:
        # Simplified tokenization - in reality should use the model's tokenizer
        # Here we're just ensuring each name gets a unique token ID
        token_id = hash(name) % (model.transformer.wte.weight.shape[0] - 100)
        name_tokens.append(token_id)
    
    name_tokens = torch.tensor(name_tokens, device=device)  # [batch]
    
    # Step 2: Get embeddings
    name_embeddings = model.transformer.wte(name_tokens)  # [batch, n_embd]
    
    # Step 3: Apply MLP0 (the first transformer block's MLP)
    # First apply layer norm
    ln0 = model.transformer.h[0].ln_2
    mlp0 = model.transformer.h[0].mlp
    
    # Add batch dimension for processing
    name_embeddings = name_embeddings.unsqueeze(1)  # [batch, 1, n_embd]
    ln_output = ln0(name_embeddings)
    mlp_output = mlp0(ln_output)
    
    # Add to residual stream (embeddings + MLP output)
    resid_after_mlp0 = name_embeddings + mlp_output  # [batch, 1, n_embd]
    
    # Step 4: Apply final layer norm to get properly scaled embeddings
    ln_final = model.transformer.ln_f
    
    # Print debug information
    print(f"Processing {len(names)} names to compute copying scores")
    print(f"Shape of residual after MLP0: {resid_after_mlp0.shape}")
    print(f"Processing scores for {n_layer-1} layers, each with {n_head} heads")
    
    # Step 5: Loop through all layers (except 0) and heads
    for layer in range(1, n_layer):
        for head in range(n_head):
            # Implement copying score computation based on model type
            if isinstance(model, GPT_MLA):
                # For MLA, we need to handle the latent space mapping
                attn = model.transformer.h[layer].attn
                
                # Apply latent encoder to get latent representation
                latent = attn.latent_enc(resid_after_mlp0)  # [batch, 1, n_latent]
                
                # Apply value projection from latent to head space 
                head_size = model.config.n_embd // model.config.n_head
                
                # For MLA, we need to extract just this head's part of the output
                # Get the full value projection
                v_output = attn.value(latent)  # [batch, 1, n_embd]
                
                # Get just this head's portion
                head_start = head * head_size
                head_end = (head + 1) * head_size
                v_output_head = v_output[:, :, head_start:head_end]  # [batch, 1, head_size]
                
                # Apply output projection for this head
                W_O = attn.c_proj.weight[:, head_start:head_end]  # [n_embd, head_size]
                o_output_pos = torch.matmul(v_output_head, W_O.t())  # [batch, 1, n_embd]
                o_output_neg = -o_output_pos  # Negative of the output for negative copying
                
            else:  # Regular GPT-2
                # For GPT-2, directly apply attention operations
                attn = model.transformer.h[layer].attn
                head_size = model.config.n_embd // model.config.n_head
                n_embd = model.config.n_embd
                
                # For GPT-2, first apply the full c_attn to get q, k, v
                qkv = attn.c_attn(resid_after_mlp0)  # [batch, 1, 3*n_embd]
                
                # Split into q, k, v
                q, k, v = qkv.split(n_embd, dim=2)
                
                # Reshape to get head-specific values
                v = v.view(-1, 1, n_head, head_size)  # [batch, 1, n_head, head_size]
                
                # Extract just this head's values
                v_head = v[:, :, head, :]  # [batch, 1, head_size]
                
                # Get the output weights for this head
                W_O_full = attn.c_proj.weight  # [n_embd, n_embd]
                head_start = head * head_size
                head_end = (head + 1) * head_size
                W_O = W_O_full[:, head_start:head_end]  # [n_embd, head_size]
                
                # Apply output projection
                o_output_pos = torch.matmul(v_head, W_O.t())  # [batch, 1, n_embd]
                o_output_neg = -o_output_pos  # Negative of the output
            
            # Apply final layer norm
            ln_pos_output = ln_final(o_output_pos)  # [batch, 1, n_embd]
            ln_neg_output = ln_final(o_output_neg)  # [batch, 1, n_embd]
            
            # Get logits by projecting through the unembedding matrix
            logits_pos = torch.matmul(ln_pos_output, model.transformer.wte.weight.t())  # [batch, 1, vocab]
            logits_neg = torch.matmul(ln_neg_output, model.transformer.wte.weight.t())  # [batch, 1, vocab]
            
            # Remove sequence dimension
            logits_pos = logits_pos.squeeze(1)  # [batch, vocab]
            logits_neg = logits_neg.squeeze(1)  # [batch, vocab]
            
            # Get top-k predictions

            topk_pos = torch.topk(logits_pos, k=5, dim=1).indices  # [batch, k]
            topk_neg = torch.topk(logits_neg, k=5, dim=1).indices  # [batch, k]
            
            # Check if original token is in top-k
            name_tokens_expanded = name_tokens.unsqueeze(1)  # [batch, 1]
            in_topk_pos = (topk_pos == name_tokens_expanded).any(dim=1)  # [batch]
            in_topk_neg = (topk_neg == name_tokens_expanded).any(dim=1)  # [batch]
            
            # Calculate scores (percentage of names appearing in top-k)
            pos_score = in_topk_pos.float().mean().item()
            neg_score = in_topk_neg.float().mean().item()
            
            # Store results - layer-1 because we're omitting layer 0
            results[0, layer-1, head] = pos_score
            results[1, layer-1, head] = neg_score
            
            # Print progress for select layers/heads
            if head == 0 or (layer == n_layer-1 and head == n_head-1):
                print(f"Layer {layer}, Head {head}: Pos={pos_score:.3f}, Neg={neg_score:.3f}")
    
    return results

def plot_copying_scores(gpt2_results, mla_results):
    """
    Plot all copying scores in a single figure with 2x2 subplots.
    
    Args:
        gpt2_results: Tensor of shape [2, n_layers-1, n_heads] with GPT-2 copying scores
        mla_results: Tensor of shape [2, n_layers-1, n_heads] with MLA copying scores
    """
    # Convert results to numpy arrays
    gpt2_pos = gpt2_results[0].cpu().numpy()
    gpt2_neg = gpt2_results[1].cpu().numpy()
    mla_pos = mla_results[0].cpu().numpy()
    mla_neg = mla_results[1].cpu().numpy()
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot GPT-2 positive scores
    im1 = ax1.imshow(gpt2_pos, cmap='viridis', aspect='auto')
    ax1.set_title('GPT-2 Positive copying scores')
    ax1.set_xlabel('Head')
    ax1.set_ylabel('Layer')
    plt.colorbar(im1, ax=ax1)
    
    # Plot GPT-2 negative scores
    im2 = ax2.imshow(gpt2_neg, cmap='viridis', aspect='auto')
    ax2.set_title('GPT-2 Negative copying scores')
    ax2.set_xlabel('Head')
    ax2.set_ylabel('Layer')
    plt.colorbar(im2, ax=ax2)
    
    # Plot MLA positive scores
    im3 = ax3.imshow(mla_pos, cmap='viridis', aspect='auto')
    ax3.set_title('MLA Positive copying scores')
    ax3.set_xlabel('Head')
    ax3.set_ylabel('Layer')
    plt.colorbar(im3, ax=ax3)
    
    # Plot MLA negative scores
    im4 = ax4.imshow(mla_neg, cmap='viridis', aspect='auto')
    ax4.set_title('MLA Negative copying scores')
    ax4.set_xlabel('Head')
    ax4.set_ylabel('Layer')
    plt.colorbar(im4, ax=ax4)
    
    # Update axes for all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        # Set layer labels (0-11)
        layer_labels = [str(i) for i in range(12)]
        ax.set_yticks(range(len(layer_labels)))
        ax.set_yticklabels(layer_labels)
        
        # Set head labels
        head_labels = [str(i) for i in range(gpt2_pos.shape[1])]
        ax.set_xticks(range(len(head_labels)))
        ax.set_xticklabels(head_labels)
    
    # Add overall title
    fig.suptitle("Copying scores of attention heads' OV circuits")
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return fig

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
    print("\nComputing copying scores for GPT-2...")
    gpt2_scores = get_copying_scores(gpt2_model)
    
    print("\nComputing copying scores for MLA...")
    mla_scores = get_copying_scores(mla_model)
    
    # Plot all results in one figure
    print("\nPlotting all copying scores...")
    plot_copying_scores(gpt2_scores, mla_scores)
    
    # Print summary statistics
    print("\nGPT-2 Statistics:")
    print(f"Average positive copying score: {gpt2_scores[0].mean().item():.2%}")
    print(f"Average negative copying score: {gpt2_scores[1].mean().item():.2%}")
    
    # Find max positive and negative scores with their positions
    max_pos_val, max_pos_idx = gpt2_scores[0].max(), gpt2_scores[0].argmax()
    layer_idx = max_pos_idx // gpt2_scores.shape[2]
    head_idx = max_pos_idx % gpt2_scores.shape[2]
    print(f"Max positive score: {max_pos_val.item():.2%} at Layer {layer_idx+1}, Head {head_idx}")
    
    max_neg_val, max_neg_idx = gpt2_scores[1].max(), gpt2_scores[1].argmax()
    layer_idx = max_neg_idx // gpt2_scores.shape[2]
    head_idx = max_neg_idx % gpt2_scores.shape[2]
    print(f"Max negative score: {max_neg_val.item():.2%} at Layer {layer_idx+1}, Head {head_idx}")
    
    print("\nMLA Statistics:")
    print(f"Average positive copying score: {mla_scores[0].mean().item():.2%}")
    print(f"Average negative copying score: {mla_scores[1].mean().item():.2%}")
    
    # Find max positive and negative scores with their positions
    max_pos_val, max_pos_idx = mla_scores[0].max(), mla_scores[0].argmax()
    layer_idx = max_pos_idx // mla_scores.shape[2]
    head_idx = max_pos_idx % mla_scores.shape[2]
    print(f"Max positive score: {max_pos_val.item():.2%} at Layer {layer_idx+1}, Head {head_idx}")
    
    max_neg_val, max_neg_idx = mla_scores[1].max(), mla_scores[1].argmax()
    layer_idx = max_neg_idx // mla_scores.shape[2]
    head_idx = max_neg_idx % mla_scores.shape[2]
    print(f"Max negative score: {max_neg_val.item():.2%} at Layer {layer_idx+1}, Head {head_idx}")

if __name__ == "__main__":
    main() 