# Configuration flags
SENTENCE = "The American flag is red, white, and blue"
REP_SEQUENCE = False  # Set to False to use SENTENCE, or a number N to generate repeated sequence of length 2N
PLOT_SUBSET = False  # Set to False to plot all points, or a number N to plot N points (or N points from each half if REP_SEQUENCE is set)
USE_OPENAI_WEIGHTS = False  # Set to True to use official OpenAI GPT-2 weights, False to use local checkpoint

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
from model import GPTConfig, GPT
from transformers import GPT2Tokenizer
import argparse
from tabulate import tabulate
from termcolor import colored
import torch.nn.functional as F
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_repeated_sequence(tokenizer, seq_len):
    """
    Generate a sequence of repeated random tokens.
    Format: [random_tokens, same_random_tokens]
    """
    # Generate random token indices (excluding special tokens)
    vocab_size = tokenizer.vocab_size
    rep_tokens_half = torch.randint(0, vocab_size - 4, (1, seq_len))  # -4 to avoid special tokens
    
    # Repeat the sequence
    rep_tokens = torch.cat([rep_tokens_half, rep_tokens_half], dim=-1)
    
    return rep_tokens.to(device)

def load_gpt2_model(use_openai=False, checkpoint_path='out/ckpt.pt'):
    """
    Load GPT-2 model either from OpenAI weights or local checkpoint
    """
    if use_openai:
        print("Loading GPT-2 from OpenAI weights...")
        model = GPT.from_pretrained('gpt2', override_args=dict(dropout=0.0))
    else:
        print("Loading GPT-2 from local checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
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
    
    model.to(device)
    model.eval()
    return model

def get_tokenizer():
    """
    Get GPT-2 tokenizer
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return tokenizer

def extract_residual_activations(model, tokens):
    """
    Extract residual stream activations at each layer and project to 2D using PCA
    """
    B, T = tokens.size()
    
    # Store residual stream activations from each layer
    residual_activations = []
    
    def residual_hook(module, input, output):
        # Get the input to the transformer block (residual stream)
        x = input[0]
        residual_activations.append(x.detach().cpu())
    
    # Register hooks for all blocks
    hooks = []
    
    # Hooks for inputs to each transformer block
    for block in model.transformer.h:
        hooks.append(block.register_forward_hook(residual_hook))
    
    # Run the model and get predictions
    with torch.no_grad():
        # Create targets for getting full logits
        if tokens.size(1) > 1:
            targets = tokens[0, 1:].clone()
            targets = torch.cat([targets, torch.tensor([-1], device=targets.device)])
        else:
            targets = torch.tensor([-1], device=tokens.device)
        targets = targets.unsqueeze(0)
        
        logits, _ = model(tokens, targets)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Process activations with PCA
    # First combine all activations to fit PCA
    all_activations = torch.stack(residual_activations, dim=0)  # [n_layers, B, T, n_embd]
    n_layers, B, T, D = all_activations.shape
    
    # Reshape to 2D array for PCA
    flat_activations = all_activations.reshape(-1, D).numpy()
    
    # Fit PCA on all activations
    pca = PCA(n_components=2)
    pca.fit(flat_activations)
    
    # Project each layer's activations using the same PCA
    projected_activations = []
    for layer_acts in residual_activations:
        layer_flat = layer_acts.reshape(-1, D).numpy()
        layer_projected = pca.transform(layer_flat)
        projected_activations.append(torch.from_numpy(layer_projected.reshape(B, T, 2)))
    
    variance_explained = [pca.explained_variance_ratio_] * n_layers
    
    return projected_activations, logits, variance_explained

def plot_residual_activations(residual_activations, tokens, tokenizer, text, variance_explained):
    """
    Plot the 2D residual stream activations for each layer
    """
    n_layers = len(residual_activations)  # Should be 12
    
    # Create a figure with 3x4 subplots (for 12 layers)
    fig = plt.figure(figsize=(22, 15))  # Made figure wider for colorbar
    gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.3)
    axes = []
    for i in range(3):
        for j in range(4):
            axes.append(fig.add_subplot(gs[i, j]))
    
    # Decode tokens for labeling points
    token_texts = []
    next_token_texts = []
    for i, token_id in enumerate(tokens[0]):
        token_text = tokenizer.decode([token_id.item()])
        token_texts.append(token_text)
        # Get next token (if not last token)
        if i < len(tokens[0]) - 1:
            next_token = tokenizer.decode([tokens[0][i + 1].item()])
        else:
            next_token = "<end>"
        next_token_texts.append(next_token)
    
    # Determine which indices to plot based on PLOT_SUBSET
    if PLOT_SUBSET:
        if REP_SEQUENCE:
            # For repeated sequences, take N points from first half and N points from second half
            seq_len = len(tokens[0]) // 2
            first_half_indices = list(range(min(PLOT_SUBSET, seq_len)))
            second_half_indices = list(range(seq_len, seq_len + min(PLOT_SUBSET, seq_len)))
            plot_indices = first_half_indices + second_half_indices
        else:
            # For regular sequences, take first N points
            plot_indices = list(range(min(PLOT_SUBSET, len(tokens[0]))))
    else:
        # Plot all points
        plot_indices = list(range(len(tokens[0])))
    
    # Create token labels for the title
    if REP_SEQUENCE:
        title = "Transformer Block Input Activations"
        if PLOT_SUBSET:
            title += f" (showing {PLOT_SUBSET} tokens from each half)"
    else:
        title_tokens = [f"[{text}]" for text in token_texts]
        title = f"Transformer Block Input Activations for: {' '.join(title_tokens)}"
        if PLOT_SUBSET:
            title += f"\n(showing first {PLOT_SUBSET} tokens)"
    
    # Set the main title
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Create color map for token positions
    cmap = plt.cm.get_cmap('jet')
    norm = mcolors.Normalize(vmin=0, vmax=len(tokens[0])-1)
    
    # Plot each layer's activations
    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        
        # Get residual activations for this layer
        z = residual_activations[layer_idx][0]  # [T, 2]
        
        # Plot selected points
        for token_idx in plot_indices:
            x, y = z[token_idx, 0], z[token_idx, 1]
            color = cmap(norm(token_idx))
            ax.scatter(x, y, c=[color], s=150, alpha=0.7)
            
            # Create two-line annotation with current and next token
            label_text = f"{token_texts[token_idx]}\n({next_token_texts[token_idx]})"
            
            # Add token text as label with both current and next token
            ax.annotate(
                label_text,
                (x, y),
                fontsize=9,
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
                linespacing=1.2  # Increase space between lines
            )
        
        # Add layer info as subtitle with variance explained
        title = f"Block {layer_idx + 1} Input\nPCA var: {variance_explained[layer_idx][0]:.2%}, {variance_explained[layer_idx][1]:.2%}"
        ax.set_title(title, fontsize=14)
        
        ax.set_xlabel("PC 1", fontsize=12)
        ax.set_ylabel("PC 2", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Adjust limits to add some padding
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Add a colorbar to show the token position mapping
    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Token Position', fontsize=12)
    
    plt.savefig("residual_activations.png", dpi=300, bbox_inches='tight')
    plt.savefig("residual_activations.svg", format='svg', bbox_inches='tight')
    plt.show()

def analyze_predictions(model, tokens, tokenizer):
    """
    Analyze model predictions and create a detailed table
    """
    with torch.no_grad():
        # Create targets for getting full logits
        if tokens.size(1) > 1:
            targets = tokens[0, 1:].clone()
            targets = torch.cat([targets, torch.tensor([-1], device=targets.device)])
        else:
            targets = torch.tensor([-1], device=tokens.device)
        targets = targets.unsqueeze(0)
        
        logits, loss = model(tokens, targets)
        
    # Get probabilities
    probs = F.softmax(logits[0], dim=-1)
    
    # Process each position
    table_data = []
    for i in range(len(tokens[0])):
        current_token = tokenizer.decode([tokens[0][i].item()])
        
        # Get next token (actual)
        if i < len(tokens[0]) - 1:
            next_token = tokenizer.decode([tokens[0][i + 1].item()])
            next_token_id = tokens[0][i + 1].item()
        else:
            next_token = "<end>"
            next_token_id = -1
            
        # Get predicted token
        pred_token_id = torch.argmax(logits[0, i]).item()
        pred_token = tokenizer.decode([pred_token_id])
        
        # Get probability of correct token
        if next_token_id != -1:
            correct_prob = probs[i, next_token_id].item()
            token_loss = -torch.log(probs[i, next_token_id]).item()
        else:
            correct_prob = 0.0
            token_loss = 0.0
        
        # Color the predicted token based on correctness
        if pred_token_id == next_token_id:
            pred_token_colored = colored(pred_token, 'green')
            prob_colored = colored(f"{correct_prob:.4f}", 'green')
        else:
            pred_token_colored = colored(pred_token, 'red')
            prob_colored = colored(f"{correct_prob:.4f}", 'red')
            
        table_data.append([
            current_token,
            f"({next_token})",
            pred_token_colored,
            prob_colored,
            f"{token_loss:.4f}"
        ])
    
    headers = ["Token", "Next Token", "Predicted", "Correct Prob", "Loss"]
    print("\nPrediction Analysis:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser(description='Visualize residual stream activations in GPT-2')
    parser.add_argument('--sentence', type=str, default=SENTENCE,
                        help='Input sentence to analyze')
    parser.add_argument('--use_openai', action='store_true',
                        help='Use OpenAI GPT-2 weights instead of local checkpoint')
    args = parser.parse_args()
    
    # Load model and tokenizer
    model = load_gpt2_model(use_openai=args.use_openai or USE_OPENAI_WEIGHTS)
    tokenizer = get_tokenizer()
    
    # Generate input sequence based on configuration
    if REP_SEQUENCE:
        input_ids = generate_repeated_sequence(tokenizer, REP_SEQUENCE)
        print(f"\nGenerated repeated sequence of length {REP_SEQUENCE * 2}:")
        # Print the tokens in the sequence
        tokens_text = [tokenizer.decode([t.item()]) for t in input_ids[0]]
        first_half = ' '.join(tokens_text[:REP_SEQUENCE])
        second_half = ' '.join(tokens_text[REP_SEQUENCE:])
        print(f"First half:  {first_half}")
        print(f"Second half: {second_half}\n")
    else:
        input_ids = tokenizer.encode(args.sentence, return_tensors='pt').to(device)
    
    # Extract residual activations and get predictions
    residual_activations, logits, variance_explained = extract_residual_activations(model, input_ids)
    
    # Plot activations
    plot_residual_activations(residual_activations, input_ids, tokenizer, args.sentence, variance_explained)
    
    # Analyze and display predictions
    analyze_predictions(model, input_ids, tokenizer)
    
    print(f"\nVisualizations saved as 'residual_activations.png' and 'residual_activations.svg'")

if __name__ == "__main__":
    main() 