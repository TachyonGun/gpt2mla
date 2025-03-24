import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import GPT, GPTConfig
from model_mla import GPT_MLA, MLAConfig
import argparse

def projection_frobenius_distance(W1: np.ndarray, W2: np.ndarray) -> float:
    """
    Compute the projection Frobenius distance between two matrices.
    This measures how similar the subspaces spanned by the rows of W1 and W2 are.
    
    Args:
        W1: First weight matrix of shape (m1, n)
        W2: Second weight matrix of shape (m2, n)
    
    Returns:
        float: A similarity score between 0 and 1, where 1 means identical subspaces
    """
    # Normalize the matrices along rows
    W1_norm = W1 / np.linalg.norm(W1, axis=1, keepdims=True)
    W2_norm = W2 / np.linalg.norm(W2, axis=1, keepdims=True)
    
    # Compute projection matrices
    P1 = W1_norm @ W1_norm.T
    P2 = W2_norm @ W2_norm.T
    
    # Compute Frobenius norm of difference
    diff_norm = np.linalg.norm(P1 - P2, ord='fro')
    
    # Convert to similarity score (1 - normalized distance)
    # Note: Maximum Frobenius norm of difference between two projection matrices is sqrt(2)
    similarity = 1 - (diff_norm / np.sqrt(2))
    
    return float(similarity)

def angular_subspace_similarity(W1: np.ndarray, W2: np.ndarray) -> float:
    """
    Compute the similarity between two subspaces using principal angles.
    
    Args:
        W1: First weight matrix of shape (m1, n)
        W2: Second weight matrix of shape (m2, n)
    
    Returns:
        float: A similarity score between 0 and 1, where 1 means identical subspaces
    """
    # Step 1: Orthonormalize the matrices using QR decomposition
    Q1, _ = np.linalg.qr(W1.T)  # Transpose because we want column space
    Q2, _ = np.linalg.qr(W2.T)
    
    # Step 2: Compute matrix of cosines of principal angles
    C = Q1.T @ Q2
    
    # Step 3: Compute singular values (cosines of principal angles)
    cosines = np.linalg.svd(C, compute_uv=False)
    
    # Step 4: Compute average of squared cosines
    k = min(W1.shape[0], W2.shape[0])  # min of ranks
    similarity = (cosines[:k]**2).mean()
    
    return float(similarity)

def extract_residual_weights(model, component: str = "attn") -> list[np.ndarray]:
    """
    Extract the residual output projection weights from each transformer block.
    
    Args:
        model: The GPT or GPT_MLA model
        component: Which component to analyze ("attn" or "mlp")
        
    Returns:
        list of weight matrices, one for each layer
    """
    weights = []
    for block in model.transformer.h:
        if component == "attn":
            # Extract attention output projection weights
            W = block.attn.c_proj.weight.detach().cpu().numpy()
        else:  # mlp
            # Extract MLP output projection weights
            W = block.mlp.c_proj.weight.detach().cpu().numpy()
        weights.append(W)
        
        # Clear CUDA cache after each extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return weights

def compute_similarity_matrix(weights: list[np.ndarray], 
                            method: str = "frobenius",
                            transpose: bool = False) -> np.ndarray:
    """
    Compute pairwise similarities between all layers' weight matrices.
    
    Args:
        weights: List of weight matrices
        method: Similarity computation method ("frobenius" or "angular")
        transpose: Whether to transpose matrices before computing similarity
        
    Returns:
        n_layers x n_layers numpy array of similarity scores
    """
    n_layers = len(weights)
    similarity_matrix = np.zeros((n_layers, n_layers))
    
    similarity_fn = angular_subspace_similarity if method == "angular" else projection_frobenius_distance
    
    for i in range(n_layers):
        for j in range(n_layers):
            W1 = weights[i].T if transpose else weights[i]
            W2 = weights[j].T if transpose else weights[j]
            similarity_matrix[i, j] = similarity_fn(W1, W2)
    
    return similarity_matrix

def plot_similarity_heatmap(similarity_matrix: np.ndarray, 
                          method: str, 
                          model_name: str,
                          component: str,
                          save_path: str = None):
    """
    Create a heatmap visualization of the similarity matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='viridis',
                xticklabels=[f'Layer {i}' for i in range(len(similarity_matrix))],
                yticklabels=[f'Layer {i}' for i in range(len(similarity_matrix))])
    plt.title(f'{component.upper()} Output Projection Similarities\n{model_name} ({method.capitalize()} Method)')
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def load_gpt_model(use_openai_weights=False):
    """Load GPT model either from local checkpoint or OpenAI weights"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_openai_weights:
        print("Loading GPT-2 from OpenAI weights...")
        model = GPT.from_pretrained('gpt2', override_args=dict(dropout=0.0))
    else:
        print("Loading GPT-2 from local checkpoint...")
        ckpt_path = os.path.join('out', 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        
        # Fix the state dict if it was saved from a DDP model
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
    
    return model.to(device)

def load_mla_model():
    """Load MLA model from checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join('out_mla', 'ckpt_mla.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    config = MLAConfig(**checkpoint['model_args'])
    model = GPT_MLA(config)
    
    # Fix the state dict if it was saved from DDP
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    return model.to(device)

def analyze_model(model, model_name: str, args):
    """Analyze a single model's residual output projections"""
    print(f"\nAnalyzing {model_name}...")
    
    # Extract weights and compute similarities
    weights = extract_residual_weights(model, args.attn_or_mlp)
    similarity_matrix = compute_similarity_matrix(weights, 
                                               method=args.method,
                                               transpose=args.transpose)
    
    # Plot and save results
    save_path = f'residout_similarities_{model_name}_{args.attn_or_mlp}_{args.method}.png'
    plot_similarity_heatmap(similarity_matrix, 
                          args.method, 
                          model_name,
                          args.attn_or_mlp,
                          save_path)
    
    # Print statistics
    print(f"\nResults for {model_name} using {args.method} method:")
    print("Average similarity between adjacent layers:", 
          np.mean([similarity_matrix[i,i+1] for i in range(len(weights)-1)]))
    print("Average similarity between all layers:", 
          np.mean(similarity_matrix[~np.eye(len(weights), dtype=bool)]))
    
    # Clear some memory
    del weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return similarity_matrix

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze residual output projection subspaces')
    parser.add_argument('--method', type=str, choices=['frobenius', 'angular'],
                      default='frobenius', help='Method to compute subspace similarities')
    parser.add_argument('--transpose', action='store_true',
                      help='Transpose matrices before computing similarities')
    parser.add_argument('--attn_or_mlp', type=str, choices=['attn', 'mlp'],
                      default='attn', help='Which component to analyze')
    parser.add_argument('--openai_weights', action='store_true',
                      help='Use OpenAI pretrained weights for GPT-2')
    args = parser.parse_args()

    # Analyze GPT-2
    gpt2_model = load_gpt_model(args.openai_weights)
    gpt2_similarities = analyze_model(gpt2_model, "GPT-2", args)
    del gpt2_model  # Free up memory
    
    # Analyze MLA
    mla_model = load_mla_model()
    mla_similarities = analyze_model(mla_model, "MLA", args)
    del mla_model  # Free up memory

if __name__ == "__main__":
    main() 