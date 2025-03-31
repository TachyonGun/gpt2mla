import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_mla import GPT_MLA, MLAConfig
import argparse

def projection_frobenius_distance(W1: torch.Tensor, W2: torch.Tensor) -> float:
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
    W1_norm = W1 / torch.norm(W1, dim=1, keepdim=True)
    W2_norm = W2 / torch.norm(W2, dim=1, keepdim=True)
    
    # Compute projection matrices
    P1 = W1_norm @ W1_norm.T
    P2 = W2_norm @ W2_norm.T
    
    # Compute Frobenius norm of difference
    diff_norm = torch.norm(P1 - P2, p='fro')
    
    # Convert to similarity score (1 - normalized distance)
    # Note: Maximum Frobenius norm of difference between two projection matrices is sqrt(2)
    similarity = 1 - (diff_norm / np.sqrt(2))
    
    return similarity.item()

def angular_subspace_similarity(W1: torch.Tensor, W2: torch.Tensor) -> float:
    """
    Compute the similarity between two subspaces using principal angles.
    
    Args:
        W1: First weight matrix of shape (m1, n)
        W2: Second weight matrix of shape (m2, n)
    
    Returns:
        float: A similarity score between 0 and 1, where 1 means identical subspaces
    """
    # Step 1: Orthonormalize the matrices using QR decomposition
    Q1, _ = torch.linalg.qr(W1.T)  # Transpose because we want column space
    Q2, _ = torch.linalg.qr(W2.T)
    
    # Step 2: Compute matrix of cosines of principal angles
    C = Q1.T @ Q2
    
    # Step 3: Compute singular values (cosines of principal angles)
    cosines = torch.linalg.svd(C)[1]
    
    # Step 4: Compute average of squared cosines
    k = min(W1.shape[0], W2.shape[0])  # min of ranks
    similarity = (cosines[:k]**2).mean().item()
    
    return similarity

def extract_latent_weights(model: GPT_MLA) -> list[torch.Tensor]:
    """
    Extract the latent encoder weight matrices from each transformer block.
    
    Args:
        model: The GPT_MLA model
        
    Returns:
        list of weight matrices, one for each layer
    """
    weights = []
    for block in model.transformer.h:
        # Extract the weight matrix from the latent encoder
        # Shape will be (n_latent, n_embd)
        W = block.attn.latent_enc.weight.detach()
        weights.append(W)
    return weights

def compute_similarity_matrix(weights: list[torch.Tensor], 
                            method: str = "frobenius") -> np.ndarray:
    """
    Compute pairwise similarities between all layers' weight matrices.
    
    Args:
        weights: List of weight matrices
        method: Similarity computation method ("frobenius" or "angular")
        
    Returns:
        n_layers x n_layers numpy array of similarity scores
    """
    n_layers = len(weights)
    similarity_matrix = np.zeros((n_layers, n_layers))
    
    similarity_fn = angular_subspace_similarity if method == "angular" else projection_frobenius_distance
    
    for i in range(n_layers):
        for j in range(n_layers):
            similarity_matrix[i, j] = similarity_fn(weights[i], weights[j])
    
    return similarity_matrix

def plot_similarity_heatmap(similarity_matrix: np.ndarray, method: str, save_path: str = None):
    """
    Create a heatmap visualization of the similarity matrix.
    
    Args:
        similarity_matrix: n_layers x n_layers numpy array of similarity scores
        method: The method used to compute similarities
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='viridis',
                xticklabels=[f'Layer {i}' for i in range(len(similarity_matrix))],
                yticklabels=[f'Layer {i}' for i in range(len(similarity_matrix))])
    plt.title(f'Latent Encoder Subspace Similarities ({method.capitalize()} Method)')
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze latent subspace similarities')
    parser.add_argument('--method', type=str, choices=['frobenius', 'angular'],
                      default='frobenius', help='Method to compute subspace similarities')
    args = parser.parse_args()

    # Load the MLA model from checkpoint
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
    model.to(device)
    model.eval()
    
    # Extract weights and compute similarities
    weights = extract_latent_weights(model)
    similarity_matrix = compute_similarity_matrix(weights, method=args.method)
    
    # Plot and save results
    plot_similarity_heatmap(similarity_matrix, args.method, f'latent_similarities_{args.method}.png')
    
    # Print some statistics
    print(f"\nResults using {args.method} method:")
    print("Average similarity between adjacent layers:", 
          np.mean([similarity_matrix[i,i+1] for i in range(len(weights)-1)]))
    print("Average similarity between all layers:", 
          np.mean(similarity_matrix[~np.eye(len(weights), dtype=bool)]))

if __name__ == "__main__":
    main() 