import os
import torch
import tiktoken
from collections import Counter
from model_mla import GPT_MLA, MLAConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_mla_model(ckpt_path):
    """Load the MLA model from checkpoint"""
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
    return mla_model

def analyze_latent_directions(model):
    """Analyze alignment between latent encoder directions and vocabulary tokens"""
    # Get the unembedding matrix (lm_head weights)
    W_E = model.lm_head.weight.detach()  # [vocab_size, n_embd]
    
    # Normalize the unembedding matrix
    W_E = W_E / W_E.norm(dim=1, keepdim=True)
    
    # Setup tokenizer for decoding
    enc = tiktoken.get_encoding("gpt2")
    decode = lambda l: enc.decode(l)
    
    # Store layer summaries for final output
    layer_summaries = []
    
    # For each layer
    for layer_idx, block in enumerate(model.transformer.h):
        print(f"\n{'='*50}")
        print(f"Analyzing Layer {layer_idx}")
        print('='*50)
        
        # Get the latent encoder weight matrix
        W_l = block.attn.latent_enc.weight.detach()  # [n_latent, n_embd]
        
        # Normalize the latent encoder matrix
        W_l = W_l / W_l.norm(dim=1, keepdim=True)
        
        # Compute alignment scores
        # W_l: [n_latent, n_embd]
        # W_E: [vocab_size, n_embd]
        # scores: [n_latent, vocab_size]
        alignment_scores = torch.matmul(W_l, W_E.t())
        
        # Get top 10 aligned tokens for each latent direction
        top_k = 20
        top_scores, top_indices = torch.topk(alignment_scores, k=top_k, dim=1)
        
        # Collect all top tokens for layer-wise summary
        all_top_tokens = []
        
        # Print results for this layer
        print("\nPer-direction analysis:")
        print('-'*30)
        for latent_idx in range(W_l.shape[0]):
            print(f"\nLatent direction {latent_idx}:")
            tokens = top_indices[latent_idx].tolist()
            scores = top_scores[latent_idx].tolist()
            
            # Add tokens to the layer collection
            all_top_tokens.extend(tokens)
            
            # Print top aligned tokens and their scores
            for token_idx, score in zip(tokens, scores):
                token_str = decode([token_idx])
                print(f"Token: {token_str!r}, Score: {score:.3f}, Index: {token_idx}")
        
        # Store layer summary
        token_counter = Counter(all_top_tokens)
        most_common_tokens = token_counter.most_common(15)  # Show top 15 most common tokens
        layer_summaries.append((layer_idx, most_common_tokens))
    
    # Print all layer summaries together
    print("\n\n" + "="*70)
    print("LAYER-WISE SUMMARIES OF MOST COMMON ALIGNED TOKENS")
    print("="*70)
    
    for layer_idx, most_common_tokens in layer_summaries:
        print(f"\nLayer {layer_idx}:")
        print('-'*30)
        print("Most common aligned tokens across all directions:")
        for token_idx, count in most_common_tokens:
            token_str = decode([token_idx])
            print(f"Token: {token_str!r} (Index: {token_idx}) - Appears in {count} directions")

def main():
    # Load MLA model from checkpoint
    ckpt_path = os.path.join('out_mla', 'ckpt_mla.pt')
    print("Loading MLA model...")
    model = load_mla_model(ckpt_path)
    
    print("\nAnalyzing latent directions...")
    with torch.no_grad():
        analyze_latent_directions(model)

if __name__ == "__main__":
    main() 