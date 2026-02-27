#!/usr/bin/env python3
"""
Simple function to count transformer LLM parameters.
Supports MHA and GQA (Grouped Query Attention).
"""

def count_params(
    d_model: int,
    num_heads: int,
    num_layers: int,
    vocab_size: int,
    intermediate_size: int,
    num_kv_heads: int = None,
    head_dim: int = None,
) -> int:
    """
    Count parameters in a transformer LLM.
    
    Args:
        d_model: Hidden dimension
        num_heads: Number of query heads
        num_layers: Number of transformer layers
        vocab_size: Vocabulary size
        intermediate_size: FFN intermediate dimension
        num_kv_heads: Number of KV heads (defaults to num_heads for MHA)
    
    Returns:
        Total parameter count
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads
        
    head_dim = head_dim or (d_model // num_heads)
    
    # Per layer
    attn_params = (
        d_model * d_model +                          # Q proj
        d_model * (num_kv_heads * head_dim) +        # K proj
        d_model * (num_kv_heads * head_dim) +        # V proj
        d_model * d_model                            # O proj
    )
    ffn_params = 3 * d_model * intermediate_size    # gate + up + down
    ln_params = 2 * d_model                         # 2 layer norms
    
    per_layer = attn_params + ffn_params + ln_params
    
    # Total
    return (
        2 * vocab_size * d_model +           # embeddings (tied with output)
        per_layer * num_layers +         # transformer blocks
        d_model                          # final layer norm
    )


# Quick test
if __name__ == '__main__':
    # Llama2-7B
    p1 = count_params(d_model=4096, num_heads=32, num_layers=32, vocab_size=32000, intermediate_size=11008)
    print(f"Llama2-7B:       {p1/1e9:.2f}B params ({p1:,})")
    
    # Llama3-8B (GQA)
    p2 = count_params(d_model=4096, num_heads=32, num_layers=32, vocab_size=128256, intermediate_size=14336, num_kv_heads=8)
    print(f"Llama3-8B (GQA): {p2/1e9:.2f}B params ({p2:,})")
    
    p3 = count_params(d_model=3072, num_heads=24, num_layers=28, vocab_size=128256, intermediate_size=8192, num_kv_heads=8, head_dim=128)
    print(f"Llama3-3B (GQA): {p3/1e9:.2f}B params ({p3:,})")
    
    p4 = count_params(d_model=2048, num_heads=32, num_kv_heads=8, head_dim=64, num_layers=16, vocab_size=128256, intermediate_size=8192 )
    print(f"Llama3-1B (GQA): {p4/1e9:.2f}B params ({p4:,})")
