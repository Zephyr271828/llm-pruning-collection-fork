# GQA (Grouped Query Attention) Support for Llama3

## Overview
This implementation adds support for Grouped Query Attention (GQA) to enable pruning of Llama3 models. GQA is an attention mechanism where multiple query heads share the same key-value heads, reducing memory and computation costs.

## Key Changes Made

### 1. **Modified Files**
- `models/composer_llama.py`: Core GQA implementation
- `configs/llama3/8b.yaml`: Example configuration for Llama3-8B

### 2. **Architecture Changes**

#### LlamaAttention Class
- Added `num_key_value_heads` parameter (defaults to `n_heads` for backward compatibility)
- Added `num_key_value_groups` to track the GQA ratio
- Modified K/V projection dimensions:
  - `wk`: `d_model → num_key_value_heads * head_dim`
  - `wv`: `d_model → num_key_value_heads * head_dim`
  - `wq`: `d_model → n_heads * head_dim` (unchanged)

#### Forward Pass
- K/V tensors are reshaped with `num_key_value_heads` instead of `n_heads`
- Added KV head expansion: each KV head is repeated `num_key_value_groups` times to match query heads
- Expansion happens after RoPE application for correctness

#### Pruning Logic
- Updated to maintain GQA ratio during pruning
- `num_key_value_heads` is updated proportionally when heads are pruned
- K/V projection dimensions are handled separately from Q projections

### 3. **Configuration**

Example for Llama3-8B:
```yaml
model:
  n_heads: 32
  num_key_value_heads: 8  # GQA ratio of 4:1
  # ... other params
```

**Important constraints:**
- `n_heads` must be divisible by `num_key_value_heads`
- When pruning, maintain the GQA ratio in the target model

### 4. **Compatibility**

- **Backward Compatible**: If `num_key_value_heads` is not specified, defaults to `n_heads` (standard MHA)
- **Llama2**: Will continue to work (uses MHA with `num_key_value_heads = n_heads`)
- **Llama3**: Requires explicit `num_key_value_heads` parameter

## Usage

### Configuration
```yaml
model:
  name: mosaic_llama3_8b
  d_model: 4096
  n_heads: 32
  num_key_value_heads: 8  # NEW: Enable GQA
  n_layers: 32
  intermediate_size: 14336
  vocab_size: 128256
  attn_impl: flash  # Recommended for GQA
  
  l0_module:
    target_model:
      n_heads: 20
      num_key_value_heads: 5  # Maintain 4:1 ratio
```

### Model Loading
```python
# The model will automatically detect and use GQA if num_key_value_heads is set
from llmshearing.models.composer_llama import ComposerMosaicLlama

model = ComposerMosaicLlama(cfg)
```

## Technical Details

### GQA Mechanism
1. **Projection**: Q uses all heads, K/V use fewer heads
2. **Expansion**: Each K/V head is repeated to match Q heads
3. **Attention**: Standard attention computation with expanded K/V

### Memory Savings
For Llama3-8B (32 query heads, 8 KV heads):
- K/V projection parameters: 4x reduction
- K/V cache during inference: 4x reduction
- Attention computation: Same (after expansion)

### Pruning Considerations
- Head pruning must maintain divisibility: `n_heads % num_key_value_heads == 0`
- The implementation automatically adjusts `num_key_value_heads` proportionally
- Example: 32→20 heads with 8→5 KV heads maintains 4:1 ratio

## Validation

To verify GQA is working correctly:
```python
# Check model configuration
print(f"Query heads: {model.model.transformer.blocks[0].attn.n_heads}")
print(f"KV heads: {model.model.transformer.blocks[0].attn.num_key_value_heads}")
print(f"GQA ratio: {model.model.transformer.blocks[0].attn.num_key_value_groups}")

# Check K/V projection dimensions
print(f"wq shape: {model.model.transformer.blocks[0].attn.wq.weight.shape}")
print(f"wk shape: {model.model.transformer.blocks[0].attn.wk.weight.shape}")
print(f"wv shape: {model.model.transformer.blocks[0].attn.wv.weight.shape}")
```

Expected output for Llama3-8B:
```
Query heads: 32
KV heads: 8
GQA ratio: 4
wq shape: torch.Size([4096, 4096])
wk shape: torch.Size([1024, 4096])  # 8 * 128 = 1024
wv shape: torch.Size([1024, 4096])
```

## Known Limitations

1. **Flash Attention**: Recommended for GQA. Standard attention works but may be slower
2. **Pruning**: Currently maintains fixed GQA ratio during pruning
3. **Mixed Precision**: Tested with bf16/fp16, should work with all supported dtypes

## References

- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Llama 3 Model Card](https://github.com/meta-llama/llama3)
