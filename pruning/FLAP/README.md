# FLAP \& Wanda-sp

This is the official implementation of FLAP & Wanda-sp by An et al, with minimal modifications to make the code compatible with GQA. For the original README, please see [here](ORIGINAL_README.md).

## Installation
```bash
bash scripts/install.sh
```

## Pruning
```bash
bash scripts/prune_llama_7b.sh
bash scripts/prune_llama3.1_8b.sh
```

## Results
Notably, the sequence length of eval data significantly affects the perplexity. Generally, if the sequence length is identical with the model's `max_position_embeddings` (4096 for Llama-2-7B, 8192 for Llama-3.1-8B), the perplexity will be much lower. However, in this section we follow the original setups in the paper and set sequence length to 128.

Base Model=Llama-7B
| Method | Ratio | WikiText2(Tested) | WikiText2(Paper) |
|:--:|:--:|:--:|:--:|
| - | - | 12.62 | 12.62 |
| FLAP | 0.2 | 15.05 | 14.62 |
| FLAP | 0.3 | 18.28 | 17.62 |
| FLAP | 0.5 | 33.00 | 31.80 |

Base Model=Llama3.1-8B
| Method | Ratio | WikiText2(Tested) | WikiText2(Paper) | 
|:--:|:--:|:--:|:--:|
| - | - | 14.31 | |
| FLAP | 0.2 | 19.32 | - |
| FLAP | 0.3 | 24.50 | - |
| FLAP | 0.5 | 42.86 | - |



