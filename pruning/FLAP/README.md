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
Base Model=Llama-7B
| Method | Ratio | WikiText2 |
| - | - | 12.62 |
| FLAP | 0.2 | 16.14 |
| FLAP | 0.3 |  |
| FLAP | 0.5 |  |



