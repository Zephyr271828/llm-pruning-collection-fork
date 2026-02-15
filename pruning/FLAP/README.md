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
| FLAP | 0.2 | 16.14 | 15.05 |
| FLAP | 0.3 | 20.52 | 18.28 |
| FLAP | 0.5 | 52.72 | 33.00 |

Base Model=Llama3.1-8B
| Method | Ratio | WikiText2 |
| - | - | 14.31 |
| FLAP | 0.2 | 19.93 |
| FLAP | 0.3 | 26.50 |
| FLAP | 0.5 | 51.29 |



