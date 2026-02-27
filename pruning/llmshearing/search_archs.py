import os
import sys; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  
from count_params import count_params

if __name__ == "__main__":
    
    
    for n_kv_heads in range(4, 8 + 1, 1):
        for d_model in range(2048, 4096 + 1, 128):
            max_mlp = -1
            for mlp_size in range(4096, 14336 + 1, 128):
                
                if mlp_size < 2 * d_model or mlp_size > 3.5 * d_model:
                    continue
                
                params = count_params(
                    d_model=d_model,
                    num_heads=4*n_kv_heads,
                    num_layers=32,
                    vocab_size=128256,
                    intermediate_size=mlp_size,
                    num_kv_heads=n_kv_heads,
                    head_dim=128,
                )
                
                if 4 < params / 1e9 < 4.5:
                    max_mlp = mlp_size
                
            if d_model == 3072 and max_mlp > 0:
                print(d_model, n_kv_heads, max_mlp)