#!/usr/bin/env python3
import pdb
import argparse
import math
import os
from tqdm import tqdm
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from streaming import StreamingDataset
except Exception as e:
    raise RuntimeError(
        "Failed to import StreamingDataset from `streaming`. "
        "Please install the `streaming` package in your environment."
    ) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-domain LM loss on tokenized RedPajama MDS shards."
    )
    parser.add_argument("--model_path", type=str, required=True, help="HF model path")
    parser.add_argument(
        "--source_tokenizer_path",
        type=str,
        default="/scratch/yx3038/cache/Llama-3.1-8B-Instruct",
        help="Tokenizer used by the tokenized dataset (Llama-3.1).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/yx3038/pruning/llm-pruning-collection-fork-llmshearing/pruning/llmshearing/llmshearing/data/redpajama/for_prune_llama3",
        help="Root folder containing domain subfolders (each with index.json + shards)",
    )
    parser.add_argument("--seq_len", type=int, default=4096, help="Evaluation sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--max_samples_per_domain",
        type=int,
        default=0,
        help="If > 0, evaluate only first N samples per domain",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="",
        help="Comma-separated domains to evaluate (default: all subfolders with index.json)",
    )
    parser.add_argument(
        "--is_uint16",
        action="store_true",
        help="Interpret token bytes as uint16 (default: int64)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string, e.g. auto/cuda/cuda:0/cpu",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="none",
        help='Set "auto" to shard model across multiple GPUs; use "none" for single-device mode.',
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def pick_torch_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


def discover_domains(data_root: str) -> List[str]:
    domains = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "index.json")):
            domains.append(name)
    return domains


def decode_tokens(raw_bytes: bytes, is_uint16: bool, seq_len: int) -> torch.Tensor:
    if is_uint16:
        arr = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.int64)
    else:
        arr = np.frombuffer(raw_bytes, dtype=np.int64)
    arr = arr[:seq_len].copy()
    return torch.from_numpy(arr)


def is_llama31_model_name(model_path: str) -> bool:
    s = model_path.lower()
    return ("llama3.1" in s) or ("llama-3.1" in s)


def get_input_device(model, fallback: torch.device) -> torch.device:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        candidate_keys = [
            "model.embed_tokens",
            "model.model.embed_tokens",
            "transformer.wte",
            "gpt_neox.embed_in",
            "model.decoder.embed_tokens",
        ]
        for key in candidate_keys:
            if key in model.hf_device_map:
                return torch.device(model.hf_device_map[key])
        return torch.device(next(iter(model.hf_device_map.values())))
    return fallback


@torch.no_grad()
def eval_domain(
    model,
    model_tokenizer,
    source_tokenizer,
    domain_path: str,
    seq_len: int,
    batch_size: int,
    max_samples_per_domain: int,
    is_uint16: bool,
    device: torch.device,
    pad_token_id: int,
) -> Dict[str, float]:
    ds = StreamingDataset(local=domain_path, shuffle=False)
    total_samples = len(ds)
    if max_samples_per_domain and max_samples_per_domain > 0:
        total_samples = min(total_samples, max_samples_per_domain)

    total_nll = 0.0
    total_tokens = 0
    input_device = get_input_device(model, device)

    batch_tokens: List[torch.Tensor] = []

    for i in tqdm(range(total_samples)):
        sample = ds[i]
        toks = decode_tokens(sample["tokens"], is_uint16=is_uint16, seq_len=seq_len)

        batch_tokens.append(toks)

        if len(batch_tokens) < batch_size and i != total_samples - 1:
            continue

        max_len = max(t.shape[0] for t in batch_tokens)
        # pdb.set_trace()
        input_ids = torch.full(
            (len(batch_tokens), max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        for j, t in enumerate(batch_tokens):
            input_ids[j, : t.shape[0]] = t

        labels = input_ids.clone()
        if pad_token_id is not None and pad_token_id >= 0:
            labels[labels == pad_token_id] = -100

        input_ids = input_ids.to(input_device)
        labels = labels.to(input_device)
        attention_mask = (input_ids != pad_token_id).long() if pad_token_id is not None and pad_token_id >= 0 else None

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].to(shift_logits.device).contiguous()

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        valid = shift_labels.view(-1) != -100

        total_nll += loss_per_token[valid].sum().item()
        total_tokens += valid.sum().item()

        batch_tokens = []

    avg_loss = total_nll / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return {
        "loss": avg_loss,
        "ppl": ppl,
        "tokens": int(total_tokens),
        "samples": int(total_samples),
    }

def main():
    args = parse_args()

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data_root does not exist: {args.data_root}")

    if args.domains.strip():
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    else:
        domains = discover_domains(args.data_root)

    if not domains:
        raise ValueError(f"No domains found under: {args.data_root}")

    torch_dtype = pick_torch_dtype(args.dtype)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=None if torch_dtype == "auto" else torch_dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        device_map=(args.device_map if args.device_map != "none" else None),
    )
    if args.device_map == "none":
        model.to(device)
    model.eval()

    model_tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    source_tokenizer = None

    pad_token_id = model.config.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.config.eos_token_id if model.config.eos_token_id is not None else 0
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0]

    print(f"model_path={args.model_path}")
    print(f"data_root={args.data_root}")
    print(f"seq_len={args.seq_len}, batch_size={args.batch_size}, is_uint16={args.is_uint16}")
    print(f"device={device}")
    print(f"device_map={args.device_map}")
    if hasattr(model, "hf_device_map"):
        print(f"hf_device_map modules: {len(model.hf_device_map)}")

    results = {}
    for domain in domains:
        domain_path = os.path.join(args.data_root, domain)
        if not os.path.isdir(domain_path):
            print(f"[skip] {domain}: folder not found")
            continue

        metrics = eval_domain(
            model=model,
            model_tokenizer=model_tokenizer,
            source_tokenizer=source_tokenizer,
            domain_path=domain_path,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_samples_per_domain=args.max_samples_per_domain,
            is_uint16=args.is_uint16,
            device=device,
            pad_token_id=pad_token_id,
        )
        results[domain] = metrics
        print(
            f"[{domain}] loss={metrics['loss']:.4f}, ppl={metrics['ppl']:.2f}, "
            f"samples={metrics['samples']}, tokens={metrics['tokens']}"
        )

    if results:
        avg_loss = sum(v["loss"] for v in results.values()) / len(results)
        print(f"[summary] mean_domain_loss={avg_loss:.4f}")


if __name__ == "__main__":
    main()
