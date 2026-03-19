#!/usr/bin/env python3
"""Run a parameter-golf experiment with MLX, with memory-safe validation."""
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

# Ensure we can import from the worktree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Import model components from train_gpt_mlx
from train_gpt_mlx import (
    GPT,
    Hyperparameters,
    SplitOptimizers,
    TokenLoader,
    build_sentencepiece_luts,
    load_data_shard,
)


def eval_val_chunked(
    args,
    compiled_loss,
    val_pattern: str,
    seq_len: int,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    max_val_tokens: int = 65536,
) -> tuple[float, float]:
    """Memory-safe validation that processes one shard at a time."""
    val_files = sorted(glob.glob(val_pattern))
    if not val_files:
        raise FileNotFoundError(f"No val files for {val_pattern}")

    total_loss = 0.0
    total_tokens = 0.0
    total_bytes = 0.0

    for vf in val_files:
        shard_tokens = load_data_shard(Path(vf))
        usable = ((shard_tokens.size - 1) // seq_len) * seq_len
        if usable <= 0:
            continue
        shard_tokens = shard_tokens[: usable + 1]

        # Process in small chunks
        batch_seqs = max(max_val_tokens // seq_len, 1)
        total_seqs = usable // seq_len

        for batch_start in range(0, total_seqs, batch_seqs):
            batch_end = min(batch_start + batch_seqs, total_seqs)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            chunk = shard_tokens[raw_start:raw_end]

            x_np = chunk[:-1].reshape(-1, seq_len)
            y_np = chunk[1:].reshape(-1, seq_len)
            x = mx.array(x_np, dtype=mx.int32)
            y = mx.array(y_np, dtype=mx.int32)

            chunk_count = float(y.size)
            loss_val = compiled_loss(x, y)
            mx.eval(loss_val)
            total_loss += float(loss_val.item()) * chunk_count
            total_tokens += chunk_count

            prev_ids = x_np.reshape(-1)
            tgt_ids = y_np.reshape(-1)
            bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
            bytes_np += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).astype(np.int16, copy=False)
            total_bytes += float(bytes_np.astype(np.float64).sum())

    if total_tokens == 0:
        raise ValueError("No validation tokens processed")

    val_loss = total_loss / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


class TeeStream:
    """Write to both a stream and a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def main():
    # Load experiment config
    exp_dir = os.environ.get("EXP_DIR")
    if not exp_dir:
        print("EXP_DIR env var required")
        sys.exit(1)

    # Tee all output to train.log in the experiment directory
    log_path = os.path.join(exp_dir, "train.log")
    log_file = open(log_path, "w")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    print(f"Config: {json.dumps(config, indent=2)}")

    args = Hyperparameters()
    mx.random.seed(args.seed)

    # Setup tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    # Build model
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    opt = SplitOptimizers(model, args)

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    print(f"Model params: {n_params}")

    # Use uncompiled functions to avoid MLX compile graph memory issues on Mac
    def loss_fn(x, y):
        return model.loss(x, y)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    compiled_loss = loss_fn
    compiled_loss_and_grad = loss_and_grad_fn

    # Train loader
    train_loader = TokenLoader(args.train_files)

    # Skip warmup on Mac to save memory - go straight to training
    print(f"Skipping warmup (Mac memory constraints)")

    # Training loop
    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step = None
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )
        if last_step:
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            if stop_after_step is not None and step < args.iterations:
                print(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        # Simple single-batch training step (no grad accum for memory)
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len)
        train_loss, grads = loss_and_grad_fn(x, y)
        mx.eval(train_loss, grads)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        step += 1

        if step <= 10 or step % 10 == 0:
            print(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_time_ms:.0f}ms step_avg:{approx_time_ms / step:.2f}ms"
            )

        if max_wallclock_ms is not None and stop_after_step is None and approx_time_ms >= max_wallclock_ms:
            stop_after_step = step

    print(f"Training done. Total train_time: {train_time_ms:.0f}ms, steps: {step}")

    # Validation - memory safe
    print("Running validation...")
    val_loss, val_bpb = eval_val_chunked(
        args,
        compiled_loss,
        args.val_files,
        args.train_seq_len,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        max_val_tokens=4096,
    )
    print(f"val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    # Write results
    result = {
        "val_bpb": val_bpb,
        "val_loss": val_loss,
        "train_loss": train_loss_value,
        "steps": step,
        "train_time_ms": train_time_ms,
        "n_params": n_params,
        "config": config,
        "platform": "mlx_apple_m3_max",
        "note": "Reduced batch/iterations for Mac; relative ranking valid for architecture comparison",
    }

    result_path = os.path.join(exp_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {result_path}")
    print(f"val_bpb: {val_bpb}")


if __name__ == "__main__":
    main()
