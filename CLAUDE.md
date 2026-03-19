# Parameter Golf - Research Experiments

> **USE `run_experiment.py` FOR MAC EXPERIMENTS, NOT `train_gpt_mlx.py`.**
> The stock `train_gpt_mlx.py` will OOM during validation on this 36GB Mac (it loads all 62M val tokens at once without `mx.eval()` per batch). `run_experiment.py` has memory-safe chunked validation. See "OOM Lessons" section below.
> Also: this is a Mac with Apple MLX — there is NO CUDA. Do not look for or try to use PyTorch/CUDA.

## What This Is
OpenAI Parameter Golf challenge: train the best language model in a 16MB artifact, measured by bits per byte (BPB) on FineWeb validation set. Lower BPB = better.

**Strategy**: Use the 36GB Mac Pro for fast local iteration — tune architecture and hyperparameters with short MLX runs. Once we find winning configs, port to 8xH100 for the scored 10-min track submission. The algorithm/architecture is what we're optimizing — not MLX-specific tricks.

## Running an Experiment

When you receive a research experiment with a `config.json`, follow these steps:

### 1. Read the config
The `config.json` in your experiment directory contains hyperparameter values (the "genome").

### 2. Run training
**Local Mac training uses `train_gpt_mlx.py` (Apple MLX framework — no CUDA).** The CUDA version is `train_gpt.py` (for H100 submissions only). Do NOT look for or try to use CUDA/PyTorch locally — this Mac has no NVIDIA GPU.

Execute `train_gpt_mlx.py` from the repo root (`/Users/tbowyer/parameter-golf`) with the genome values as environment variables.

Map config keys to env vars (all uppercase):
- `num_layers` → `NUM_LAYERS`
- `model_dim` → `MODEL_DIM`
- `num_heads` → `NUM_HEADS` (derive as `model_dim / 64` if not in config)
- `num_kv_heads` → `NUM_KV_HEADS`
- `mlp_mult` → `MLP_MULT`
- `train_seq_len` → `TRAIN_SEQ_LEN`
- `matrix_lr` → `MATRIX_LR`
- `tied_embed_lr` → `TIED_EMBED_LR`
- `scalar_lr` → `SCALAR_LR`
- `muon_momentum` → `MUON_MOMENTUM`
- `warmup_steps` → `WARMUP_STEPS`
- `warmdown_iters` → `WARMDOWN_ITERS`
- `rope_base` → `ROPE_BASE`
- `logit_softcap` → `LOGIT_SOFTCAP`

Always set (fixed, not genome parameters):
- `DATA_PATH=/Users/tbowyer/parameter-golf/data/datasets/fineweb10B_sp1024`
- `TOKENIZER_PATH=/Users/tbowyer/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`
- `VAL_LOSS_EVERY=100` (validate periodically)
- `TRAIN_LOG_EVERY=50`

Memory management (36GB M3 Max, ~29GB available after OS/apps):
- `MLX_MAX_MICROBATCH_TOKENS=1024` (safe default — the script default of 8192 will OOM on larger configs)
- `GRAD_ACCUM_STEPS=32` (more accumulation = less peak memory; script default of 8 is too aggressive)
- `TRAIN_BATCH_TOKENS=524288` (total batch size, unchanged)
- `VAL_BATCH_SIZE=131072` (smaller val batches to avoid OOM during validation)
- If OOM still occurs: try `MLX_MAX_MICROBATCH_TOKENS=512` first, then reduce `GRAD_ACCUM_STEPS=64`
- Configs at the upper genome range (dim=576, layers=11, mlp_mult=3) use ~2x memory vs baseline — always use conservative settings for these

### OOM Lessons from gen-6 (2026-03-18)
The gen-6 experiment (30.7M params, dim=576, layers=11, mlp_mult=3) OOM'd **5 times** before succeeding. Key findings:

1. **Validation is the #1 OOM killer, not training.** `train_gpt_mlx.py` loads ALL 62M validation tokens into memory at once, then runs inference without calling `mx.eval()` per batch. The MLX computation graph accumulates unboundedly and OOMs.
2. **`mx.compile()` hangs on large models.** The first `run_experiment.py` attempt used `mx.compile()` and hung for 20+ minutes with no output. Fix: use uncompiled functions for Mac runs.
3. **Reducing `VAL_BATCH_SIZE` alone doesn't fix it.** Even `VAL_BATCH_SIZE=4096` OOM'd because the stock validation loop still loads all val tokens and doesn't force evaluation per batch.
4. **What actually worked:** `run_experiment.py` with:
   - Uncompiled loss/grad functions (no `mx.compile()`)
   - Per-shard validation: load one val shard at a time, process in small chunks, call `mx.eval()` after each batch
   - Skip warmup to save memory
   - `TRAIN_BATCH_TOKENS=8192`, `GRAD_ACCUM_STEPS=1` for training
5. **Use `python3 -u` for unbuffered output** — otherwise stdout is buffered and you can't see progress until the process dies.
6. **numpy architecture mismatch**: system Python 3.9 may have x86_64 numpy on arm64 Mac. Fix: `python3 -m pip install --force-reinstall numpy`

**Use `run_experiment.py` instead of `train_gpt_mlx.py` for Mac experiments** — it handles memory-safe validation automatically.

Iteration counts by phase:
- `ITERATIONS=500` — phase 1 architecture sweeps (quick signal)
- `ITERATIONS=5000` — phase 2 optimizer tuning
- `ITERATIONS=20000+` — phase 3 final convergence runs

### 3. Parse results
After training completes, parse the log output for the final validation BPB. Look for lines like:
```
step=500 | val_loss=X.XXXX | val_bpb=X.XXXX
```

### 4. Write result.json
Write `result.json` to your experiment directory:
```json
{
  "val_bpb": 1.2345,
  "iterations_completed": 500,
  "notes": "any observations about training"
}
```

### 5. Train log
`run_experiment.py` automatically tees all stdout/stderr to `train.log` in the experiment directory. Each experiment should have three files:
- `config.json` — the genome/hyperparameters
- `train.log` — full training output (loss curves, validation results, errors)
- `result.json` — parsed final metrics

## Genome Search Strategy

### Phase 1 — Architecture sweep (near baseline)
**IMPORTANT: Not all combinations fit in 16MB.** Always estimate params before training (see Constraints). The genome ranges below include oversized combos — you MUST reject any config where estimated params > 15.3M.

Genome parameters:
- `num_layers`: 8–11
- `model_dim`: 384–576 (step 64)
- `num_kv_heads`: 1, 2, or 4
- `mlp_mult`: 1–3 (but mlp_mult=3 only fits with dim≤448)
- `matrix_lr`: 0.02–0.07 (log scale)
- `tied_embed_lr`: 0.03–0.08 (log scale)
- `scalar_lr`: 0.02–0.07 (log scale)
- `muon_momentum`: 0.92–0.97
- `warmup_steps`: 10–30
- `warmdown_iters`: 800–1600

Fixed during phase 1:
- `num_heads` = `model_dim / 64` (keeps head_dim=64)
- `train_seq_len` = 1024
- `rope_base` = 10000
- `logit_softcap` = 30

### Phase 2 — Optimizer tuning on best architectures
Widen LR ranges, sweep `warmdown_iters` up to 10000, increase iterations to 5000.

### Phase 3 — Convergence and exotic knobs
Add `rope_base`, `logit_softcap`, `train_seq_len` back into the genome. Run 20000+ iterations.

## Experiment Runner
`run_experiment.py` — standalone script that reads a config, runs training, and writes results. Does NOT pre-check artifact size, so agents must validate size constraints before invoking it.

## Experiment Log
- **gen-6 / f3789760**: layers=11, dim=576, mlp_mult=3, kv_heads=1 → 30.7M params (2x over 16MB limit). val_bpb=2.52 at 100 steps. **Rejected** — way too large. Lesson: the genome ranges allow oversized configs; always estimate params first.

## Baseline Reference
- Architecture: 9 layers, dim 512, 8 heads, 4 KV heads, mlp_mult 2
- LRs: matrix=0.04, scalar=0.04, tied_embed=0.05
- Optimizer: muon_momentum=0.95, warmup=20, warmdown=1200
- Result: val_bpb=1.2244 (10 min on 8xH100), 15,863,489 bytes artifact

## Important Constraints
- The final artifact (code + int8-quantized weights + zlib) must fit in 16MB (16,000,000 bytes)
- `num_heads` must evenly divide `model_dim`
- `num_kv_heads` must evenly divide `num_heads`
- Larger `model_dim` x `num_layers` = more parameters = risk exceeding 16MB
- The baseline (9 layers, dim 512) fits in ~15.8MB — stay close to this budget
- Keep `train_batch_tokens / grad_accum_steps` divisible by `train_seq_len`
- **MUST reject configs before training if estimated size > 15.5MB post-quantization**
- Quick size estimate: `params ≈ vocab_size*model_dim + num_layers*(4*model_dim² + 2*mlp_mult*model_dim² + 3*model_dim) + model_dim`. Int8 = ~1 byte/param. Add ~200KB for code/overhead. If params > 15.3M, the config won't fit.
- Example: 11 layers, dim 576, mlp_mult 3 → ~30.7M params → ~30MB artifact → REJECTED (2x over budget)
- Safe combos that fit 16MB: (9, 512), (10, 448), (11, 384–448), (8, 512–576 with mlp_mult≤2)
