# RX 7900 XTX (gfx1100) edition

This branch (`rx7900xtx`) is upstream Strata plus speedups measured on one AMD Radeon RX 7900 XTX. Every change keeps
the model's output **bit-identical**: teacher-forced KL against the engine without them is 0.0000, with the same top
token at every position. Nothing here trades quality for speed.

Measured on: RX 7900 XTX 24 GB (gfx1100), Ryzen 7 9800X3D, 62 GiB RAM, Linux 7.2, ROCm 10.2 (TheRock nightly
10.2.0a20260930), Qwen3.8-Flash-Next GSQ-RCO IQ3_S, 262,144-token context, no control vector.

## Results

`tools/strata_bench.py --long`, same day, same config, alternating rounds, medians of 3 runs per test.

| Test | Upstream main `a0116b27` | This branch | Change |
| --- | ---: | ---: | ---: |
| Short chat, writes answers | 86.8 tokens/s | 95.8 tokens/s | **+10%** |
| Short chat, first token | 1.12 s | 0.83 s | **-26%** |
| 4K prompt, reads | 1,069 tokens/s | 1,427 tokens/s | **+34%** |
| 16K prompt, reads | 1,381 tokens/s | 1,827 tokens/s | **+32%** |
| 32K prompt, reads | 1,421 tokens/s | 1,844 tokens/s | **+30%** |
| Agent session, first token per turn | 3.90 s | 2.92 s | **-25%** |
| Agent session, writes answers | 96.6 tokens/s | 107.0 tokens/s | **+11%** |

Reports: [upstream and PR #2231](../bench/results/2026-10-13-community-rx-7900-xtx-gfx1100-exact-speedups/),
[PR #2231 and this branch](../bench/results/2026-10-13-community-rx-7900-xtx-gfx1100-gu-exact/). In the prompt and
first-token rows every round of this branch beat every round of the engine it was compared with; decode moves a few
percent between runs (the drafts follow measured timings), so read its rows as approximate.

The config is a strict one: every switch that changes the output (KL > 0) is off, including upstream's own opt-in
ones (`STRATA_HC_Q8`, `STRATA_PF_GEMM`, `STRATA_PF_FUSED`, WMMA prompt attention). With those on, both engines are
faster and no longer identical to each other.

## What is in it

### 1. Decode and prompt-path kernels (upstream PR #2231)

[Niko1221/Strata#2231](https://github.com/Niko1221/Strata/pull/2231): launch shapes of latency-bound kernels (router,
GDN norm, hyper-connection up, QSA top-k, the drafter's row select), DPP lane exchanges, the multi-block argmax on HIP,
the PCIe experts' landed flag written by the copy stream, the adaptive tier's copies over both DMA engines, warp
dequant of Q6_K / IQ4_XS. On by default, each with an `=0` switch; plus opt-ins listed below.

### 2. The experts' gate/up product, exact and fast (`STRATA_PF_GU_EXACT=1`)

In a prompt about half the GPU time is the routed experts' gate/up product (FP16 x FP16 -> FP32, 1280 x 2560 per
expert). It has no hipBLASLt tuning, so it falls back to rocBLAS's generic Tensile kernels, which do not use the matrix
cores and reach 6-13 TFLOPS on this card. Any faster kernel adds the products in another order, which changes the
output a little (that is why `STRATA_PF_GEMM` has KL 0.0115).

This branch reproduces rocBLAS's order instead. Read from its kernels and checked bit for bit:

- **65 tokens or more per expert** (`MT64x32x8`, GSU 1): each output is one fused-multiply-add chain over the 2,560
  inputs, starting at input `((n / 64) % 16) * 128` (Tensile's StaggerU) and wrapping around.
- **2 to 64 tokens** (`MT128x32x16`, GSU 3): the inputs are split three ways in interleaved steps of 16 (split `g`
  takes steps `g, g+3, g+6, ...`), each split is one chain starting at its step `((n / 128) % 4) * 8`, and the result
  is `(p0 + p1) + p2`.
- **1 token** is a GEMV and stays on rocBLAS.

With the order fixed, the work is free to run faster:

- The prompt path collects 16 experts and computes them together (`src/prefill/gu_exact.cu`).
- From 65 tokens the products run on a **Tensile-generated kernel** (`src/prefill/gu_tensile_gfx1100.s`). Tensile
  tuned everything except the parameters that define the order (MacroTile0 64, GlobalSplitU 1, InnerUnroll 1,
  StaggerU 32 / 256 bytes / mapping 0, no matrix instructions). The chosen kernel, MT64x128x32, reaches 26.5 TFLOPS at
  512 tokens and 28.7 at 1,024 (rocBLAS's fallback: 12.9 / 13.1), about what rocBLAS's own tuned FP32 GEMM does on
  this card. One launch per expert, spread over 8 streams: about 24 TFLOPS on a typical group.

How it was checked:

- Offline against `hipblasGemmEx`: every token count 2..4,096 and every 37th to 16,384 for gu_exact's own kernels;
  for the Tensile kernel 748 token counts through its own client (65 to ~3,300) and 656 through this launcher (65 to
  4,100), with a negative control that must differ (it does).
- At every start: the grouped kernels against the engine's own `Gemm::f16` call at 25 token counts, bit for bit. A
  difference, a different GEMM path or another GPU turns the feature off and says so in the log.
- In the engine: kl.py (2,557 positions after 16K / 16K / 64K prompt reads) KL 0.0000, 100% same top token; the
  session-state fingerprint after short prompts identical.

### 3. Deterministic decode: the same tokens whatever the drafter does (`STRATA_IQ_MT_MIN=1`)

Strata verifies the drafter's guesses in windows of 1-8 tokens, and the window a token lands in follows the drafts, which
follow measured timings. Without this switch a token's logits depend on its window: reading the same text through windows
of 2, 3 or 4 tokens instead of 1 gives KL 0.010-0.015 (max 8 at single positions), so the same prompt can give a slightly
different reply from run to run. The cause on this card is the CPU expert rows, which were grouped by window.

Upstream's `STRATA_IQ_MT_MIN=1` computes every CPU row independently of the window. With it, teacher-forced reading
through windows of **1, 2, 3, 4, 5, 6 and 8 tokens gives byte-identical log-probabilities** (2,557 positions, KL 0.0000,
the same 36 GDN layer states), so every token is exactly what one-token-at-a-time decoding would give, and the reply is
the same every run. Checked with the debug knob `STRATA_READ_WINDOW=N` (read through windows of at most N tokens) and
`kl.py`; end to end, a 2,425-token file rewrite returned byte-identical replies under four draft policies.

That makes the draft policy a pure speed setting. On rewrite turns (the model returns a file with small edits):

| Draft policy (all with `STRATA_IQ_MT_MIN=1`) | Decode, 2,425-token rewrite |
| --- | ---: |
| `--spec-min-p 0.70`, no lookup | 134 tokens/s (median of 3) |
| `--spec-min-p 0.5 --lookup-chain 3` | 145 tokens/s (median of 3), same bytes |

Ordinary chat and agent turns are unchanged by the policy (within noise); `STRATA_IQ_MT_MIN` itself costs 0-5% there.
The reference this matches is one-token decoding, not the engine without the switch (whose replies follow the drafts).

## Use it

Build the HIP engine for gfx1100 (the interleaved-copy path needs `-DSTRATA_HIP_MMVQ_IL`):

```bash
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      "-DCMAKE_HIP_FLAGS=-DSTRATA_HIP_EXP_LAYOUTS -DSTRATA_HIP_MMVQ_IL"
cmake --build build-hip --target strata
```

Then add to the `env` of your `strata-*.json` (all bit-exact):

```json
"STRATA_PF_GU_EXACT": "1",
"STRATA_WAIT_FUSE": "1", "STRATA_MMVQ_IL_PLAIN": "1", "STRATA_COMBINE_HITS": "1", "STRATA_ROUTE_PUB": "1",
"STRATA_WAIT_COMBINE": "1", "STRATA_PF_LEAN_ISSUE": "1", "STRATA_PF_COPY2": "1", "STRATA_IO_THREADS": "64",
"STRATA_IQ_MT_MIN": "1"
```

and, for deterministic decode with faster rewrite turns, to its `args`: `--spec-min-p 0.5 --lookup-chain 3`.

The log says whether it is on:

```
strata prefill: STRATA_PF_GU_EXACT: the experts' gate/up products from 2 tokens, 16 experts a launch (checked bit for bit against rocBLAS at 25 token counts); from 65 tokens on the Tensile kernel
```

`STRATA_PF_GU_EXACT=G` sets the group size (1 = 16, at most 64; it borrows about 10 MB of VRAM per expert while a
prompt is read). `STRATA_PF_GU_TENSILE=0` keeps the 65+ token experts on gu_exact's own kernel.

## Limits

- Measured on one PC. The Tensile kernel is gfx1100 only; other cards keep gu_exact's own kernels, which follow
  rocBLAS's fallback order generally but were only checked here.
- The bit-exactness is against this ROCm's rocBLAS fallback. The start-up check is there for a different ROCm: if
  rocBLAS's order changes, the feature turns itself off.
- Not built for CUDA or SYCL. The new code is HIP-only and compiles to stubs elsewhere.
- Decode is unchanged by part 2; its gains are part 1's.
