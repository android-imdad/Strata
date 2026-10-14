# RX 7900 XTX (gfx1100) edition

This branch (`rx7900xtx`) is upstream Strata plus speedups measured on one AMD Radeon RX 7900 XTX. Every change keeps
the model's output **bit-identical**: reading the same texts, the engine with them gives the same log-probabilities
and the same top token at every position as the engine without them, so the same perplexity. Nothing here trades quality for speed. The one exception is opt-in and built separately:
part 4 (`-DSTRATA_SWIGLU_SEXP` + `STRATA_CPU_PARITY=1`) shifts the output once against today's engine so that from then
on it no longer depends on where the experts sit.

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

The config is a strict one: every switch that changes the output (even slightly) is off, including upstream's own opt-in
ones (`STRATA_HC_Q8`, `STRATA_PF_GEMM`, `STRATA_PF_FUSED`, WMMA prompt attention). With those on, both engines are
faster and no longer identical to each other.

## What is in it

### 1. Decode and prompt-path kernels (from PR #2231, withdrawn)

[Niko1221/Strata#2231](https://github.com/Niko1221/Strata/pull/2231): launch shapes of latency-bound kernels (router,
GDN norm, hyper-connection up, QSA top-k, the drafter's row select), DPP lane exchanges, the multi-block argmax on HIP,
the PCIe experts' landed flag written by the copy stream, the adaptive tier's copies over both DMA engines, warp
dequant of Q6_K / IQ4_XS. On by default, each with an `=0` switch; plus opt-ins listed below.

### 2. The experts' gate/up product, exact and fast (`STRATA_PF_GU_EXACT=1`)

In a prompt about half the GPU time is the routed experts' gate/up product (FP16 x FP16 -> FP32, 1280 x 2560 per
expert). It has no hipBLASLt tuning, so it falls back to rocBLAS's generic Tensile kernels, which do not use the matrix
cores and reach 6-13 TFLOPS on this card. Any faster kernel adds the products in another order, which changes the
output a little (that is why `STRATA_PF_GEMM` moves perplexity from 4.447 to 4.508).

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
- In the engine: kl.py (2,557 positions after 16K / 16K / 64K prompt reads) identical log-probabilities and perplexity, 100% same top token; the
  session-state fingerprint after short prompts identical.

### 3. Deterministic decode: the same tokens whatever the drafter does (`STRATA_IQ_MT_MIN=1`)

Strata verifies the drafter's guesses in windows of 1-8 tokens, and the window a token lands in follows the drafts, which
follow measured timings. Without this switch a token's logits depend on its window: reading the same text through windows
of 2, 3 or 4 tokens instead of 1 gives slightly different log-probabilities, so the same prompt can give a slightly different reply from run to run. The cause on this card is the CPU expert rows, which were grouped by window.

Upstream's `STRATA_IQ_MT_MIN=1` computes every CPU row independently of the window. With it, teacher-forced reading
through windows of **1, 2, 3, 4, 5, 6 and 8 tokens gives byte-identical log-probabilities** (2,557 positions, the same perplexity,
the same 36 GDN layer states), so every token is exactly what one-token-at-a-time decoding would give: the reply no
longer depends on the drafts or on timing. It can still change when the expert placement changes (how many experts fit
in VRAM, which other programs' VRAM use can shift by a few slots), because an expert computed on the CPU differs from the
GPU's result in the last bits; for the same placement it is the same (part 4 removes that difference too). Checked with the debug knob `STRATA_READ_WINDOW=N` (read through windows of at most N tokens) and
`kl.py`; end to end, a 2,425-token file rewrite returned byte-identical replies under four draft policies.

That makes the draft policy a pure speed setting. On rewrite turns (the model returns a file with small edits):

| Draft policy (all with `STRATA_IQ_MT_MIN=1`) | Decode, 2,425-token rewrite |
| --- | ---: |
| `--spec-min-p 0.70`, no lookup | 134 tokens/s (median of 3) |
| `--spec-min-p 0.5 --lookup-chain 3` | 145 tokens/s (median of 3), same bytes |

The MTP drafter's own depth is a speed setting too now. With `--spec 4` it stops at 3 guesses although its acceptance is
still ~85-90% there; with `--spec 6` acceptance stays ~80% down to the fifth guess. Two alternating rounds of thinking-on
questions, a coding task and a rewrite (slots pinned): `--spec 6` was fastest in every request, about +5-20% over
`--spec 4` (thinking turns most), for ~6 MiB of VRAM; `--spec 8` drafts too deep, and `--spec-min-p` 0.4 / 0.3 did not
beat 0.5. Ordinary chat and agent turns gain nothing from the lookup chain; `STRATA_IQ_MT_MIN` itself costs 0-5% there.
The reference this matches is one-token decoding, not the engine without the switch (whose replies follow the drafts).

### 4. The same output whatever the expert placement (`STRATA_CPU_PARITY=1`, build with `-DSTRATA_SWIGLU_SEXP`)

Each routed expert runs either on the GPU (from VRAM or over PCIe) or on the CPU, and today the two give results that
differ in the last bits. So the reply depends on the placement: how many expert slots fit in VRAM (other programs' VRAM
use moves it by a few slots), the adaptive tier's swaps, and VRAM-saving options such as `--kv-resident`. On the same
engine, moving 50 / 200 slots moves perplexity from 4.425 to 4.461 / 4.419.

This part makes the CPU compute exactly what the GPU computes:

- **A portable exponential in the GPU SwiGLU** (`include/strata/kernels/sexp.hpp`, selected at build time with
  `-DSTRATA_SWIGLU_SEXP`): a Cody-Waite `exp` from `fma` / `rint` / `ldexp` only, at most 0.94 ulp (`__expf` on
  gfx1100: up to 64 ulp). `g / (1 + sexp(-g))` gives the same bits on the host and on the GPU for all 2^32 inputs.
- **A CPU replica of the GPU expert arithmetic** (`src/kernels/cpu/replica_avx512.cpp`, AVX-512 VNNI): the same q8_1
  activations, each part's integer dot and scale step, `(dw * d8) * sumi` accumulated by FMA (the GPU contracts it) in
  the GPU's lane and butterfly order, for every format the model uses (gate/up IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS; down
  IQ4_NL, Q2_0). The tool `expert_replica` checks it against `native_expert_grouped`: 192/192 bit for bit (48 layers x
  1-4 tokens, VRAM and PCIe launches); plain multiply + add, the negative control, differs everywhere.

The switch turns itself off (and says why in the log) without AVX-512, if a layer's format is not in the replica, or on a
GPU build without `-DSTRATA_SWIGLU_SEXP`. Without the switch nothing changes.

Checked with kl.py (2,557 positions, slots pinned): with parity on, 7,200 slots against 7,150 / 7,000 / 6,500 slots
gives identical log-probabilities, 100% same top token, perplexity 4.447 at every placement (without parity the
placements disagree). With `--kv-resident 32768` against without it, also identical: perplexity 4.447 = 4.447. Together with part 3 the reply
no longer depends on the drafts, the timing, the expert placement or VRAM use.

**The cost:** the new exponential rounds differently from `__expf`, a one-time shift against today's output: perplexity
4.425 -> 4.469, 96.6% same top token, the same size as the placement drift above (perplexity 4.419-4.461).
Tool calls: 36/36 valid with and without it.

**The gain:** placement becomes a free speed setting, like the drafts in part 3. `--kv-resident 32768` keeps only the
last 32K tokens of KV in VRAM and frees it for ~1,600 more expert slots (8,798 against ~7,200 here). Official benchmark,
4 alternating rounds, medians:

| Engine | Chat, writes answers | Agent session, writes answers | 16K prompt, reads |
| --- | ---: | ---: | ---: |
| Parts 1-3 | 95.4 tokens/s | 98.1 tokens/s | 1,852 tokens/s |
| + parity | 94.6 tokens/s (-0.9%) | 98.5 tokens/s (+0.4%) | 1,857 tokens/s |
| + parity + `--kv-resident 32768` | 94.8 tokens/s (-0.6%) | **105.8 tokens/s (+7.8%)** | 1,850 tokens/s |

Parity alone costs nothing measurable. With `--kv-resident` every agent round (105.4-108.0) beat the best baseline round
(103.7); chat decode was flat in this run (+4% in an earlier sweep).

## Use it

Build the HIP engine for gfx1100 (the interleaved-copy path needs `-DSTRATA_HIP_MMVQ_IL`):

```bash
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      "-DCMAKE_HIP_FLAGS=-DSTRATA_HIP_EXP_LAYOUTS -DSTRATA_HIP_MMVQ_IL"
cmake --build build-hip --target strata
```

For part 4 add `-DSTRATA_SWIGLU_SEXP` to `CMAKE_HIP_FLAGS` (a separate build directory keeps today's output available).

Then add to the `env` of your `strata-*.json` (all bit-exact):

```json
"STRATA_PF_GU_EXACT": "1",
"STRATA_WAIT_FUSE": "1", "STRATA_MMVQ_IL_PLAIN": "1", "STRATA_COMBINE_HITS": "1", "STRATA_ROUTE_PUB": "1",
"STRATA_WAIT_COMBINE": "1", "STRATA_PF_LEAN_ISSUE": "1", "STRATA_PF_COPY2": "1", "STRATA_IO_THREADS": "64",
"STRATA_IQ_MT_MIN": "1"
```

and, for deterministic decode with faster drafting, set its `args` to `--spec 6 --spec-min-p 0.5 --lookup-chain 3`.
With the part 4 build, also `"STRATA_CPU_PARITY": "1"` in `env` and `--kv-resident 32768` in `args`.

The log says whether it is on:

```
strata prefill: STRATA_PF_GU_EXACT: the experts' gate/up products from 2 tokens, 16 experts a launch (checked bit for bit against rocBLAS at 25 token counts); from 65 tokens on the Tensile kernel
strata generate: STRATA_CPU_PARITY: on (the CPU expert rows in the GPU's arithmetic, bit for bit)
```

`STRATA_PF_GU_EXACT=G` sets the group size (1 = 16, at most 64; it borrows about 10 MB of VRAM per expert while a
prompt is read). `STRATA_PF_GU_TENSILE=0` keeps the 65+ token experts on gu_exact's own kernel.

## Limits

Why this is a fork rather than an upstream pull request, and thanks to Strata: see the
[README](../README.md#why-a-fork-and-not-merged-upstream).

- Measured on one PC. The Tensile kernel is gfx1100 only; other cards keep gu_exact's own kernels, which follow
  rocBLAS's fallback order generally but were only checked here.
- The bit-exactness is against this ROCm's rocBLAS fallback. The start-up check is there for a different ROCm: if
  rocBLAS's order changes, the feature turns itself off.
- Not built for CUDA or SYCL. The new code is HIP-only and compiles to stubs elsewhere.
- Decode is unchanged by part 2; its gains are part 1's.
- Part 4 needs AVX-512 (VNNI) on the CPU and was checked only for this model's formats. Its replica follows the GPU
  kernels' order as built here; a change to those kernels needs the `expert_replica` check again.
