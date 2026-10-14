<h1 align="center">Strata for the RX 7900 XTX</h1>

<p align="center"><b>A faster <a href="https://github.com/Niko1221/Strata">Strata</a> for the AMD Radeon RX 7900 XTX, with output
bit-identical to the original</b><br>
RX 7900 XTX (gfx1100) · Linux · ROCm · free and open source</p>

[Strata](https://github.com/Niko1221/Strata) runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**,
a 125-billion-parameter model, on one gaming PC. This fork tunes it for the RX 7900 XTX. Every speedup keeps the output
**bit-identical** to upstream (the same log-probabilities and the same token at every position, so the same perplexity): faster, with nothing traded
for quality. The one exception is an opt-in build, described under [Install](#install).

Upstream Strata 0.1.44.1 against this fork (merged with 0.1.44.1), same PC, same day, alternating rounds, medians of 3
(RX 7900 XTX 24 GB, Ryzen 7 9800X3D, 62 GiB RAM, Linux, ROCm 10.2, IQ3_S):

| | Upstream 0.1.44.1 | RX 7900 XTX edition | |
| --- | ---: | ---: | ---: |
| Writes answers (short chat) | 88.8 tokens/s | 105.0 tokens/s | **+18%** |
| First token (short chat) | 1.12 s | 0.82 s | **-27%** |
| Reads a 4K prompt | 1,081 tokens/s | 1,443 tokens/s | **+33%** |
| Reads a 16K prompt | 1,397 tokens/s | 1,921 tokens/s | **+37%** |
| Reads a 32K prompt | 1,406 tokens/s | 1,990 tokens/s | **+42%** |
| Agent turn, first token | 3.88 s | 2.93 s | **-24%** |
| Agent session, writes answers | 96.4 tokens/s | 102.5 tokens/s | **+6%** |

The fork runs the settings below, including the opt-in parity build, exact 32K prompt chunks and the draft rule
`STRATA_SPEC_CUM=0.5` (engine `d4b184ee`). Upstream runs its defaults with every output-changing switch off. Every
round of the fork beat every round of upstream in every row; writing speed still moves a few percent between runs.
Reports: [bench/results/2026-10-14-...-0.1.44-cum](bench/results/2026-10-14-community-rx-7900-xtx-gfx1100-0.1.44-cum/)
(the earlier comparison, before `STRATA_SPEC_CUM`:
[2026-10-14-...-0.1.44](bench/results/2026-10-14-community-rx-7900-xtx-gfx1100-0.1.44/)).

Decode is also deterministic: the same question gives the same answer whatever the drafter does (and, with an opt-in
build, whatever the expert placement).

<p align="center"><a href="https://ko-fi.com/imdadismail"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support this fork on Ko-fi" width="446"></a><br>
<sub>This fork is free. If it runs well on your RX 7900 XTX, a coffee keeps the benchmarking going.</sub></p>

<details>
<summary><b>Read more:</b> how it works</summary>

Faster decode kernels (first opened as [PR #2231](https://github.com/Niko1221/Strata/pull/2231)), and the prompt's expert
products computed in exactly rocBLAS's summation order, but grouped and on a Tensile-generated kernel for this card
(`STRATA_PF_GU_EXACT=1`), about 2x faster for that step. With `STRATA_IQ_MT_MIN=1` decode is also **deterministic**:
every token equals one-token-at-a-time decoding whatever the drafter guesses (for a given expert placement), so faster draft policies (prompt lookup
for code rewrites, +8%) change only the speed. An opt-in build (`-DSTRATA_SWIGLU_SEXP` + `STRATA_CPU_PARITY=1`) makes
the CPU's expert rows bit-identical to the GPU's, so the output no longer depends on expert placement either; that
shifts the output once (perplexity 4.425 -> 4.469) and lets `--kv-resident 32768` free VRAM for agent sessions with
identical output (+8% agent decode). `STRATA_SPEC_CUM=0.5` keeps drafts while the whole chain is likely to hold
(+6.4% short chat, +2.1% agent decode, same output). Details, flags and how it was checked:
**[docs/RX7900XTX.md](docs/RX7900XTX.md)**.

</details>

## What you need

| | |
| --- | --- |
| **Graphics card** | AMD Radeon **RX 7900 XTX** (gfx1100, 24 GB). Other cards: use [upstream Strata](https://github.com/Niko1221/Strata). |
| **RAM** | 64 GB for the IQ3_S model measured here (it was tested with 62 GiB usable). |
| **Processor** | Any x86-64 with AVX2. The optional placement parity build (below) needs **AVX-512** (Zen 4 / Zen 5). |
| **System** | Linux with ROCm. Measured with ROCm 10.2 (TheRock nightly 10.2.0a20260930). |
| **Disk** | About 80 GB free, on an SSD. |

## Install

**1. Model and config.** Clone this repository and run upstream's installer, which downloads the model and writes a
`strata-<model>.json` config:

```bash
git clone https://github.com/android-imdad/Strata-7900xtx.git Strata
cd Strata
./setup.sh
```

The installer may set up upstream's engine; the next step replaces it.

**2. Build this fork's engine** for gfx1100:

```bash
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      "-DCMAKE_HIP_FLAGS=-DSTRATA_HIP_EXP_LAYOUTS -DSTRATA_HIP_MMVQ_IL"
cmake --build build-hip --target strata
```

For the placement parity build, add `-DSTRATA_SWIGLU_SEXP` to `CMAKE_HIP_FLAGS` and use another build directory (for
example `build-sexp`). It shifts the output once against the other builds (perplexity 4.425 -> 4.469), so
keep both if you want to compare.

**3. Point the config at it.** In `strata-<model>.json`, set `"exe"` to the engine you built
(`build-hip/strata` or `build-sexp/strata`) and add the settings below. `./setup.sh --setup` and `./update.sh` can
write upstream's engine back into the config, so check `"exe"` after running them.

## Settings

Add to `env` (all bit-exact):

```json
"STRATA_PF_GU_EXACT": "1",
"STRATA_WAIT_FUSE": "1", "STRATA_MMVQ_IL_PLAIN": "1", "STRATA_COMBINE_HITS": "1", "STRATA_ROUTE_PUB": "1",
"STRATA_WAIT_COMBINE": "1", "STRATA_PF_LEAN_ISSUE": "1", "STRATA_PF_COPY2": "1", "STRATA_IO_THREADS": "64",
"STRATA_IQ_MT_MIN": "1"
```

and to `args`, for deterministic decode with faster drafting:

```json
"--spec", "6", "--spec-min-p", "0.5", "--lookup-chain", "3"
```

With the parity build, also add `"STRATA_CPU_PARITY": "1"` to `env` and `"--kv-resident", "32768"` to `args`
(+8% decode in agent sessions).

Faster drafting with the same output: add `"STRATA_SPEC_CUM": "0.5"` to `env`. Drafts are kept while the chance that
the whole chain of drafts holds stays at 0.5 or more, instead of judging each draft on its own: short chat **+6.4%**,
agent sessions **+2.1%** writing answers (official benchmark, 3 alternating rounds, every round faster).

Optional, from upstream 0.1.44: `"STRATA_MTP_HIP_PRIO": "1"` gives the drafter its own GPU queue (agent decode +3% in one
benchmark, short chat within noise - inside the run-to-run spread, so not a proven gain; the output does not change). Do not set `STRATA_COPY_HIP_PRIO=1` on this card: it
hangs the engine.

Faster long prompts with the same output: add `"STRATA_PF_PIECE": "8192"` to `env` and change `"--prefill", "auto"` to
`"--prefill", "auto:32768"` in `args`. Prompts are read in 32K chunks but computed exactly as 8K chunks, so the output
and the conversation checkpoints are bit-identical: a fresh 32K prompt +13-16%, long documents +10-12%.

The engine's log (`strata-<model>.log`) shows that they are on:

```
strata prefill: STRATA_PF_GU_EXACT: the experts' gate/up products from 2 tokens, 16 experts a launch (checked bit for bit against rocBLAS at 25 token counts); from 65 tokens on the Tensile kernel
strata generate: STRATA_CPU_PARITY: on (the CPU expert rows in the GPU's arithmetic, bit for bit)
```

If a check fails at start-up (another ROCm, another GPU), the feature turns itself off and the log says why. Every
switch, how each one was checked and its limits: **[docs/RX7900XTX.md](docs/RX7900XTX.md)**.

## More speed, if you accept a small change in output

The settings above are all exact. The switches below change the output slightly, so it is no longer bit-identical to
upstream. Each was turned on alone against the exact setup (parity build, `--kv-resident 32768`, the expert cache
pinned at 8,600 slots): perplexity over 2,557 positions of code and docs after 16K, 16K and 64K prompts (lower is
better; the exact setup scores **4.447**), and prompt reading speed from the same runs (one run each, so about +/-3%).
For scale: before the parity build, shifting only 200 expert slots moved perplexity between 4.419 and 4.461.

| Switch | Perplexity | Same top token | 16K prompt read | 48K prompt read |
| --- | ---: | ---: | ---: | ---: |
| exact (the settings above) | 4.447 | 100% | - | - |
| `--prefill auto:32768` (`args`): 32K prompt chunks instead of 8K (with `STRATA_PF_PIECE=8192`: 4.447, exact, about half the gain) | 4.469 | 96.2% | **+13%** | **+21%** |
| `STRATA_HIP_WMMA=1` + `STRATA_SELECT_WMMA=1` + `STRATA_PA_FAST=1` (`env`): prompt attention on the matrix cores | 4.492 | 95.9% | **+11%** | **+16%** |
| `STRATA_HC_UPMIX=1` + `STRATA_PF_SWITCH_MIN_T=4096` (`env`) | 4.501 | 94.7% | +2% | -1% |
| `STRATA_PF_GEMM=1` + `STRATA_PF_SWITCH_MIN_T=4096` (`env`) | 4.508 | 94.2% | **-21%** | **-21%** |

- **`--prefill auto:32768`** is the best trade: the smallest change in perplexity and the largest gain on long prompts.
  Prompt attention on the matrix cores (WMMA) is next. Both speed up reading prompts only; writing answers is unchanged.
- **Do not use `STRATA_PF_GEMM`** on this fork: it takes the experts' products off rocBLAS, which turns
  `STRATA_PF_GU_EXACT` off (the log says so), and the prompt path ends up 21% slower. All the switches together
  (perplexity 4.502) were slower than `--prefill auto:32768` alone for the same reason.
- **No effect here:** `STRATA_HC_Q8` converts nothing on this build (the log reports 0.00 GiB), and `STRATA_PF_FUSED`
  changed neither the output nor the speed. `STRATA_PREFILL_CPU_SHARE=auto` acts only on short prompt chunks (agent
  tool results), which this test does not read, so it is not measured here.

Each switch is a single `env` or `args` entry; remove it to go back to exact output.

Never set `STRATA_FUSE_HEAD_GR=1`. On this card it breaks the model (perplexity 15).

## Run and measure

Start it with `./setup.sh` as usual. The app opens at `http://127.0.0.1:8080`, and apps and coding agents connect to
`http://127.0.0.1:8080/v1` (OpenAI) or `/v1/messages` (Anthropic). Everything about using it is the same as upstream:
[upstream README](https://github.com/Niko1221/Strata#using-it), [docs/INSTALL.md](docs/INSTALL.md),
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

To measure your own card, run `./bench.sh` (5-10 minutes: chat, 4K / 16K prompts and an agent-style session). The
numbers above come from this benchmark.

Problems with this fork's changes: open an [issue here](https://github.com/android-imdad/Strata-7900xtx/issues). For
everything else, see [upstream's issues](https://github.com/Niko1221/Strata/issues).

## Why a fork and not merged upstream

- **The fastest part is specific to this card.** The prompt speedup comes from a Tensile-generated gfx1100 assembly
  kernel, tuned and checked on one RX 7900 XTX with one ROCm build. On other GPUs it stays off, and its fallback
  kernels were checked only here. It is bit-exact because it copies the summation order inside rocBLAS's fallback
  kernels, and nothing guarantees that order: a ROCm update can change it (a check at start-up then turns the feature
  off). Upstream would carry a kernel that its maintainers cannot test or regenerate without this card and a patched
  Tensile setup.
- **Part of it changes the output on purpose.** CPU/GPU expert parity shifts the output once (perplexity 4.425 -> 4.469) for every build
  that uses it and needs AVX-512. Its CPU replica copies the GPU expert kernels' arithmetic exactly, so every future
  change to those kernels would have to be checked against it again. That cost belongs in an opt-in fork, not in
  every upstream change.
- **Upstream merges slowly compared with how much it receives.** On 2026-10-14 Strata had 525 open pull requests
  (the oldest from 2026-10-06) and 30 merged through GitHub in total, none in the past week, while 655 new ones were
  opened. Exact-speed work goes stale quickly while it waits, since every upstream kernel change needs the exactness checks
  again. A fork can ship it now and merge upstream on its own schedule. The decode kernels were opened as
  [PR #2231](https://github.com/Niko1221/Strata/pull/2231), later withdrawn; they live here instead.
- **It is measured on one PC.** Upstream serves NVIDIA and AMD cards on Windows and Linux. Everything here was tuned
  and checked on a single RX 7900 XTX under Linux and ROCm 10.2, and it should be judged as that.

This branch follows upstream and takes in its changes from time to time. Everything here is MIT licensed, like Strata,
so upstream or anyone else is welcome to take any part of it.

## Thanks

To **[Niko](https://github.com/Niko1221)** and **[Strata](https://github.com/Niko1221/Strata)**. Strata is what makes a
125-billion-parameter model usable on one gaming PC at all, and everything in this fork builds on it: the engine, the
expert packs, the multi-token drafting and the official benchmark used for every number here. Deterministic
decode is upstream's own `STRATA_IQ_MT_MIN` switch. Thanks also to Strata's contributors, and to AMD's ROCm, rocBLAS
and Tensile, which generated the prompt kernel.

## Credits and license

Strata is by [Niko](https://github.com/Niko1221) and its contributors. The model is
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team, compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) and Unsloth.
Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Open source under the [MIT License](LICENSE); a few parts and
every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).
