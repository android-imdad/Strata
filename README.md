<h1 align="center">Strata for the RX 7900 XTX</h1>

<p align="center"><b>A faster <a href="https://github.com/Niko1221/Strata">Strata</a> for the AMD Radeon RX 7900 XTX, with output
bit-identical to the original</b><br>
RX 7900 XTX (gfx1100) · Linux · ROCm · free and open source</p>

[Strata](https://github.com/Niko1221/Strata) runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**,
a 125-billion-parameter model, on one gaming PC. This fork tunes it for the RX 7900 XTX. Every speedup keeps the output
**bit-identical** to upstream (teacher-forced KL 0.0000, the same token at every position): faster, with nothing traded
for quality. The one exception is an opt-in build, described under [Install](#install).

Same PC, same day, same settings (RX 7900 XTX 24 GB, Ryzen 7 9800X3D, 62 GiB RAM, Linux, ROCm 10.2, IQ3_S):

| | Upstream | RX 7900 XTX edition | |
| --- | ---: | ---: | ---: |
| Writes answers | 86.8 tokens/s | 95.8 tokens/s | **+10%** |
| Reads a 4K prompt | 1,069 tokens/s | 1,427 tokens/s | **+34%** |
| Reads a 16K prompt | 1,381 tokens/s | 1,827 tokens/s | **+32%** |
| Reads a 32K prompt | 1,421 tokens/s | 1,844 tokens/s | **+30%** |
| Agent turn, first token | 3.90 s | 2.92 s | **-25%** |

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
shifts the output once (KL 0.016, perplexity 4.425 -> 4.469) and lets `--kv-resident 32768` free VRAM for agent
sessions at KL 0 (+8% agent decode). Details, flags and how it was checked:
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
example `build-sexp`). It shifts the output once against the other builds (KL 0.016, perplexity 4.425 -> 4.469), so
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

The engine's log (`strata-<model>.log`) shows that they are on:

```
strata prefill: STRATA_PF_GU_EXACT: the experts' gate/up products from 2 tokens, 16 experts a launch (checked bit for bit against rocBLAS at 25 token counts); from 65 tokens on the Tensile kernel
strata generate: STRATA_CPU_PARITY: on (the CPU expert rows in the GPU's arithmetic, bit for bit)
```

If a check fails at start-up (another ROCm, another GPU), the feature turns itself off and the log says why. Every
switch, how each one was checked and its limits: **[docs/RX7900XTX.md](docs/RX7900XTX.md)**.

## Run and measure

Start it with `./setup.sh` as usual. The app opens at `http://127.0.0.1:8080`, and apps and coding agents connect to
`http://127.0.0.1:8080/v1` (OpenAI) or `/v1/messages` (Anthropic). Everything about using it is the same as upstream:
[upstream README](https://github.com/Niko1221/Strata#using-it), [docs/INSTALL.md](docs/INSTALL.md),
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

To measure your own card, run `./bench.sh` (5-10 minutes: chat, 4K / 16K prompts and an agent-style session). The
numbers above come from this benchmark.

## Why a fork and not merged upstream

- **The fastest part is specific to this card.** The prompt speedup comes from a Tensile-generated gfx1100 assembly
  kernel, tuned and checked on one RX 7900 XTX with one ROCm build. On other GPUs it stays off, and its fallback
  kernels were checked only here. It is bit-exact because it copies the summation order inside rocBLAS's fallback
  kernels, and nothing guarantees that order: a ROCm update can change it (a check at start-up then turns the feature
  off). Upstream would carry a kernel that its maintainers cannot test or regenerate without this card and a patched
  Tensile setup.
- **Part of it changes the output on purpose.** CPU/GPU expert parity shifts the output once (KL 0.016) for every build
  that uses it and needs AVX-512. Its CPU replica copies the GPU expert kernels' arithmetic exactly, so every future
  change to those kernels would have to be checked against it again. That cost belongs in an opt-in fork, not in
  every upstream change.
- **Upstream merges slowly compared with how much it receives.** On 2026-10-14 Strata had 525 open pull requests
  (the oldest from 2026-10-06) and 30 merged through GitHub in total, none in the past week, while 655 new ones were
  opened. Exact-speed work goes stale quickly while it waits, since every upstream kernel change needs the KL checks
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
