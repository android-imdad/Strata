# Community benchmark on AMD Radeon

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 24 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 62.4 GiB; storage: not measured
- OS: Linux 7.2.5-3-omarchy; GPU driver: 7.2.5-3-omarchy; CUDA: not read
- Strata: project 0.1.44.1, commit 296b59d2; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-up-296b59d2 --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 262144 --kv int8 --vram-reserve-mib 1024 --pcie-frac 0.35 --spec-min-p 0.70 --pool-workers 7 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --conversation-cache-mib 4096 --conversation-cache-slots 4 --conversation-cache-min-free-mib 3072 --pcie-mode dma --aux-cpus 8,9,10,11,12,13,14,15
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100500.txt`, `STRATA_GDN_HEAD=1`, `STRATA_GDN_PP=2`, `STRATA_GDN_CONVL2=1`, `STRATA_GDN_NOY=1`, `STRATA_CVEC_FUSE=1`, `STRATA_HCD_EXACT=1`, `STRATA_PF_PAD=1`, `STRATA_MMVF_ROWS=1`, `STRATA_ATTN_LANECELL=0`, `STRATA_EXPERT_V2=1`, `STRATA_TSUM=1`, `STRATA_LFUSE=1`, `STRATA_GDN_SPLIT=1`, `STRATA_QFUSE=1`, `STRATA_PLE_BATCH=1`, `STRATA_PREFILL_RING=512`, `HIP_VISIBLE_DEVICES=0`, `ROCR_VISIBLE_DEVICES=0`, `STRATA_WAIT_FUSE=1`, `STRATA_MMVQ_IL_PLAIN=1`, `STRATA_ATTN_PRE75=1`, `STRATA_VERIFY_QDEDUP=1`, `STRATA_DF_PLE=1`, `STRATA_COMBINE_HITS=1`, `STRATA_ROUTE_PUB=1`, `STRATA_WAIT_COMBINE=1`, `STRATA_PF_LEAN_ISSUE=1`, `STRATA_SH_STREAM=1`, `STRATA_DF_BRANCH=1`, `STRATA_PF_COPY2=1`, `STRATA_IO_THREADS=64`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 88 (85-93) | 0 | 100 (100-100) | 9 | not measured | 84.8 (72.5-103.8) | 1.14 (1.12-1.16) |
| 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 1045.3 (1043.4-1045.4) | not measured | 3.87 (3.87-3.87) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1350.0 (1339.0-1354.0) | not measured | 11.92 (11.88-12.02) |
| 32K prompt, fresh | 32050 (32050-32050) | 0 (0-0) | 24 cap | 3 | 1370.7 (1364.7-1372.0) | not measured | 23.41 (23.39-23.52) |
| agent session, follow-up turns | 14640 (14621-14660) at the last turn | see below | up to 64 | 3 | see below | 91.0 (73.9-109.0) | 4.03 (3.13-4.36) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 74 (72-74) (earlier turns that the cache did not hold; the 13730 (13729-13731) new tokens of the files are read once); first turn (nothing cached) first token: 2.63 s (2.57-2.64).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 647 | 0 | 647 | 253.5 | 2.571 | 70.3 |
| 1 | 3792 | 672 | 3120 | 767.0 | 4.089 | 84.5 |
| 2 | 5325 | 3855 | 1470 | 473.2 | 3.128 | 89.1 |
| 3 | 8769 | 5318 | 3451 | 795.7 | 4.36 | 91.0 |
| 4 | 11695 | 8832 | 2863 | 719.1 | 4.006 | 96.4 |
| 5 | 14621 | 11722 | 2899 | 723.5 | 4.032 | 83.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 225 s.
