# Community benchmark on AMD Radeon

Measured on 2026-10-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 24 GiB VRAM
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 62.4 GiB; storage: not measured
- OS: Linux 7.2.5-3-omarchy; GPU driver: 7.2.5-3-omarchy; CUDA: not read
- Strata: project 0.1.44.1, commit 8bdc04e5; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_s
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-sync-8bdc04e5 --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:32768 --spec 6 --mtp rt --max-context 262144 --kv int8 --vram-reserve-mib 1024 --pcie-frac 0.35 --spec-min-p 0.5 --pool-workers 7 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --conversation-cache-mib 4096 --conversation-cache-slots 4 --conversation-cache-min-free-mib 3072 --pcie-mode dma --aux-cpus 8,9,10,11,12,13,14,15 --lookup-chain 3 --kv-resident 32768
```

Environment of the config: `STRATA_HIPBLASLT_TUNING=gfx1100-hipblaslt-100500.txt`, `STRATA_GDN_HEAD=1`, `STRATA_GDN_PP=2`, `STRATA_GDN_CONVL2=1`, `STRATA_GDN_NOY=1`, `STRATA_CVEC_FUSE=1`, `STRATA_HCD_EXACT=1`, `STRATA_PF_PAD=1`, `STRATA_MMVF_ROWS=1`, `STRATA_ATTN_LANECELL=0`, `STRATA_EXPERT_V2=1`, `STRATA_TSUM=1`, `STRATA_LFUSE=1`, `STRATA_GDN_SPLIT=1`, `STRATA_QFUSE=1`, `STRATA_PLE_BATCH=1`, `STRATA_PREFILL_RING=512`, `HIP_VISIBLE_DEVICES=0`, `ROCR_VISIBLE_DEVICES=0`, `STRATA_WAIT_FUSE=1`, `STRATA_MMVQ_IL_PLAIN=1`, `STRATA_ATTN_PRE75=1`, `STRATA_VERIFY_QDEDUP=1`, `STRATA_DF_PLE=1`, `STRATA_COMBINE_HITS=1`, `STRATA_ROUTE_PUB=1`, `STRATA_WAIT_COMBINE=1`, `STRATA_PF_LEAN_ISSUE=1`, `STRATA_SH_STREAM=1`, `STRATA_DF_BRANCH=1`, `STRATA_PF_COPY2=1`, `STRATA_IO_THREADS=64`, `STRATA_PF_GU_EXACT=1`, `STRATA_IQ_MT_MIN=1`, `STRATA_CPU_PARITY=1`, `STRATA_MTP_HIP_PRIO=1`, `STRATA_PF_PIECE=8192`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 88 (85-93) | 0 | 100 (100-100) | 9 | not measured | 93.2 (73.9-112.9) | 0.84 (0.81-0.85) |
| 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 1387.2 (1371.5-1395.5) | not measured | 2.92 (2.90-2.95) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1871.2 (1869.6-1873.5) | not measured | 8.61 (8.59-8.61) |
| 32K prompt, fresh | 32050 (32050-32050) | 0 (0-0) | 24 cap | 3 | 1943.3 (1934.9-1945.5) | not measured | 16.52 (16.50-16.59) |
| agent session, follow-up turns | 14784 (14773-14792) at the last turn | see below | up to 64 | 3 | see below | 97.7 (83.4-110.2) | 3.01 (2.23-3.27) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-69.1); earlier-turn tokens read again per session: 143 (3-144) (earlier turns that the cache did not hold; the 13868 (13867-13869) new tokens of the files are read once); first turn (nothing cached) first token: 1.80 s (1.80-1.84).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 647 | 0 | 647 | 355.7 | 1.838 | 58.3 |
| 1 | 3924 | 666 | 3258 | 1045.1 | 3.139 | 96.5 |
| 2 | 5457 | 3917 | 1540 | 662.7 | 2.366 | 97.4 |
| 3 | 8901 | 5450 | 3451 | 1064.4 | 3.265 | 88.0 |
| 4 | 11827 | 8965 | 2862 | 972.1 | 2.968 | 97.3 |
| 5 | 14773 | 11874 | 2899 | 968.9 | 3.017 | 97.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 169 s.
