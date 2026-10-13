# gu_exact / Tensile kernel: checks and regeneration

Standalone tools behind `STRATA_PF_GU_EXACT` (see `docs/RX7900XTX.md`). Build each with the ROCm clang, for example:

    clang++ -x hip --offload-arch=gfx1100 -O3 -std=c++17 X.cpp -lhipblas -lamdhip64 -o X

| File | What it shows |
| --- | --- |
| `order_gsu1_check.cpp` | rocBLAS's fallback for 65+ tokens (MT64x32x8, GSU1) is one FMA chain per output starting at `((n/64)%16)*128`: a naive replica, bitwise for every T, and timings |
| `order_gsu3_check.cpp` | 2..64 tokens (MT128x32x16, GSU3): three interleaved splits, rotation `((n/128)%4)*8`, `(p0+p1)+p2`, bitwise for every T |
| `tensile_launch_check.cpp` | loads a Tensile code object, packs its 128-byte kernarg (ContractionSolution::solve's values), checks it bitwise against hipblasGemmEx for T 65..4100 and times both. Usage: `tensile_launch_check <k.co> <kernel name> 128 32 8` |
| `tensile_streams_bench.cpp` | the same kernel for a group of experts over 1/2/4/8 streams |
| `tensile_sweep.yaml` | the Tensile tuning config: the order-defining parameters pinned (MT0 64, GSU 1, InnerUnroll 1, StaggerU 32/256/0, no MatrixInstruction), the rest forked |
| `validate_tensile.sh`, `tensile_dump_check.cpp` | bitwise check of a tuned solution through the Tensile client's tensor dumps (slow: the client computes on the CPU) |

Regenerating the kernel: run Tensile (ROCm/rocm-libraries `shared/tensile`) on `tensile_sweep.yaml` with a built
`tensile-client` (`--prebuilt-client`), pick a solution, take its `.s` from `build_tmp/SOURCE/assembly/`, assemble it as
described at the top of `src/prefill/gu_tensile_gfx1100.s`, check it with `tensile_launch_check`, and regenerate
`gu_tensile_gfx1100.inc`.
