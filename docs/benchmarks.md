# Performance Benchmarks

xad-forge JIT compilation provides significant speedups for repeated evaluation workloads. This page documents benchmark results comparing tape-based AD to JIT-compiled execution.

## LIBOR Swaption Portfolio Benchmark

This benchmark uses a realistic quantitative finance workload: pricing a portfolio of European swaptions under the LIBOR Market Model with full sensitivity computation.

**Benchmark source:** [CI workflow run](https://github.com/da-roth/forge/actions/runs/21132764692/job/60767569466)

### Environment

| | |
|---|---|
| **Platform** | Linux (Azure) |
| **CPU** | AMD EPYC 7763 64-Core Processor |
| **RAM** | 16 GB |
| **SIMD** | SSE3, SSE4.1, SSE4.2, AVX, AVX2 |
| **Compiler** | GCC 13.3.0 |

### Workload

| | |
|---|---|
| **Portfolio** | 15 European swaptions |
| **Maturities** | 4, 8, 20, 28, 40 years (3 each) |
| **Model** | LIBOR Market Model (lognormal forwards) |
| **Sensitivities** | 161 total (1 delta + 80 volatilities + 80 forward rates) |

### Methods Compared

| Method | Description |
|--------|-------------|
| **FD** | Finite Differences (bump-and-revalue baseline, paths ≤ 1000 only) |
| **XAD** | XAD tape-based reverse-mode AAD |
| **JIT** | Forge JIT-compiled native x86-64 code (ScalarBackend) |
| **JIT-AVX** | Forge JIT + AVX2 SIMD via AVXBackend (4 paths per instruction) |

### Results

All times in milliseconds (mean of 3 iterations after 2 warmup iterations):

| Paths | FD | XAD (tape) | JIT | JIT-AVX |
|------:|-------:|-------:|-------:|-------:|
| 10 | 29.37 | **1.74** | 25.89 | 30.31 |
| 100 | 293.42 | **14.75** | 34.84 | 33.46 |
| 1K | 2933.93 | 143.17 | 125.58 | **66.59** |
| 10K | - | 1444.58 | 1023.37 | **386.64** |
| 50K | - | 7170.51 | 5028.22 | **1840.27** |
| 100K | - | 14336.53 | 9905.43 | **3505.08** |
| 400K | - | 57375.55 | 40203.81 | **14507.45** |

### Analysis

**Crossover point:** JIT becomes faster than tape-based AD at approximately **1,000 paths**. Below this threshold, tape-based AD is faster due to JIT compilation overhead.

**Speedup at scale:**

| Paths | JIT vs XAD | JIT-AVX vs XAD |
|------:|------:|------:|
| 1K | 1.1x | **2.1x** |
| 10K | 1.4x | **3.7x** |
| 100K | 1.4x | **4.1x** |
| 400K | 1.4x | **4.0x** |

**Key observations:**

1. **Low path counts (10-100):** XAD tape-based AD is 10-20x faster than JIT. The compilation cost dominates.

2. **Medium path counts (1K):** Crossover point. JIT catches up as compilation cost is amortized over more evaluations.

3. **High path counts (10K+):** JIT provides consistent 1.4x speedup over tape replay. AVX2 SIMD adds another 2.5-4x.

4. **AVX2 impact:** Processing 4 paths per SIMD instruction provides substantial additional speedup, especially valuable for Monte Carlo workloads.

5. **Validation:** All methods produce identical derivatives (161/161 match against finite differences), confirming correctness.

### Interpretation

For a typical Monte Carlo pricing workflow:
- **< 100 evaluations:** Use tape-based AD (XAD default)
- **100-1000 evaluations:** Either approach is reasonable
- **> 1000 evaluations:** Use xad-forge JIT backends, preferably AVXBackend

The exact crossover depends on graph complexity. Simpler graphs have lower compilation cost and cross over earlier; complex graphs may require more evaluations to amortize.

## See Also

- [When to Use JIT](../README.md#when-to-use-jit) — Decision guide in main README
- [Forge](https://github.com/da-roth/forge) — Underlying JIT compiler
- [QuantLib-Risks-Cpp-Forge](https://github.com/da-roth/QuantLib-Risks-Cpp-Forge) — QuantLib-Risks with Forge JIT integration
