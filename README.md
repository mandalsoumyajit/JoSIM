# JoSIM
### Superconductor Circuit Simulator

---

## Full Documentation

- **Upstream Documentation:** [JoSIM Documentation](https://joeydelp.github.io/JoSIM)
- **Fork Usage, Options & Benchmark Guide:** [**`docs/usage_and_benchmarks.md`**](docs/usage_and_benchmarks.md)
- **GPU & Nodal Solver Architecture:** [**`docs/gpu_architecture.md`**](docs/gpu_architecture.md)

## Overview

JoSIM was developed under IARPA contract SuperTools (via the U.S. Army Research Office grant W911NF-17-1-0120).

JoSIM is a SPICE syntax circuit simulator specifically created to handle superconducting elements such as the Josephson junction. It reads in a standard SPICE deck, creates an $A$ matrix and solves the linear algebra problem $Ax=b$ using SuiteSparse KLU (or `cuSolverRF` on GPU). JoSIM builds both a command-line executable (`josim-cli`) and a C++ library (`libjosim`), and outputs results in text formats (`.csv`, `.dat`, `.raw`) or high-speed binary formats (`.npy`, `.bin`, `.rawb`).

Referencing: [JoSIM — Superconductor SPICE Simulator](https://ieeexplore.ieee.org/document/8633946/)

If you find this tool useful, you can help keep it alive by \
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/joeydelp) \
or via \
[![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/JoeyDelp)

---

## Inherited Upstream (`v2.7`) vs. New in This Fork (`v2.8`)

This repository is an enhanced fork of [`JoeyDelp/JoSIM`](https://github.com/JoeyDelp/JoSIM) branched from upstream **`v2.7` (commit `02a34ee`)**. It preserves 100% numerical equivalence with upstream `v2.7` while adding the following capabilities (see [**`docs/usage_and_benchmarks.md`**](docs/usage_and_benchmarks.md) for full details):

- **Inherited from Upstream `v2.7` (`02a34ee`):**
  - SPICE netlist parser, RCSJ Josephson junction physics (`RTYPE=0/1`, Haberkorn tunneling, multi-harmonic CPR, `.temp`/`.neb` thermal noise), transmission lines (`T`), mutual inductances (`K`), and controlled sources.
  - Phase-mode (`-a 1`, default) and Voltage-mode (`-a 0`) Modified Nodal Analysis (MNA) with 2nd-order Gear (BDF2) integration, serial SuiteSparse KLU solve, single-run `.spread`, `.iv`, and ASCII text output (`.csv`, `.dat`, `.raw`).
- **New in This Fork (`v2.8`):**
  1. **Schur-Complement Nodal Admittance & Series $R+L$ Chain Elimination (`-n` CPU & `-g 4` GPU):** Eliminates internal branch rows (`JJ`, `R`, `C`, uncoupled `L`, `T`, grounded `V`/`P`) *plus* internal degree-2 nodes along series $R+L$, $R+R$, and $L+L$ damping/bias chains into a **2.25x–5.16x smaller** nodal matrix $A_{\text{red}}$, reconstructing `.print` traces on the fly (**2.2x–3.2x faster** on 1 thread for production macros; **3.1x–4.4x** on 2D JJ grids).
  2. **Per-Partition Nodal PTL Block-Parallel Solver & Short-PTL Coarsening (`-p -n` OpenMP & `-g` CUDA GPU):** Combines Nodal matrix reduction (`-n`) with PTL block-parallel windowing (`-p`), automatically coarsening passive PTL-via islands and ultra-short PTLs ($k < 4$) while decoupling $A_{\text{red}}$ across transmission lines (`T`) and shared Dirichlet supply rails into $P$ tiny independent KLU blocks (e.g. `47,425 -> 9,195` eqns coarsened from `583` to `126` blocks on `adder`, $W = 3 \to 5$, max block `164` eqns; `38,826 -> 9,870` eqns across `464` blocks on `coupler`, max block `30` eqns) and advancing them in parallel across the causal delay window $W = \min_{p_1 \neq p_2}(T_d / \Delta t)$ (**up to `75.3x` OpenMP speedup** and **`21.5x` GPU speedup**), with optional quiescent partition bypass (`.option bypass`).
  3. **Sub-Timestep PTL Lumped $\pi$-Replacement (`.option lumpptl`, `.option minptl`):** Automatically replaces lossless transmission lines whose delay $T_d$ is smaller than a timestep ($T_d < 0.5\Delta t$ by default, e.g. `1 fs` PTL stubs in `r10forig`) with their exact matched lumped $L$-$C$ $\pi$-equivalent ($L_{\text{eq}} = Z_0 T_d$, $C_{\text{port}} = T_d / (2 Z_0)$) before step-size clamping, eliminating artificial $40\times$ global timestep crushing (**`591.3x` speedup vs `v2.6.5` / `79.8x` vs `v2.7` on `r10forig`**).
  4. **Direct In-Loop Trace Recording, Zero-Allocation Streaming `.npy`/`.bin` Output & Built-In SFQ Event/Pulse Extractor (`-e`, `-E`):** Drops unused JJ `variableIndex_` trace vectors, folds exclusive `V(n1, n2)` / `P(n1, n2)` node-difference pairs (`traceType = 4`), parallelizes NUMA trace allocation across OpenMP threads, and streams `.npy`/`.bin`/`.rawb` directly from `results.xVector` via a 64×8 L1 cache-tiled transpose (**cuts `adder` full-waveform peak RAM from `26.5 GB` (`v2.6.5`) / `8.04 GB` to `1.72 GB`**). In **Events-Only Mode** (`-e out.events.npy` without `-o`, plus optional pulse metrics `-E [pulse|all|t_peak,v_peak,fwhm,dt_pi,area,time=peak]`), tracks $2\pi$ SFQ phase slips in-loop without allocating full waveform buffers (**cuts `adder -p -n -e` to `4.63 s` wall / `0.19 GB` RAM / `606 KB` output (`117x` faster & `136x` less RAM vs `v2.6.5`) and `coupler -p -n -e` to `2.58 s` wall / `0.22 GB` RAM / `512 KB` output**).
  5. **Batched Monte Carlo (`.spread`) & Deterministic Parameter/Margin Sweep (`.sweep`, `-S`) Solver (`-b <N>`, `-p -n`, `-g`):** Solves $N$ perturbed fabrication-yield trials or deterministic 1D/2D parameter/margin sweep points (`.sweep`, `.option sweep=...`, `-S <spec>`) simultaneously in one run across `.param` identifiers, `@BIAS`/`@JJ`/`@IND`/`@RES`/`@CAP`/`@SRC`/`@GLOBAL` categories, and wildcard device patterns (`IB*`, `B*`), with Batched Schur-Complement Nodal reduction (`-p -n`), Welford online ensemble statistics, and automatic SFQ operating margin window detection (**`33.3x–51.8x` speedup vs 64 sequential MNA runs**; **up to `83.7x` with `-p -n -b 64`** and **`19.7x` GPU speedup**).
  6. **Fully GPU-Resident `cuSolverRF` Monolithic Solver (`-g 3`, `-g 4`):** Zero-PCIe in-VRAM transient loop (**`2.5x–4.2x` GPU speedup** on 2D JJ grids).
  7. **Core Single-Thread, Memory & Local Rollback Optimizations:** $O(1)$ hash-map lookups (`deviceMap`, `csMap`), in-place CSR updates + `klu_l_refactor` on `RTYPE=1` switching, compact PTL circular ring buffers (`hist_`), in-loop `PRSTART`/`PRSTEP` decimation (**1.5x–3.4x faster 1T MNA** and **2x–17.6x lower peak RAM**), and **in-memory rolling checkpoints for local step-halving rollback** (`.option rollback[=1|2|window|stay]`, `.option norollback`, **2.0x–3.9x faster** when dynamic step-halving triggers).
  8. **High-Speed Binary I/O & New Controls:** `.npy` (NumPy structured array), `.bin` (`JOSIMBIN` float64), `.rawb`, [`scripts/josim_io.py`](scripts/josim_io.py), `.savestate`/`.loadstate` checkpointing, `.option txinterp` fractional PTL interpolation, `.option kcutoff`, `-c` predictor-corrector, and `-f` fixed-step control.

---

## Summary of Results (Production `spec2si-seeqc` Suite on BNL `exxact`)

End-to-end wall-clock runtime (and peak RSS memory) comparing official `v2.6.5`, upstream `v2.7` (`02a34ee`), and this fork (`v2.8` with `.npy` output):

| Deck (Scale & Partitions) | `v2.6.5` (CSV) | `v2.7` (CSV) | **Fork 1T (`-n`)** | **Fork OMP (`-p -n`)** | **Speedup vs `v2.6.5` (`v2.7`)** |
| :--- | ---: | ---: | ---: | ---: | :---: |
| **`r5cal`** (3.5k JJs, $P=1$) | 762 s (0.29 GB) | 385 s | **82.8 s** (0.28 GB) | **82.8 s** | **9.2x** (4.6x) |
| **`r10f1n`** (10k JJs, $P=2$) | 337 s (1.31 GB) | 173 s (0.73 GB) | **27.4 s** (**0.28 GB**) | **27.4 s** | **12.3x** (6.3x) |
| **`r10forig`** (10k JJs, 1 fs PTL) | 1,904 s (6.0 GB) | 257 s (0.87 GB) | **3.22 s** (**0.28 GB**) | **3.22 s** | **591.3x** (79.8x) |
| **`adder`** (5.5k JJs, $P=583\to126$) | 542 s (26.5 GB) | 325 s (15.2 GB) | **26.2 s** (**1.72 GB**) | **7.20 s** (**1.72 GB**) / **`4.63 s` (`-e`, 0.19 GB)** | **75.3x** / **`117.1x` (`-e`)** |
| **`coupler`** (2.1k JJs, $P=464$) | 156 s (7.4 GB) | 126 s | **12.6 s** (**1.43 GB**) | **4.08 s** (**1.42 GB**) / **`2.58 s` (`-e`, 0.22 GB)** | **38.2x** / **`60.5x` (`-e`)** |
| **`xadd4`** (3.5k JJs, $P=1$) | 85 s (0.42 GB) | 39 s | **9.83 s** (**0.22 GB**) | **9.83 s** (**0.22 GB**) | **8.6x** (4.0x) |
| **`dff`** (33 JJs, $P=1$) | 0.82 s | 0.46 s | **0.20 s** (**0.13 GB**) | **0.20 s** (**0.13 GB**) | **4.1x** (2.3x) |

---

## Quick-Start Solver Selection Guide

| Workload / Circuit Type | Recommended Command | Speedup & Notes |
| :--- | :--- | :--- |
| **Single-Core / Cluster Pool Job** *(1 thread)* | `josim-cli -n -o out.npy deck.cir` | **2.2x–75.8x** via Nodal + $R/L$-chain & sub-step PTL reduction (`-n`). |
| **Universal Multi-Core CPU** *(PTL or monolithic)* | `josim-cli -p -n -o out.npy deck.cir` | **15x–75x** if $P \ge 4$ (Coarsened Nodal PTL); uses `-n` if $P \le 2$. |
| **Digital SFQ Timing / Event Verification** | `josim-cli -p -n -e events.npy [-E all] deck.cir` | **Zero waveform allocation** (`195 MB` RAM, `606 KB` output, **`4.63 s` on `adder`**). |
| **Large PTL Tile Array on GPU** *($P \ge 64, W \ge 8$)* | `josim-cli -g -o out.bin deck.cir` | **Up to 21.5x** (1 CUDA warp per partition). |
| **Monolithic 2D/3D JJ Grid on GPU** *(0 PTLs)* | `josim-cli -g 3 -o out.bin deck.cir`<br>`josim-cli -g 4 -o out.bin deck.cir` | **2.5x–4.2x** (`-g 3` for `RTYPE=0`, `-g 4` for switching `RTYPE=1`). |
| **Batched `.spread` Yield Sweep** *($B$ trials)* | `josim-cli -p -n -b 256 -o mc.npy deck.cir`<br>`josim-cli -g -b 256 -o mc.npy deck.cir` | **23x–84x** on OpenMP CPU (`-p -n`); **12x–20x** on GPU. |
| **Deterministic Parameter / Margin Sweep** | `josim-cli -p -n -S "@BIAS:MARGIN:0.3:21" -o sw.npy deck.cir`<br>`josim-cli -p -n -S "Scaling:LIN:21:0.7:1.3" -o sw.npy deck.cir` | **33x–52x** vs sequential MNA; automatic SFQ operating margin window report. |

- **New CLI Flags (`v2.8`):** `-n` (`--nodal`), `-p` (`--parallel`), `-e [file]` (`--events=`), `-E [spec]` (`--event-metrics=`), `-S <spec>` (`--sweep=`), `-g [1|2|3|4]` (`--gpu=`), `-b <N>` (`--batch=<N>`), `-c [N]` (`--corrector=`), `-f` (`--fixed-step`), and `-o <file.npy|.bin|.rawb>`.
- **New Netlist `.option` & Controls (`v2.8`):** `.option nodal`, `.option events[=file]`, `.option eventsonly[=file]`, `.option eventmetrics[=spec]`, `.option eventpulse`, `.sweep <target> ...`, `.option sweep=<spec>`, `.option sweepall`, `.option lumpptl[=0|1]`, `.option nolumpptl`, `.option minptl=<val>`, `.option rollback[=0|1|2|window|stay]`, `.option norollback`, `.option gpu[=1|2|3|4]`, `.option bypass`, `.option bypasstol=<val>`, `.option mc=<N>`, `.option mcall`, `.option txinterp`, `.option kcutoff=<val>`, `.option corrector=<N>`, `.option fixedstep`, `.option warnstep`, `.savestate`, and `.loadstate`.
- **Loading Binary & Event Outputs in Python:**
  ```python
  import numpy as np
  from scripts.josim_io import load_josim, load_josim_events

  arr = np.load("out.npy", mmap_mode="r")          # Standard NumPy structured waveform array
  data = load_josim("out.bin")                     # Dict of trace_name -> 1D float64 array
  ev = load_josim_events("events.npy")             # Structured array: trace_id, trace, time, polarity, winding, phase (+ t_peak, v_peak, fwhm, dt_pi, area)
  ```

For the complete CLI and `.option` reference tables and full benchmark suites across all solver modes, see [**`docs/usage_and_benchmarks.md`**](docs/usage_and_benchmarks.md) and [**`docs/gpu_architecture.md`**](docs/gpu_architecture.md).

---

## Changelog

### v2.8 (Enhanced Fork)
- **Schur-Complement Nodal Admittance & Series $R+L$ Chain Reduction Solver** on CPU (`-n`, `.option nodal`, `-p -n`) and GPU (`-g 4`, `.option gpu=4`), **Quiescent Partition Bypass** (`.option bypass`), and NUMA-aware OpenMP scaling.
- **Multi-Step Windowed PTL Block-Parallel Solver & Short-PTL Coarsening** for OpenMP CPU (`-p`, `-p -n`) and CUDA GPU (`-g`), plus automatic **Sub-Timestep PTL Lumped $\pi$-Replacement** (`.option lumpptl`, `.option minptl`).
- **Fully GPU-Resident `cuSolverRF` Monolithic Solver** (`-g 3`, `-g 4`, `.option gpu=3/4`) and **Unified GPU Auto-Dispatcher** (`-g`).
- **Batched GPU (`-g`), OpenMP (`-p`), & Batched Nodal (`-p -n`) Monte Carlo (`.spread`) and Deterministic Parameter/Margin Sweep (`.sweep`, `-S`) Solver** (`-b <N>`, `.option mc=<N>`, `.option mcall`, `.option sweep=<spec>`, `.option sweepall`, `JOSIM_SEED`) with Welford online ensemble statistics and automatic SFQ operating margin window extraction.
- **Zero-Allocation Streaming Binary Output** (`.npy`, `.bin`, `.rawb`), **Built-In SFQ Switching Event & Pulse Metric Extractor** (`-e`, `-E`, `.option events`, `.option eventmetrics`), and Python loader [`scripts/josim_io.py`](scripts/josim_io.py).
- **Single-Core, Memory & Local Rollback Optimizations:** $O(1)$ hash-map lookups (`deviceMap`, `csMap`), in-place CSR updates + `klu_l_refactor`, self-contained PTL circular ring buffers (`hist_`), in-loop `PRSTART`/`PRSTEP` decimation, homogeneous device arrays, cached `PWL`/`PWS` segments, pre-computed `CUS` splines, and in-memory rolling checkpoints for local step-halving rollback (`.option rollback[=1|2|window|stay]`, `.option norollback`).
- **New Controls:** `.savestate` / `.loadstate` checkpointing, `.option txinterp` fractional PTL delay interpolation, `.option kcutoff=<val>`, `-c` implicit JJ predictor-corrector, and `-f` / `.option fixedstep` / `.option warnstep`.

### v2.7 (Upstream `02a34ee`)
- Added `.option seed=` for RNG, verbose seed reporting, piecewise sinusoidal (`PWS`) sources, and expanded sinusoid parameters.
- See [`docs/index.md`](docs/index.md) for earlier release notes (`v1.2` – `v2.6.10`).
