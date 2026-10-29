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
  1. **Tier 4 + Phase 5 Schur-Complement Nodal & Series $R+L$ Chain Elimination (`-n` CPU & `-g 4` GPU):** Eliminates internal branch rows (`JJ`, `R`, `C`, uncoupled `L`, `T`, grounded `V`/`P`) *plus* internal degree-2 nodes along series $R+L$, $R+R$, and $L+L$ damping/bias chains into a **2.25x–5.16x smaller** nodal matrix $A_{\text{red}}$, reconstructing `.print` traces on the fly (**2.2x–3.2x faster** on 1 thread for production macros; **3.1x–4.4x** on 2D JJ grids).
  2. **Per-Partition Nodal PTL Block-Parallel Solver (`-p -n` OpenMP & `-g` CUDA GPU):** Combines Nodal matrix reduction (`-n`) with PTL block-parallel windowing (`-p`), decoupling $A_{\text{red}}$ across transmission lines (`T`) and shared Dirichlet supply rails into $P$ tiny independent KLU blocks (e.g. `47,425 -> 9,195` eqns across `583` blocks on `adder`, max block `85` eqns; `38,826 -> 9,870` eqns across `464` blocks on `coupler`, max block `30` eqns) and advancing them in parallel across the causal delay window $W = \min(T_d / \Delta t)$ (**up to `27.1x` OpenMP speedup** and **`21.5x` GPU speedup**), with optional quiescent partition bypass (`.option bypass`).
  3. **Tier 2 Batched Monte Carlo (`.spread`) Solver (`-b <N>`, `.option mc=<N>`, `.option mcall`):** Solves $N$ perturbed fabrication-yield trials simultaneously in one run with Welford online ensemble statistics (**up to `31.1x` OpenMP speedup** and **`19.7x` GPU speedup**).
  4. **Tier 3 Fully GPU-Resident `cuSolverRF` Monolithic Solver (`-g 3`):** Zero-PCIe in-VRAM transient loop (**`2.5x–4.2x` GPU speedup** on 2D JJ grids).
  5. **Core Single-Thread & Memory Optimizations:** $O(1)$ hash-map lookups (`deviceMap`, `csMap`), in-place CSR updates + `klu_l_refactor` on `RTYPE=1` switching, compact PTL circular ring buffers (`hist_`), and in-loop `PRSTART`/`PRSTEP` decimation (**1.5x–3.4x faster 1T MNA** and **2x–17.6x lower peak RAM**).
  6. **High-Speed Binary I/O & New Controls:** `.npy` (NumPy structured array), `.bin` (`JOSIMBIN` float64), `.rawb`, [`scripts/josim_io.py`](scripts/josim_io.py), `.savestate`/`.loadstate` checkpointing, `.option txinterp` fractional PTL interpolation, `.option kcutoff`, `-c` predictor-corrector, and `-f` fixed-step control.

---

## Summary of Results (Production `spec2si-seeqc` Suite on BNL `exxact`)

End-to-end wall-clock runtime (and peak RSS memory) comparing official `v2.6.5`, upstream `v2.7` (`02a34ee`), and this fork (`v2.8` with `.npy` output):

| Deck (Scale & Partitions) | `v2.6.5` (CSV) | `v2.7` (CSV) | **Fork 1T (`-n`)** | **Fork OMP (`-p -n`)** | **Speedup vs `v2.6.5` (`v2.7`)** |
| :--- | ---: | ---: | ---: | ---: | :---: |
| **`r5cal`** (3.5k JJs, $P=1$) | 762 s (0.29 GB) | 385 s | **82.8 s** (0.28 GB) | **82.8 s** | **9.2x** (4.6x) |
| **`r10f1n`** (10k JJs, $P=2$) | 337 s (1.31 GB) | 173 s (0.73 GB) | **35.5 s** (**0.21 GB**) | **35.5 s** | **9.5x** (4.9x) |
| **`r10forig`** (10k JJs, 1 fs PTL) | 1,904 s (6.0 GB) | 257 s (0.87 GB) | **41.6 s** (**0.34 GB**) | **41.6 s** | **45.8x** (6.2x) |
| **`adder`** (5.5k JJs, $P=583$) | 542 s (26.5 GB) | 325 s (15.2 GB) | **36.3 s** (**8.0 GB**) | **18.96 s** | **28.6x** (17.1x) |
| **`coupler`** (2.1k JJs, $P=464$) | 156 s (7.4 GB) | 126 s | **15.3 s** (**4.3 GB**) | **5.73 s** | **27.2x** (22.0x) |
| **`xadd4`** (3.5k JJs, $P=1$) | 85 s (0.42 GB) | 39 s | **9.77 s** (**0.32 GB**) | **9.77 s** | **8.7x** (4.0x) |
| **`dff`** (33 JJs, $P=1$) | 0.82 s | 0.46 s | **0.20 s** | **0.20 s** | **4.1x** (2.3x) |

---

## Quick-Start Solver Selection Guide

| Workload / Circuit Type | Recommended Command | Speedup & Notes |
| :--- | :--- | :--- |
| **Single-Core / Cluster Pool Job** *(1 thread)* | `josim-cli -n -o out.npy deck.cir` | **2.2x–4.4x** via Nodal + $R/L$-chain reduction (`-n`). |
| **Universal Multi-Core CPU** *(PTL or monolithic)* | `josim-cli -p -n -o out.npy deck.cir` | **6.6x–27x** if $P \ge 4$ (Per-Partition Nodal PTL); uses `-n` if $P \le 2$. |
| **Large PTL Tile Array on GPU** *($P \ge 64, W \ge 8$)* | `josim-cli -g -o out.bin deck.cir` | **Up to 21.5x** (1 CUDA warp per partition). |
| **Monolithic 2D/3D JJ Grid on GPU** *(0 PTLs)* | `josim-cli -g 3 -o out.bin deck.cir`<br>`josim-cli -g 4 -o out.bin deck.cir` | **2.5x–4.2x** (`-g 3` for `RTYPE=0`, `-g 4` for switching `RTYPE=1`). |
| **Batched `.spread` Yield Sweep** *($B$ trials)* | `josim-cli -p -b 256 -o mc.npy deck.cir`<br>`josim-cli -g -b 256 -o mc.npy deck.cir` | **23x–31x** on OpenMP CPU; **12x–20x** on GPU. |

- **New CLI Flags (`v2.8`):** `-n` (`--nodal`), `-p` (`--parallel`), `-g [1|2|3|4]` (`--gpu=`), `-b <N>` (`--batch=<N>`), `-c [N]` (`--corrector=`), `-f` (`--fixed-step`), and `-o <file.npy|.bin|.rawb>`.
- **New Netlist `.option` & Controls (`v2.8`):** `.option nodal`, `.option gpu[=1|2|3|4]`, `.option bypass`, `.option bypasstol=<val>`, `.option mc=<N>`, `.option mcall`, `.option txinterp`, `.option kcutoff=<val>`, `.option corrector=<N>`, `.option fixedstep`, `.option warnstep`, `.savestate`, and `.loadstate`.
- **Loading Binary Outputs in Python:**
  ```python
  import numpy as np
  from scripts.josim_io import load_josim

  arr = np.load("out.npy", mmap_mode="r")       # Standard NumPy structured array
  data = load_josim("out.bin")                  # Dict of trace_name -> 1D float64 array
  ```

For the complete CLI and `.option` reference tables and full benchmark suites across Tiers 1–4, see [**`docs/usage_and_benchmarks.md`**](docs/usage_and_benchmarks.md) and [**`docs/gpu_architecture.md`**](docs/gpu_architecture.md).

---

## Changelog

### v2.8 (Enhanced Fork)
- **Tier 4 Schur-Complement Nodal Admittance Solver** on CPU (`-n`, `.option nodal`, `-p -n`) and GPU (`-g 4`, `.option gpu=4`), **Quiescent Partition Bypass** (`.option bypass`), and NUMA-aware OpenMP scaling.
- **Tier 3 Fully GPU-Resident `cuSolverRF` Monolithic Solver** (`-g 3`, `.option gpu=3`) and **Unified GPU Auto-Dispatcher** (`-g`).
- **Tier 2 Batched GPU (`-g -b <N>`) & OpenMP (`-p -b <N>`) Monte Carlo (`.spread`) Solver** (`.option mc=<N>`, `.option mcall`, `JOSIM_SEED`) with Welford online ensemble statistics.
- **Tier 1 Multi-Step Windowed PTL Block-Parallel Solver** for CUDA GPU (`-g`) and OpenMP CPU (`-p`).
- **High-Speed Binary Output Formats** (`.npy`, `.bin`, `.rawb`) and Python loader [`scripts/josim_io.py`](scripts/josim_io.py).
- **Single-Core & RAM Optimizations:** $O(1)$ hash-map lookups (`deviceMap`, `csMap`), in-place CSR updates + `klu_l_refactor`, self-contained PTL circular ring buffers (`hist_`), in-loop `PRSTART`/`PRSTEP` decimation, homogeneous device arrays, cached `PWL`/`PWS` segments, and pre-computed `CUS` splines.
- **New Controls:** `.savestate` / `.loadstate` checkpointing, `.option txinterp` fractional PTL delay interpolation, `.option kcutoff=<val>`, `-c` implicit JJ predictor-corrector, and `-f` / `.option fixedstep` / `.option warnstep`.

### v2.7 (Upstream `02a34ee`)
- Added `.option seed=` for RNG, verbose seed reporting, piecewise sinusoidal (`PWS`) sources, and expanded sinusoid parameters.
- See [`docs/index.md`](docs/index.md) for earlier release notes (`v1.2` – `v2.6.10`).
