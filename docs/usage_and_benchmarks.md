# JoSIM Fork (`v2.8`): Usage Guide, New Features & Benchmark Results

This document details the new solver modes, command-line switches, netlist `.option` directives, and benchmark results introduced in this enhanced fork of JoSIM (branched from upstream [`JoeyDelp/JoSIM`](https://github.com/JoeyDelp/JoSIM) `v2.7`, commit `02a34ee`).

For low-level GPU kernel and linear-algebra implementation details, see [`docs/gpu_architecture.md`](gpu_architecture.md).

---

## 1. Inherited Upstream (`v2.7`) vs. New in This Fork (`v2.8`)

### What Was Inherited from Upstream `JoeyDelp/JoSIM` (`v2.7`, `02a34ee`)
- **Core SPICE Syntax & Superconducting Device Models:** RCSJ Josephson junction model (`RTYPE=0` and `RTYPE=1`, Haberkorn tunneling, multi-harmonic current-phase relations, thermal noise `.temp`/`.neb`), lossless/lossy transmission lines (`T`), mutual inductances (`K`), passive `R`/`L`/`C` elements, controlled sources (`CCCS`, `CCVS`, `VCCS`, `VCVS`), and `PWL`/`PULSE`/`SIN`/`CUS`/`PWS` waveforms.
- **Modified Nodal Analysis (MNA) & BDF2 Integration:** Phase-mode (`-a 1`, default) and Voltage-mode (`-a 0`) MNA formulation using 2nd-order Gear (BDF2) integration and single-matrix SuiteSparse KLU (`klu_l_factor` / `klu_l_tsolve`) or SuperLU (`-x 1`).
- **Standard Controls & Text Output:** `.tran`, `.print`/`.plot`/`.save`, `.param`, `.subckt`, `.include`, `.iv`, single-run `.spread`, `.option seed=<int>`, and ASCII text output (`.csv`, `.dat`, `.raw`).

### What Is New in This Fork (`v2.8` — Added Since `02a34ee`)
1. **Tier 4 + Phase 5 Schur-Complement Nodal & Series $R+L$ Chain Elimination (`-n` CPU & `-g 4` GPU):**
   - Analytically eliminates internal branch rows (`JJ` `variableIndex_` and `currentIndex_`, `R`, `C`, uncoupled `L`, `T` port currents, and grounded Dirichlet `V`/`P` sources) *plus* internal degree-2 nodes along series $R+L$, $R+R$, and $L+L$ damping/bias chains into a **2.25x–5.16x smaller** nodal matrix $A_{\text{red}}$ ($N_{\text{red}} \ll N$) and reconstructs requested `.print` branch currents on the fly.
   - Delivers **2.2x–3.2x additional single-thread speedup** on production macros (`r5cal`, `r10f1n`, `r10forig`, `xadd4`, `adder`, `coupler`) and **3.1x–4.4x speedup** on 2D JJ grids.
2. **Tier 1 + Phase 5 Per-Partition Nodal PTL Block-Parallel Solver & Short-PTL Coarsening (`-p -n` OpenMP & `-g` CUDA GPU):**
   - Combines Nodal matrix reduction (`-n`) with PTL block-parallel windowing (`-p`): automatically coarsens 0-JJ passive PTL-via islands and ultra-short PTLs ($k < 4$) while partitioning the reduced matrix $A_{\text{red}}$ across transmission lines (`T`) and shared Dirichlet supply rails (`VS`/`PS`) into $P$ tiny independent KLU blocks (`47,425 -> 9,195` eqns coarsened from `583` to `126` blocks on `adder`, $W = 3 \to 5$, max block `164` eqns; `38,826 -> 9,870` eqns across `464` blocks on `coupler`, max block `30` eqns) and advances all $P$ blocks in parallel across the causal delay window $W = \min_{p_1 \neq p_2}(T_d / \Delta t)$ steps in L1/L2 cache (`-p -n`, **up to `47.8x` CPU speedup**) or in GPU shared memory (`-g`, **up to `21.5x` GPU speedup**).
   - Includes **activity-driven quiescent partition bypass** (`.option bypass`) and NUMA-aware thread capping (`16` threads).
3. **Sub-Timestep PTL Lumped $\pi$-Replacement (`.option lumpptl`, `.option minptl`):**
   - Automatically replaces any lossless transmission line (`T`) whose propagation delay $T_d$ is smaller than a timestep ($T_d < 0.5\Delta t$ by default, or $T_d < \text{minptl}$ via `.option minptl=<td_min>`) with its exact matched lumped $L$-$C$ $\pi$-equivalent ($L_{\text{eq}} = Z_0 T_d$ in series, $C_{\text{port}} = T_d / (2 Z_0)$ shunting each port) prior to step-size clamping.
   - Eliminates artificial $40\times$ global timestep crushing from `1 fs` PTL via/pad stubs (e.g. `r10forig` drops from `1,904 s` in `v2.6.5` / `257 s` in `v2.7` / `41.6 s` previously to **`3.39 s` wall / `2.81 s` transient** — **`561.7x` faster than `v2.6.5` and `75.8x` faster than `v2.7`** with `0.000 fs` event shift).
4. **Tier 2 Batched Monte Carlo (`.spread`) Robustness Solver (`-b <N>`, `.option mc=<N>`, `.option mcall`):**
   - Parses and symbolically analyzes the netlist **once**, then executes $B = 32\text{–}4,096$ perturbed fabrication-yield trials simultaneously across OpenMP threads (`-p -b <N>`, **up to `31.1x` speedup**) or in a coalesced CUDA kernel (`-g -b <N>`, **up to `19.7x` speedup**).
   - Computes Welford online ensemble statistics (`<Trace>`, `<Trace>_STD`, `<Trace>_MIN`, `<Trace>_MAX`) and prints a **Monte Carlo Robustness Summary** table. Fixed an upstream bug in `JJ.cpp` where default `area=1.0` junctions ignored `.spread`.
5. **Tier 3 Fully GPU-Resident `cuSolverRF` Monolithic Solver (`-g 3`, `.option gpu=3`):**
   - Keeps the entire transient loop in GPU VRAM with zero per-timestep PCIe transfers (**`2.5x–4.2x` GPU speedup** on 2D JJ arrays up to `65,536` JJs).
6. **Single-Core Engine & Memory Footprint Optimizations:**
   - $O(1)$ hash-map lookups (`deviceMap`, `csMap`) for mutual inductances (`K`) and `.print` traces; in-place CSR updates and `klu_l_refactor` on `RTYPE=1` switching; pre-reserved subcircuit flattening; linear token scanning; cached `PWL`/`PWS` segments; pre-computed `CUS` splines; and homogeneous device arrays (**1.5x–3.4x faster single-thread MNA** over `v2.6.5`/`v2.7`).
   - Replaced full-history PTL `results.xVector` forcing with compact circular ring buffers (`hist_`) of length $k+4$ and added in-loop `PRSTART`/`PRSTEP` decimation (**reducing peak RAM by 2x–17.6x**).
7. **High-Speed Binary Output (`.npy`, `.bin`, `.rawb`) & New Controls:**
   - `.npy` (NumPy structured array, **2.4x–2.5x faster end-to-end runtime** on 2.5k–3k trace decks and **23x faster Python load**), `.bin` (`JOSIMBIN` column-major float64, **~37x faster write than CSV**), `.rawb` (SPICE3 binary raw), and Python loader [`scripts/josim_io.py`](../scripts/josim_io.py).
   - State checkpointing (`.savestate` / `.loadstate`), fractional PTL delay interpolation (`.option txinterp` / `interp=1`), mutual inductance cutoff (`.option kcutoff=<val>`), implicit JJ predictor-corrector (`-c [N]` / `.option corrector=<N>`), and fixed-step / step-halving diagnostics (`-f` / `.option fixedstep` / `.option warnstep`).

---

## 2. Solver Selection Guide

All solver modes produce machine-precision equivalent waveforms ($\le 10^{-14}$ relative difference). Choose the mode that matches your circuit topology and compute environment:

### Recommended Modes by Workload

- **1. Default Single-Core / Cluster Pool Job (`-n`):**
  ```bash
  josim-cli -n -o out.npy deck.cir
  ```
  Use whenever each job is allocated **1 CPU thread** (e.g. cluster pools) or when simulating monolithic circuits with $P = 1\text{–}2$ blocks (`r5cal`, `r10f1n`, `xadd4`, `dff`). Shrinks the matrix by 2x–6x via Schur-complement nodal reduction (**1.5x–2.5x faster** on production macros, **3.1x–4.2x faster** on 2D grids).

- **2. Universal Multi-Core CPU Mode (`-p -n`):**
  ```bash
  josim-cli -p -n -o out.npy deck.cir
  ```
  Best single command when multiple CPU cores are available:
  - If the circuit partitions into $P \ge 4$ blocks across PTLs or independent channels (e.g. `coupler`, `adder`), it runs the **Windowed PTL OpenMP Solver (`3.6x–27.1x` speedup)**.
  - If the circuit is monolithic ($P = 1\text{–}2$), it automatically runs the **Nodal Admittance Solver (`-n`, `1.5x–4.4x` speedup)** with zero OpenMP barrier overhead.

- **3. Multi-Block PTL Chip on NVIDIA GPU (`-g`):**
  ```bash
  josim-cli -g -o out.bin ptl_chip.cir
  ```
  For large PTL-interconnected tile arrays ($P \ge 64$ balanced blocks of $\le 512$ eqns and $W \ge 8$ steps), assigns 1 CUDA warp per partition in shared memory (**up to `21.5x` GPU speedup**). Automatically falls back if partitions are unbalanced or $W < 8$.

- **4. Large Monolithic 2D/3D JJ Array on NVIDIA GPU (`-g 3` or `-g 4`):**
  ```bash
  josim-cli -g 3 -o out.bin grid_rtype0.cir
  josim-cli -g 4 -o out.bin grid_rtype1_switching.cir
  ```
  Runs the entire transient loop in GPU VRAM via `cuSolverRF` (**`2.5x–4.2x` speedup** on `14k–65k` JJ 2D grids; `-g 4` is **`1.98x` faster than `-g 3`** when `RTYPE=1` junctions switch dynamically).

- **5. Batched Monte Carlo (`.spread`) Yield Sweep (`-p -b <N>` or `-g -b <N>`):**
  ```bash
  JOSIM_SEED=42 josim-cli -p -b 256 -o yield.npy spread_deck.cir
  JOSIM_SEED=42 josim-cli -g -b 256 -o yield.npy spread_deck.cir
  ```
  Solves $N$ perturbed fabrication-yield trials simultaneously in a single invocation (**`23x–31x` OpenMP speedup**, **`12x–20x` GPU speedup**—even on $P = 1$ monolithic circuits).

---

## 3. Command-Line (CLI) & Netlist `.option` Reference

### A. Command-Line Switches (`josim-cli [OPTIONS] <input.cir>`)

| CLI Switch | Origin | Summary |
| :--- | :--- | :--- |
| `-a <0\|1>` / `--analysis=` | Inherited (`v2.7`) | `1` = Phase mode (default), `0` = Voltage mode. |
| **`-n` / `--nodal`** | **New (`v2.8`)** | Tier 4 Schur-Complement Nodal Admittance solver ($2\times–6\times$ smaller matrix). |
| **`-p` / `--parallel`** | **Enhanced (`v2.8`)** | Multi-core OpenMP solver (Windowed PTL blocks, large meshes, batched MC). Combine as `-p -n`. |
| **`-g [1\|2\|3\|4]` / `--gpu=`** | **New (`v2.8`)** | CUDA GPU solver: `1` = Auto, `2` = Hybrid, `3` = `cuSolverRF` MNA, `4` = `cuSolverRF` Nodal. |
| **`-b <N>` / `--batch=<N>`** | **New (`v2.8`)** | Tier 2 Batched Monte Carlo solver ($N$ simultaneous `.spread` trials). |
| **`-o <file>` / `--output=`** | **Enhanced (`v2.8`)** | Output file: **`.npy`**, **`.bin`**, **`.rawb`** *(New binary formats)* or `.csv`, `.dat`, `.raw` *(Inherited)*. |
| **`-c [N]` / `--corrector=`** | **New (`v2.8`)** | Implicit predictor-corrector JJ iterations per step (default $N=2$ if value omitted). |
| **`-f` / `--fixed-step`** | **New (`v2.8`)** | Disables automatic global timestep halving on $>0.4\pi$ phase jumps. |
| `-m [0\|1]` / `--minimal` | Inherited (`v2.7`) | Suppresses progress bars and console status output. |
| `-V <1\|2\|3>` / `--Verbose=` | Inherited (`v2.7`) | Verbosity level (`1` = stats/warnings, `2` = params, `3` = flattened netlist). |
| `-s [SUBCKT]` | Inherited (`v2.7`) | Experimental subcircuit node connectivity sanity check. |
| `-i` / `--input` | Inherited (`v2.7`) | Reads netlist from `stdin` until `.end` or `EOF`. |
| `-h` / `-v` | Inherited (`v2.7`) | Displays help menu (`-h`) or version info (`-v`). |

### B. Netlist `.option` & Control Cards

| Netlist Directive | Origin | CLI Equivalent & Description |
| :--- | :--- | :--- |
| **`.option nodal`** | **New (`v2.8`)** | Same as `-n`. Enables Schur-complement nodal reduction ($A_{\text{red}}$). |
| **`.option lumpptl[=0\|1]`** | **New (`v2.8`)** | Enables/disables automatic lumped $L$-$C$ $\pi$-replacement of sub-step PTLs (default `1`; disable via `.option nolumpptl`). |
| **`.option minptl=<val>`** | **New (`v2.8`)** | Custom delay threshold $T_{d,\min}$ (seconds) below which lossless PTLs are replaced by $\pi$-equivalents (default $0.5\Delta t$). |
| **`.option gpu[=1\|2\|3\|4]`** | **New (`v2.8`)** | Same as `-g [1\|2\|3\|4]` (aliases: `.option gpumono`, `.option gpunodal`, `.option cusolverrf`). |
| **`.option bypass`** | **New (`v2.8`)** | Skips `klu_l_tsolve` on quiescent PTL partitions in `-p` mode (alias: `.option latency`). |
| **`.option bypasstol=<val>`** | **New (`v2.8`)** | Activity threshold for `.option bypass` (default `1e-12`). |
| **`.option mc=<N>`** | **New (`v2.8`)** | Same as `-b <N>` or `.spread ... runs=<N>`. Runs `N` batched Monte Carlo trials. |
| **`.option mcall`** | **New (`v2.8`)** | Appends individual `<Trace>_MC0 .. MC{N-1}` waveforms to batched Monte Carlo output. |
| **`.savestate <f> [t]`** | **New (`v2.8`)** | Saves full simulator & PTL state to binary file `<f>` at time `t` (alias: `.savebias`). |
| **`.loadstate <f>`** | **New (`v2.8`)** | Restores simulator & PTL state from `<f>` at $t=0$ (alias: `.loadbias`). |
| **`.option txinterp`** | **New (`v2.8`)** | Linearly interpolates fractional PTL delays (or per-line `interp=1`; alias: `.option fractx`). |
| **`.option kcutoff=<val>`** | **New (`v2.8`)** | Prunes mutual inductances with $\|k\| < \text{val}$ during matrix setup (alias: `.option kmin`). |
| **`.option corrector=<N>`** | **New (`v2.8`)** | Same as `-c <N>`. Enables up to `N` implicit JJ predictor-corrector iterations. |
| **`.option correctortol=<v>`** | **New (`v2.8`)** | Phase convergence threshold for `corrector` (default `1e-3` rad). |
| **`.option fixedstep`** | **New (`v2.8`)** | Same as `-f`. Disables automatic timestep halving. |
| **`.option warnstep`** | **New (`v2.8`)** | Prints diagnostic warning naming the JJ and time that triggered step halving. |
| `.option seed=<uint64>` | Enhanced (`v2.8`) | Deterministic RNG seed for noise and `.spread` (also settable via `JOSIM_SEED` env var). |

---

## 4. Benchmark Results

### A. Production `spec2si-seeqc` Benchmarks (`BNL exxact`)

End-to-end wall-clock runtime (and peak RSS memory) comparing official `v2.6.5` (`c81027a`), upstream `v2.7` (`02a34ee`), and this fork (`v2.8`) using `.npy` binary output:

| Deck (Scale & Partitions) | `v2.6.5` (CSV) | `v2.7` (CSV) | **Fork 1T MNA** | **Fork 1T `-n`** | **Fork OMP `-p -n`** | **Speedup vs `v2.6.5` (`v2.7`)** |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **`r5cal`** (3.5k JJs, $P=1$) | 762 s (0.29 GB) | 385 s | 212.6 s (0.28 GB) | **82.8 s** | **82.8 s** | **9.2x** (4.6x) |
| **`r10f1n`** (10k JJs, $P=2$) | 337 s (1.31 GB) | 173 s (0.73 GB) | 80.9 s (**0.21 GB**) | **35.5 s** | **35.5 s** | **9.5x** (4.9x) |
| **`r10forig`** (10k JJs, 1 fs PTL) | 1,904 s (6.0 GB) | 257 s (0.87 GB) | 9.46 s (**0.20 GB**) | **3.39 s** | **3.39 s** | **561.7x** (75.8x) |
| **`adder`** (5.5k JJs, $P=583\to126$) | 542 s (26.5 GB) | 325 s (15.2 GB) | 91.9 s (**8.0 GB**) | **36.3 s** | **11.33 s** | **47.8x** (28.7x) |
| **`coupler`** (2.1k JJs, $P=464$) | 156 s (7.4 GB) | 126 s | 39.5 s (**4.3 GB**) | **15.3 s** | **5.58 s** | **28.0x** (22.6x) |
| **`xadd4`** (3.5k JJs, $P=1$) | 85 s (0.42 GB) | 39 s | 24.6 s (**0.32 GB**) | **9.77 s** | **9.77 s** | **8.7x** (4.0x) |
| **`dff`** (33 JJs, $P=1$) | 0.82 s | 0.46 s | 0.29 s | **0.20 s** | **0.20 s** | **4.1x** (2.3x) |

#### Transient Solve Time & Matrix Reduction on `spec2si-seeqc` (`Fork v2.8`)

| Deck | Full MNA ($N$) | Nodal ($N_{\text{red}}$) | Serial MNA Solve | **1T Nodal (`-n`) Solve** | **OpenMP (`-p -n`) Solve** | **Best Solve Speedup** |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **`r5cal`** ($P=1, W=1$) | 64,400 | **29,622** (`2.17x`) | 212.10 s | **82.04 s** | **82.04 s** | **2.59x** (`-n`) |
| **`r10f1n`** ($P=2, W=1$) | 89,806 | **39,976** (`2.25x`) | 80.29 s | **34.90 s** | **34.90 s** | **2.30x** (`-n`) |
| **`r10forig`** ($P=2, W=1$) | 89,810 | **39,978** (`2.25x`) | 130.63 s (`8.89 s` lumped) | **2.81 s** | **2.81 s** | **46.49x** (`-n`) |
| **`xadd4`** ($P=1, W=1$) | 31,813 | **9,041** (`3.52x`) | 24.06 s | **9.18 s** | **9.18 s** | **2.62x** (`-n`) |
| **`coupler`** ($P=464, W=420$) | 38,826 | **9,870** (`3.93x`) | 36.19 s | **12.14 s** (`2.98x`) | **2.71 s** | **13.38x** (`-p -n`) |
| **`adder`** ($P=583\to126, W=3\to5$) | 47,425 | **9,195** (`5.16x`) | 83.34 s | **28.06 s** (`2.97x`) | **5.47 s** | **15.24x** (`-p -n`) |
| **`dff`** ($P=1, W=1$) | 504 | **174** (`2.90x`) | 0.234 s | **0.141 s** | **0.141 s** | **1.66x** (`-n`) |

---

### B. Tier 4 Nodal (`-n`, `-g 4`) & Tier 3 (`-g 3`) on 2D Monolithic JJ Grids (`BNL exxact`)

1,001-timestep transient solve time on 0-PTL single-partition ($P=1$) 2D Josephson junction grids with dynamic `RTYPE=1` switching:

| 2D Grid (`RTYPE=1` Switching) | Eqns ($N \to N_{\text{red}}$) | Serial MNA | **CPU Nodal (`-n`)** | **OMP Nodal (`-p -n`)** | GPU Tier 3 (`-g 3`) | **GPU Tier 4 (`-g 4`)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`Mono-2D-Grid-14kJJ`** ($120\times120$) | $86\text{k} \to \mathbf{14.4\text{k}}$ (`6.0x`) | 4.64 s | **1.11 s (`4.19x`)** | **1.06 s (`4.37x`)** | 3.47 s (`1.34x`) | **3.09 s (`1.50x`)** |
| **`Mono-2D-Grid-25kJJ`** ($160\times160$) | $153\text{k} \to \mathbf{25.6\text{k}}$ (`6.0x`) | 8.88 s | **2.84 s (`3.13x`)** | **2.21 s (`4.01x`)** | 4.41 s (`2.02x`) | **3.82 s (`2.32x`)** |
| **`Mono-2D-Grid-64kJJ`** ($256\times256$) | $393\text{k} \to \mathbf{65.5\text{k}}$ (`6.0x`) | 29.04 s | **9.29 s (`3.13x`)** | **8.65 s (`3.36x`)** | 34.09 s (`0.85x`) | **17.22 s (`1.69x`, `1.98x` vs `-g 3`)** |

When junctions do not dynamically switch (`RTYPE=0` or quiescent bias), Tier 3 `cuSolverRF` (`-g 3` / `-g` auto) achieves **`3.86x–4.21x` GPU speedup** on `14kJJ` (`4.81 s -> 1.14 s`), **`3.70x`** on `25kJJ` (`8.87 s -> 2.40 s`), and **`2.57x`** on `64kJJ` (`30.46 s -> 11.87 s`).

---

### C. Tier 1 Windowed PTL Block-Parallel & Multi-Channel Benchmarks (`BNL exxact`)

Reproducible via [`scripts/benchmark_gpu_exxact.py`](../scripts/benchmark_gpu_exxact.py) and [`scripts/benchmark_non_ptl_exxact.py`](../scripts/benchmark_non_ptl_exxact.py):

| Circuit | Scale & Topology | Serial (1T) | **OpenMP (`-p`)** | **CUDA GPU (`-g`)** | **OpenMP Speedup** | **GPU Speedup** |
| :--- | :--- | ---: | ---: | ---: | :---: | :---: |
| **`PTL-Tile-64`** | 2,048 JJs, 10.6k eqns ($P=64, W=100$) | 7.16 s | **0.61 s** | 5.69 s | **11.78x** | **1.26x** |
| **`PTL-Tile-256`** | 8,192 JJs, 42.2k eqns ($P=256, W=100$) | 29.88 s | **2.06 s** | **6.00 s** | **14.54x** | **4.98x** |
| **`PTL-Tile-1024`** | 32,768 JJs, 169k eqns ($P=1024, W=100$) | 104.90 s | **4.20 s** | **6.95 s** | **24.95x** | **15.10x** |
| **`PTL-Tile-4096`** | 131,072 JJs, 676k eqns ($P=4096, W=100$) | 232.89 s | **8.58 s** | **10.85 s** | **27.15x** | **21.47x** |
| **`NoPTL-SQUID-1024`** | 32k JJs, 98k `K`, 299k eqns (**0 PTL**, $P=1024$) | 75.82 s | **3.52 s** | **11.20 s** | **21.55x** | **6.77x** |
| **`NoPTL-SQUID-4096`** | 131k JJs, 393k `K`, 1.20M eqns (**0 PTL**, $P=4096$) | 185.79 s | **7.44 s** | **18.04 s** | **24.98x** | **10.30x** |
| **`RSFQ-ALU-Macros-4096`** | 131k JJs, 393k `K`, 131k `C`, 1.21M eqns ($P=4096$) | 157.08 s | **7.50 s** | **18.29 s** | **20.93x** | **8.59x** |

---

### D. Tier 2 Batched Monte Carlo (`.spread`) Yield Benchmarks (`BNL exxact`)

Reproducible via [`scripts/benchmark_mc_exxact.py`](../scripts/benchmark_mc_exxact.py):

| Design (`.spread` Yield Analysis) | Base $P$ | Trials ($B$) | Est. $B \times$ Serial 1T | **Batched OpenMP (`-p`)** | **Batched GPU (`-g`)** | **OpenMP Speedup** | **GPU Speedup** |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| **`ex_mitll_xor_wr`** (MIT-LL XOR) | 2 | 1024 | 8.03 s | **0.30 s** | **1.23 s** | **26.48x** | **6.53x** |
| **`ex_mitll_dff_wr`** (MIT-LL DFF) | 2 | 1024 | 11.30 s | **0.40 s** | **1.56 s** | **28.16x** | **7.24x** |
| **`Mono-RSFQ-Cell-Macro`** (128 JJs, 0-PTL) | **1** | 1024 | 16.37 s | **0.67 s** | **4.97 s** | **24.32x** | **3.29x** |
| **`Mono-RLK-SQUID-Macro`** (96 JJs, 191 `K`, 0-PTL) | **1** | 1024 | 19.36 s | **0.80 s** | **5.51 s** | **24.34x** | **3.52x** |
| **`ex_ksa4bit`** (4-bit KSA Adder, 695 JJs) | 66 | 64 | 51.25 s | **2.04 s** | **4.31 s** | **25.07x** | **11.89x** |
| **`ex_ksa4bit`** (4-bit KSA Adder, 695 JJs) | 66 | 256 | 204.24 s | **6.56 s** | **10.39 s** | **31.14x** | **19.65x** |
| **`RSFQ-Pipeline-64B`** (2,048 JJs, 4.85M eqns) | 64 | 256 | 48.57 s | **1.68 s** | **3.19 s** | **28.87x** | **15.21x** |
