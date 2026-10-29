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
1. **Schur-Complement Nodal Admittance & Series $R+L$ Chain Elimination (`-n` CPU & `-g 4` GPU):**
   - Analytically eliminates internal branch rows (`JJ` `variableIndex_` and `currentIndex_`, `R`, `C`, uncoupled `L`, `T` port currents, and grounded Dirichlet `V`/`P` sources) *plus* internal degree-2 nodes along series $R+L$, $R+R$, and $L+L$ damping/bias chains into a **2.25x–5.16x smaller** nodal matrix $A_{\text{red}}$ ($N_{\text{red}} \ll N$) and reconstructs requested `.print` branch currents on the fly.
   - Delivers **2.2x–3.2x additional single-thread speedup** on production macros (`r5cal`, `r10f1n`, `r10forig`, `xadd4`, `adder`, `coupler`) and **3.1x–4.4x speedup** on 2D JJ grids.
2. **Per-Partition Nodal PTL Block-Parallel Solver & Short-PTL Coarsening (`-p -n` OpenMP & `-g` CUDA GPU):**
   - Combines Nodal matrix reduction (`-n`) with PTL block-parallel windowing (`-p`): automatically coarsens 0-JJ passive PTL-via islands and ultra-short PTLs ($k < 4$) while partitioning the reduced matrix $A_{\text{red}}$ across transmission lines (`T`) and shared Dirichlet supply rails (`VS`/`PS`) into $P$ tiny independent KLU blocks (`47,425 -> 9,195` eqns coarsened from `583` to `126` blocks on `adder`, $W = 3 \to 5$, max block `164` eqns; `38,826 -> 9,870` eqns across `464` blocks on `coupler`, max block `30` eqns) and advances all $P$ blocks in parallel across the causal delay window $W = \min_{p_1 \neq p_2}(T_d / \Delta t)$ steps in L1/L2 cache (`-p -n`, **up to `75.3x` CPU speedup**) or in GPU shared memory (`-g`, **up to `21.5x` GPU speedup**).
   - Includes **activity-driven quiescent partition bypass** (`.option bypass`) and NUMA-aware thread capping (`16` threads).
3. **Sub-Timestep PTL Lumped $\pi$-Replacement (`.option lumpptl`, `.option minptl`):**
   - Automatically replaces any lossless transmission line (`T`) whose propagation delay $T_d$ is smaller than a timestep ($T_d < 0.5\Delta t$ by default, or $T_d < \text{minptl}$ via `.option minptl=<td_min>`) with its exact matched lumped $L$-$C$ $\pi$-equivalent ($L_{\text{eq}} = Z_0 T_d$ in series, $C_{\text{port}} = T_d / (2 Z_0)$ shunting each port) prior to step-size clamping.
   - Eliminates artificial $40\times$ global timestep crushing from `1 fs` PTL via/pad stubs (e.g. `r10forig` drops from `1,904 s` in `v2.6.5` / `257 s` in `v2.7` / `41.6 s` previously to **`3.22 s` wall / `2.66 s` transient** — **`591.3x` faster than `v2.6.5` and `79.8x` faster than `v2.7`** with `0.000 fs` event shift).
4. **Direct In-Loop Trace Recording, Zero-Allocation Streaming `.npy`/`.bin` Output & Built-In SFQ Event/Pulse Extractor (`-e`, `-E`):**
   - Drops unused JJ `variableIndex_` trace vectors from `relevantIndices`, folds exclusive `V(n1, n2)` and `P(n1, n2)` node-difference pairs into single difference vectors (`traceType = 4`), parallelizes NUMA trace allocation across OpenMP threads, and streams `.npy`/`.bin`/`.rawb` directly from `results.xVector` via a 64×8 L1 cache-tiled transpose without allocating `Output::traces`.
   - Cuts `adder` full-waveform peak RSS memory from **`26.5 GB` (`v2.6.5`) / `8.04 GB` to `1.72 GB` (`15.4x` lower vs `v2.6.5`)** and `coupler` peak RSS from **`7.4 GB` (`v2.6.5`) / `4.3 GB` to `1.42 GB`**.
   - **Built-In SFQ Switching Event & Pulse Metric Extractor (`-e [file]`, `-E [spec]`, `.option events`, `.option eventmetrics`):** Detects $2\pi$ Josephson phase-slip crossings across $(2k-1)\pi$ barriers with sub-step linear interpolation (`time`, `polarity`, `winding`, `phase`) and user-selectable pulse metrics (`t_peak`, `v_peak`, `fwhm`, `dt_pi`, `area`, and `time=peak` vs `time=cross`). When invoked without `-o` (**Events-Only Mode**), tracks events in-loop without allocating `results.xVector`, cutting `adder -p -n -e` to **`4.63 s` wall / `195 MB` RSS / `606 KB` `.events.npy` (`5,250x` smaller than `.npy`, `117.1x` faster than `v2.6.5`)** and `coupler -p -n -e` to **`2.58 s` wall / `216 MB` RSS / `512 KB` `.events.npy` (`2,503x` smaller)**.
5. **Batched Monte Carlo (`.spread`) & Deterministic Parameter/Margin Sweep (`.sweep`, `-S`) Solver (`-b <N>`, `-p -n`, `-g`):**
   - Parses and symbolically analyzes the netlist **once**, then executes $B = 2\text{–}4,096$ perturbed fabrication-yield trials (`.spread`) or deterministic 1D/2D parameter/margin sweep points (`.sweep`, `.option sweep=...`, `-S <spec>`) simultaneously across OpenMP threads with Batched Schur-Complement Nodal reduction (`-p -n`, **up to `83.7x` Monte Carlo speedup** and **`33.3x–51.8x` sweep speedup vs sequential MNA**), OpenMP MNA (`-p`, **up to `31.1x` speedup**), or a coalesced CUDA kernel (`-g`, **up to `19.7x` speedup**).
   - Supports sweeping `.param` identifiers (re-evaluating all dependent parameter expressions and inline `{...}` expressions per sweep point), device categories (`@BIAS`, `@JJ`/`@B`, `@IND`/`@L`, `@RES`/`@R`, `@CAP`/`@C`, `@SRC`, `@ISRC`, `@VSRC`, `@PSRC`, `@GLOBAL`), and specific device labels or glob wildcards (`IB*`, `VBIAS*`, `B01*`, `L*`, `R*`) via linear steps (`<start> <stop> <step>` or `LIN <count> <start> <stop>`), relative margin windows (`MARGIN <delta> [<count>]`), explicit value lists (`LIST <v1> ... <vB>`), and 2D Cartesian products (two `.sweep` cards or two `-S` flags).
   - Computes Welford online ensemble statistics (`<Trace>`, `<Trace>_STD`, `<Trace>_MIN`, `<Trace>_MAX`), optional per-point waveforms (`<Trace>_MC0 .. <Trace>_MC{B-1}` via `.option mcall` / `.option sweepall`), and prints a **Monte Carlo Robustness Summary** or **Parameter Sweep / Margin Summary** table with automatic SFQ operating pass/fail margin window detection around nominal $N_{\text{SFQ}} = \text{round}(\Delta\phi / 2\pi)$. Fixed an upstream bug in `JJ.cpp` where default `area=1.0` junctions ignored `.spread`.
6. **Fully GPU-Resident `cuSolverRF` Monolithic Solver (`-g 3`, `-g 4`, `.option gpu=3/4`):**
   - Keeps the entire transient loop in GPU VRAM with zero per-timestep PCIe transfers (**`2.5x–4.2x` GPU speedup** on 2D JJ arrays up to `65,536` JJs).
7. **Single-Core Engine, Memory Footprint & Local Rollback Optimizations:**
   - $O(1)$ hash-map lookups (`deviceMap`, `csMap`) for mutual inductances (`K`) and `.print` traces; in-place CSR updates and `klu_l_refactor` on `RTYPE=1` switching; pre-reserved subcircuit flattening; linear token scanning; cached `PWL`/`PWS` segments; pre-computed `CUS` splines; and homogeneous device arrays (**1.5x–3.4x faster single-thread MNA** over `v2.6.5`/`v2.7`).
   - Replaced full-history PTL `results.xVector` forcing with compact circular ring buffers (`hist_`) of length $k+4$ and added in-loop `PRSTART`/`PRSTEP` decimation (**reducing peak RAM by 2x–17.6x**).
   - **In-Memory Rolling Checkpoints for Local Step-Halving Rollback (`.option rollback[=1|2|window|stay]`, `.option norollback`):** Maintains a double-buffered 128-step rolling checkpoint (`ckpt_[2]`, `<2 MB`) in both Standard MNA and CPU Nodal (`-n`). When a junction phase jump exceeds $0.4\pi$ (`needsTR_`), JoSIM rolls back only $32\text{–}256$ base steps to the nearest safe checkpoint, rescales BDF2 two-step history ($h_{\text{old}} \to h_{\text{new}}$) and matrix diagonal entries in place (`klu_l_refactor`), and sub-steps locally ($M = 2, 4, 8, 16$) instead of restarting from $t = 0$ (**2.02x–3.93x faster** across Phase & Voltage modes).
8. **High-Speed Binary Output (`.npy`, `.bin`, `.rawb`) & New Controls:**
   - `.npy` (NumPy structured array, **2.4x–2.5x faster end-to-end runtime** on 2.5k–3k trace decks and **23x faster Python load**), `.bin` (`JOSIMBIN` column-major float64, **~37x faster write than CSV**), `.rawb` (SPICE3 binary raw), and Python loader [`scripts/josim_io.py`](../scripts/josim_io.py).
   - State checkpointing (`.savestate` / `.loadstate`), local step-halving rollback (`.option rollback[=1|2|window|stay]` / `.option norollback`), fractional PTL delay interpolation (`.option txinterp` / `interp=1`), mutual inductance cutoff (`.option kcutoff=<val>`), implicit JJ predictor-corrector (`-c [N]` / `.option corrector=<N>`), and fixed-step / step-halving diagnostics (`-f` / `.option fixedstep` / `.option warnstep`).

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

- **5. Batched Monte Carlo (`.spread`) Yield Sweep & Deterministic Parameter/Margin Sweep (`.sweep`, `-S`):**
  ```bash
  JOSIM_SEED=42 josim-cli -p -n -b 256 -o yield.npy spread_deck.cir
  JOSIM_SEED=42 josim-cli -g -b 256 -o yield.npy spread_deck.cir
  josim-cli -p -n -S "@BIAS:MARGIN:0.3:21" -o margin.npy deck.cir
  josim-cli -p -n -S "Scaling:LIN:21:0.7:1.3" -o sweep.npy deck.cir
  ```
  Solves $N$ perturbed fabrication-yield trials or deterministic 1D/2D parameter/margin sweep points simultaneously in a single invocation (**`23x–84x` OpenMP CPU speedup (`-p -n`)**, **`33x–52x` deterministic sweep speedup vs sequential MNA**, **`12x–20x` GPU speedup**—even on $P = 1$ monolithic circuits).

---

## 3. Command-Line (CLI) & Netlist `.option` Reference

### A. Command-Line Switches (`josim-cli [OPTIONS] <input.cir>`)

| CLI Switch | Origin | Summary |
| :--- | :--- | :--- |
| `-a <0\|1>` / `--analysis=` | Inherited (`v2.7`) | `1` = Phase mode (default), `0` = Voltage mode. |
| **`-n` / `--nodal`** | **New (`v2.8`)** | Schur-Complement Nodal Admittance solver ($2\times–6\times$ smaller matrix). |
| **`-p` / `--parallel`** | **Enhanced (`v2.8`)** | Multi-core OpenMP solver (Windowed PTL blocks, large meshes, batched MC). Combine as `-p -n`. |
| **`-g [1\|2\|3\|4]` / `--gpu=`** | **New (`v2.8`)** | CUDA GPU solver: `1` = Auto, `2` = Hybrid, `3` = `cuSolverRF` MNA, `4` = `cuSolverRF` Nodal. |
| **`-b <N>` / `--batch=<N>`** | **New (`v2.8`)** | Batched Monte Carlo solver ($N$ simultaneous `.spread` trials; combine with `-p -n` for Batched Nodal). |
| **`-S <spec>` / `--sweep=<spec>`** | **New (`v2.8`)** | Deterministic parameter/margin sweep (e.g. `-S "@BIAS:MARGIN:0.3:21"` or `-S "Scaling:LIN:21:0.7:1.3"`). Pass twice for 2D Cartesian grid. |
| **`-o <file>` / `--output=`** | **Enhanced (`v2.8`)** | Output file: **`.npy`**, **`.bin`**, **`.rawb`** *(New binary formats)* or `.csv`, `.dat`, `.raw` *(Inherited)*. |
| **`-e [file]` / `--events=`** | **New (`v2.8`)** | Extracts $2\pi$ SFQ phase-slip events to `.npy` or `.csv`. Without `-o`, runs in zero-allocation **Events-Only Mode**. |
| **`-E [spec]` / `--event-metrics=`** | **New (`v2.8`)** | Selects SFQ pulse metrics: `pulse` (default: `t_peak,v_peak,fwhm,dt_pi`), `all` (`+area`), or comma-separated list including `time=peak` / `time=cross`. |
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
| **`.option events[=file]`** | **New (`v2.8`)** | Same as `-e [file]`. Extracts $2\pi$ SFQ switching events (use `.option eventsonly[=file]` to force Events-Only mode). |
| **`.option eventmetrics[=spec]`** | **New (`v2.8`)** | Same as `-E [spec]` (alias: `.option eventpulse`). Appends `t_peak`, `v_peak`, `fwhm`, `dt_pi`, `area`, or sets `time=peak`. |
| **`.option lumpptl[=0\|1]`** | **New (`v2.8`)** | Enables/disables automatic lumped $L$-$C$ $\pi$-replacement of sub-step PTLs (default `1`; disable via `.option nolumpptl`). |
| **`.option minptl=<val>`** | **New (`v2.8`)** | Custom delay threshold $T_{d,\min}$ (seconds) below which lossless PTLs are replaced by $\pi$-equivalents (default $0.5\Delta t$). |
| **`.option gpu[=1\|2\|3\|4]`** | **New (`v2.8`)** | Same as `-g [1\|2\|3\|4]` (aliases: `.option gpumono`, `.option gpunodal`, `.option cusolverrf`). |
| **`.option bypass`** | **New (`v2.8`)** | Skips `klu_l_tsolve` on quiescent PTL partitions in `-p` mode (alias: `.option latency`). |
| **`.option bypasstol=<val>`** | **New (`v2.8`)** | Activity threshold for `.option bypass` (default `1e-12`). |
| **`.option mc=<N>`** | **New (`v2.8`)** | Same as `-b <N>` or `.spread ... runs=<N>`. Runs `N` batched Monte Carlo trials. |
| **`.option mcall`** | **New (`v2.8`)** | Appends individual `<Trace>_MC0 .. MC{N-1}` waveforms to batched Monte Carlo or `.sweep` output (alias: `.option sweepall`). |
| **`.sweep <target> ...`** | **New (`v2.8`)** | Deterministic parameter/margin sweep card: `.sweep <target> <start> <stop> <step>`, `.sweep <target> LIN <count> <start> <stop>`, `.sweep <target> MARGIN <delta> [<count>]`, or `.sweep <target> LIST <v1> ... <vB>`. Use two `.sweep` cards for a 2D Cartesian grid. |
| **`.option sweep=<spec>`** | **New (`v2.8`)** | Same as `-S <spec>` / `.sweep` (e.g. `.option sweep=@BIAS:MARGIN:0.3:21`). |
| **`.option sweepall`** | **New (`v2.8`)** | Same as `.option mcall`. Appends individual `<Trace>_MC0 .. MC{B-1}` sweep waveforms to output. |
| **`.savestate <f> [t]`** | **New (`v2.8`)** | Saves full simulator & PTL state to binary file `<f>` at time `t` (alias: `.savebias`). |
| **`.loadstate <f>`** | **New (`v2.8`)** | Restores simulator & PTL state from `<f>` at $t=0$ (alias: `.loadbias`). |
| **`.option txinterp`** | **New (`v2.8`)** | Linearly interpolates fractional PTL delays (or per-line `interp=1`; alias: `.option fractx`). |
| **`.option kcutoff=<val>`** | **New (`v2.8`)** | Prunes mutual inductances with $\|k\| < \text{val}$ during matrix setup (alias: `.option kmin`). |
| **`.option corrector=<N>`** | **New (`v2.8`)** | Same as `-c <N>`. Enables up to `N` implicit JJ predictor-corrector iterations. |
| **`.option correctortol=<v>`** | **New (`v2.8`)** | Phase convergence threshold for `corrector` (default `1e-3` rad). |
| **`.option fixedstep`** | **New (`v2.8`)** | Same as `-f`. Disables automatic timestep halving. |
| **`.option rollback[=m]`** | **New (`v2.8`)** | Controls in-memory rolling checkpoint rollback on $>0.4\pi$ phase jumps: `1` or `window` (default: local rollback + adaptive sub-step window), `2` or `stay` (local rollback + keep sub-step rate for remainder of run), `0` (disable; alias: `.option norollback`, full restart from $t=0$). |
| **`.option warnstep`** | **New (`v2.8`)** | Prints diagnostic warning naming the JJ and time that triggered step halving. |
| `.option seed=<uint64>` | Enhanced (`v2.8`) | Deterministic RNG seed for noise and `.spread` (also settable via `JOSIM_SEED` env var). |

---

## 4. Benchmark Results

*Test Platform: Dual-socket AMD EPYC 7413 (2 × 24 cores / 96 threads across 2 NUMA nodes), 512 GB DDR4 RAM, NVIDIA RTX A6000 GPU (48 GB GDDR6 VRAM).*

### A. Production `spec2si-seeqc` Benchmarks

End-to-end wall-clock runtime (and peak RSS memory) comparing official `v2.6.5` (`c81027a`), upstream `v2.7` (`02a34ee`), and this fork (`v2.8`) using `.npy` binary output:

| Deck (Scale & Partitions) | `v2.6.5` (CSV) | `v2.7` (CSV) | **Fork 1T MNA** | **Fork 1T `-n`** | **Fork OMP `-p -n`** | **Speedup vs `v2.6.5` (`v2.7`)** |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **`r5cal`** (3.5k JJs, $P=1$) | 762 s (0.29 GB) | 385 s | 212.6 s (0.28 GB) | **82.8 s** (0.27 GB) | **82.8 s** | **9.2x** (4.6x) |
| **`r10f1n`** (10k JJs, $P=2$) | 337 s (1.31 GB) | 173 s (0.73 GB) | 80.0 s (**0.30 GB**) | **27.4 s** (**0.28 GB**) | **27.4 s** | **12.3x** (6.3x) |
| **`r10forig`** (10k JJs, 1 fs PTL) | 1,904 s (6.0 GB) | 257 s (0.87 GB) | 9.00 s (**0.28 GB**) | **3.22 s** (**0.28 GB**) | **3.22 s** | **591.3x** (79.8x) |
| **`adder`** (5.5k JJs, $P=583\to126$) | 542 s (26.5 GB) | 325 s (15.2 GB) | 83.9 s (**3.24 GB**) | **26.2 s** (**1.72 GB**) | **7.20 s** (**1.72 GB**) | **75.3x** (45.1x) |
| **`coupler`** (2.1k JJs, $P=464$) | 156 s (7.4 GB) | 126 s | 35.2 s (**2.12 GB**) | **12.6 s** (**1.43 GB**) | **4.08 s** (**1.42 GB**) | **38.2x** (30.9x) |
| **`xadd4`** (3.5k JJs, $P=1$) | 85 s (0.42 GB) | 39 s | 24.4 s (**0.25 GB**) | **9.83 s** (**0.22 GB**) | **9.83 s** | **8.6x** (4.0x) |
| **`dff`** (33 JJs, $P=1$) | 0.82 s | 0.46 s | 0.31 s | **0.20 s** (**0.13 GB**) | **0.20 s** | **4.1x** (2.3x) |

#### Transient Solve Time & Matrix Reduction on `spec2si-seeqc` (`Fork v2.8`)

| Deck | Full MNA ($N$) | Nodal ($N_{\text{red}}$) | Serial MNA Solve | **1T Nodal (`-n`) Solve** | **OpenMP (`-p -n`) Solve** | **Best Solve Speedup** |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **`r5cal`** ($P=1, W=1$) | 64,400 | **28,093** (`2.29x`) | 212.10 s | **82.04 s** | **82.04 s** | **2.59x** (`-n`) |
| **`r10f1n`** ($P=2, W=1$) | 89,806 | **39,976** (`2.25x`) | 79.42 s | **26.81 s** | **26.81 s** | **2.96x** (`-n`) |
| **`r10forig`** ($P=2, W=1$) | 89,810 | **39,978** (`2.25x`) | 130.63 s (`8.41 s` lumped) | **2.66 s** | **2.66 s** | **49.17x** (`-n`) |
| **`xadd4`** ($P=1, W=1$) | 31,813 | **9,041** (`3.52x`) | 24.07 s | **9.49 s** | **9.49 s** | **2.54x** (`-n`) |
| **`coupler`** ($P=464, W=420$) | 38,826 | **9,870** (`3.93x`) | 33.44 s | **10.94 s** (`3.06x`) | **2.41 s** | **13.85x** (`-p -n`) |
| **`adder`** ($P=583\to126, W=3\to5$) | 47,425 | **9,195** (`5.16x`) | 79.92 s | **22.65 s** (`3.53x`) | **3.79 s** | **21.11x** (`-p -n`) |
| **`dff`** ($P=1, W=1$) | 504 | **174** (`2.90x`) | 0.234 s | **0.137 s** | **0.137 s** | **1.71x** (`-n`) |

#### Built-In SFQ Switching Event & Pulse Metric Extractor (`-e`, `-E all`) on `spec2si-seeqc`

| Deck (Events Extracted) | Full `.npy` (`-o + -e`) | **Events-Only (`-e`)** | **Events + Pulse Metrics (`-e -E all`)** | **Output Size: `.npy` $\to$ `-e` (`-E all`)** | **Median Extracted Pulse Metrics (`-E all`)** |
| :--- | ---: | ---: | ---: | :---: | :--- |
| **`adder`** (`11,920` events) | 9.16 s (1.72 GB) | **4.63 s** (**0.19 GB**) | **4.79 s** (**0.23 GB**) | `3.18 GB` $\to$ **`605.5 KB`** (`1.07 MB`) | $\|V_{\text{peak}}\|=0.302\text{ mV}$, $\text{FWHM}=4.49\text{ ps}$, $\Delta t_\pi=7.65\text{ ps}$, $\|\text{area}\|=1.81\text{ fV}\cdot\text{s}$ |
| **`coupler`** (`12,784` events) | 4.59 s (1.42 GB) | **2.58 s** (**0.21 GB**) | **2.43 s** (**0.27 GB**) | `1.28 GB` $\to$ **`512.0 KB`** (`1.01 MB`) | $\|V_{\text{peak}}\|=0.426\text{ mV}$, $\text{FWHM}=3.31\text{ ps}$, $\Delta t_\pi=7.03\text{ ps}$, $\|\text{area}\|=2.14\text{ fV}\cdot\text{s}$ |
| **`xadd4`** (`56` events) | 9.58 s (0.22 GB) | **9.33 s** (**0.17 GB**) | **9.87 s** (**0.18 GB**) | `112.5 MB` $\to$ **`3.1 KB`** (`5.5 KB`) | $\|V_{\text{peak}}\|=0.319\text{ mV}$, $\text{FWHM}=4.94\text{ ps}$, $\Delta t_\pi=5.37\text{ ps}$, $\|\text{area}\|=1.83\text{ fV}\cdot\text{s}$ |
| **`r10f1n`** (`4` events) | 27.36 s (0.28 GB) | **27.04 s** (**0.29 GB**) | **27.02 s** (**0.29 GB**) | `10.0 MB` $\to$ **`0.3 KB`** (`0.6 KB`) | $\|V_{\text{peak}}\|=0.319\text{ mV}$, $\text{FWHM}=4.01\text{ ps}$, $\Delta t_\pi=11.25\text{ ps}$, $\|\text{area}\|=2.33\text{ fV}\cdot\text{s}$ |
| **`r10forig`** (`4` events) | 3.18 s (0.28 GB) | **3.18 s** (**0.29 GB**) | **3.18 s** (**0.29 GB**) | `1.0 MB` $\to$ **`0.3 KB`** (`0.6 KB`) | $\|V_{\text{peak}}\|=0.319\text{ mV}$, $\text{FWHM}=4.01\text{ ps}$, $\Delta t_\pi=11.25\text{ ps}$, $\|\text{area}\|=2.33\text{ fV}\cdot\text{s}$ |
| **`dff`** (`9` events) | 0.21 s (0.13 GB) | **0.20 s** (**0.13 GB**) | **0.20 s** (**0.13 GB**) | `1.7 MB` $\to$ **`0.5 KB`** (`1.0 KB`) | $\|V_{\text{peak}}\|=0.277\text{ mV}$, $\text{FWHM}=4.69\text{ ps}$, $\Delta t_\pi=6.78\text{ ps}$, $\|\text{area}\|=2.09\text{ fV}\cdot\text{s}$ |

#### In-Memory Rolling Checkpoints & Local Step-Halving Rollback (`.option rollback`)

When a Josephson junction exceeds the $0.4\pi$ per-step phase-jump threshold (`needsTR_`), upstream JoSIM discards all computed work and restarts the entire transient simulation from $t = 0$ (`.option norollback`). With **in-memory rolling checkpoints** (`.option rollback=1` [`window`, default] or `.option rollback=2` [`stay`]), JoSIM rolls back only $32\text{–}256$ base steps to the nearest 128-step rolling checkpoint and sub-steps locally ($M = 2, 4, 8, 16$):

| Circuit & Step-Halving Scenario | Solver Mode | `.option norollback` (Restart from $t=0$) | **`.option rollback=1` (`window`, default)** | **`.option rollback=2` (`stay`)** | **Best Speedup (Final $\Phi$ Error)** |
| :--- | :--- | ---: | ---: | ---: | :---: |
| **166-JJ MIT-LL DCSFQ + 80-JTL Chain**<br>(`tstop=2000ps`, `tstep=1.0ps`, 3 halvings $\to 0.125\text{ ps}$) | Standard MNA (Phase)<br>Standard MNA (Voltage)<br>CPU Nodal (`-n`, Phase)<br>CPU Nodal (`-n`, Voltage)<br>Events-Only (`-n`, `eventsonly=peak,pulse`) | 1,077.2 ms<br>1,168.2 ms<br>501.4 ms<br>654.8 ms<br>493.3 ms | **387.6 ms (`2.78x`)**<br>**401.3 ms (`2.91x`)**<br>231.2 ms (`2.17x`)<br>**264.1 ms (`2.48x`)**<br>228.6 ms (`2.16x`) | 411.7 ms (`2.62x`)<br>443.9 ms (`2.63x`)<br>**227.5 ms (`2.20x`)**<br>280.9 ms (`2.33x`)<br>**223.4 ms (`2.21x`)** | **`2.78x`** (`0.0000 rad`)<br>**`2.91x`** (`0.0000 rad`)<br>**`2.20x`** (`0.0000 rad`)<br>**`2.48x`** (`0.0000 rad`)<br>**`2.21x`** (`8/8` events) |
| **2,006-JJ MIT-LL DCSFQ + 1,000-JTL Chain**<br>(`tstop=2000ps`, `tstep=0.5ps`, burst at $1600\text{ ps}$) | CPU Nodal (`-n`, Phase)<br>CPU Nodal (`-n`, Voltage)<br>Standard MNA (Phase)<br>Standard MNA (Voltage) | 2,430.0 ms<br>3,280.9 ms<br>6,580.9 ms<br>12,422.1 ms | 1,204.5 ms (`2.02x`)<br>**1,539.9 ms (`2.13x`)**<br>**2,912.3 ms (`2.26x`)**<br>4,947.2 ms (`2.51x`) | **1,192.9 ms (`2.04x`)**<br>1,563.2 ms (`2.10x`)<br>3,100.5 ms (`2.12x`)<br>**4,840.3 ms (`2.57x`)** | **`2.04x`** (`0.0000 rad`)<br>**`2.13x`** (`0.0000 rad`)<br>**`2.26x`** (`0.0000 rad`)<br>**`2.57x`** (`0.0000 rad`) |
| **2,006-JJ MIT-LL DCSFQ + 1,000-JTL Chain**<br>(`tstop=2000ps`, `tstep=1.0ps`, 2 halvings $\to 0.25\text{ ps}$) | CPU Nodal (`-n`, Phase)<br>CPU Nodal (`-n`, Voltage)<br>Standard MNA (Phase)<br>Standard MNA (Voltage) | 2,631.5 ms<br>3,460.5 ms<br>6,591.5 ms<br>13,128.2 ms | 1,201.8 ms (`2.19x`)<br>1,591.2 ms (`2.17x`)<br>3,152.3 ms (`2.09x`)<br>5,584.9 ms (`2.35x`) | **1,197.1 ms (`2.20x`)**<br>**1,558.3 ms (`2.22x`)**<br>**3,003.1 ms (`2.19x`)**<br>**3,336.8 ms (`3.93x`)** | **`2.20x`** (`0.0000 rad`)<br>**`2.22x`** (`0.0000 rad`)<br>**`2.19x`** (`0.0000 rad`)<br>**`3.93x`** (`0.0000 rad`) |

---

### B. Nodal (`-n`, `-g 4`) & GPU Monolithic (`-g 3`) Solvers on 2D JJ Grids

1,001-timestep transient solve time on 0-PTL single-partition ($P=1$) 2D Josephson junction grids with dynamic `RTYPE=1` switching:

| 2D Grid (`RTYPE=1` Switching) | Eqns ($N \to N_{\text{red}}$) | Serial MNA | **CPU Nodal (`-n`)** | **OMP Nodal (`-p -n`)** | GPU MNA (`-g 3`) | **GPU Nodal (`-g 4`)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`Mono-2D-Grid-14kJJ`** ($120\times120$) | $86\text{k} \to \mathbf{14.4\text{k}}$ (`6.0x`) | 4.64 s | **1.11 s (`4.19x`)** | **1.06 s (`4.37x`)** | 3.47 s (`1.34x`) | **3.09 s (`1.50x`)** |
| **`Mono-2D-Grid-25kJJ`** ($160\times160$) | $153\text{k} \to \mathbf{25.6\text{k}}$ (`6.0x`) | 8.88 s | **2.84 s (`3.13x`)** | **2.21 s (`4.01x`)** | 4.41 s (`2.02x`) | **3.82 s (`2.32x`)** |
| **`Mono-2D-Grid-64kJJ`** ($256\times256$) | $393\text{k} \to \mathbf{65.5\text{k}}$ (`6.0x`) | 29.04 s | **9.29 s (`3.13x`)** | **8.65 s (`3.36x`)** | 34.09 s (`0.85x`) | **17.22 s (`1.69x`, `1.98x` vs `-g 3`)** |

When junctions do not dynamically switch (`RTYPE=0` or quiescent bias), the GPU-resident `cuSolverRF` solver (`-g 3` / `-g` auto) achieves **`3.86x–4.21x` GPU speedup** on `14kJJ` (`4.81 s -> 1.14 s`), **`3.70x`** on `25kJJ` (`8.87 s -> 2.40 s`), and **`2.57x`** on `64kJJ` (`30.46 s -> 11.87 s`).

---

### C. Windowed PTL Block-Parallel & Multi-Channel Benchmarks

Reproducible via the benchmark scripts in [`scripts/`](../scripts/):

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

### D. Batched Monte Carlo (`.spread`) & Deterministic Parameter/Margin Sweep (`.sweep`, `-S`) Benchmarks

| Design (`.spread` Yield Analysis) | Base $P$ | Trials ($B$) | Est. $B \times$ Serial 1T | **Batched OpenMP (`-p`)** | **Batched GPU (`-g`)** | **OpenMP Speedup** | **GPU Speedup** |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| **`ex_mitll_xor_wr`** (MIT-LL XOR) | 2 | 1024 | 8.03 s | **0.30 s** | **1.23 s** | **26.48x** | **6.53x** |
| **`ex_mitll_dff_wr`** (MIT-LL DFF) | 2 | 1024 | 11.30 s | **0.40 s** | **1.56 s** | **28.16x** | **7.24x** |
| **`Mono-RSFQ-Cell-Macro`** (128 JJs, 0-PTL) | **1** | 1024 | 16.37 s | **0.67 s** | **4.97 s** | **24.32x** | **3.29x** |
| **`Mono-RLK-SQUID-Macro`** (96 JJs, 191 `K`, 0-PTL) | **1** | 1024 | 19.36 s | **0.80 s** | **5.51 s** | **24.34x** | **3.52x** |
| **`ex_ksa4bit`** (4-bit KSA Adder, 695 JJs) | 66 | 64 | 51.25 s | **2.04 s** | **4.31 s** | **25.07x** | **11.89x** |
| **`ex_ksa4bit`** (4-bit KSA Adder, 695 JJs) | 66 | 256 | 204.24 s | **6.56 s** | **10.39 s** | **31.14x** | **19.65x** |
| **`RSFQ-Pipeline-64B`** (2,048 JJs, 4.85M eqns) | 64 | 256 | 48.57 s | **1.68 s** | **3.19 s** | **28.87x** | **15.21x** |

#### Batched Nodal (`-p -n`) & 64-Point Deterministic Parameter/Margin Sweeps (`.sweep`, `-S`)

Combining Batched Nodal reduction (`Simulation::trans_sim_mc_nodal`, `-p -n`) with deterministic parameter/margin sweeps (`-S "@BIAS:MARGIN:0.25:64"`, `.sweep Scaling LIN 21 0.7 1.3`) or Monte Carlo (`-b 64`) achieves **`0.000e+00`–`1.000e-14` waveform parity** across MNA, Nodal, and GPU, automatically extracts the SFQ operating margin window (`Nom SFQ`, `Pass Pts`, `[vLo, vHi]`), and accelerates 64-point sweeps by **`33.3x–51.8x` over 64 sequential MNA runs** (**`1.46x–1.84x` faster than Batched MNA**):

| Circuit | 1x MNA | 1x Nodal (`-n`) | 64x Seq MNA Est. | **64-pt Sweep MNA (`-p`)** | **64-pt Sweep Nodal (`-p -n`)** | **64-trial MC MNA (`-p -b 64`)** | **64-trial MC Nodal (`-p -n -b 64`)** | **Sweep Speedup vs 64x Seq MNA** | **Nodal vs Batched MNA** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| **`ex_mitll_dff_wr` (`23` JJs)** | `67.7 ms` | `59.7 ms` | `4,330.2 ms` | `122.1 ms` (`35.5x`) | **`83.6 ms`** | `127.7 ms` | **`93.0 ms`** | **`51.8x`** | **`1.46x` / `1.37x`** |
| **`ex_ksa4bit` (`518` JJs)** | `1,508.5 ms` | `709.2 ms` | `96,542.2 ms` | `5,011.3 ms` (`19.3x`) | **`2,779.9 ms`** | `2,001.9 ms` | **`1,087.7 ms`** | **`34.7x`** | **`1.80x` / `1.84x`** |
| **`ex_jtl_string` (`2,000` JJs)** | `2,363.7 ms` | `1,103.6 ms` | `151,274.6 ms` | — | **`4,541.0 ms`** | — | **`1,807.6 ms`** (`83.7x`) | **`33.3x`** | **`15.6x` vs 64x Seq Nodal** |

