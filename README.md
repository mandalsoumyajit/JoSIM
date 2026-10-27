# JoSIM
### Superconductor Circuit Simulator

---

## Full Documentation

Please see full documentation here:
[JoSIM Documentation](https://joeydelp.github.io/JoSIM)

## Overview

JoSIM was developed under IARPA contract SuperTools(via the U.S. Army Research Office grant W911NF-17-1-0120).

JoSIM is a SPICE syntax circuit simulator specifically created to handle superconducting elements such as the Josephson junction. It reads in a standard SPICE deck, creates an A matrix and solves the linear algebra problem *Ax=b*. The linear algebra package KLU by [Tim Davis](http://faculty.cse.tamu.edu/davis/suitesparse.html) is used to solve the system of equations.

A two stage distribution platform is implemented in JoSIM with the first being a command line interface (CLI) binary that is always built as standard during compilation. This is accompanied by a C++ library (*libjosim*).

JoSIM allows output of results in standard text formats (`.csv`, `.dat`, ASCII SPICE `.raw`) as well as high-speed binary formats (`.bin` native column-major IEEE-754 float64, `.npy` NumPy structured array, and `.rawb` SPICE3 binary raw). JoSIM has inherent support for `.PARAM` commands that allow components to have variable values as well as expression parsing. JoSIM implements the RCSJ model of the Josephson junction (with optional implicit predictor-corrector iteration), lossless transmission lines (with optional fractional-step delay interpolation), and state checkpointing (`.savestate` / `.loadstate`) for fast bias conditioning.

JoSIM takes a *.cir* file as input and produces a *.csv*, *.dat*, *.raw*, *.bin*, *.npy*, or *.rawb* file as output.

Prebuilt binaries can be found here:

https://github.com/JoeyDelp/JoSIM/releases

Referencing:

[JoSIM — Superconductor SPICE Simulator](https://ieeexplore.ieee.org/document/8633946/)

If you find this tool useful, you can help keep it alive by \
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/joeydelp) \
or via \
[![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/JoeyDelp)

---

## Performance, Binary I/O & Enhanced Features

This enhanced fork of JoSIM delivers significant single-core, multi-threaded (`-p`), numerical integration, and binary I/O improvements (see [`ENHANCEMENT_PLAN.md`](ENHANCEMENT_PLAN.md) for architectural details):

### 1. Speed, Multi-Core OpenMP (`-p`) & CUDA GPU (`-g`) Parallelism
- **Up to `31.11x` OpenMP CPU speedup (`-p -b 256`) and `19.54x` CUDA GPU speedup (`-g -b 256`)** on batched Monte Carlo (`.spread`) robustness studies (`Simulation::trans_sim_mc`, `advance_mc_coalesced_kernel`), and **up to `27.15x` OpenMP speedup (`-p`) and `21.47x` CUDA GPU speedup (`-g`)** on single-run PTL-interconnected superconducting circuits via the **Multi-Step Windowed PTL Block-Parallel Solver** ("Exact Waveform Relaxation"; see [`docs/gpu_architecture.md`](docs/gpu_architecture.md)).
- **2.84x faster on the 23-test CTest suite** (`20.20 s -> 7.11 s` in WSL with 100% bit-identical outputs).
- **2.2x–3.2x faster end-to-end simulation + output** on medium/large extracted superconducting IC netlists when combining the solver optimizations with binary output (`.bin` / `.npy`).
- **$O(1)$ hash-map lookups** (`deviceMap`, `csMap`) for mutual inductances (`K`) and `.print`/`.save` traces, eliminating $O(N^2)$ setup bottlenecks on extracted layouts with thousands of mutual couplings.
- **In-place CSR updates & `klu_l_refactor`** whenever `RTYPE=1` Josephson junctions cross their gap voltage threshold ($V_g \pm \frac{1}{2}\Delta V$), avoiding per-transition matrix rebuilds and heap allocations.
- **Multi-Step Windowed PTL Block-Parallel Solver (`-p` OpenMP & `-g` CUDA GPU)**: Automatically partitions the MNA matrix and all circuit devices into $P$ connected components separated by lossless transmission lines (`T` elements) and computes the exact causal lookahead window $W = \min(T_d / \Delta t)$ steps. Because PTLs couple ports strictly via delayed history ($k \ge W$ steps), all $P$ partitions advance across $W$ consecutive timesteps completely independently before synchronizing PTL ring buffers—either in L1/L2 cache across OpenMP CPU threads (`-p`) or inside a single fused shared-memory/register-cached CUDA kernel (`-g`, 1 warp per partition with full $L_p + U_p$ factors in shared memory).
- **Self-Contained Transmission-Line Ring Buffers & Mid-Flight Checkpointing**: Each transmission line maintains a compact circular ring buffer (`hist_`) of length $k + 4$ instead of forcing all 6 of its port/current variables into `results.xVector` for the entire simulation length ($N_{\text{steps}}$), reducing transmission-line memory by **100x–1,000x** and supporting exact `.savestate`/`.loadstate` resumption even while SFQ pulses are mid-flight inside PTLs.
- **In-Loop Print Windowing & Decimation (`PRSTART` / `PRSTEP`)**: When `.tran` specifies `PRSTART > 0` or `PRSTEP > TSTEP` (without FIR windowing), `Simulation::trans_sim` stores only the required output steps (plus 2-step BDF2 lookback points when phase-to-voltage differentiation is requested), slashing RAM usage on long transient runs.
- **Homogeneous device arrays & zero-allocation hot loop**, cached `PWL`/`PWS` time segments, pre-computed `CUS` cubic splines, fast pre-reserved subcircuit flattening, and functional OpenMP multi-threading (`-p`) across device evaluation, block KLU solves, and output formatting.

### 2. High-Speed Binary Output Formats (`.bin`, `.npy`, `.rawb`)
Specify the output format via the file extension in `-o <file>` or `.file <file>`:
- **`.bin` (`JOSIMBIN`)**: Native column-major IEEE-754 `float64` binary format. Performs a zero-copy contiguous `fwrite` per trace directly from `Trace::data_`, making pure file writing **~37x faster than CSV** and reducing file size by ~38% while preserving full 64-bit precision.
- **`.npy`**: Standard NumPy v1.0/v2.0 structured array with named `<f8` columns (`time`, `"V(1)"`, etc.). Loadable directly in Python via `np.load("out.npy")` up to **23x faster than parsing CSV**.
- **`.rawb`**: Standard SPICE3 binary `.raw` format (`Binary:\n` preamble followed by interleaved `float64` rows).

Use [`scripts/josim_io.py`](scripts/josim_io.py) in Python to load `.bin`, `.npy`, or `.csv` files into a dict of NumPy arrays or a Pandas `DataFrame`:
```python
from scripts.josim_io import load_josim, load_josim_bin, load_josim_npy

data = load_josim("output.bin")          # Returns dict[str, np.ndarray]
df = load_josim("output.bin", as_dataframe=True)
arr = load_josim_npy("output.npy")       # Zero-copy memory-mapped structured array
```

### 3. New CLI Switches & Netlist Controls
- **Fractional Transmission-Line Delay Interpolation (`.option txinterp` / `INTERP=1`)**:
  Linearly interpolates delayed transmission-line port states when $T_d / T_{\text{step}}$ is non-integer, eliminating delay-rounding waveform drift across timestep sweeps:
  ```spice
  .option txinterp
  T01 1 0 2 0 td=5.07p z0=50 interp=1
  ```
- **State Checkpointing (`.savestate` / `.loadstate`)**:
  Save the complete MNA solution vector and all multi-step device & transmission-line histories after bias conditioning, and reload instantly in subsequent runs:
  ```spice
  .savestate bias_6ns.state 6000p
  * In subsequent runs:
  .loadstate bias_6ns.state
  ```
  (`.savebias` and `.loadbias` are also supported as aliases.)
- **Implicit Predictor-Corrector JJ Iteration (`-c [N]` / `.option corrector=N`)**:
  Performs up to `N` damped Picard/BDF2 corrector iterations per timestep on active Josephson junctions (with optional `.option correctortol=<tol>`, default `1e-3` rad).
- **Fixed-Step Enforcement & Step-Halving Diagnostics (`-f` / `.option fixedstep` / `.option warnstep`)**:
  Disable automatic step-halving restarts (`-f` / `--fixed-step` or `.option fixedstep`) or print detailed diagnostics naming the junction that triggered step reduction (`.option warnstep` or `-V 1`).
- **Mutual Inductance Coupling Cutoff (`.option kcutoff=<val>`)**:
  Automatically prunes parasitic mutual inductance entries (`K` lines) with $|k| < \text{kcutoff}$ during matrix construction to prevent dense LU fill-in on 3D field-extracted (e.g. InductEx) netlists:
  ```spice
  .option kcutoff=1e-3
  ```
- **CUDA GPU & Multi-Step Windowed PTL Solver (`-g` / `--gpu`, `.option gpu`, `-p` / `--parallel`)**:
  Build with `-DUSING_CUDA=ON -DUSING_OPENMP=ON` (`CMAKE_CUDA_ARCHITECTURES` automatically defaults to `"native"` on CMake $\ge 3.24$ across Volta, Turing, Ampere, Ada, Hopper, and Blackwell GPUs) to enable CUDA GPU acceleration (`-g` or `.option gpu`) alongside multi-core OpenMP windowed PTL acceleration (`-p`).
- **Batched Monte Carlo (`.spread`) Robustness Solver (`-b <N>` / `--batch=<N>`, `.option mc=<N>`, `.spread ... runs=<N>`, `.option mcall`)**:
  Executes `N` independent fabrication-variation trials simultaneously in a single run (parse-once, symbolic-once) across OpenMP CPU threads (`-p -b <N>`) or in a coalesced thread-per-trial CUDA GPU kernel (`-g -b <N>`), outputting Welford ensemble `<Trace>`, `<Trace>_STD`, `<Trace>_MIN`, and `<Trace>_MAX` statistics (plus `<Trace>_MC0..MC{N-1}` with `.option mcall`) and a stdout **Monte Carlo Robustness Summary** table.

### 4. Benchmark Results (Medium, Large & Very-Large Designs)

#### A. Tier 1 Windowed PTL Block-Parallel Solver Scaling (`Dual AMD EPYC 7413 48C/96T + NVIDIA RTX A6000`)

Reproducible via [`scripts/build_exxact_gpu.sh`](scripts/build_exxact_gpu.sh) and [`scripts/benchmark_gpu_exxact.py`](scripts/benchmark_gpu_exxact.py) (see [`docs/gpu_architecture.md`](docs/gpu_architecture.md) for full details and numerical equivalence verification to $\le 5.1 \times 10^{-15}$):

| Design | JJs | MNA Eqns | Steps | Serial (1T) | OpenMP (8T) | OpenMP (32T) | GPU (RTX A6000) | OpenMP Speedup | GPU Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **PTL-Tile-64** | 2,048 | 10,560 | 40,000 | `7.16 s` | `2.10 s` | **`0.61 s`** | `5.69 s` | **11.78x** | **1.26x** |
| **PTL-Tile-256** | 8,192 | 42,240 | 40,000 | `29.88 s` | `7.89 s` | **`2.06 s`** | `6.00 s` | **14.54x** | **4.98x** |
| **PTL-Tile-1024** | 32,768 | 168,960 | 20,000 | `104.90 s` | `15.54 s` | **`4.20 s`** | `6.95 s` | **24.95x** | **15.10x** |
| **PTL-Tile-4096** | 131,072 | 675,840 | 10,000 | `232.89 s` | `33.55 s` | **`8.58 s`** | `10.85 s` | **27.15x** | **21.47x** |

#### B. Non-PTL & Diverse Superconducting Circuit Topologies (`Dual AMD EPYC 7413 48C/96T + NVIDIA RTX A6000`)

Reproducible via [`scripts/benchmark_non_ptl_exxact.py`](scripts/benchmark_non_ptl_exxact.py) and [`scripts/run_non_ptl_exxact.sh`](scripts/run_non_ptl_exxact.sh) across **0-PTL monolithic circuits ($P=1$)**, **0-PTL multi-channel SQUID/detector arrays ($P \ge 2$)**, and **3D-extracted RLK/capacitor RSFQ ALU macros** up to **`1,212,412` MNA equations** (all matching serial output to **`1.90e-18`–`2.46e-16`**):

| Design | Topology (`K`, `C`, PTLs) | MNA Eqns | Steps | Serial (1T) | OpenMP (`-p`) | GPU (`-g`) | OpenMP Speedup | GPU Speedup | Max Diff |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Mono-2D-JJ-Grid** | $128\times128$ 2D JJ Grid (**0 PTL**, $P=1$) | 106,242 | 2,000 | `11.47 s` | `10.79 s` | `11.94 s` | **1.06x** | 0.96x | `9.08e-17` |
| **Mono-RLK-49kK-Volt** | 3D-Extracted 49k `K` + 8k `C` (**0 PTL**, $P=1$, `-a 0`) | 65,540 | 2,000 | `2.35 s` | `2.02 s` | `3.32 s` | **1.16x** | 0.71x | `2.38e-18` |
| **NoPTL-SQUID-256** | 256-Ch SQUID Array, 24k `K`, 8k `C` (**0 PTL**, $P=256$) | 74,752 | 20,000 | `28.06 s` | **`1.76 s`** | **`9.21 s`** | **15.90x** | **3.05x** | `2.46e-16` |
| **NoPTL-SQUID-256-Volt** | 256-Ch SQUID Array, 24k `K`, 8k `C` (**0 PTL**, `-a 0`) | 74,752 | 20,000 | `29.49 s` | **`1.72 s`** | **`9.21 s`** | **17.15x** | **3.20x** | `1.90e-18` |
| **NoPTL-SQUID-1024** | 1024-Ch SQUID Array, 98k `K`, 32k `C` (**0 PTL**, $P=1024$) | 299,008 | 10,000 | `75.82 s` | **`3.52 s`** | **`11.20 s`** | **21.55x** | **6.77x** | `7.46e-17` |
| **NoPTL-SQUID-4096** | 4096-Ch SQUID Array, 393k `K`, 131k `C` (**0 PTL**, $P=4096$) | 1,196,032 | 5,000 | `185.79 s` | **`7.44 s`** | **`18.04 s`** | **24.98x** | **10.30x** | `5.24e-17` |
| **RSFQ-ALU-Macros-256** | 256 RSFQ ALU Macros, 24k `K`, 8k `C` ($P=256$) | 75,772 | 20,000 | `27.99 s` | **`1.77 s`** | **`9.33 s`** | **15.80x** | **3.00x** | `2.12e-16` |
| **RSFQ-ALU-Macros-1024** | 1024 RSFQ ALU Macros, 98k `K`, 32k `C` ($P=1024$) | 303,100 | 10,000 | `72.81 s` | **`3.54 s`** | **`11.39 s`** | **20.54x** | **6.39x** | `1.03e-16` |
| **RSFQ-ALU-Macros-4096** | 4096 RSFQ ALU Macros, 393k `K`, 131k `C` ($P=4096$) | 1,212,412 | 5,000 | `157.08 s` | **`7.50 s`** | **`18.29 s`** | **20.93x** | **8.59x** | `6.96e-17` |

#### C. Tier 2 Batched Monte Carlo (`.spread`) Robustness Benchmarks (`Dual AMD EPYC 7413 48C/96T + NVIDIA RTX A6000`)

Reproducible via [`scripts/benchmark_mc_exxact.py`](scripts/benchmark_mc_exxact.py) and [`scripts/run_mc_exxact.sh`](scripts/run_mc_exxact.sh) across MIT-LL standard cells (`ex_jtl_basic`, `ex_mitll_xor_wr`, `ex_mitll_dff_wr`), real production RSFQ logic (`ex_ksa4bit`), 0-PTL single-partition monolithic circuits ($P=1$), and PTL-coupled RSFQ processor pipelines up to **`4,848,640` simultaneous MNA equations** ($B \in \{64, 256, 1024\}$ trials):

| Design (`.spread` Yield Analysis) | Base $P$ | Batch $B$ | Est. $B \times$ Serial 1T | Batched OpenMP (`-p`) | Batched GPU (`-g`) | OpenMP Speedup | GPU Trans Speedup | GPU Wall Speedup | CPU vs GPU Rel Diff |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **`ex_mitll_xor_wr` (MIT-LL XOR)** | 2 | 1024 | `8.025 s` | **`0.303 s`** | **`1.229 s`** | **26.48x** | **6.53x** | **4.30x** | `1.17e-14` |
| **`ex_mitll_dff_wr` (MIT-LL DFF)** | 2 | 1024 | `11.301 s` | **`0.401 s`** | **`1.560 s`** | **28.16x** | **7.24x** | **4.96x** | `6.67e-15` |
| **`Mono-RSFQ-Cell-Macro` (128 JJs, 0-PTL)** | **1** | 1024 | `16.374 s` | **`0.673 s`** | **`4.972 s`** | **24.32x** | **3.29x** | **2.93x** | `2.44e-13` |
| **`Mono-RLK-SQUID-Macro` (96 JJs, 191 K, 0-PTL)** | **1** | 1024 | `19.364 s` | **`0.796 s`** | **`5.507 s`** | **24.34x** | **3.52x** | **3.16x** | `1.67e-13` |
| **`ex_ksa4bit` (4-bit KSA Adder, 695 JJs)** | 66 | 64 | `51.250 s` | **`2.044 s`** | **`4.309 s`** | **25.07x** | **11.89x** | **10.13x** | `1.04e-15` |
| **`ex_ksa4bit` (4-bit KSA Adder, 695 JJs)** | 66 | 256 | `204.237 s` | **`6.559 s`** | **`10.394 s`** | **31.14x** | **19.65x** | **18.33x** | `1.04e-15` |
| **`RSFQ-Pipeline-16B` (512 JJs, 16 Blocks)** | 16 | 256 | `12.165 s` | **`0.431 s`** | **`1.226 s`** | **28.22x** | **9.92x** | **6.50x** | `1.19e-17` |
| **`RSFQ-Pipeline-64B` (2,048 JJs, 64 Blocks)** | 64 | 64 | `12.178 s` | **`0.510 s`** | **`1.243 s`** | **23.87x** | **9.80x** | **6.34x** | `1.45e-17` |
| **`RSFQ-Pipeline-64B` (2,048 JJs, 64 Blocks)** | 64 | 256 | `48.566 s` | **`1.682 s`** | **`3.192 s`** | **28.87x** | **15.21x** | **12.55x** | `1.45e-17` |

#### D. Upstream `v2.7` vs. Enhanced Single/Multi-Thread CPU Benchmark (WSL)

Reproducible via [`scripts/benchmark_medium_large_wsl.sh`](scripts/benchmark_medium_large_wsl.sh) in WSL (`-O3` Release build with Ninja, comparing upstream `v2.7` / `02a34ee` against this enhanced version):

#### End-to-End Wall-Clock Runtime

| Design | Scale & Description | Upstream CSV | Enhanced 1T CSV | Enhanced `-p` CSV | Enhanced `-p` `.bin` | Speedup (`-p` CSV) | Speedup (`-p` `.bin`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ex_ksa4bit` | Medium: 4-bit RSFQ Kogge-Stone Adder | `2.029 s` | `1.409 s` | `1.499 s` | `1.466 s` | **1.35x–1.44x** | **1.38x** |
| `ex_jtl_string` | Medium: Long RSFQ JTL chain (`100 ps`) | `3.069 s` | `2.103 s` | `2.137 s` | `2.109 s` | **1.44x–1.64x** | **1.46x–1.73x** |
| `ex_gen_pp_ipht` | Medium: IPHT SFQ pulse generator & logic | `5.962 s` | `2.563 s` | `2.547 s` | `2.570 s` | **2.34x** | **2.32x** |
| `med_ext_6k_K` | Medium: 3D-extracted macro (500 JJs, 2.5k `L`, 6k `K`) | `0.826 s` | `0.557 s` | `0.360 s` | `0.357 s` | **2.29x** | **2.31x–2.56x** |
| `large_10k_jtl` | Large: 10,000-JJ JTL (~70k devices, ~90k eqns) | `1.894 s` | `1.173 s` | `1.059 s` | `1.064 s` | **1.79x** | **1.78x** |
| `large_64blk_ptl` | Large: 64-tile PTL fabric (3,072 JJs, 512 PTLs) | `1.375 s` | `0.912 s` | `0.900 s` | `0.840 s` | **1.53x** | **1.64x** |
| `large_2k_windowed` | Large: 2,000 JJs, 4k steps (`PRSTART=800p, PRSTEP=2p`) | `1.413 s` | `0.734 s` | `0.773 s` | `0.712 s` | **1.83x** | **1.98x** |
| `vlarge_io_2.5Mpts` | Very Large: 2,500 JJs, 1,250 traces $\times$ 2k steps | `1.219 s` | `0.996 s` | `1.047 s` | `0.574 s` | **1.16x** | **2.12x** |
| `vlarge_512ptl_10k` | Very Large: 512 PTLs over 10,000 steps (`2.5 ns`) | `0.548 s` | `0.341 s` | `0.344 s` | `0.343 s` | **1.59x** | **1.60x** |

#### Peak Memory (RSS) Reduction

| Design | Upstream Peak RSS | Enhanced Peak RSS | RAM Reduction | Primary Optimization Driver |
| :--- | ---: | ---: | ---: | :--- |
| `vlarge_512ptl_10k` (512 PTLs, 10k steps) | `165.4 MB` | **`11.1 MB`** | **14.89x less RAM** | Self-contained TX circular ring buffers (`hist_`) of length $k + 4$ instead of 6 $\times N_{\text{steps}}$ vectors per PTL |
| `large_2k_windowed` (`PRSTART=800p, PRSTEP=2p`) | `85.1 MB` | **`33.5 MB`** | **2.54x less RAM** | In-loop print windowing & decimation (`keepStep_`) skipping unprinted transient steps |
| `large_64blk_ptl` (3,072 JJs, 512 PTLs, 2.5k steps) | `77.8 MB` | **`47.1 MB`** | **1.65x less RAM** | TX circular ring buffers (`hist_`) across 512 PTL interconnects |
| `ex_ksa4bit` (4-bit Adder) | `60.9 MB` | **`43.9 MB`** | **1.39x less RAM** | Pre-reserved subcircuit expansion + TX circular ring buffers |
| `vlarge_io_2.5Mpts` (1,250 traces $\times$ 2k steps) | `75.0 MB` | **`68.3 MB`** | **1.10x less RAM** | Direct streaming binary `.bin` output |

### 5. WSL Build, Benchmark & Verification Scripts
- [`scripts/build_wsl.sh`](scripts/build_wsl.sh): Configures and builds JoSIM in WSL using Ninja (`-DCMAKE_BUILD_TYPE=Release -DPARALLEL=ON`) and runs the 23-test CTest suite.
- [`scripts/benchmark_wsl.sh`](scripts/benchmark_wsl.sh): Benchmarks upstream vs. optimized JoSIM and verifies 100% bit-identical CSV outputs.
- [`scripts/benchmark_medium_large_wsl.sh`](scripts/benchmark_medium_large_wsl.sh): Benchmarks wall-clock runtime and peak memory (RSS) across 9 medium-, large-, and very-large-scale superconducting IC designs.
- [`scripts/test_phase3_wsl.sh`](scripts/test_phase3_wsl.sh): Tests and benchmarks `.bin`/`.npy`/`.rawb` binary outputs, fractional TX delay interpolation, `.savestate`/`.loadstate` checkpointing, `-f` fixed-step control, and `-c` JJ predictor-corrector iteration.
- [`scripts/test_phase4_wsl.sh`](scripts/test_phase4_wsl.sh): Tests transmission-line circular ring buffers, mid-flight PTL `.savestate`/`.loadstate` checkpointing, in-loop `PRSTART`/`PRSTEP` decimation, `.option kcutoff`, and OpenMP block-partitioned KLU solving across PTL boundaries.
- [`scripts/build_exxact_gpu.sh`](scripts/build_exxact_gpu.sh) & [`scripts/benchmark_gpu_exxact.py`](scripts/benchmark_gpu_exxact.py): Builds JoSIM with CUDA (`-DUSING_CUDA=ON`) and OpenMP (`-DUSING_OPENMP=ON`), verifies numerical equivalence across Phase and Voltage modes, and benchmarks scaling across PTL-coupled circuits up to 131k JJs (`675k` MNA equations).
- [`scripts/benchmark_non_ptl_exxact.py`](scripts/benchmark_non_ptl_exxact.py) & [`scripts/run_non_ptl_exxact.sh`](scripts/run_non_ptl_exxact.sh): Benchmarks non-PTL monolithic circuits ($P=1$), multi-channel 0-PTL SQUID arrays ($P \ge 2$), and rich RLK/capacitor RSFQ ALU macros up to 1.21M MNA equations.
- [`scripts/benchmark_mc_exxact.py`](scripts/benchmark_mc_exxact.py) & [`scripts/run_mc_exxact.sh`](scripts/run_mc_exxact.sh): Benchmarks Tier 2 Batched Monte Carlo (`.spread`) across 16 circuit/batch configurations up to 2.72M simultaneous MNA equations.

---

## Changelog
### v2.8
- Implemented **Tier 2 Batched GPU (`-g`) & Multi-Core OpenMP (`-p`) Monte Carlo (`.spread`) Robustness Solver** (`-b <N>` / `--batch=<N>`, `.option mc=<N>`, `.spread ... runs=<N>`, `.option mcall`, `JOSIM_SEED`), with hardware-adaptive CUDA block sizing across all NVIDIA GPU architectures (`advance_mc_coalesced_kernel`) and Welford online ensemble statistics (`<Trace>`, `<Trace>_STD`, `<Trace>_MIN`, `<Trace>_MAX`), achieving up to **31.11x OpenMP speedup** and **19.54x CUDA GPU speedup**.
- Implemented **Tier 1 Multi-Step Windowed PTL Block-Parallel Solver** ("Exact Waveform Relaxation") for both **CUDA GPU (`-g` / `--gpu`, `.option gpu`, `-DUSING_CUDA=ON`)** and **Multi-Core OpenMP CPU (`-p` / `--parallel`, `-DUSING_OPENMP=ON`)**, achieving up to **27.15x OpenMP speedup** and **21.47x CUDA GPU speedup** on 131k-JJ PTL-interconnected circuits.
- Added high-speed binary output formats: `.bin` (column-major `JOSIMBIN` float64), `.npy` (NumPy structured array), and `.rawb` (SPICE3 binary raw), along with Python loader [`scripts/josim_io.py`](scripts/josim_io.py).
- Implemented $O(1)$ hash-map lookups for mutual inductance (`K`) and `.print`/`.save` trace resolution.
- Implemented in-place CSR value updates and `klu_l_refactor` on `RTYPE=1` Josephson junction state transitions.
- Implemented block-partitioned KLU solver (`KluPartition`) across lossless transmission lines (`T` elements) with selective per-block `klu_l_refactor` and OpenMP parallel block solves (`-p`).
- Replaced full-history `results.xVector` forcing for transmission lines with self-contained circular ring buffers (`hist_`) of length $k + 4$, and added v2 `.savestate`/`.loadstate` serialization for mid-flight PTL pulses.
- Implemented in-loop `PRSTART`/`PRSTEP` print windowing and decimation in `Simulation::trans_sim` to avoid storing unprinted transient steps in RAM.
- Added `.option kcutoff=<val>` (`kmin=<val>`) to prune weak parasitic mutual inductances during matrix setup.
- Partitioned devices into contiguous homogeneous arrays (`Simulation::setup`) and eliminated per-timestep heap allocations in `Simulation::trans_sim` and `Simulation::setup_b`.
- Cached `PWL`/`PWS` active segment indices for $O(1)$ amortized source evaluation and pre-computed `CUS` cubic splines once during setup.
- Replaced per-line/per-token `std::regex` construction in netlist parsing with linear whitespace scanning, optimized hierarchical subcircuit flattening, and fixed `ProgressBar` busy-wait CPU spin.
- Fixed `Resistor::update_timestep` scaling factor in Phase mode (`factor * factor`).
- Implemented OpenMP parallelization (`-p` / `--parallel`) across device RHS updates, block-partitioned KLU solves, and output formatting (`Output::write_output`).
- Added fractional delay interpolation for transmission lines via `.option txinterp` (`.option fractx`) or per-element `interp=1`.
- Added binary simulation state checkpointing via `.savestate <file> [time]` (`.savebias`) and `.loadstate <file>` (`.loadbias`).
- Added implicit predictor-corrector JJ iteration via `-c [N]` (`--corrector=N`), `.option corrector=<N>`, and `.option correctortol=<tol>`.
- Added explicit step-halving control and diagnostics via `-f` (`--fixed-step`), `.option fixedstep`, and `.option warnstep`.
### v2.7
- Added ability to specify seed for RNG through `.option seed=` command.
- Current run seed can be shown using verbose command.
- Added piece-wise sinusoidal (PWS) syntax to syntax guide documentation.
- Expanded sinusoid parameters to include phase offset and number of cycles. Now allows non-continuous sinusoid.
- Collection of bug fixes, code & code cleanup.
- Cleaned up warnings during compilation.

### v2.6.10
- Fixed a bug resulting in sub optimal performance in the custom function.

### v2.6.9
- Fixed a bug resulting in bad optional access when shorting transmission line components.

### v2.6.8
- Add a fix to warn of subcircuit redefenitions. Redefintions now overwrite the previous definition.
- Add a fix when parameters are used in subcircuits but defined within the global scope.

### v2.6.7
- Added ability to parameterize time points specified for `.tran` and `pwl` commands.

### v2.6.6
- Fixed a bug in noise source generation when noise step is less than or equal to the simulation step.
- Reduced aggressiveness of timestep scale down when close to transmission line step delay size. Now only scales down if TX delay is less than 1 step (previously 4)
- Added output dimensions option to josim-plot script when writing directly to PDF, SVG, EPS, etc.

### v2.6.5
- Updated the CPR specification in the JJ model to allow multiple simulataneous harmonics.
- Added a Notepad++ UDL syntax highlighting script to the scripts folder.

### v2.6.4
- Added the ability to specify CPR in the JJ model to enable non-sinusoidal CPRs needed for pi-junctions.

### v2.6.3
- Fixed a bug where Windows and CentOS parameters do not coincide. Issue was due to expression substitution.

### v2.6.2
- Fixed sp_generator.py having invalid tab spacing
- Fixed Area= and Ic= JJ parameters not producing the same result for the same critical current

### v2.6.1
- Fixed an issue with custom waveform where it wasn't reading from the file properly
- Added spline method for cubic interpolation of custom waveform
- Moved entire codebase to 64bit integers to allow larger/longer simulations
- Fixed parameters
- Fixed transmission line integer step delay rounding issue that caused an error build up

### v2.6
- Full Johnson-Nyquist temperature noise for Resistors and Resistive branch of RCSJ model
- Phi-junction model with verified Pi junction results
- Various bug fixes and code clean up

### v2.5.8
- Updated KLU to a 64-bit version capable of handling non-zero count larger than the maximum value of a 32-bit integer.
- Included SuperLU 64-bit that can be activated using the *-x 1* switch (default 0 for KLU).
- Bug fixes to pulse command.

### v2.5.7
- Fixed an issue with comparison of double values for small numbers between various compilers.

### v2.5.6
- Fixed a initial values when converting between phase and voltage when simulation output does not start at 0.

### v2.5.5
- Fixed a bug in output conversion when plotting voltage in a phase simulation and vice versa.

### v2.5.4
- Fixed comparison function for GCC interpretation of double values

### v2.5.3
- Fixed an issue with the pulse command producing incorrect pulses
- Fixed an issue where timesteps smaller than 10fs were causing out-of-range issues
- Various other bug fixes

### v2.5.2
- Fixed a major issue with relative paths not being correctly evaluated causing files to not be found even though the correct path was specified.

### v2.5.1 - 18/05/2021
- Added .spread command. See documentation for more details.
- Added .file command. See documentation for more detauls.
- Added .iv command. See documentation for more details.
- Fixed a bug where .neb value was not being read properly
- Fixed a bug where pulse source type was outputting incorrect number of pulses.
- Fixed a bug where custom waveform was not allowing cubic interpolation method.
- Code cleanup.

### v2.5 - 05/03/2021
- Phase mode is now the standard analysis mode. Produces the exact same results as voltage.
- Trapezoidal method has been replaced by superior 2nd order Gear method.
- JoSIM now internally determines the convention. (-c option removed)
- New progress indicators added. Spawns a seperate non blocking thread.
- New verbose modes. 1: Circuit statistics. 2: 1 + Parameters and evaluated values. 3: 2 + Complete expanded netlist.
- New DC source type. I.e. V01 1 0 DC 5V
- JJ has new IC= parameters that supersedes AREA= but ultimately performs similar model parameter scaling.
- Fixed NOISE source type to have syntax NOISE(AMP TD TSTEP)
- New '.temp' and '.neb' commands for noise thermal noise analysis
- Transient simulation command now allows PRSTART and PRSTEP commands. .tran TSTEP TSTOP PRSTART PSTEP
- Various memory improvements and overall speed-up
- Reduced output file size.
- Rewrote documentation: [JoSIM Documentation](https://joeydelp.github.io/JoSIM)
- Various bug fixes

### v2.4.1 - 09/03/2020
- Fixed an issue where "GND" was not recognized as ground, creating a new node causing matrix singularity.
- Fixed an issue with multiport devices (TX, CCCS, CCVS, VCCS, VCVS) not adding the 3rd and 4th node to the node map.
- Fixed an issue with multiport devices not expanding properly if part of subcircuits.
- Various other bug fixes

### v2.4 - 07/02/2020
- Complete rewrite of core internals
- Modularization of each component allowing easier implementation of integration methods
- Error reporting major improvment. Less random exits with no warnings. Better exits.
- Focus on object orientation for improved library access
- Speed up of matrix creation algorithm
- Improvements to output format
- Inclusion of '.include' control card
- Capability to read from standard in using '-i' command
- Various bug fixes

### v2.3 - 10/05/2019
- Major changes to JoSIM internal engine
- Removal of plotting engines in favor of external tools such as Python
- Introduction of pyJoSIM and libjosim packages
- Immense speed-up of internals
- Various bug fixes

### v2.2 - 12/02/2019
- Major rewrite of most of JoSIM code to accommodate library creation for use in C++ and Python
- Implementation of a phase source for use in phase-mode simulation

### v2.1 - 16/01/2019
- Inclusion of the Haberkorn junction current which allows for user switching between ballistic and normal electron tunneling
- Various bug fixes

### v2.0 - 10/10/2018
- New analysis method that allows Modified Nodal Phase Analysis (see ReadMe)
- Major memory improvement by not creating intermediate A matrix
- Various bug fixes

### v1.3 - 28/07/2018
- Addition of mutual inductance with the netlist label **K**
- Addition of the sinusoidal source (Voltage and Current)
- Various bug fixes
- AQFP now works

### v1.2 - 17/07/2018
- New Release brings RType=1 to the RCSJ model of the Josephson Junction.
- Various bugs have been fixed. 
- Data structures have been majorly altered.
- Other changes and fixes

### v1.1 - 04/06/2018
- Ability to change between plotting libraries (NONE, FLTK or Matplotlib [Default])
- Major bug fixes
- JoSIM now supports WRSpice syntax (mostly)
- Parameterization of component values with expression parsing
- Code cleanup and commenting

### v1.0 - 03/05/2018
- Initial release of JoSIM
- Default FLTK plotting engine
- Replaces JSIM as default simulator
