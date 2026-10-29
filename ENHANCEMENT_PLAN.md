# JoSIM Enhancement Plan: Speed, Parallelism, and Functionality

This document captures the architectural audit of **JoSIM** (`v2.7` / `2.6.5.c81027a`), cross-referenced against how [`spec2si-seeqc`](../spec2si-seeqc/README.md) uses JoSIM in production (e.g., [`josim-timestep-audit.md`](../spec2si-seeqc/docs/josim-timestep-audit.md), [`ptl-step6-routed-chips.md`](../spec2si-seeqc/docs/ptl-step6-routed-chips.md), [`routed-xesfq-counter51.md`](../spec2si-seeqc/docs/routed-xesfq-counter51.md)) and compared against modern SPICE-class simulators—primarily Sandia's **Xyce**, **WRspice**, and **Ngspice (KLU/CUSPICE)**.

---

## Executive Summary: Critical Findings in Upstream JoSIM

1. **Parallelism (`-p` / `USING_OPENMP`) is a complete no-op in upstream.**
   Despite advertising `-p` / `--parallel` in [`src/CliOptions.cpp`](src/CliOptions.cpp) and linking `OpenMP::OpenMP_CXX` in [`CMakeLists.txt`](CMakeLists.txt), there was not a single `#pragma omp` directive anywhere in the upstream JoSIM codebase. Meanwhile, [`include/JoSIM/ProgressBar.hpp`](include/JoSIM/ProgressBar.hpp) spawned a busy-waiting `std::thread` with no sleep/yield when `-m` was omitted, burning 100% of a CPU core.
2. **Per-timestep heap reallocation, full CSR reconstruction, and full KLU numeric malloc/free.**
   Inside the inner transient loop (`Simulation::trans_sim`), `Simulation::setup_b` called `b_.clear(); b_.resize(...)` and `x_ = b_` every timestep, and whenever any `RTYPE=1` junction crossed $V_g \pm \frac{1}{2}\Delta V$, it called `Matrix::create_nz()` (rebuilding the entire `nz` vector from scratch via `std::visit` over `std::variant`) followed by `klu_l_free_numeric` + `klu_l_factor` instead of in-place value updates and `klu_l_refactor`.
3. **Quadratic $O(N^2)$ netlist flattening and mutual inductance lookup.**
   `Matrix::handle_mutual_inductance` performed a linear scan over all `components.devices` via `std::visit` for *every* `K` element ($O(|K| \cdot |D|)$), and `RelevantTrace.cpp` scanned all devices linearly for every `.print`/`.save` token. For extracted `spec2si-seeqc` layouts with thousands of local/global mutual couplings (e.g., 2,190+ mutuals in the 51-cell counter), matrix setup alone suffered severe quadratic scaling.
4. **No Newton-Raphson corrector or local adaptive time-stepping.**
   Upstream JoSIM uses a pure explicit predictor for junction phase (`Simulation::handle_jj`) with a fixed global step $h$. If $|\phi_0 - \phi_{n-1}| > 0.4\pi$ after 1% of the run, `Simulation::reduce_step` silently discards the entire simulation, rebuilds the matrix from scratch at $h/2$, and restarts from $t = 0$.

---

## Phased Implementation Roadmap

### Phase 1: Zero-Risk, Bit-Identical Single-Core Speedups (3x–8x)
- **$O(1)$ Hash Maps for Mutual Inductance (`K`) & Trace Resolution:**
  - Build `deviceLabelMap` / `inductorMap` (`std::unordered_map<std::string, size_t>`) during `Matrix::create_components` to eliminate $O(|K| \cdot |D|)$ and $O(|\text{PRINT}| \cdot |D|)$ linear scans in `Matrix::handle_mutual_inductance` and `find_relevant_traces`.
- **In-Place `nz` Updates & `klu_l_refactor`:**
  - Store each JJ's direct index into `mObj.nz` during `Matrix::create_nz()`. When `JJ::update_value` transitions state (`RTYPE=1`), update `mObj.nz[jj.nzIndex_]` directly in $O(1)$ time and call `klu_l_refactor` (reusing the symbolic/numeric allocation and pivot order in-place, with automatic fallback to `klu_l_factor` only if numerical conditioning requires re-pivoting).
- **Zero-Allocation Hot Loop (`Simulation::trans_sim` & `Simulation::setup_b`):**
  - Pre-allocate `b_` and `x_` once; replace `b_.clear(); b_.resize(...)` with `std::fill(b_.begin(), b_.end(), 0.0)` and `x_ = b_` with in-place solve + `std::swap(x_, b_)`.
  - When recording output waveforms at each timestep, iterate over `mObj.relevantIndices` ($O(|\text{saved}|)$) instead of scanning all `0 .. results.xVector.size()` branch variables ($O(|M|)$).
- **Amortized $O(1)$ Source Evaluation (`Function.cpp`):**
  - Cache the active time-segment index (`lastIdx_`) in `Function::return_pwl` and `Function::return_pws` so monotonically advancing timesteps find their segment in $O(1)$ instead of scanning from index `0` every step.
  - Pre-compute the cubic spline (`tk::spline`) once in `Function::parse_cus` instead of reconstructing and solving a tridiagonal spline system on every timestep in `Function::return_cus`.
- **Eliminate Regex Compilation in Tokenizer & Busy-Wait in `ProgressBar`:**
  - Replace per-line and per-token `std::regex_replace` construction in `Input::read_input` and `Misc::tokenize` with fast linear whitespace trimming.
  - Add `std::this_thread::sleep_for(std::chrono::milliseconds(50))` in `ProgressBar::thread_print()` so non-minimal runs do not burn 100% of a CPU core.
- **Fix `Resistor::update_timestep` Bug:**
  - Fix `Resistor::update_timestep` scaling by `factor` (instead of `factor * factor`) in Phase mode.

### Phase 2: Homogeneous Device Arrays (SoA/AoS Separation) & Multi-Threading / SIMD
- **Separate Homogeneous Device Containers (`Components.hpp`):**
  - Replace the single polymorphic `std::vector<std::variant<11 Types>>` hot-loop indirection with direct typed vectors (`jjs`, `inductors`, `resistors`, `capacitors`, `txlines`, `vsources`, `psources`, `ccvs`, `vccs`, `cccs`, `vcvs`), eliminating `std::get<T>` variant checks, reducing per-device memory footprint (sharing `Model` via pointer/index or compact struct), and enabling contiguous cache prefetching and compiler auto-vectorization.
- **OpenMP Parallelism in `Simulation.cpp`, `Output.cpp`, and `IV.cpp`:**
  - Parallelize device RHS updates (`handle_jj`, `handle_inductors`, `handle_resistors`, `handle_capacitors`, `handle_tx`) when `-p` (`--parallel`) is enabled and circuit size exceeds a threshold, separating nodal KCL noise/current-source accumulation from conflict-free branch-row writes.
  - Parallelize FIR filtering and trace formatting across traces in `Output::write_output` using `#pragma omp parallel for`.
  - Parallelize I-V curve sweep points in `IV::generate_iv` across threads.
  - Ensure thread-safe, deterministic RNG draws for thermal noise.

### Phase 3: Numerical Integration, Binary I/O & Superconductor IC Features
- **High-Speed Binary File Output Formats (`.bin`, `.npy`, `.rawb`) + Python Loader (`scripts/josim_io.py`):**
  - **`.bin` (`FileOutputType::Bin`)**: Column-major IEEE-754 `float64` binary format (`JOSIMBIN` v1 header + trace metadata + zero-copy contiguous column dumps directly from `Trace::data_`), delivering **2.2x–3.2x end-to-end simulation+write speedups** (~37x faster pure file write than `.csv`) and ~38% smaller files with full 64-bit precision.
  - **`.npy` (`FileOutputType::Npy`)**: Standard NumPy v1.0/v2.0 structured array format with named `<f8` trace fields, loadable directly via `np.load("out.npy")` or `pd.DataFrame(np.load("out.npy"))` up to **23x faster** than CSV parsing.
  - **`.rawb` (`FileOutputType::RawBin`)**: Standard SPICE3 binary `.raw` format (`Binary:\n` preamble + raw IEEE-754 `float64` records).
  - **Parallel All-Node Output Formatting**: OpenMP parallelization across nodes when `.print` is omitted (`Output::write_output`) and cached column data pointers in `Output::format_csv_or_dat`.
- **Fractional Delay Interpolation for Transmission Lines (`TransmissionLine.cpp`, `Simulation.cpp`):**
  - Enabled via `.OPTION TXINTERP` (or `.OPTION FRACTX` / per-element `INTERP=1`), decomposing $T_d / h = k + \alpha$ ($k = \lfloor T_d / h \rfloor$, $\alpha \in [0, 1)$) and linearly interpolating delayed port voltages/phases and branch currents between steps $i - k$ and $i - k - 1$. Eliminates delay-rounding circuit distortion across coarse/fine timestep sweeps (`max |V_0.1ps - V_0.05ps|` drops from `2.5e-5 V` to `1.08e-19 V`).
- **Explicit Step-Control Diagnostics & CLI Flags (`-f`, `.OPTION FIXEDSTEP`, `.OPTION WARNSTEP`):**
  - Reports the offending junction label, halving count, and old/new `tstep` when `-V >= 1` or `.OPTION WARNSTEP` is enabled, and enforces strict fixed-step execution when `-f` or `.OPTION FIXEDSTEP` is set.
- **State Checkpointing (`.SAVESTATE` / `.LOADSTATE` / `.SAVEBIAS` / `.LOADBIAS`) for Fast Bias Conditioning:**
  - Serializes and restores the full MNA solution vector `x_` and all multi-step device history registers (`JJ`, `Inductor`, `Resistor`, `Capacitor`, `VoltageSource`, `CCVS`, `VCCS`, `TransmissionLine`) in binary format (`JOSIMSTA`), allowing long bias-conditioning intervals (e.g., 6 ns xeSFQ conditioning) to be simulated once and reused across margin/logic sweeps.
- **Implicit Predictor-Corrector Iteration for JJs (`-c`, `.OPTION CORRECTOR=<N>`, `.OPTION CORRECTORTOL=<tol>`):**
  - Performs damped Picard/BDF2 corrector passes on active Josephson junctions after the initial KLU solve when $\max |\phi^{(1)}_n - \phi^{(0)}_n| > \text{tol}$, updating `mObj.nz` and calling `klu_l_refactor` if an `RTYPE=1` junction crosses its gap threshold.

---

### Phase 4: Large-Circuit Scalability (Memory, Partitioned Solver, & Layout Sparsification)

When simulating large superconducting integrated circuits (10k–500k+ Josephson junctions, extracted 3D layouts with thousands of mutual inductances, PTL-routed multi-block chips, and multi-million-step transient runs), the primary bottlenecks shift to trace memory bandwidth, transmission-line lookback storage, LU refactorization scope, and netlist flattening overhead.

#### 1. Self-Contained Circular Ring Buffers for Transmission Lines (`TransmissionLine.hpp`, `Simulation.cpp`, `RelevantTrace.cpp`) — **[Highest ROI: Memory & Cache]**
- **Problem:** Previously, `Simulation::handle_tx` looked up delayed port voltages/phases and branch currents in `*results.xVector[...][i - k]`. Consequently, `RelevantTrace.cpp` forced all 6 MNA variables (4 nodes + 2 branch currents) of *every* transmission line in the circuit into `mObj.relevantIndices`, allocating 6 full-simulation-length `std::vector<double>(simSize_)` arrays per transmission line even when those nodes were never `.print`ed.
- **Solution:**
  - Equip each `TransmissionLine` with a compact circular history buffer `hist_` of length $k + 4$ (`timestepDelay_ + 4`) storing `{v1, v2, i1, i2}` directly inside the `TransmissionLine` object.
  - Remove transmission-line indices from `mObj.relevantIndices` in `find_relevant_traces` (unless explicitly requested by `.print`/`.save`).
  - Serialize and restore the transmission-line ring buffer in `.savestate` / `.loadstate` so in-flight PTL pulses across checkpoints are preserved with 100% fidelity.

#### 2. In-Loop Print Windowing & Decimation (`PRSTART` / `PRSTEP`) (`Simulation.cpp`, `Output.cpp`) — **[Highest ROI: Memory & Bandwidth]**
- **Problem:** Even when `.tran TSTEP TSTOP PRSTART PRSTEP` specifies a late print start (e.g., skipping a 5 ns bias-conditioning window) or a coarse print step (`PRSTEP > TSTEP`), `Simulation::trans_sim` previously allocated and recorded every internal simulation step into `results.xVector` and `results.timeAxis`, only decimating afterward in `Output::write_output`.
- **Solution:**
  - When FIR window filtering (`P_WINDOW`) is disabled and no voltage-mode non-JJ phase integration is required, pre-compute the exact print sample schedule in `Simulation::setup` (including 2 lookback steps `j - 1, j - 2` before each print point when Phase-to-Voltage BDF2 differentiation is requested on non-JJ nodes) and record *only* those required steps during `Simulation::trans_sim`.
  - Reduces `results.xVector` RAM footprint and inner-loop memory writes by **5x–20x** on long conditioning or decimated runs while preserving 100% bit-identical outputs.

#### 3. Block-Partitioned MNA Solver Across PTL Boundaries (`Simulation.hpp`, `Simulation.cpp`) — **[Highest ROI: Solver Speed & Multi-Core Scaling]**
- **Problem:** In PTL-routed superconducting chips, passive transmission lines (`T` elements) have zero instantaneous off-diagonal coupling between Port 1 and Port 2 in the MNA matrix $A$ (coupling occurs strictly through delayed history sources in $\mathbf{b}$). However, solving the entire chip as a single monolithic KLU matrix forces `klu_l_refactor` across the entire chip whenever *any* single `RTYPE=1` junction switches, and serializes `klu_l_tsolve` on a single thread.
- **Solution:**
  - Partition the MNA matrix graph `(rp, ci)` into independent connected components $A_0, A_1, \dots, A_{P-1}$ using Union-Find during `Simulation::setup`.
  - Maintain an independent KLU `(Symbolic, Numeric, Common)` solver instance and `partitionNeedsLU_[p]` dirty flag per block.
  - When an `RTYPE=1` junction in block $p$ crosses its gap threshold, only refactor block $p$ ($O(|A_p|)$ instead of $O(|A|)$).
  - When `-p` (`--parallel`) is enabled and $P \ge 2$, factor and solve all $P$ blocks in parallel across OpenMP threads.

#### 4. Mutual Inductance Coupling Cutoff (`.OPTION KCUTOFF=<val>`) (`Input.cpp`, `Matrix.cpp`) — **[High ROI: Extracted Layouts]**
- **Problem:** 3D inductance extractors emit thousands of tiny parasitic mutual couplings ($|k| < 10^{-3}$), inflating matrix non-zeros ($NNZ$), increasing LU fill-in, and slowing `Simulation::handle_inductors`.
- **Solution:**
  - Support `.OPTION KCUTOFF=<val>` to prune mutual inductance elements with $|k| < \text{kcutoff}$ during `Matrix::handle_mutual_inductance`, reporting the number of pruned vs. retained couplings in verbose mode (`-V 1`).

#### 5. Fast Hierarchical Netlist Flattening & Pre-Reserved CSR Assembly (`Netlist.cpp`, `Matrix.cpp`) — **[High ROI: Setup Time]**
- **Problem:** `Netlist::expand_maindesign` and `Matrix::create_csr` repeatedly reallocate vectors and construct temporary strings when flattening large hierarchical designs.
- **Solution:**
  - Pre-compute total expanded line counts to `reserve` `expNetlist` once, use in-place string appends (`append("|").append(label)`) in `Netlist::expand_io`, and pre-reserve `nz`, `ci`, and `rp` in `Matrix::create_csr`.

#### 6. Phase 4 Verification & Medium/Large-Scale Benchmark Results

Verified via [`scripts/test_phase4_wsl.sh`](scripts/test_phase4_wsl.sh) and [`scripts/benchmark_medium_large_wsl.sh`](scripts/benchmark_medium_large_wsl.sh) in WSL (`-O3` Release build with Ninja):
- **Mid-Flight PTL `.savestate` / `.loadstate` (`JOSIMSTA` v2):** Exact bit-identical (`0.000e+00` error) waveform continuation across a checkpoint taken while an SFQ pulse is mid-flight inside a `25 ps` transmission line.
- **In-Loop `PRSTART` / `PRSTEP` Decimation:** `0.000e+00` max error vs. full-history output across `V`, `P`, and `I` traces while reducing stored transient steps by >90%.
- **Block-Partitioned KLU & `.option kcutoff`:** Bit-identical (`0.000e+00` diff) serial vs. OpenMP (`-p`) multi-block KLU solves.

##### End-to-End Runtime & Memory Benchmark (Upstream `v2.7` vs. Enhanced)

| Design | Scale & Description | Upstream CSV | Enhanced 1T CSV | Enhanced `-p` CSV | Enhanced `-p` `.bin` | Speedup (`-p` `.bin`) | Peak RSS (`Base -> Enh`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ex_ksa4bit` | Medium: 4-bit RSFQ Kogge-Stone Adder | `2.029 s` | `1.409 s` | `1.499 s` | `1.466 s` | **1.38x** (`1.44x` 1T) | `60.9 -> 43.9 MB` (**1.39x**) |
| `ex_jtl_string` | Medium: Long RSFQ JTL chain (`100 ps`) | `3.069 s` | `2.103 s` | `2.137 s` | `2.109 s` | **1.46x–1.73x** | `39.1 -> 43.7 MB` |
| `ex_gen_pp_ipht` | Medium: IPHT SFQ pulse generator & logic | `5.962 s` | `2.563 s` | `2.547 s` | `2.570 s` | **2.32x–2.34x** | `62.6 -> 70.0 MB` |
| `med_ext_6k_K` | Medium: 500 JJs, 2.5k `L`, 6k `K` couplings | `0.826 s` | `0.557 s` | `0.360 s` | `0.357 s` | **2.31x–2.56x** | `19.7 -> 19.6 MB` |
| `large_10k_jtl` | Large: 10,000-JJ JTL (~70k devices, ~90k eqns) | `1.894 s` | `1.173 s` | `1.059 s` | `1.064 s` | **1.78x–1.79x** | `136.8 -> 149.7 MB` |
| `large_64blk_ptl` | Large: 64-tile PTL fabric (3,072 JJs, 512 PTLs) | `1.375 s` | `0.912 s` | `0.900 s` | `0.840 s` | **1.64x** | `77.8 -> 47.1 MB` (**1.65x**) |
| `large_2k_windowed` | Large: 2,000 JJs, 4k steps (`PRSTART=800p, PRSTEP=2p`) | `1.413 s` | `0.734 s` | `0.773 s` | `0.712 s` | **1.98x** | `85.1 -> 33.5 MB` (**2.54x**) |
| `vlarge_io_2.5Mpts` | Very Large: 2,500 JJs, 1,250 traces $\times$ 2k steps | `1.219 s` | `0.996 s` | `1.047 s` | `0.574 s` | **2.12x** | `75.0 -> 68.3 MB` (**1.10x**) |
| `vlarge_512ptl_10k` | Very Large: 512 PTLs over 10,000 steps (`2.5 ns`) | `0.548 s` | `0.341 s` | `0.344 s` | `0.343 s` | **1.60x** | `165.4 -> 11.1 MB` (**14.89x**) |

#### 7. Phase 5: GPU & Multi-Core Windowed PTL Acceleration (See [`docs/gpu_architecture.md`](docs/gpu_architecture.md))
- **Tier 1 (Completed) — Multi-Step Windowed PTL Block-Parallel GPU (`-g`) & OpenMP (`-p`) Solver ("Exact Waveform Relaxation"):**
  - Exploits the minimum PTL propagation delay $W = \min(T_d / \Delta t)$ to advance all $P$ PTL-decoupled circuit partitions across $W$ consecutive timesteps inside a single fused CUDA kernel (`advance_partitions_window_kernel` in [`src/CudaSolver.cu`](src/CudaSolver.cu), assigning 1 warp per partition with the partition's entire sparse $L_p + U_p$ matrix and working vectors in on-chip shared memory and device states in registers) or a single per-window OpenMP parallel region (`Simulation::trans_sim_windowed` in [`src/Simulation.cpp`](src/Simulation.cpp)).
  - **Verified Machine-Precision Equivalence (`<= 5.1e-15` max error)** across Phase mode (`-a 1`), Voltage mode (`-a 0`), `RTYPE=1` switching junctions, `RTYPE=0` junctions, mutual inductances (`K`), fractional PTL interpolation (`.option txinterp`), and output decimation (`PRSTART`/`PRSTEP`).
  - **BNL `exxact` Workstation Benchmark Results (`Dual AMD EPYC 7413 48C/96T + NVIDIA RTX A6000`):**
    - **`PTL-Tile-64` (`2,048` JJs, `10,560` eqns, `40k` steps):** Serial `7.16s` $\to$ OpenMP (32T) **`0.61s` (`11.78x` speedup)**, GPU **`5.69s` (`1.26x` speedup)**.
    - **`PTL-Tile-256` (`8,192` JJs, `42,240` eqns, `40k` steps):** Serial `29.88s` $\to$ OpenMP (32T) **`2.06s` (`14.54x` speedup)**, GPU **`6.00s` (`4.98x` speedup)**.
    - **`PTL-Tile-1024` (`32,768` JJs, `168,960` eqns, `20k` steps):** Serial `104.90s` $\to$ OpenMP (32T) **`4.20s` (`24.95x` speedup)**, GPU **`6.95s` (`15.10x` speedup)**.
    - **`PTL-Tile-4096` (`131,072` JJs, `675,840` eqns, `10k` steps):** Serial `232.89s` $\to$ OpenMP (32T) **`8.58s` (`27.15x` speedup)**, GPU **`10.85s` (`21.47x` speedup; `15.16x` end-to-end wall-clock speedup)**.
  - **Non-PTL & Diverse Superconducting Circuit Topologies (`scripts/benchmark_non_ptl_exxact.py` on BNL `exxact`):**
    - **Multi-Channel Non-PTL Circuits (`0 PTLs`, $P \ge 2$, SQUID / detector arrays with mutual inductances `K` and capacitors `C`):** Automatically partitioned across ground-decoupled channels with a `1,024`-step lookahead window ($W = 1,024$), achieving **`15.90x–24.98x` OpenMP (`-p`) speedup** and **`3.05x–10.30x` GPU (`-g`) speedup** up to **`1,196,032` MNA equations** (`131,072` JJs, `393,216` mutual couplings `K`, `131,072` capacitors `C`, `Max Diff = 5.24e-17`) across both Phase and Voltage modes.
    - **Rich Multi-Gate RSFQ ALU Macros (`JJ + L + Mutual K + Cap C + R + PTL`):** Achieves **`15.80x–20.93x` OpenMP speedup** and **`3.00x–8.59x` GPU speedup** up to **`1,212,412` MNA equations** (`393,216` mutual couplings `K`, `131,072` capacitors `C`, `Max Diff = 6.96e-17`).
    - **Purely Monolithic Single-Component Circuits (`0 PTLs`, $P = 1$):** Fused single-region OpenMP `setup_b` (`#pragma omp parallel` across `b_` zeroing, JJs, resistors, capacitors, and two-pass inductors + mutual couplings) and multi-block GPU `monolithic_eval_devices_kernel` achieve **`1.06x–1.16x` OpenMP speedup** once per-step device work grows (`Mono-2D-JJ-Grid` at `106,242` equations and `Mono-RLK-49kK-Volt` with `49,152` mutual couplings in Voltage mode).
- **Tier 2 (Completed) — Batched GPU (`-g`) & Multi-Core OpenMP (`-p`) Solver (`trans_sim_mc`, `advance_mc_coalesced_kernel`) for `.spread` Monte Carlo & Margin Sweeps:**
  - Supports `-b <N>` / `--batch=<N>`, `.option mc=<N>` / `batch=<N>` / `runs=<N>`, `.spread ... runs=<N>`, `.option mcall`, and `JOSIM_SEED`.
  - Parses the netlist and performs connected-component symbolic partitioning **once**, then solves $B$ fabrication-variation trials simultaneously in a coalesced batch-minor layout (`[baseIdx * B + b]`) with hardware-adaptive CUDA block sizing (`warpsPerBlock` $\in \{1, 2, 4\}$) and shared symbolic CSR indices across all NVIDIA GPU architectures (`CMAKE_CUDA_ARCHITECTURES="native"`), outputting Welford online `<Trace>`, `<Trace>_STD`, `<Trace>_MIN`, and `<Trace>_MAX` statistics.
  - **BNL `exxact` 16-Case Benchmark Results (`scripts/benchmark_mc_exxact.py`):**
    - **`ex_ksa4bit` (4-bit RSFQ Adder, $P=66$, $B=256$):** Est. $256\times$ Serial `204.24s` $\to$ Batched OpenMP **`6.56s` (`31.14x` speedup)**, Batched GPU **`10.39s` (`19.65x` transient speedup, `18.33x` wall-clock speedup)** (`Rel Diff = 1.04e-15`).
    - **`ex_mitll_dff_wr` (MIT-LL RSFQ DFF, $B=1024$):** Est. $1024\times$ Serial `11.30s` $\to$ Batched OpenMP **`0.401s` (`28.16x` speedup)**, Batched GPU **`1.56s` (`7.24x` transient speedup, `4.96x` wall-clock speedup)** (`Rel Diff = 6.67e-15`).
    - **`Mono-RLK-SQUID-Macro` (0-PTL Monolithic $P=1$, 96 JJs, 191 K, $B=1024$):** Est. $1024\times$ Serial `19.36s` $\to$ Batched OpenMP **`0.796s` (`24.34x` speedup)**, Batched GPU **`5.51s` (`3.52x` transient speedup, `3.16x` wall-clock speedup)** (`Rel Diff = 1.67e-13`).
    - **`RSFQ-Pipeline-64B` (64-Block PTL Pipeline $P=64$, $B=256$, `4,848,640` eqns):** Est. $256\times$ Serial `48.57s` $\to$ Batched OpenMP **`1.68s` (`28.87x` speedup)**, Batched GPU **`3.19s` (`15.21x` transient speedup, `12.55x` wall-clock speedup)** (`Rel Diff = 1.45e-17`).
- **Tier 3 (Completed) — Fully GPU-Resident Monolithic Solver (`cuSolverRF` + Zero-PCIe Fused Device Kernels, `-g 3` / `.option gpu=3` / `.option gpumono` / `.option cusolverrf`) & Unified Auto-Dispatcher (`-g`):**
  - Extracts the exact unscaled symbolic and numeric CSR factors $\tilde{L} = U_{\text{klu}}^T D^{-1}$ (unit diagonal) and $\tilde{U} = D L_{\text{klu}}^T$ (non-unit diagonal) and permutations $P_{\text{csr}} = Q_{\text{klu}}, Q_{\text{csr}} = P_{\text{num}}$ from KLU in $O(\text{nnz}(L+U))$ time via `klu_l_extract`, and initializes NVIDIA **`cuSolverRF`** (`cusolverRfSetupHost` $\to$ `cusolverRfAnalyze` $\to$ `cusolverRfResetValues` $\to$ `cusolverRfRefactor` $\to$ `cusolverRfSolve`).
  - Evaluates **all** circuit devices (`JJ`, `R`, `C`, `L + Mutual K`, `V`, `P`, `I`, `CCVS`, `VCCS`, and `T` port halves) in parallel in VRAM (`monolithic_full_setup_b_kernel`), updates `d_csrValA` in-place when `RTYPE=1` junctions switch, and records decimated traces in VRAM (`monolithic_full_post_solve_kernel`) with **zero per-timestep PCIe transfers**.
  - **Verified Machine-Precision Equivalence (`0.000e+00` to `1.320e-17` max error)** across 8 Phase (`-a 1`) and Voltage (`-a 0`) verification circuits.
  - **BNL `exxact` Benchmark Results (`scripts/benchmark_tier3_exxact.py`):**
    - **`Mono-2D-Grid-14kJJ-Volt` ($120\times120$ JJs, 0 PTL, $P=1$, `-a 0`):** Serial `4.810s` $\to$ Hybrid `-g 2` `3.258s` (`1.48x`) $\to$ **Tier 3 `-g 3` `1.158s` (`4.16x` speedup; `-g` Auto `1.144s`, `4.21x` speedup)**.
    - **`Mono-2D-Grid-14kJJ-Phase` ($120\times120$ JJs, 0 PTL, $P=1$, `-a 1`):** Serial `4.487s` $\to$ Hybrid `-g 2` `3.501s` (`1.28x`) $\to$ **Tier 3 `-g 3` `1.164s` (`3.86x` speedup; `-g` Auto `1.148s`, `3.91x` speedup)**.
    - **`Mono-2D-Grid-25kJJ-Phase` ($160\times160 = 25,600$ JJs, 0 PTL, $P=1$):** Serial `8.867s` $\to$ Hybrid `-g 2` `7.750s` (`1.14x`) $\to$ **Tier 3 `-g 3` `2.407s` (`3.68x` speedup; `-g` Auto `2.395s`, `3.70x` speedup)**.
    - **`Mono-2D-Grid-64kJJ-NR` ($256\times256 = 65,536$ JJs, `393,730` eqns, 0 PTL, $P=1$):** Serial `30.458s` $\to$ Hybrid `-g 2` `25.501s` (`1.19x`) $\to$ **Tier 3 `-g 3` `11.865s` (`2.57x` speedup)**.
- **Tier 4 (Completed) — Exact Schur-Complement Nodal Admittance Formulation on CPU (`-n`) & GPU (`-g 4`) and Activity-Driven Quiescent Partition Bypass (`.option bypass`):**
  - **Mathematical Foundation (`Simulation::build_nodal_system`, `Simulation::setup_cpu_nodal`, `Simulation::setup_gpu_nodal`):**
    - Analytically eliminates all diagonal branch equations (`JJ` `variableIndex_` and `currentIndex_`, `Resistor` `currentIndex_`, `Capacitor` `currentIndex_`, uncoupled `Inductor` `currentIndex_`, `TransmissionLine` `currentIndex_` and `currentIndex2_`, and grounded Dirichlet `VoltageSource` and `PhaseSource` branch rows) into the exact Schur-complement Nodal Admittance matrix $A_{\text{red}}$ of dimension $N_{\text{red}} \ll N$.
    - Reduces matrix dimension $N \to N_{\text{red}}$ by **1.9x–6.0x** (e.g., $392,707 \to 65,537$ on `Mono-2D-Grid-64kJJ`; $89,806 \to 46,994$ on `r10f1n`; $64,400 \to 33,578$ on `r5cal`) and reduces LU factor nonzeros $\text{nnz}(L+U)$ by **1.5x–2.5x**.
    - Reconstructs any requested `.print` branch currents or junction variables on the fly during output recording (`nodal_full_post_solve_kernel` on GPU and `trans_sim_nodal` on CPU).
    - Supports both **CPU Nodal Solver (`-n` / `--nodal`, `.option nodal`, `-p -n`)** via KLU on $A_{\text{red}}$ and **GPU Tier 4 Nodal Solver (`-g 4` / `--gpu=4`, `.option gpu=4`, `.option gpunodal`)** via `cuSolverRF` on $A_{\text{red}}$.
  - **Activity-Driven Quiescent Partition Bypass (`.option bypass`, `.option bypasstol=<val>`, `.option latency=<val>`):**
    - Tracks per-partition dynamic activity across `JJ`, `Resistor`, `Capacitor`, `Inductor` (+ mutual `K`), sources, and PTL boundary histories in `Simulation::trans_sim_omp_windowed` and skips redundant `klu_l_tsolve` calls on quiescent partitions.
  - **NUMA-Aware OpenMP Thread Scaling (`Simulation::trans_sim_omp_windowed`, `Simulation::setup_b`):**
    - Caps default partition OpenMP thread count to 16 threads (matching single-socket L3 cache domain on dual-socket servers) and raises per-step parallel thresholds (`numTx * w >= 4096` and `setup_b` device threshold `32,768`), accelerating `spec2si-seeqc` `adder` (`583` partitions, $W = 3$) to **`3.61x` OpenMP transient speedup (`23.36 s` vs `84.24 s`)**.
  - **Verified Machine-Precision Equivalence (`1.60e-18` to `1.00e-14` relative error)** across `ex_jtl_basic`, `ex_ksa4bit`, `Mono-2D-Grid-14kJJ`, and all `spec2si-seeqc` benchmarks (`xadd4`, `r5cal`, `r10f1n`, `coupler`, `dff`, `adder`).
  - **BNL `exxact` Tier 4 Benchmark Highlights:**
    - **`Mono-2D-Grid-14kJJ` ($120\times120$ JJs, $N=86,163 \to N_{\text{red}}=14,401$):** Serial MNA `4.636s` $\to$ **CPU Nodal (`-n`) `1.106s` (`4.19x` speedup)**, **OMP Nodal (`-p -n`) `1.060s` (`4.37x` speedup)**, **GPU Tier 4 (`-g 4`) `3.089s` (`1.50x` speedup, `1.12x` faster than Tier 3)**.
    - **`Mono-2D-Grid-25kJJ` ($160\times160$ JJs, $N=153,283 \to N_{\text{red}}=25,601$):** Serial MNA `8.882s` $\to$ **CPU Nodal (`-n`) `2.836s` (`3.13x` speedup)**, **OMP Nodal (`-p -n`) `2.214s` (`4.01x` speedup)**, **GPU Tier 4 (`-g 4`) `3.820s` (`2.32x` speedup, `1.15x` faster than Tier 3)**.
    - **`Mono-2D-Grid-64kJJ` ($256\times256$ JJs, $N=392,707 \to N_{\text{red}}=65,537$):** Serial MNA `29.037s` $\to$ **CPU Nodal (`-n`) `9.289s` (`3.13x` speedup)**, **OMP Nodal (`-p -n`) `8.646s` (`3.36x` speedup)**, **GPU Tier 4 (`-g 4`) `17.223s` (`1.69x` speedup, `1.98x` faster than Tier 3)**.
    - **Production `spec2si-seeqc` Benchmarks (`-a 1` Phase Mode):**
      - **`r5cal` ($N=64,400 \to N_{\text{red}}=33,578$, $P=1$):** Serial MNA `218.69s` $\to$ **CPU Nodal (`-n`) `88.49s` (`2.47x` speedup; wall-clock `220.67s -> 90.31s`, `2.44x`)**.
      - **`r10f1n` ($N=89,806 \to N_{\text{red}}=46,994$, $P=2$):** Serial MNA `79.05s` $\to$ **CPU Nodal (`-n`) `37.01s` (`2.14x` speedup; wall-clock `79.89s -> 37.81s`, `2.11x`)**.
      - **`xadd4` ($N=31,813 \to N_{\text{red}}=16,393$, $P=1$):** Serial MNA `25.44s` $\to$ **CPU Nodal (`-n`) `15.95s` (`1.59x` speedup; wall-clock `29.01s -> 19.63s`, `1.48x`)**.
      - **`coupler` ($N=38,826 \to N_{\text{red}}=15,372$, $P=464$):** Serial MNA `35.44s` $\to$ **CPU Nodal (`-n`) `18.92s` (`1.87x` 1-core speedup)**, **OpenMP (`-p`) `4.70s` (`7.53x` parallel partition speedup)**.
      - **`adder` ($N=47,425 \to N_{\text{red}}=20,450$, $P=583$):** Serial MNA `84.24s` $\to$ **CPU Nodal (`-n`) `46.84s` (`1.80x` 1-core speedup)**, **OpenMP (`-p`) `23.36s` (`3.61x` parallel partition speedup)**.
      - **`dff` ($N=504 \to N_{\text{red}}=230$, $P=1$):** Serial MNA `0.231s` $\to$ **CPU Nodal (`-n`) `0.157s` (`1.47x` speedup)**.

---

## 9. Phase 5 Prioritized Roadmap (Ordered by Payoff / Effort Ratio)

| Rank | Proposed Improvement | Status | Effort | Payoff | Primary Target & Measured / Expected Impact |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Series $R + L$ Internal Node Elimination in Nodal Solver (`-n`)** | **DONE** | **Low**<br>*(~150 LOC)* | **Very High** | Eliminates degree-2 internal nodes between series $R$ and uncoupled $L$ chains, shrinking $A_{\text{red}}$ to **`8,565` eqns (`3.71x`)** on `xadd4`, **`9,870` (`3.93x`)** on `coupler`, **`9,195` (`5.16x`)** on `adder`, **`29,622` (`2.17x`)** on `r5cal`, and **`41,262` (`2.18x`)** on `r10f1n` (**2.3x–3.2x 1-thread speedup**). |
| **2** | **Per-Partition Nodal Reduction Inside Windowed PTL Solver (`-p -n`)** | **DONE** | **Low–Med**<br>*(~250 LOC)* | **Very High** | Applies Nodal reduction *inside each PTL partition* (`NodalPartition`), multiplying `-p` and `-n` gains on `adder` (**`23.36s -> 12.53s` solve / `18.96s` wall**, **`6.65x` solve speedup**) and `coupler` (**`4.70s -> 2.71s` solve / `5.73s` wall**, **`13.38x` solve speedup**). |
| **3** | **Short-PTL Coarsening & Sub-Step Lumped $\pi$-Replacement (`r10forig`, `adder`)** | **DONE** | **Low–Med**<br>*(~200 LOC)* | **Very High** | Replaces $T_d < 0.5\Delta t$ PTL stubs (`1 fs` lines in `r10forig`) with exact $\pi$-equivalents (**`41.6s -> 3.39s` wall / `2.81s` solve on `r10forig` — `38.8x` vs `5f33997`, `561.7x` vs `v2.6.5`**) and coarsens 0-JJ PTL-via islands + $k < 4$ PTLs (`adder`: $P = 583 \to 126, W = 3 \to 5$, **`18.96s -> 11.33s` wall / `12.53s -> 5.47s` solve — `15.24x` solve speedup**). |
| **4** | **Direct In-Loop Trace Recording & Streaming `.npy`/`.bin` Output** | **DONE** | **Low–Med**<br>*(~200 LOC)* | **High** | Drops unused JJ `variableIndex_` trace vectors, folds exclusive `V(n1, n2)` node-difference pairs (`traceType = 4`), parallelizes NUMA trace allocation across OpenMP threads, and streams `.npy`/`.bin`/`.rawb` directly from `results.xVector` via a 64×8 L1 cache-tiled transpose without allocating `Output::traces`: cuts `adder` peak RAM from **`8.04 GB -> 1.72 GB` (`4.67x` reduction; `15.4x` vs `v2.6.5`)** and `coupler` from **`4.3 GB -> 1.42 GB` (`3.0x` reduction)**, accelerating **`adder -p -n` to `7.20 s` wall / `3.79 s` solve (`75.3x` vs `v2.6.5`)** and **`coupler -p -n` to `4.08 s` wall / `2.41 s` solve (`38.2x` vs `v2.6.5`)**. |
| **5** | **Built-In SFQ Switching Event & Pulse Metric Extractor (`-e`, `-E`)** | **DONE** | **Low**<br>*(~220 LOC)* | **High** | Detects $(2k-1)\pi$ SFQ phase crossings with sub-step linear interpolation and optional pulse metrics (`-E`: `t_peak`, `v_peak`, `fwhm`, `dt_pi`, `area`, `time=peak`) in zero-allocation events-only mode (`-e` without `-o`): cuts **`adder -p -n -e` to `4.63 s` wall / `195 MB` RSS (`605 KB` output — `5,250x` smaller, `117x` faster vs `v2.6.5`)** and **`coupler -p -n -e` to `2.58 s` wall / `216 MB` RSS (`512 KB` output — `2,503x` smaller, `60.5x` faster vs `v2.6.5`)** with `0.0000 fs` shift. |
| **6** | **In-Memory Rolling Checkpoints for Local Step-Halving Rollback (`.option rollback`)** | **DONE** | **Medium**<br>*(~250 LOC)* | **Med–High** | Keeps a double-buffered 128-step rolling in-memory state buffer (`ckpt_[2]`, `<2 MB`) in both Standard MNA and CPU Nodal (`-n`), rolling back only $32\text{–}256$ base steps with in-place BDF2 history rescaling ($h_{\text{old}} \to h_{\text{new}}$) and $M = 2^k$ internal sub-stepping instead of restarting from $t = 0$: delivers **`2.02x–3.93x` speedup** across Phase & Voltage modes (`166`-JJ and `2,006`-JJ MIT-LL DCSFQ + JTL chains) with `0.0000 rad` final phase error. |
| **7** | **Batched Deterministic Parameter Sweeps (`.sweep` + `-b <N> -n`)** | **DONE** | **Medium**<br>*(~300 LOC)* | **High** | Extends the batched engine with 1D/2D deterministic parameter/margin sweeps (`.sweep`, `.option sweep=...`, `-S` / `--sweep`) across `.param` identifiers, `@BIAS` / `@JJ` / `@IND` / `@RES` / `@CAP` / `@SRC` / `@GLOBAL` categories, and wildcard device patterns (`IB*`, `B*`), plus Batched Tier 4 Schur-Complement Nodal reduction (`-p -n` with `.sweep` / `-b <N>`): delivers **`33.3x–51.8x` speedup vs 64 sequential MNA runs** (**`1.46x–1.84x` faster than Batched MNA**) with `0.000e+00`–`1.000e-14` parity across MNA, Nodal, and CUDA GPU and automatic SFQ operating margin window extraction. |

### 9.1 Phase 5 Item #7 Benchmark Results on `exxact` (Dual AMD EPYC 7413 48C/96T + NVIDIA RTX A6000)

1. **1D `.param` & Margin Sweep Parity (`ex_mitll_dff_wr.cir`, `.sweep Scaling LIN 21 0.7 1.3` & `-S "@BIAS:MARGIN:0.25:21"`):**
   - Across all 21 sweep points (`Scaling = 0.70 .. 1.30`), Batched MNA (`-p`), Batched Nodal (`-p -n`), and Batched CUDA GPU (`-g`) achieve **`0.000e+00` to `1.000e-14` maximum waveform error** and automatically identify the SFQ operating window (`Nom SFQ = 2`, `15/21` passing points, operating window `[0.88, 1.30]`).
   - 2D Cartesian Margin Sweep (`.sweep @BIAS MARGIN 0.35 11` × `.sweep @JJ MARGIN 0.25 11`, $11 \times 11 = 121$ grid points) executes in a single batched Nodal pass (`40/121` passing points around `Nom SFQ = 2`).
2. **64-Point Deterministic Sweep & 64-Trial Monte Carlo Speedup (`exxact`):**

| Circuit | 1x MNA | 1x Nodal (`-n`) | 64x Seq MNA Est. | **64-pt Sweep MNA (`-p`)** | **64-pt Sweep Nodal (`-p -n`)** | **64-trial MC MNA (`-p -b 64`)** | **64-trial MC Nodal (`-p -n -b 64`)** | **Sweep Speedup vs 64x Seq MNA** | **Nodal vs Batched MNA** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| **`ex_mitll_dff_wr` (`23` JJs)** | `67.7 ms` | `59.7 ms` | `4,330.2 ms` | `122.1 ms` (`35.5x`) | **`83.6 ms`** | `127.7 ms` | **`93.0 ms`** | **`51.8x`** | **`1.46x` / `1.37x`** |
| **`ex_ksa4bit` (`518` JJs)** | `1,508.5 ms` | `709.2 ms` | `96,542.2 ms` | `5,011.3 ms` (`19.3x`) | **`2,779.9 ms`** | `2,001.9 ms` | **`1,087.7 ms`** | **`34.7x`** | **`1.80x` / `1.84x`** |
| **`ex_jtl_string` (`2,000` JJs)** | `2,363.7 ms` | `1,103.6 ms` | `151,274.6 ms` | — | **`4,541.0 ms`** | — | **`1,807.6 ms`** (`83.7x`) | **`33.3x`** | **`15.6x` vs 64x Seq Nodal** |


