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
- **Tier 2 — Batched GPU Solver (`Batch-KLU`) for `.spread` Monte Carlo & Margin Sweeps:** Solve $B = 64\text{–}4,096$ parameter/noise trajectories in SIMT lockstep sharing a single symbolic factorization.
- **Tier 3 — Fully GPU-Resident Monolithic Solver (`cuDSS` / `GLU 3.0` + Fused SoA Kernels):** Keep device SoA arrays and CSR matrices in VRAM for single monolithic circuits $>50\text{k}$ JJs.
- **Tier 4 — Explicit / Semi-Implicit Nodal Formulation & Future Roadmap:**
  - Series $R + L$ internal node elimination and Norton inductor companion stamping.
  - In-memory rolling checkpoints & local timestep rollback on `needsTR_`.
  - Activity-driven / event-driven subcircuit bypass for quiescent PTL partitions.
  - Streaming binary output in constant $O(1)$ memory.
