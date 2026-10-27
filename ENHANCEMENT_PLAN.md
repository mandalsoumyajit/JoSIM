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

#### 6. Future Large-Circuit Roadmap (Phase 5+)
- **Series $R + L$ Internal Node Elimination & Compact Nodal Stamping:** Fold uncoupled two-terminal resistors/capacitors and series $R$-$L$ JJ shunt branches into nodal/composite stamps to shrink MNA dimension by 35%–50%.
- **In-Memory Rolling Checkpoints & Local Timestep Rollback:** Replace full $t = 0$ restarts on `needsTR_` with local rollback to a recent in-memory state snapshot using variable-step BDF2 coefficients.
- **Activity-Driven / Event-Driven Subcircuit Bypass:** Skip device evaluation and block KLU solves on quiescent PTL-decoupled partitions whose terminal states and inputs are idle ($\Delta \phi < \epsilon$).
- **Streaming Binary Output & Multi-Run `.param` / `.spread` Batch Engine:** Flush binary trace chunks to disk in constant $O(1)$ memory and reuse `klu_l_symbolic` across parallel Monte Carlo / margin sweeps.
