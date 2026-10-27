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

### Phase 3: Numerical Integration & Superconductor IC Features
- **Fractional Delay Interpolation for Transmission Lines (`TransmissionLine.cpp`)**
- **Explicit Step-Control Diagnostics & CLI Flags (`--fixed-step`, warning on step halving)**
- **State Checkpointing (`.SAVE_STATE` / `.LOAD_STATE`) for Fast xeSFQ Conditioning**
- **Newton-Raphson JJ Corrector & Adaptive LTE Time-Stepping**
- **Native SFQ `.MEASURE` (Fluxon Phase-Slip Timestamps & Margin Sweeps)**
