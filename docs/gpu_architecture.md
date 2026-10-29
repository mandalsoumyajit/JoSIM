# GPU Acceleration Architecture & SPICE Solver Survey for JoSIM

## 1. Survey of GPU-Accelerated SPICE-Like Simulators

Modern SPICE-class simulators face a fundamental tension on GPUs: transistor/device equations are data-parallel, whereas Modified Nodal Analysis (MNA) matrices are hypersparse (~3–5 nonzeros per row), unsymmetric, and produce deep, narrow elimination trees during LU factorization and forward/backward triangular substitution (`SpSV`).

| Simulator / Solver | GPU Architecture & Strategy | Key Lessons & Bottlenecks |
| :--- | :--- | :--- |
| **NGSPICE (`CUSPICE`)** | Converts device structs into flattened Structure-of-Arrays (SoA) in GPU VRAM; evaluates models (`BSIM4`) and stamps the RHS/Jacobian on the GPU; solves $Ax=b$ via **`cuSolverRF`** or **`GLU 3.0`**. | **Zero PCIe transfers in the timestep loop is mandatory.** Copying $\mathbf{b}$ or $\mathbf{x}$ across PCIe every timestep ($5\text{–}10\,\mu\text{s}$ latency) destroys speedup. |
| **Xyce (Sandia) + Trilinos (`Kokkos` / `Basker` / `Tacho`)** | Uses `Kokkos::View` for portable GPU device evaluation and couples to Block Triangular Form (BTF) solvers (**`ShyLU-Basker`**) or **`NVIDIA cuDSS`**. | Published Sandia benchmarks show a single monolithic circuit needs **$>50\text{k–100k}$ devices** before a GPU sparse direct solver beats CPU KLU/Basker, *unless* the matrix breaks into many independent BTF blocks. |
| **`GLU 3.0` / `cuSolverRF` / `NVIDIA cuDSS`** | Hybrid sparse direct solvers: perform symbolic analysis, AMD reordering, and BTF pivoting **once on the CPU**, then execute level-scheduled numeric refactorization (`Rf`) and sparse triangular solves (`SpSV`) on the GPU. | Single-matrix triangular solves (`L\b`, `U\y`) on narrow circuit elimination trees suffer from low warp occupancy unless many independent sub-trees or blocks exist. |
| **`Batch-KLU` / `Ginkgo` Batched Solvers** | Packs hundreds of identical-sparsity matrices (or decoupled subcircuit blocks) into a batched CSR format `[nnz_idx][batch_idx]` so each CUDA warp/block solves a distinct system with 100% coalesced memory access. | Eliminates dependency-graph starvation completely; achieves near-peak GPU memory bandwidth on small- and medium-scale circuits when running multi-parameter or multi-tile workloads. |
| **Commercial GPU SPICE (`Spectre X GPU`, `PrimeSim GPU`)** | Combines GPU device evaluation with **Partitioned Waveform Relaxation (WR)** / multi-rate harmonic balance and **Batched Monte Carlo**. | Partitioning across weakly coupled or delay-decoupled boundaries lets GPU thread-blocks advance independent partitions over multiple timesteps without global synchronization. |

---

## 2. Why Superconducting SPICE (JoSIM) Differs Mathematically from CMOS SPICE

1. **Lightweight Device Physics (~30 FLOPs vs. ~5,000 FLOPs for CMOS BSIM4)**
   - In CMOS SPICE, evaluating a single `BSIM4` or `PSP` transistor takes thousands of floating-point operations (60%–75% of total runtime), saturating GPU compute cores even on small circuits.
   - In JoSIM (`Simulation::handle_jj`), the RCSJ Josephson junction update is a BDF2 phase prediction plus a single `sin()` call (~30 FLOPs). Furthermore, JoSIM’s MNA matrix $\mathbf{A}$ is **constant on $>95\%$ of timesteps** (it only refactors when an `RTYPE=1` junction crosses $V_g \pm \frac{1}{2}\Delta V$).
   - Consequently, on most timesteps JoSIM only executes `setup_b` + a sparse triangular solve (`klu_l_tsolve`). A naive GPU port that only offloads `handle_jj` or performs single-matrix `cuSPARSE` triangular solves on a narrow tree is slower than CPU L3-cached KLU for $N < 50\text{k}$.

2. **Exact Multi-Step Temporal Decoupling Across PTLs (`T` Lines)**
   - In CMOS SPICE, wires are RC networks that couple all blocks instantaneously at every timestep, requiring iterative Waveform Relaxation (which can fail to converge on tight feedback loops).
   - In superconducting digital circuits (RSFQ, ERSFQ, xeSFQ, AQFP), logic tiles communicate via passive transmission lines (`T` elements) with propagation delay $T_d = k \cdot \Delta t$ (typically $k = 20\text{–}100$ timesteps).
   - Because PTLs have **zero instantaneous coupling** between Port 1 and Port 2 in $\mathbf{A}$, if the minimum PTL delay in a chip is $W = k_{\min} = \min(T_d / \Delta t)$ steps, **every PTL-bounded circuit partition is 100% mathematically independent for $W$ consecutive timesteps!**

---

## 3. Tiered GPU Acceleration Architecture for JoSIM

### Tier 1: Multi-Step Windowed PTL Block-Parallel Solver ("Exact Waveform Relaxation" on GPU & CPU) — **[10x–25x Speedup on Large Multi-Tile Chips]**
- **Target Workload:** Large PTL-interconnected superconducting processors and multi-tile fabrics ($5\text{k}–500\text{k}$ JJs partitioned into $P = 16 \text{ to } 4,096$ PTL-coupled islands).
- **Mathematical Foundation:**
  1. Partition the MNA matrix AND all non-TX circuit devices (`JJ`, `Inductor`, `Resistor`, `Capacitor`, `VoltageSource`, `PhaseSource`, `CurrentSource`, `CCVS`, `VCCS`) into $P$ independent connected components $p \in \{0, \dots, P-1\}$ separated only by transmission lines (`T` elements).
  2. Compute the **exact lookahead window** $W = \max(1, \min_{T_m} k_m)$ where $k_m = \lfloor T_{d,m} / \Delta t \rfloor$ (or $k_m - 1$ when fractional interpolation `.option txinterp` is active).
  3. Within any time window $[t_0, t_0 + W - 1]$:
     - Every transmission line lookup `effStep - k_m` reads strictly from steps $< t_0$, which are **already completed and immutable** in `tx.hist_`!
     - Therefore, partition $p$ can advance across all $W$ timesteps $t = t_0, t_0 + 1, \dots, t_0 + W - 1$ **completely independently of all other partitions**!
  4. **CUDA Execution (`#ifdef JOSIM_CUDA`) & OpenMP Windowed Execution:**
     - **On GPU (CUDA):** Each CUDA thread-block (or warp) is assigned one partition $p$. It loads partition $p$'s sparse $L_p, U_p$ factors, permutation vectors, local solution vector $\mathbf{x}_p$, and local device state registers into **GPU Shared Memory (SRAM) / registers** and loops over $s = 0 \dots W-1$ timesteps *inside a single fused CUDA kernel*:
       - Evaluates local devices (`JJ`, `L`, `R`, `C`, sources) and pre-computed PTL boundary contributions into $\mathbf{b}_p$
       - Refactors $L_p, U_p$ in-place on the GPU if any local `RTYPE=1` junction switched
       - Solves $L_p U_p \mathbf{x}_p = P_p Q_p \mathbf{b}_p$ in shared memory
       - Writes newly solved PTL port voltages/currents into `tx.hist_` and requested `.print` traces into the GPU output buffer.
     - **On Multi-Core CPU (OpenMP):** Even without a GPU, advancing each partition $p$ for $W$ consecutive steps inside a single OpenMP parallel region reduces OpenMP thread-barrier frequency by a factor of $W$ (e.g. $20\times$ fewer barriers) and keeps partition $p$'s KLU factors and device arrays hot in per-core L1/L2 cache!

### Tier 2: Batched GPU Solver for `.spread` Monte Carlo, Margin Optimization & Parameter Sweeps — **[50x–100x Throughput]**
- **Target Workload:** Critical margin analysis, yield estimation (`.spread`), bias optimization, and `.iv` curves, where the same netlist is simulated $B = 64 \text{ to } 4,096$ times with perturbed parameters.
- **Architecture:** All $B$ runs share the exact same sparsity graph (`rp`, `ci`) and symbolic elimination tree (`Symbolic_`). State vectors and nonzeros are stored in interleaved batch-major layout (`x[var_idx * B + batch_id]`), so each CUDA warp executes 32 independent circuit simulations in lockstep with 100% coalesced memory access.

### Tier 3: Fully GPU-Resident Monolithic Solver (`NVIDIA cuDSS` / `GLU 3.0` + Fused SoA Kernels) — **[3x–6x Speedup for $>50\text{k}$ JJs]**
- **Target Workload:** Single massive tightly-coupled circuits without PTL breaks ($>50,000$ JJs and $>250,000$ MNA equations).
- **Architecture:** Structure-of-Arrays (SoA) device kernels in VRAM paired with `NVIDIA cuDSS` / `cuSolverRF` for level-scheduled GPU sparse LU refactorization and triangular solves.

### Tier 4: Exact Schur-Complement Nodal Admittance Formulation on CPU (`-n`) & GPU (`-g 4`) + Quiescent Partition Bypass (`.option bypass`) — **[1.5x–4.4x Monolithic Speedup]**
- **Target Workload:** Large monolithic single-partition circuits ($P = 1\text{–}2$, e.g. `spec2si-seeqc` `r5cal`, `r10f1n`, `xadd4` and 2D/3D JJ arrays `14kJJ–64kJJ`) where standard MNA inflates the matrix dimension by $2\times–6\times$ with internal branch equations (`JJ` `variableIndex_` and `currentIndex_`, `R`/`C`/uncoupled `L`/`T` branch currents, and grounded `V`/`P` sources).
- **Architecture:** Analytically eliminates all diagonal internal branch equations via exact Schur-complement reduction into the reduced Nodal Admittance matrix $A_{\text{red}}$ of dimension $N_{\text{red}} \ll N$ (reducing matrix rows by **1.9x–6.0x** and LU factor nonzeros $\text{nnz}(L+U)$ by **1.5x–2.5x**), solves $A_{\text{red}} \mathbf{x}_{\text{red}} = \mathbf{b}_{\text{red}}$ via KLU on CPU (`-n` / `.option nodal`) or `cuSolverRF` in GPU VRAM (`-g 4` / `.option gpu=4`), and reconstructs any requested `.print` branch currents on the fly. Also adds activity-driven quiescent partition bypass (`.option bypass`) and NUMA-aware OpenMP scaling (`adder -p` `3.61x`).

---

## 4. Tier 1 Implementation Details & BNL `exxact` Benchmark Results

### 4.1 Implementation Architecture (`Simulation.cpp`, `CudaSolver.cu`)
Tier 1 is implemented for both **Multi-Core OpenMP CPU (`-p` / `--parallel`, `-DUSING_OPENMP=ON`)** and **NVIDIA CUDA GPU (`-g` / `--gpu`, `.option gpu`, `-DUSING_CUDA=ON`)**:

1. **Full Device & Matrix Partitioning (`Simulation::setup_partitions`):**
   - Partitions the MNA matrix $\mathbf{A}$ into $P$ connected components across lossless/lossy transmission lines (`T` elements) and classifies every circuit device (`JJ`, `Inductor` + mutual `M`, `Resistor`, `Capacitor`, `VoltageSource`, `PhaseSource`, `CurrentSource`, `CCVS`, `VCCS`, and `TransmissionLine` Port 1 / Port 2 halves) into its owning partition $p \in \{0, \dots, P-1\}$.
   - Computes the exact causal lookahead window $W = \min_{T_m} (k_m - \mathbb{I}[\alpha_m > 0])$, bounded by `[1, 256]`, and builds the in-loop print decimation table `keptSlot_[i]` (`PRSTART` / `PRSTEP`).
   - Automatically detects **zero-state startup** (all sources zero at $t \le 0$, zero `phiOff`, no thermal noise) to skip redundant $t < 0$ stabilization steps.

2. **OpenMP Multi-Step Windowed CPU Solver (`Simulation::trans_sim_windowed`):**
   - Precomputes delayed remote PTL boundary terms `txRemote1_`, `txRemote2_` for all $s \in [0, W)$, then launches a single `#pragma omp parallel for` across partitions $p \in [0, P)$ where each thread advances its partition across all $W$ timesteps in L1/L2 cache using `advance_partition_step` before synchronizing at the window boundary.

3. **CUDA Warp-Per-Partition Shared-Memory GPU Solver (`CudaPartitionSolver` in `src/CudaSolver.cu`):**
   - **Symbolic-Once + CSR $L_p + U_p$ Extraction (`Simulation::setup_gpu_solver`):** Uses SuiteSparse KLU's fill-reducing permutation $(P_p, Q_p)$ (`btf = 0, scale = 0`) to construct a static CSR $(L_p + U_p)$ non-zero pattern with symbolic fill-in closure for each partition, mapping each `RTYPE=1` JJ's variable conductance directly to its partition nonzero slot.
   - **Kernel Evolution (`v1` $\to$ `v3`):**
     - **`v1` (1 Thread/Partition in Global Memory):** Suffered from 32-way uncoalesced global memory striding across partitions and low SM occupancy (`~20s` runtime across 256–4,096 tiles).
     - **`v2` (1 Warp/Partition + Shared-Memory Working Vectors `sx, sb, sy`):** Parallelized device evaluations across all 32 lanes of a warp with `sx, sb, sy` in shared memory, cutting small-circuit GPU runtime from `10.37s` to `6.20s`.
     - **`v3` (1 Warp/Partition + Full Shared-Memory $L_p + U_p$ Factors + Register-Cached Devices):** Coalesce-loads each partition's entire sparse $L_p + U_p$ matrix (`s_luVal`, `s_luCi`, `s_luRp`, `s_luDiag`, `s_P`, `s_Q`) into on-chip shared memory (~13.7 KB/warp for 165-equation tiles, enabling up to 7 concurrent warps/SM with `cudaFuncAttributePreferredSharedMemoryCarveout = 100`) and caches per-lane `JJ`, `Resistor`, and `TxHalf` states in hardware registers across all $W$ steps of the window.

### 4.2 Numerical Equivalence Verification (`exxact` Workstation)
Verified via [`scripts/benchmark_gpu_exxact.py`](../scripts/benchmark_gpu_exxact.py) across Phase mode (`-a 1`), Voltage mode (`-a 0`), `RTYPE=1` piecewise-linear switching junctions, `RTYPE=0` non-switching junctions, mutual inductances (`K`), fractional-delay interpolated PTLs (`.option txinterp`), and output decimation (`PRSTART`/`PRSTEP`):

| Verification Suite | `max \|Serial - OpenMP (-p)\|` | `max \|Serial - GPU (-g)\|` | Status |
| :--- | :---: | :---: | :---: |
| **Phase mode (`-a 1`) + `RTYPE=1` + Mutual `L` + Decimation** | `1.776e-15` | `3.664e-15` | **PASS** |
| **Phase mode (`-a 1`) + `RTYPE=0` + Fractional PTL (`INTERP=1`)** | `1.665e-15` | `3.664e-15` | **PASS** |
| **Voltage mode (`-a 0`) + `RTYPE=1` + Mutual `L` + Decimation** | `0.000e+00` | `5.107e-15` | **PASS** |

### 4.3 Scaling Benchmark Results on BNL `exxact` (`Dual AMD EPYC 7413 48C/96T + NVIDIA RTX A6000`)
Benchmarked on PTL-coupled SFQ processor tile arrays ($W = 100$ step PTL lookahead window, 32 JJs and 165 MNA equations per tile, `RTYPE=1` dynamic switching with in-place LU refactorization):

#### A. Transient Solver Runtime (`Simulation::trans_sim`)

| Design | JJs | MNA Eqns | Steps | Serial (1T) | OpenMP (8T) | OpenMP (32T) | GPU (RTX A6000) | OpenMP Speedup | GPU Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **PTL-Tile-64** | 2,048 | 10,560 | 40,000 | 7.16 s | 2.10 s | **0.61 s** | 5.69 s | **11.78x** | **1.26x** |
| **PTL-Tile-256** | 8,192 | 42,240 | 40,000 | 29.88 s | 7.89 s | **2.06 s** | 6.00 s | **14.54x** | **4.98x** |
| **PTL-Tile-1024** | 32,768 | 168,960 | 20,000 | 104.90 s | 15.54 s | **4.20 s** | 6.95 s | **24.95x** | **15.10x** |
| **PTL-Tile-4096** | 131,072 | 675,840 | 10,000 | 232.89 s | 33.55 s | **8.58 s** | 10.85 s | **27.15x** | **21.47x** |

#### B. Total End-to-End Wall-Clock Time (Parse + Matrix + Factorize + Sim + Binary Output)

| Design | JJs | MNA Eqns | Steps | Serial (1T) | OpenMP (8T) | OpenMP (32T) | GPU (RTX A6000) | OpenMP Speedup | GPU Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **PTL-Tile-64** | 2,048 | 10,560 | 40,000 | 7.21 s | 2.16 s | **0.66 s** | 6.29 s | **10.89x** | **1.15x** |
| **PTL-Tile-256** | 8,192 | 42,240 | 40,000 | 30.08 s | 8.11 s | **2.28 s** | 6.78 s | **13.19x** | **4.44x** |
| **PTL-Tile-1024** | 32,768 | 168,960 | 20,000 | 105.78 s | 16.53 s | **5.32 s** | 8.46 s | **19.88x** | **12.51x** |
| **PTL-Tile-4096** | 131,072 | 675,840 | 10,000 | 236.79 s | 37.92 s | **12.90 s** | 15.62 s | **18.35x** | **15.16x** |

#### Key Hardware & Roofline Takeaways
1. **Up to `27.15x` OpenMP CPU Speedup & `21.47x` CUDA GPU Speedup:**
   - The Tier 1 Multi-Step Windowed PTL Solver reduces the 131,072-JJ (`675,840`-equation) simulation runtime from **232.89 s** down to **8.58 s** on 32 EPYC CPU threads (**`27.15x` speedup**) and **10.85 s** on a single RTX A6000 GPU (**`21.47x` speedup**, and **`3.09x` faster than 8-thread OpenMP**).
2. **Workstation GA102 (`sm_86`) vs. Data-Center GA100/GH100 (`sm_80`/`sm_90`) FP64 Throughput:**
   - The NVIDIA RTX A6000 uses the workstation **GA102** die, which has a **1:64 FP64-to-FP32 ratio** (only **2 FP64 cores per SM × 84 SMs = 168 FP64 cores**, ~0.61 TFLOPS peak FP64, and microcoded ~60-cycle double-precision divisions during backward substitution).
   - Data-center GPUs (**NVIDIA A100 / H100 / B200**) feature full **1:2 FP64 Tensor/FP64 pipelines** (16x–55x higher FP64 throughput per SM than GA102), where the exact same shared-memory `advance_partitions_window_kernel` will scale proportionately higher on $10^5–10^6$ JJ netlists.

---

## 5. Benchmarks on Non-PTL & Diverse Superconducting Circuit Topologies

To evaluate performance on circuits that **do not rely solely on PTLs**—including circuits with **zero transmission lines (`0 PTLs`)**, dense 3D field-extracted mutual inductance meshes (`K`), explicit shunt/parasitic capacitors (`C`), and both Phase (`-a 1`) and Voltage (`-a 0`) simulation modes—[`scripts/benchmark_non_ptl_exxact.py`](../scripts/benchmark_non_ptl_exxact.py) benchmarks 11 representative circuits across three topological categories on BNL `exxact`:

1. **Category 1 — Purely Monolithic Tightly-Coupled Non-PTL Circuits (`0 PTLs`, $P = 1$):**
   - `Mono-2D-JJ-Grid`: $128 \times 128$ 2D Josephson junction array (`16,384` JJs, `32,512` inductors, `8,192` capacitors, `106,242` MNA equations, $P = 1$).
   - `Mono-RSFQ-Logic`: Monolithic RSFQ splitter/merger/DRO pipeline wired purely with inductors and mutual couplings (`8,192` JJs, `8,192` mutual couplings `K`, `4,096` capacitors `C`, `47,108` MNA equations, $P = 1$).
   - `Mono-RLK-49kK-Phase` & `Mono-RLK-49kK-Volt`: Monolithic 3D-extracted RLK mesh with `8,192` JJs, `16,384` inductors, **`49,152` mutual inductance entries**, and `8,192` capacitors (`65,540` MNA equations, $P = 1$) in both Phase mode and Voltage mode.
   - For $P = 1$ circuits, JoSIM uses a **fused single-region OpenMP `setup_b`** (`#pragma omp parallel` across `b_` zeroing, JJs, resistors, capacitors, and two-pass inductors + mutual inductances) and a **multi-block CUDA device evaluator (`monolithic_eval_devices_kernel`)** paired with KLU sparse solves.
2. **Category 2 — Multi-Channel Non-PTL Superconducting Circuits (`0 PTLs`, $P \ge 2$):**
   - Multi-channel SQUID / detector readout arrays and parallel bit-slices (`NoPTL-SQUID-256`, `NoPTL-SQUID-256-Volt`, `NoPTL-SQUID-1024`, `NoPTL-SQUID-4096`) containing **zero PTLs (`0 PTLs`)**, up to **`131,072` JJs**, **`393,216` mutual inductance terms `K`**, **`131,072` capacitors `C`**, and **`1,196,032` MNA equations**.
   - Because parallel readout channels or independent bit-slices only share the ground reference node (`0`, which is not an MNA row), JoSIM's connected-component partitioner (`setup_partitions`) automatically decomposes them into $P$ independent KLU partitions with a multi-step lookahead window of **$W = 1,024$ steps** even when **zero transmission lines** are present.
3. **Category 3 — Rich Multi-Gate RSFQ ALU Macros (`JJ + L + Mutual K + Cap C + R + PTL`):**
   - Multi-gate RSFQ ALU/logic macros (`RSFQ-ALU-Macros-256`, `RSFQ-ALU-Macros-1024`, `RSFQ-ALU-Macros-4096`) combining `32` JJs, `64` inductors, `96` mutual inductance couplings `K`, `32` capacitors `C`, and `66` resistors `R` per macro (`296` MNA equations/macro, up to **`1,212,412` MNA equations**).

### 5.1 Transient Solver Runtime (`Simulation::trans_sim`) & Numerical Equivalence

| Design | Category | MNA Eqns | Steps | Serial (1T) | OpenMP (`-p`) | GPU (`-g`) | OpenMP Speedup | GPU Speedup | Max Diff vs Serial |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Mono-2D-JJ-Grid (16k JJs, 0 PTL)** | Monolithic ($P=1$, 0 PTL) | 106,242 | 2,000 | 11.47 s | 10.79 s | 11.94 s | **1.06x** | 0.96x | `9.08e-17` |
| **Mono-RSFQ-Logic (8k JJs, 0 PTL)** | Monolithic ($P=1$, 0 PTL) | 47,108 | 2,000 | 1.42 s | 1.76 s | 3.30 s | 0.80x | 0.43x | `1.47e-16` |
| **Mono-RLK-49kK-Phase (0 PTL)** | Monolithic ($P=1$, 0 PTL) | 65,540 | 2,000 | 1.84 s | 2.17 s | 3.29 s | 0.85x | 0.56x | `1.10e-16` |
| **Mono-RLK-49kK-Volt (0 PTL)** | Monolithic ($P=1$, 0 PTL, `-a 0`) | 65,540 | 2,000 | 2.35 s | 2.02 s | 3.32 s | **1.16x** | 0.71x | `2.38e-18` |
| **NoPTL-SQUID-256 (8k JJs, 24k K)** | Multi-Channel (**0 PTL**, $P=256$) | 74,752 | 20,000 | 28.06 s | **1.76 s** | **9.21 s** | **15.90x** | **3.05x** | `2.46e-16` |
| **NoPTL-SQUID-256-Volt (24k K)** | Multi-Channel (**0 PTL**, $P=256$, `-a 0`) | 74,752 | 20,000 | 29.49 s | **1.72 s** | **9.21 s** | **17.15x** | **3.20x** | `1.90e-18` |
| **NoPTL-SQUID-1024 (32k JJs, 98k K)** | Multi-Channel (**0 PTL**, $P=1024$) | 299,008 | 10,000 | 75.82 s | **3.52 s** | **11.20 s** | **21.55x** | **6.77x** | `7.46e-17` |
| **NoPTL-SQUID-4096 (131k JJs, 393k K)** | Multi-Channel (**0 PTL**, $P=4096$) | 1,196,032 | 5,000 | 185.79 s | **7.44 s** | **18.04 s** | **24.98x** | **10.30x** | `5.24e-17` |
| **RSFQ-ALU-Macros-256 (24k K, 8k C)** | Rich RSFQ Macros ($P=256$) | 75,772 | 20,000 | 27.99 s | **1.77 s** | **9.33 s** | **15.80x** | **3.00x** | `2.12e-16` |
| **RSFQ-ALU-Macros-1024 (98k K, 32k C)** | Rich RSFQ Macros ($P=1024$) | 303,100 | 10,000 | 72.81 s | **3.54 s** | **11.39 s** | **20.54x** | **6.39x** | `1.03e-16` |
| **RSFQ-ALU-Macros-4096 (393k K, 131k C)** | Rich RSFQ Macros ($P=4096$) | 1,212,412 | 5,000 | 157.08 s | **7.50 s** | **18.29 s** | **20.93x** | **8.59x** | `6.96e-17` |

### 5.2 Total End-to-End Wall-Clock Time (Parse + Matrix + Factorize + Sim + Binary Output)

| Design | MNA Eqns | Steps | Serial (1T) | OpenMP (`-p`) | GPU (`-g`) | OpenMP Speedup | GPU Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Mono-2D-JJ-Grid (16k JJs, 0 PTL)** | 106,242 | 2,000 | 12.16 s | 11.65 s | 13.22 s | **1.04x** | 0.92x |
| **Mono-RSFQ-Logic (8k JJs, 0 PTL)** | 47,108 | 2,000 | 1.61 s | 1.99 s | 4.05 s | 0.81x | 0.40x |
| **Mono-RLK-49kK-Phase (0 PTL)** | 65,540 | 2,000 | 2.17 s | 2.54 s | 4.17 s | 0.85x | 0.52x |
| **Mono-RLK-49kK-Volt (0 PTL)** | 65,540 | 2,000 | 2.69 s | 2.39 s | 4.23 s | **1.12x** | 0.64x |
| **NoPTL-SQUID-256 (8k JJs, 24k K)** | 74,752 | 20,000 | 28.44 s | **2.17 s** | **10.16 s** | **13.12x** | **2.80x** |
| **NoPTL-SQUID-256-Volt (24k K)** | 74,752 | 20,000 | 29.87 s | **2.12 s** | **10.19 s** | **14.11x** | **2.93x** |
| **NoPTL-SQUID-1024 (32k JJs, 98k K)** | 299,008 | 10,000 | 77.42 s | **5.22 s** | **13.48 s** | **14.84x** | **5.74x** |
| **NoPTL-SQUID-4096 (131k JJs, 393k K)** | 1,196,032 | 5,000 | 193.79 s | **14.61 s** | **25.97 s** | **13.27x** | **7.46x** |
| **RSFQ-ALU-Macros-256 (24k K, 8k C)** | 75,772 | 20,000 | 28.37 s | **2.16 s** | **10.29 s** | **13.13x** | **2.76x** |
| **RSFQ-ALU-Macros-1024 (98k K, 32k C)** | 303,100 | 10,000 | 74.43 s | **5.24 s** | **13.71 s** | **14.21x** | **5.43x** |
| **RSFQ-ALU-Macros-4096 (393k K, 131k C)** | 1,212,412 | 5,000 | 164.04 s | **14.80 s** | **26.33 s** | **11.08x** | **6.23x** |

### 5.3 Architectural Conclusions Across Topologies
1. **Multi-Channel Non-PTL Circuits (`0 PTLs`, $P \ge 2$) Achieve `15.9x–25.0x` OpenMP Speedup and `3.1x–10.3x` GPU Speedup:**
   - A circuit does **not** need PTLs to benefit from JoSIM's multi-step windowed solver: any multi-channel detector array, SQUID array, or parallel bit-slice netlist whose channels only share ground (`0`) is decomposed into $P$ independent KLU partitions with a `1,024`-step lookahead window, accelerating a **1.20-million-equation, 393k-mutual-inductance, 0-PTL circuit** from **185.79 s to 7.44 s (`24.98x`) on OpenMP** and **18.04 s (`10.30x`) on GPU** with `5.24e-17` numerical error.
2. **Rich Component Mixes (`JJ + L + Mutual K + Cap C + R`) Scale Up to 1.21M Equations:**
   - Adding 393,216 mutual couplings (`K`) and 131,072 explicit capacitors (`C`) increases each partition's matrix size from `165` equations (`PTL-Tile-*`) to `296` equations (`RSFQ-ALU-Macros-*`, $nnz(L_p + U_p) \approx 1,350$), achieving **`20.93x` OpenMP speedup** and **`8.59x` GPU speedup** at 1.21M equations.
3. **Purely Monolithic Single-Component Circuits ($P = 1$, `0 PTLs`) Confirm the Sandia Xyce / `CUSPICE` Roofline:**
   - When all nodes are galvanically/inductively coupled into a single component ($P = 1$) with zero PTLs, a single global triangular solve `klu_l_tsolve` must execute at every timestep. For $47\text{k}–65\text{k}$ equations in Phase mode, one serial timestep takes only `0.7–0.9 ms` inside CPU L3 cache, so thread synchronization or PCIe transfers exceed the device-evaluation savings. Once per-step device work grows via Voltage-mode mutual inductance sweeps (`Mono-RLK-49kK-Volt`, **`1.16x` OMP**) or $>100\text{k}$ equations (`Mono-2D-JJ-Grid`, **`1.06x` OMP**), parallel device evaluation reaches net speedup, while confirming why **partition-level parallelism ($P \ge 2$) or batched `.spread` execution (Tier 2)** is essential for order-of-magnitude speedups.

---

## 6. Tier 2 Implementation: Batched GPU (`-g`) & Multi-Core OpenMP (`-p`) Monte Carlo (`.spread`) Robustness Solver

### 6.1 Architectural Design (`Simulation::trans_sim_mc`, `advance_mc_coalesced_kernel`)
Superconducting IC design requires extensive Monte Carlo yield and critical-margin analysis (`.spread`), simulating a circuit across $B = 32\text{ to }4,096$ fabrication-variation trials where every inductor (`L`), Josephson junction (`B` area/$I_c$), resistor (`R`), and capacitor (`C`) is perturbed around its nominal value. Running JoSIM $B$ times sequentially from the CLI repeats netlist parsing, subcircuit flattening, CSR assembly, and symbolic factorization $B$ times, and leaves single-component or modest-partition circuits ($P = 1\text{–}66$) unable to saturate a multi-core CPU or GPU.

Tier 2 (`Simulation::trans_sim_mc` in [`src/Simulation.cpp`](../src/Simulation.cpp) and `advance_mc_coalesced_kernel` in [`src/CudaSolver.cu`](../src/CudaSolver.cu)) solves all $B$ Monte Carlo trials simultaneously in a single invocation:

1. **CLI & Netlist Controls:**
   - `-b <N>` / `--batch=<N>` CLI flag
   - `.option mc=<N>` / `.option batch=<N>` / `.option runs=<N>`
   - `.spread <global> [L=...] [B=...] [C=...] [R=...] [runs=<N>]`
   - `.option mcall` (outputs all individual trial waveforms `<Trace>_MC0 .. <Trace>_MC{B-1}` alongside ensemble statistics)
   - `JOSIM_SEED=<uint64>` environment variable or `.option seed=<uint64>` for deterministic reproducibility.
2. **Parse-Once, Symbolic-Once Setup:**
   - Each device stores its unperturbed nominal value (`nominalValue_`, `nominalArea_`, `nominalIc_`, `baseModel_`) and resolved spread fraction (`spread_`). We also fixed an upstream bug in [`src/JJ.cpp`](../src/JJ.cpp) where `AREA=` was parsed before `SPREAD=` and default `area=1.0` junctions ignored `.spread`.
   - The base circuit topology and connected-component partitions ($P_{\text{base}} \ge 1$) are analyzed **once**. Each trial $b \in \{0, \dots, B-1\}$ receives a deterministic 64-bit splitmix seed $\text{seed}_b = \text{baseSeed} \oplus (b \cdot \text{0x9E3779B97F4A7C15})$ via thread-safe `Spread::spread_value_rng`.
3. **Thread-Per-Trial Coalesced CUDA Kernel (`advance_mc_coalesced_kernel`):**
   - **Why Coalesced Batch-Minor Layout (`[baseIdx * B + b]`) Beats Warp-Per-Partition:**
     - Assigning 1 thread per `(trial b, base partition p)` and transposing all per-trial state vectors (`d_xLocal`, `d_bLocal`, `d_yLocal`), matrix values (`d_luVal`, `d_partNz`), and device arrays (`d_jjs`, `d_inds`, etc.) into **`[baseIdx * B + b]`** layout ensures that adjacent threads $t = 0 \dots 31$ in a warp execute the **exact same instruction** on consecutive memory addresses (`b = blockIdx.x * blockDim.x + tid`).
     - This achieves **100% coalesced global memory reads/writes**, eliminates all warp-shuffle reductions (`__shfl_down_sync`), and allows all 32 lanes of a warp to execute forward/backward triangular solves (`L\b`, `U\y`) and `RTYPE=1` in-place LU refactorizations simultaneously in lockstep (achieving an **`11.4x` faster GPU kernel** than warp-per-partition on `ex_ksa4bit`).
4. **Hardware-Adaptive Generalization Across All NVIDIA GPU Architectures:**
   - **Why 32 is Fundamental (`warpSize` & Shared-Memory Banks) while Block Sizing is Dynamic:**
     - Across every NVIDIA CUDA architecture ever shipped (Volta `sm_70`, Turing `sm_75`, Ampere `sm_80`/`sm_86`, Ada `sm_89`, Hopper `sm_90`, Blackwell `sm_100`), the hardware SIMT warp width (`warpSize`) and the number of shared-memory banks are both **32**. Striding shared-memory working vectors as `s_x[r * 32 + lane]` maps each lane to its own distinct bank (`0..31`), guaranteeing **zero shared-memory bank conflicts** on any NVIDIA GPU.
   - **Runtime Device Introspection (`cudaDeviceGetAttribute`):**
     - At initialization, `CudaPartitionSolver` queries `numSMs_` (`cudaDevAttrMultiProcessorCount`), `warpSize_` (`cudaDevAttrWarpSize`), `maxShmemBytes_` (`cudaDevAttrMaxSharedMemoryPerBlockOptin` / `cudaDevAttrMaxSharedMemoryPerBlock`), and `maxShmemPerSM_` (`cudaDevAttrMaxSharedMemoryPerMultiprocessor`).
   - **Dynamic Multi-Warp Block Sizing (`warpsPerBlock` $\in \{1, 2, 4\}$) with Shared Symbolic CSR:**
     - `step_window` automatically selects `warpsPerBlock` $\in \{1, 2, 4\}$ (`32, 64, or 128` threads/block) based on the active GPU's SM count, shared-memory capacity per block/SM, and total trial warps (`(B / 32) * P_base`).
     - When `warpsPerBlock > 1`, all warps in the thread block solve trials for the same base partition $p = \text{blockIdx.y}$ and **share a single copy** of the symbolic CSR structure (`s_luRp`, `s_luDiag`, `s_P`, `s_Q`, `s_luCi`, `s_luSrcNz`) in shared memory while each warp indexes its own bank-conflict-free vector slice (`s_warpBase = s_mem_mc + warpInBlock * 32 * 3 * n`).
   - **Multi-Architecture CMake Configuration:**
     - [`CMakeLists.txt`](../CMakeLists.txt) automatically defaults `CMAKE_CUDA_ARCHITECTURES` to `"native"` on CMake $\ge 3.24$ (auto-detecting the installed V100, T4, A100, RTX A6000, L40S, H100, or B200 GPU) or `"70;75;80;86"` on older CMake when not overridden by the user.
5. **Numerically Stable Welford Ensemble Statistics & Robustness Summary (`Output::write_output`):**
   - Uses **Welford's online algorithm** across all $B$ trials at each output timestep to compute `<Trace>` (ensemble mean), `<Trace>_STD` (sample standard deviation), `<Trace>_MIN`, and `<Trace>_MAX` without catastrophic cancellation on constant $2\pi$ superconducting phase plateaus, plus a formatted **Monte Carlo Robustness Summary** table on stdout.

### 6.2 Tier 2 Batched Monte Carlo Benchmark Results on BNL `exxact` (`Dual AMD EPYC 7413 + NVIDIA RTX A6000`)

Verified via [`scripts/benchmark_mc_exxact.py`](../scripts/benchmark_mc_exxact.py) and [`scripts/run_mc_exxact.sh`](../scripts/run_mc_exxact.sh) across **16 benchmark configurations** spanning:
- Real production MIT-LL standard cells & multi-gate RSFQ logic (`ex_jtl_basic`, `ex_mitll_xor_wr`, `ex_mitll_dff_wr`, and `ex_ksa4bit` 4-bit Kogge-Stone Adder with 695 JJs, $P=66$ partitions)
- Purely monolithic 0-PTL circuits ($P=1$: `Mono-RSFQ-Cell-Macro` 128-JJ coupled logic macro and `Mono-RLK-SQUID-Macro` 96-JJ SQUID sensor array with 191 mutual inductances `K`)
- PTL-coupled RSFQ ALU processor pipelines (`RSFQ-Pipeline-16B` $P=16$ and `RSFQ-Pipeline-64B` $P=64$, 2,048 JJs, 18,940 equations/trial $\times B=256$ = **`4,848,640` simultaneous MNA equations**)
- Batch sizes $B \in \{64, 256, 1024\}$.

| Design (`.spread` Yield Analysis) | Base $P$ | Batch $B$ | Est. $B \times$ Serial 1T | Batched OpenMP (`-p`) | Batched GPU (`-g`) | OpenMP Speedup | GPU Trans Speedup | GPU Wall Speedup | CPU vs GPU Rel Diff |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **`ex_jtl_basic` (Mono 0-PTL)** | **1** | 256 | 0.262 s | **0.058 s** | **0.230 s** | **4.52x** | **1.14x** | 0.33x | `1.26e-13` |
| **`ex_jtl_basic` (Mono 0-PTL)** | **1** | 1024 | 1.046 s | **0.207 s** | **0.351 s** | **5.04x** | **2.98x** | **1.26x** | `1.25e-13` |
| **`ex_mitll_xor_wr` (MIT-LL XOR)** | 2 | 256 | 2.012 s | **0.086 s** | **0.823 s** | **23.42x** | **2.44x** | **1.45x** | `1.07e-15` |
| **`ex_mitll_xor_wr` (MIT-LL XOR)** | 2 | 1024 | 8.025 s | **0.303 s** | **1.229 s** | **26.48x** | **6.53x** | **4.30x** | `1.17e-14` |
| **`ex_mitll_dff_wr` (MIT-LL DFF)** | 2 | 256 | 2.823 s | **0.105 s** | **1.058 s** | **26.86x** | **2.67x** | **1.74x** | `3.68e-15` |
| **`ex_mitll_dff_wr` (MIT-LL DFF)** | 2 | 1024 | 11.301 s | **0.401 s** | **1.560 s** | **28.16x** | **7.24x** | **4.96x** | `6.67e-15` |
| **`Mono-RSFQ-Cell-Macro` (128 JJs, 0-PTL)** | **1** | 256 | 4.092 s | **0.178 s** | 4.611 s | **22.93x** | 0.89x | 0.80x | `6.20e-13` |
| **`Mono-RSFQ-Cell-Macro` (128 JJs, 0-PTL)** | **1** | 1024 | 16.374 s | **0.673 s** | **4.972 s** | **24.32x** | **3.29x** | **2.93x** | `2.44e-13` |
| **`Mono-RLK-SQUID-Macro` (96 JJs, 191 K, 0-PTL)** | **1** | 256 | 4.842 s | **0.212 s** | 5.037 s | **22.79x** | 0.96x | 0.86x | `1.43e-13` |
| **`Mono-RLK-SQUID-Macro` (96 JJs, 191 K, 0-PTL)** | **1** | 1024 | 19.364 s | **0.796 s** | **5.507 s** | **24.34x** | **3.52x** | **3.16x** | `1.67e-13` |
| **`ex_ksa4bit` (4-bit KSA Adder, 695 JJs)** | 66 | 64 | 51.250 s | **2.044 s** | **4.309 s** | **25.07x** | **11.89x** | **10.13x** | `1.04e-15` |
| **`ex_ksa4bit` (4-bit KSA Adder, 695 JJs)** | 66 | 256 | 204.237 s | **6.559 s** | **10.394 s** | **31.14x** | **19.65x** | **18.33x** | `1.04e-15` |
| **`RSFQ-Pipeline-16B` (512 JJs, 16 Blocks)** | 16 | 64 | 3.036 s | **0.136 s** | **0.903 s** | **22.37x** | **3.36x** | **2.08x** | `1.19e-17` |
| **`RSFQ-Pipeline-16B` (512 JJs, 16 Blocks)** | 16 | 256 | 12.165 s | **0.431 s** | **1.226 s** | **28.22x** | **9.92x** | **6.50x** | `1.19e-17` |
| **`RSFQ-Pipeline-64B` (2,048 JJs, 64 Blocks)** | 64 | 64 | 12.178 s | **0.510 s** | **1.243 s** | **23.87x** | **9.80x** | **6.34x** | `1.45e-17` |
| **`RSFQ-Pipeline-64B` (2,048 JJs, 64 Blocks)** | 64 | 256 | 48.566 s | **1.682 s** | **3.192 s** | **28.87x** | **15.21x** | **12.55x** | `1.45e-17` |

### 6.3 Key Takeaways from Tier 2 Batched Monte Carlo
1. **Unlocks GPU & Multi-Core Speedups Even on Single-Partition Monolithic Circuits ($P = 1$) and Small Standard Cells:**
   - Whereas a single run ($B=1$) of a monolithic $P=1$ circuit or an MIT-LL standard cell (`ex_mitll_xor_wr`, `ex_mitll_dff_wr`) cannot saturate a GPU, batching $B = 1024$ Monte Carlo trials achieves **`24.3x–28.2x` OpenMP speedup** and **`3.3x–7.2x` CUDA GPU speedup** (`4.96x` total wall-clock GPU speedup on `ex_mitll_dff_wr`). Notice that on the GPU, increasing $B$ by $4\times$ (from `256` to `1024`) on `Mono-RSFQ-Cell-Macro` increases GPU transient time by only `7.8%` (`4.611s -> 4.972s`) as additional trial warps fill previously idle SMs!
2. **Up to `31.14x` OpenMP Speedup & `19.65x` CUDA GPU Speedup (`18.33x` Wall-Clock Speedup) on Real Production RSFQ Circuits (`ex_ksa4bit`):**
   - Simulating 256 Monte Carlo fabrication-yield trials of the 4-bit RSFQ Kogge-Stone Adder (695 JJs/trial) drops from **`204.24 s` sequentially** to **`6.56 s` (`31.14x` speedup) on OpenMP (`-p -b 256`)** and **`10.39 s` (`19.65x` transient speedup, `18.33x` end-to-end wall-clock speedup) on CUDA GPU (`-g -b 256`)**, with a relative difference of `1.04e-15`.
3. **4.85 Million Simultaneous Equations in 3.19 Seconds on GPU (`RSFQ-Pipeline-64B`, $B=256$):**
   - On the 64-partition RSFQ ALU pipeline across $B=256$ trials (`4,848,640` MNA equations), the coalesced GPU solver completes all timesteps in **`3.19 s` (`15.21x` GPU transient speedup, `12.55x` wall-clock speedup)** and **`1.68 s` (`28.87x` OpenMP speedup)** with `1.45e-17` relative error.

---

## 7. Tier 3 Implemented & Benchmarked: Fully GPU-Resident Monolithic Solver (`cuSolverRF`, `-g 3`) & Unified GPU Auto-Dispatcher (`-g`)

### 7.1 Motivation & Architectural Design (`Simulation::setup_gpu_monolithic_full`, `CudaPartitionSolver::step_monolithic_full_window`)

When a circuit contains **zero transmission lines ($P = 1$ single connected component)** and is simulated as a single run ($B = 1$), Tier 1's PTL block partitioning produces only a single global partition. If that monolithic circuit is a large 2D/3D mesh or Josephson junction array ($N \ge 50,000$ equations with 2D fill-in $\text{nnz}(L+U) \ge 10 N$), two bottlenecks limit hybrid CPU/GPU execution (`-g 2`):
1. **Per-Timestep PCIe Transfers:** Copying the solution vector $\mathbf{x} \in \mathbb{R}^N$ (`H2D`) and right-hand side $\mathbf{b} \in \mathbb{R}^N$ (`D2H`) across PCIe at every timestep incurs ~15–40 $\mu\text{s}$ of latency per step.
2. **Host KLU Triangular Solve Memory Bandwidth:** On 2D/3D meshes (`120x120` to `256x256` JJs), $L$ and $U$ contain `1,000,000` to `5,500,000+` FP64 nonzeros (`12–45 MB`), spilling out of CPU L2/L3 cache and making `klu_l_tsolve` memory-bandwidth bound on the host (`4.5 s` to `30.5 s` for 1,000 steps).

To eliminate both bottlenecks, **Tier 3 (`-g 3` / `--gpu=3`, `.option gpu=3`, `.option gpumono`, `.option cusolverrf`)** keeps the **entire transient simulation loop 100% resident in GPU VRAM** with **zero per-timestep PCIe transfers**:

1. **Exact $O(\text{nnz}(L+U))$ KLU Factor Extraction (`klu_l_extract`):**
   - During setup (`Simulation::setup_gpu_monolithic_full` in [`src/Simulation.cpp`](../src/Simulation.cpp)), KLU factors the unscaled monolithic CSR matrix ($A$, interpreted by KLU as CSC of $A^T$) with `Common_.btf = 0; Common_.scale = 0`:
     $$P_{\text{klu}} A^T Q_{\text{klu}} = L_{\text{klu}} U_{\text{klu}} \implies Q_{\text{klu}}^T A P_{\text{klu}}^T = U_{\text{klu}}^T L_{\text{klu}}^T = \underbrace{(U_{\text{klu}}^T D^{-1})}_{\tilde{L}} \underbrace{(D L_{\text{klu}}^T)}_{\tilde{U}}$$
   - Using `klu_l_extract`, JoSIM extracts the exact symbolic and numeric CSR factors $\tilde{L}$ (lower-triangular with stored unit diagonal `CUSOLVERRF_UNIT_DIAGONAL_STORED_L`), $\tilde{U}$ (upper-triangular with non-unit diagonal $D = \text{diag}(U_{\text{klu}})$), row permutation $P_{\text{csr}}[r] = Q_{\text{klu}}[r]$, and column permutation $Q_{\text{csr}}[c] = P_{\text{num}}[c]$ in milliseconds.
   - JoSIM passes $(\tilde{L}, \tilde{U}, P_{\text{csr}}, Q_{\text{csr}}, A_{\text{csr}})$ to NVIDIA **`cuSolverRF`** (`cusolverRfSetupHost` $\to$ `cusolverRfAnalyze` $\to$ `cusolverRfResetValues` $\to$ `cusolverRfRefactor`), which constructs the level-scheduled DAG for parallel GPU LU refactorization and sparse triangular solve (`cusolverRfSolve`).
2. **Fused All-Device GPU RHS Kernel (`monolithic_full_setup_b_kernel` in [`src/CudaSolver.cu`](../src/CudaSolver.cu)):**
   - Evaluates **all** circuit devices in parallel directly in VRAM at each timestep: Josephson junctions (`JJ` RCSJ + sin/cos CPR + temperature-dependent $I_c(T)$ + `RTYPE=1` piecewise-linear conductance state machine), resistors (`R`), capacitors (`C`), inductors (`L` + arbitrary mutual inductance couplings `K` via double-buffered `d_indIn2Old_`/`d_indIn2New_`), voltage sources (`V`), phase sources (`P`), current sources (`I` via `atomicAdd`), controlled sources (`CCVS`/`VCCS`), and transmission line port halves (`T`).
   - Whenever any `RTYPE=1` junction switches conductance state, the kernel updates `d_csrValA[jj.nzIdx]` **in-place in VRAM** and increments `d_numChangedNz_`.
   - If `hasSwitchingJJs_` is true and `h_numChanged > 0`, `step_monolithic_full_window` calls `cusolverRfResetValues` + `cusolverRfRefactor` directly on the GPU in VRAM; when `!hasSwitchingJJs_` (e.g., `RTYPE=0` or passive circuits), zero host-device synchronization occurs across the entire window $W = 1,024$ steps.
3. **In-VRAM Triangular Solve & Decimated Trace Recording (`cusolverRfSolve` + `monolithic_full_post_solve_kernel`):**
   - `cusolverRfSolve` solves $A \mathbf{x}_i = \mathbf{b}_i$ in-place in VRAM (`std::swap(d_xLocal_, d_bLocal_)`), and `monolithic_full_post_solve_kernel` records only kept/decimated output trace slots into `d_traceOut_`.

### 7.2 Complementary GPU Modes & Automatic Topology Dispatcher (`-g` / `-g 1`, `-g 2`, `-g 3`)

| CLI / Netlist Option | Solver Mode | Target Circuit Topology & Characteristics |
| :--- | :--- | :--- |
| **`-g` or `-g 1`** (`.option gpu`) | **Unified Auto-Dispatcher** (Default) | Automatically inspects partition count $P$, equation count $N$, and LU fill-in density $\text{nnz}(L+U)/N$:<br>• **$P \ge 2$ partitions:** Dispatches to **Tier 1 Windowed Partitioned GPU Solver**.<br>• **$P = 1$ monolithic mesh ($\text{nnz}(L+U) \ge 10 N$ and $\text{nnz}(L+U) \ge 200\text{k}$):** Dispatches to **Tier 3 Fully GPU-Resident `cuSolverRF` Monolithic Solver (`-g 3`)**.<br>• **$P = 1$ monolithic 1D chain/tree ($\text{nnz}(L+U) < 10 N$):** Dispatches to **Hybrid Device-GPU + Host-KLU Monolithic Solver (`-g 2`)**. |
| **`-g 2`** (`--gpu=2`, `.option gpu=2`) | **Hybrid Device-GPU + Host-KLU Monolithic Solver** | Best for **1D chains, trees, and low-fill-in monolithic circuits** ($\text{nnz}(L+U)/N \approx 4.5\text{–}8$, e.g., long 1D RSFQ pipelines or RLK ladders) where $L$ and $U$ fit in CPU L2/L3 cache (`klu_l_tsolve` takes `<0.4 ms/step`) while nonlinear JJ `sin/cos` and mutual inductance $K$ evaluations run in parallel on the GPU. |
| **`-g 3`** (`--gpu=3`, `.option gpu=3`, `.option gpumono`, `.option cusolverrf`) | **Tier 3 Fully GPU-Resident `cuSolverRF` Monolithic Solver** | Best for **large 2D/3D Josephson junction grids, arrays, and high-fill-in monolithic meshes** ($\text{nnz}(L+U)/N \ge 10$, $\text{nnz}(L+U) \ge 200\text{k}$), where CPU KLU triangular solve is memory-bandwidth bound on millions of FP64 nonzeros while `cuSolverRF` exploits wide 2D/3D level-set parallelism and 768 GB/s GDDR6 bandwidth with zero per-step PCIe traffic. |

### 7.3 Tier 3 Numerical Verification & Benchmark Results on BNL `exxact` (`2x Xeon Gold 6338 / EPYC + NVIDIA RTX A6000`)

Reproducible via [`scripts/benchmark_tier3_exxact.py`](../scripts/benchmark_tier3_exxact.py):

#### Phase 1: Numerical Accuracy Verification (Serial `1T` vs. Tier 3 `cuSolverRF` `-g 3`)
All 8 verification circuits across Phase mode (`-a 1`) and Voltage mode (`-a 0`), including `RTYPE=1` switching junctions, mutual inductances (`K`), passive $RLC$ meshes, and transmission lines (`T`), match the serial CPU KLU solver to machine precision (**`0.000e+00` to `1.320e-17` max absolute difference**):

| Verification Circuit | Mode | MNA Eqns ($N$) | $\text{nnz}(L+U)$ | Window $W$ | Max $\|\text{Serial} - \text{GPU Tier 3}\|$ |
| :--- | :---: | ---: | ---: | ---: | :---: |
| `Verify-2D-Grid-Phase` ($30\times30$ JJs, `RTYPE=1`) | Phase (`-a 1`) | 5,402 | 41,648 | 399 | **`1.320e-17`** |
| `Verify-2D-Grid-Volt` ($30\times30$ JJs, `RTYPE=1`) | Voltage (`-a 0`) | 5,402 | 40,748 | 399 | **`2.000e-19`** |
| `Verify-RSFQ-K-Phase` (800 JJs, 200 `K`, `RTYPE=1`) | Phase (`-a 1`) | 4,404 | 22,000 | 399 | **`1.000e-16`** |
| `Verify-RSFQ-K-Volt` (800 JJs, 200 `K`, `RTYPE=1`) | Voltage (`-a 0`) | 4,404 | 20,606 | 399 | **`1.000e-17`** |
| `Verify-RLK-Mesh-Phase` (300 `L`, 1,350 `K`, 60 JJs) | Phase (`-a 1`) | 1,623 | 10,442 | 399 | **`0.000e+00`** |
| `Verify-RLK-Mesh-Volt` (300 `L`, 1,350 `K`, 60 JJs) | Voltage (`-a 0`) | 1,623 | 10,382 | 399 | **`0.000e+00`** |
| `Verify-PTL-JTL-Phase` (16 PTL blocks, 240 JJs) | Phase (`-a 1`) | 1,230 | 4,888 | 20 | **`1.000e-17`** |
| `Verify-PTL-JTL-Volt` (16 PTL blocks, 240 JJs) | Voltage (`-a 0`) | 1,230 | 4,648 | 20 | **`1.000e-18`** |

#### Phase 2: Transient Solver Performance Comparison (`1,000` Timesteps)

| Circuit | Topology & Scale | Serial `1T` | OpenMP `-p` | GPU Hybrid `-g 2` | GPU `cuSolverRF` `-g 3` | GPU Auto `-g` | `-g 2` Speedup | `-g 3` Speedup | `-g` Auto Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`Mono-2D-Grid-14kJJ-Phase`** | $120\times120$ 2D Grid (`14,400` JJs, 0 PTL, $P=1$, `-a 1`) | `4.487 s` | `3.898 s` | `3.501 s` | **`1.164 s`** | **`1.148 s`** | 1.28x | **3.86x** | **3.91x** |
| **`Mono-2D-Grid-14kJJ-Volt`** | $120\times120$ 2D Grid (`14,400` JJs, 0 PTL, $P=1$, `-a 0`) | `4.810 s` | `3.992 s` | `3.258 s` | **`1.158 s`** | **`1.144 s`** | 1.48x | **4.16x** | **4.21x** |
| **`Mono-2D-Grid-25kJJ-Phase`** | $160\times160$ 2D Grid (`25,600` JJs, 0 PTL, $P=1$, `-a 1`) | `8.867 s` | `8.673 s` | `7.750 s` | **`2.407 s`** | **`2.395 s`** | 1.14x | **3.68x** | **3.70x** |
| **`Mono-2D-Grid-25kJJ-NR`** | $160\times160$ 2D Grid (`25,600` JJs, 0 PTL, `RTYPE=0`) | `8.804 s` | `8.624 s` | `8.147 s` | **`2.379 s`** | **`2.373 s`** | 1.08x | **3.70x** | **3.71x** |
| **`Mono-2D-Grid-64kJJ-Phase`** | $256\times256$ 2D Grid (`65,536` JJs, 0 PTL, $P=1$, `-a 1`) | `29.138 s` | `35.176 s` | `24.019 s` | **`11.930 s`** | **`11.904 s`** | 1.21x | **2.44x** | **2.45x** |
| **`Mono-2D-Grid-64kJJ-NR`** | $256\times256$ 2D Grid (`65,536` JJs, 0 PTL, `RTYPE=0`) | `30.458 s` | `28.171 s` | `25.501 s` | **`11.865 s`** | **`11.873 s`** | 1.19x | **2.57x** | **2.57x** |
| **`Mono-RSFQ-12kJJ-Phase`** | 1D Monolithic RSFQ Chain (`12,000` JJs, 3k `K`, 0 PTL) | `1.192 s` | `1.152 s` | `0.740 s` | *1D chain* | **`0.673 s`** | 1.61x | — | **1.77x** |
| **`Mono-RLK-49kK-Phase`** | 1D Coupled RLK Mesh (`3,500` `L`, `49,000` `K`, `-a 1`) | `0.323 s` | `0.379 s` | `0.338 s` | *1D chain* | **`0.238 s`** | 0.96x | — | **1.36x** |
| **`Mono-RLK-49kK-Volt`** | 1D Coupled RLK Mesh (`3,500` `L`, `49,000` `K`, `-a 0`) | `0.409 s` | `0.429 s` | `0.333 s` | *1D chain* | **`0.267 s`** | 1.23x | — | **1.54x** |
| **`PTL-256x30JTL-Phase`** | 256-Block PTL Network (`7,680` JJs, $P=256$) | `0.667 s` | `0.616 s` | `0.487 s` | *Partitioned* | **`0.115 s`** | 1.37x | — | **5.82x** |
| **`PTL-512x30JTL-Phase`** | 512-Block PTL Network (`15,360` JJs, $P=512$) | `2.180 s` | `0.681 s` | `0.922 s` | *Partitioned* | **`0.150 s`** | 2.37x | — | **14.49x** |
| **`PTL-800x30JTL-Phase`** | 800-Block PTL Network (`24,000` JJs, $P=800$) | `4.170 s` | `0.664 s` | `1.460 s` | *Partitioned* | **`0.300 s`** | 2.86x | — | **13.89x** |

### 7.4 Key Takeaways from Tier 3 (`cuSolverRF` Monolithic Solver & Auto-Dispatcher)
1. **`2.44x–4.21x` GPU Speedup on Purely Monolithic 0-PTL 2D Josephson Junction Arrays (`14k` to `65k` JJs):**
   - Whereas OpenMP (`-p`, `1.02x–1.20x`) and Hybrid Device-Only GPU (`-g 2`, `1.08x–1.48x`) are bottlenecked by host KLU triangular solves on 2D meshes (`N = 86,402` to `393,730`, $\text{nnz}(L+U) = 1.0\text{M}$ to $5.5\text{M}$), keeping the entire simulation loop in VRAM via Tier 3 `cuSolverRF` (`-g 3`) accelerates transient simulation by **`3.86x–4.21x` on $120\times120$ grids**, **`3.68x–3.71x` on $160\times160$ grids (`8.87s -> 2.39s`)**, and **`2.45x–2.57x` on $256\times256$ grids (`30.46s -> 11.87s`)**—roughly **`2.1x–3.4x` faster than Hybrid `-g 2`**.
2. **Seamless Unified `-g` Auto-Dispatcher Across All Superconducting Circuit Topologies:**
   - Simply passing `-g` (or `.option gpu`) now delivers strong GPU speedups across the entire spectrum of superconducting circuits:
     - **Multi-block PTL circuits ($P \ge 2$):** Dispatches to Tier 1 (`5.82x–21.47x` speedup).
     - **Monolithic 2D/3D arrays ($P = 1$, high fill-in):** Dispatches to Tier 3 `cuSolverRF` (`2.45x–4.21x` speedup).
     - **Monolithic 1D chains & mutual-inductance macros ($P = 1$, low fill-in):** Dispatches to Hybrid `-g 2` (`1.36x–1.77x` speedup).
     - **Batched Monte Carlo robustness studies (`-b N -g`):** Dispatches to Tier 2 Coalesced Batch Solver (`up to 19.65x` GPU speedup).

---

## 8. Tier 4 Implemented & Benchmarked: Exact Schur-Complement Nodal Admittance Solver (`-n` CPU & `-g 4` GPU) and Quiescent Partition Bypass (`.option bypass`)

### 8.1 Motivation: Why Standard MNA Inflates Superconducting Circuit Matrices by $2\times–6\times$
In JoSIM's standard Modified Nodal Analysis (MNA) formulation:
- Every Josephson junction (`JJ`) adds **two auxiliary rows**: `variableIndex_` ($v_J$ or $\phi_J$, with $-1$ at the node columns and $+1$ on the diagonal) and `currentIndex_` ($I_J$, with $g_J$ on `variableIndex_` and $-\tilde{R}_{\text{sub}}$ on the diagonal).
- Every resistor (`R`), capacitor (`C`), uncoupled inductor (`L`), and transmission-line port half (`T`) adds an explicit **branch-current row** (`currentIndex_` with $-1$ at the node columns and diagonal impedance $-\tilde{Z}_d$).
- Every grounded Dirichlet voltage source (`V`) or phase source (`P`) adds both a constrained node row and a branch-current row.

As a result, a circuit with $14,400$ nodes and $14,400$ JJs (`Mono-2D-Grid-14kJJ`) has **$N = 86,163$ MNA equations** even though only **$N_{\text{red}} = 14,401$ independent nodal equations** exist! Similarly, on real production superconducting IC designs (`spec2si-seeqc`: `xadd4`, `r5cal`, `r10f1n`, `coupler`, `adder`), auxiliary branch rows account for **48%–60% of the entire MNA matrix** and **35%–55% of all LU nonzeros $\text{nnz}(L+U)$**. Crucially, when a circuit has only $P = 1$ or $P = 2$ PTL partitions (`xadd4`, `r5cal`, `r10f1n`), partition-level OpenMP cannot parallelize the global triangular solve (`klu_l_tsolve`), making matrix size reduction the primary lever for both single-core CPU and GPU performance.

### 8.2 Exact Schur-Complement Nodal Admittance Formulation (`Simulation::build_nodal_system`)
Tier 4 (`Simulation::build_nodal_system`, `Simulation::setup_cpu_nodal`, `Simulation::setup_gpu_nodal`, `Simulation::trans_sim_nodal`, and `Simulation::trans_sim_gpu_nodal` in [`src/Simulation.cpp`](../src/Simulation.cpp); `nodal_full_setup_b_kernel` and `nodal_full_post_solve_kernel` in [`src/CudaSolver.cu`](../src/CudaSolver.cu)) analytically eliminates all diagonal internal branch equations prior to factorization:

1. **Two-Row Josephson Junction Elimination (`variableIndex_` and `currentIndex_`):**
   - From row $v = \text{variableIndex\_}$:
     $$x_v - (x_{n^+} - x_{n^-}) = b_v \implies x_v = (x_{n^+} - x_{n^-}) + b_v$$
   - From row $c = \text{currentIndex\_}$ with diagonal $a_{cc} = \text{curNzVal} < 0$ ($g_{\text{cur}} = -1/a_{cc} > 0$) and coupling $a_{cv} = g_{\text{scale}} \cdot g_{\text{jj}} \ge 0$:
     $$a_{cv} x_v + a_{cc} I_J = b_c \implies I_J = g_J (x_{n^+} - x_{n^-}) + g_{\text{cur}} (a_{cv} b_v - b_c), \quad g_J = g_{\text{scale}} g_{\text{cur}} g_{\text{jj}} \ge 0$$
   - Stamping $I_J$ into KCL at nodes $(n^+, n^-)$ contributes symmetric nodal conductance $+g_J$ on $(n^+, n^+)$ and $(n^-, n^-)$, $-g_J$ on $(n^+, n^-)$ and $(n^-, n^+)$, and RHS Norton current $i_{\text{eq}} = g_{\text{cur}}(b_c - a_{cv} b_v)$.
   - When an `RTYPE=1` junction switches conductance state ($\Delta g_J = g_{J,\text{new}} - g_{J,\text{old}}$), the 4 nodal CSR entries (`nzNodePP`, `nzNodePM`, `nzNodeMP`, `nzNodeMM`) are updated in-place in $A_{\text{red}}$ (`redNz_` on CPU or `d_csrValA` in GPU VRAM) and refactored via `klu_l_refactor` or `cusolverRfRefactor`.
2. **One-Row Passive & PTL Branch Elimination (`R`, `C`, Uncoupled `L`, and `T` Port Halves):**
   - Any branch row $c$ with $-1$ at $n^+$, $+1$ at $n^-$, and diagonal $a_{cc} = -\tilde{Z}_d < 0$ satisfies $I_{\text{br}} = g_d (x_{n^+} - x_{n^-}) - g_d b_c$ where $g_d = -1/a_{cc} > 0$, stamping $+g_d$ into $(n^+, n^-)$ of $A_{\text{red}}$ (once at setup) and adding $+g_d b_c$ into $\mathbf{b}_{\text{red}}$ at each step.
   - Inductors that participate in mutual inductance (`K`) or serve as the controlling branch of a `CCVS` retain their branch row in $A_{\text{red}}$ so mutual coupling matrices remain exact.
3. **Grounded Dirichlet Voltage & Phase Source Elimination (`V` and `P`):**
   - Grounded sources where the branch current is not needed by another device eliminate both the constrained node row and the branch-current row, substituting the known source waveform $v_s(t)$ directly into adjacent KCL equations (`nodalDirichletStamps_`).
4. **Exact On-the-Fly Reconstruction of `.print` / `.save` Outputs:**
   - Because $x_v$ and $I_{\text{br}}$ are affine functions of $(x_{n^+} - x_{n^-})$ and the current step's companion RHS $b_c$, any `.print` trace or device history (`ind.in2New`, `tx.hist_`) that references an eliminated variable is reconstructed exactly on the fly after the reduced triangular solve.

### 8.3 Activity-Driven Quiescent Partition Bypass (`.option bypass`) & NUMA-Aware OpenMP Scaling
1. **Quiescent Partition Bypass (`.option bypass`, `.option bypasstol=<val>`, `.option latency=<val>`):**
   - In `Simulation::trans_sim_omp_windowed`, each partition $p \in [0, P)$ tracks whether any internal device (`JJ`, `R`, `C`, `L` + mutual `K`, sources) or incoming PTL boundary wave (`txRemote1_`, `txRemote2_`) changed its companion RHS or state above `bypassTol_` (default `1e-12`) compared to the previous step (`partPrevB_`, `partPrevX_`). When a partition is completely quiescent and had zero step change on the previous step, `klu_l_tsolve` is bypassed for that partition at that timestep.
2. **NUMA-Aware OpenMP Thread Cap (`Simulation::trans_sim_omp_windowed`, `Simulation::setup_b`):**
   - On dual-socket 96-thread servers (`exxact`), spawning 96 threads across two NUMA sockets for small-window PTL circuits (such as `spec2si-seeqc` `adder` with $P = 583$ tiny partitions and lookahead window $W = 3$) previously incurred cross-socket barrier and cache-coherency overhead. Capping default partition OpenMP threads to 16 (single-socket L3 domain) and raising the per-window PTL precomputation threshold (`numTx * w >= 4096`) and monolithic `setup_b` parallel threshold (`32,768` devices) accelerates `adder -p` to **`3.61x` transient speedup (`23.36 s` vs `84.24 s`)**.

### 8.4 Tier 4 CLI & Netlist Options
- **CPU Schur-Complement Nodal Solver:**
  - CLI: `-n` or `--nodal` (combine with `-p` for OpenMP device evaluation + nodal KLU solve: `-p -n`)
  - Netlist: `.option nodal`
- **GPU Tier 4 Schur-Complement Nodal `cuSolverRF` Solver:**
  - CLI: `-g 4` or `--gpu=4`
  - Netlist: `.option gpu=4` or `.option gpunodal`
- **Quiescent Partition Bypass:**
  - Netlist: `.option bypass`, `.option bypasstol=<val>` (or `.option latency=<val>`)

### 8.5 Tier 4 Numerical Verification & Benchmark Results on BNL `exxact`

#### A. Matrix Dimension Reduction & Machine-Precision Equivalence (`-n` CPU Nodal & `-g 4` GPU Nodal vs. Baseline MNA)

| Circuit | Full MNA Eqns ($N$) | Nodal Eqns ($N_{\text{red}}$) | Matrix Reduction | Reduced $\text{nnz}(L+U)$ | Rel Diff (`-n` vs MNA) | Rel Diff (`-g 4` vs MNA) |
| :--- | ---: | ---: | :---: | ---: | :---: | :---: |
| **`ex_jtl_basic` (`-a 1` Phase)** | 27 | **10** | **2.70x** | 43 | **`1.00e-17`** | **`1.00e-17`** |
| **`ex_jtl_basic` (`-a 0` Voltage)** | 27 | **10** | **2.70x** | 43 | **`1.00e-16`** | **`1.60e-18`** |
| **`ex_ksa4bit` (`-a 0` Voltage)** | 24,410 | **8,982** | **2.72x** | 44,328 | **`1.00e-15`** | **`1.00e-15`** |
| **`ex_ksa4bit` (`-a 1` Phase)** | 24,410 | **8,982** | **2.72x** | 44,328 | **`1.00e-14`** | **`1.00e-15`** |
| **`Mono-2D-Grid-14kJJ` ($120\times120$)** | 86,163 | **14,401** | **5.98x** | 642,620 | **`1.00e-17`** | **`1.97e-18`** |
| **`Mono-2D-Grid-25kJJ` ($160\times160$)** | 153,283 | **25,601** | **5.99x** | 1,306,896 | **`1.00e-17`** | **`2.10e-18`** |
| **`Mono-2D-Grid-64kJJ` ($256\times256$)** | 392,707 | **65,537** | **5.99x** | 3,942,792 | **`1.00e-17`** | **`2.40e-18`** |

#### B. 2D Monolithic Josephson Junction Grid Benchmarks (`1,001` Timesteps, $P = 1$, 0 PTLs, `RTYPE=1` Switching)

| Circuit ($N \to N_{\text{red}}$) | Serial MNA | OpenMP MNA (`-p`) | **CPU Nodal (`-n`)** | **OMP Nodal (`-p -n`)** | GPU Tier 3 (`-g 3`) | **GPU Tier 4 (`-g 4`)** | Best CPU Nodal Speedup | `-g 4` vs `-g 3` Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| **`Mono-2D-Grid-14kJJ`**<br>($86,163 \to 14,401$ eqns) | `4.636 s`<br>*(wall `5.19s`)* | `3.981 s`<br>*(1.16x)* | **`1.106 s`**<br>*(wall `1.58s`)* | **`1.060 s`**<br>*(wall `1.59s`)* | `3.470 s`<br>*(1.34x)* | **`3.089 s`**<br>*(1.50x)* | **`4.37x` tran**<br>**(`3.28x` wall)** | **`1.12x` faster** |
| **`Mono-2D-Grid-25kJJ`**<br>($153,283 \to 25,601$ eqns) | `8.882 s`<br>*(wall `9.91s`)* | `8.375 s`<br>*(1.06x)* | **`2.836 s`**<br>*(wall `3.70s`)* | **`2.214 s`**<br>*(wall `3.15s`)* | `4.406 s`<br>*(2.02x)* | **`3.820 s`**<br>*(2.32x)* | **`4.01x` tran**<br>**(`3.15x` wall)** | **`1.15x` faster** |
| **`Mono-2D-Grid-64kJJ`**<br>($392,707 \to 65,537$ eqns) | `29.037 s`<br>*(wall `32.12s`)* | `26.890 s`<br>*(1.08x)* | **`9.289 s`**<br>*(wall `11.64s`)* | **`8.646 s`**<br>*(wall `11.19s`)* | `34.091 s`<br>*(0.85x)* | **`17.223 s`**<br>*(1.69x)* | **`3.36x` tran**<br>**(`2.87x` wall)** | **`1.98x` faster** |

#### C. Production `spec2si-seeqc` Benchmarks (`-a 1` Phase Mode on BNL `exxact`)

Evaluated across all 6 circuits from the `spec2si-seeqc` benchmark suite (`xadd4`, `r5cal`, `r10f1n`, `coupler`, `dff`, `adder`):

| Circuit | JJs | Full MNA ($N$) | Nodal ($N_{\text{red}}$) | Partitions ($P$) | Serial MNA | **CPU Nodal (`-n`)** | **OpenMP (`-p`)** | **OMP Nodal (`-p -n`)** | **Single-Core `-n` Speedup** | **Best Multi-Core Speedup** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| **`r5cal`** | 7,178 | 64,400 | **33,578** (`1.92x`) | 1 ($W=1$) | `218.69 s`<br>*(wall `220.67s`)* | **`88.49 s`**<br>*(wall `90.31s`)* | `218.50 s` | **`88.49 s`**<br>*(wall `90.31s`)* | **`2.47x` tran**<br>**(`2.44x` wall)** | **`2.47x`** (`-n` / `-p -n`) |
| **`r10f1n`** | 10,000 | 89,806 | **46,994** (`1.91x`) | 2 ($W=1$) | `79.05 s`<br>*(wall `79.89s`)* | **`37.01 s`**<br>*(wall `37.81s`)* | `78.90 s` | **`37.01 s`**<br>*(wall `37.81s`)* | **`2.14x` tran**<br>**(`2.11x` wall)** | **`2.14x`** (`-n` / `-p -n`) |
| **`xadd4`** | 3,522 | 31,813 | **16,393** (`1.94x`) | 1 ($W=1$) | `25.44 s`<br>*(wall `29.01s`)* | **`15.95 s`**<br>*(wall `19.63s`)* | `25.43 s` | **`15.95 s`**<br>*(wall `19.63s`)* | **`1.59x` tran**<br>**(`1.48x` wall)** | **`1.59x`** (`-n` / `-p -n`) |
| **`coupler`** | 5,104 | 38,826 | **15,372** (`2.53x`) | 464 ($W=420$) | `35.44 s`<br>*(wall `38.12s`)* | **`18.92 s`**<br>*(wall `21.62s`)* | **`4.70 s`**<br>*(wall `7.36s`)* | **`4.70 s`**<br>*(wall `7.36s`)* | **`1.87x` tran** | **`7.53x` tran**<br>**(`5.18x` wall)** |
| **`adder`** | 5,451 | 47,425 | **20,450** (`2.32x`) | 583 ($W=3$) | `84.24 s`<br>*(wall `93.27s`)* | **`46.84 s`**<br>*(wall `55.68s`)* | **`23.36 s`**<br>*(wall `30.95s`)* | **`23.36 s`**<br>*(wall `30.95s`)* | **`1.80x` tran**<br>**(`1.68x` wall)** | **`3.61x` tran**<br>**(`3.01x` wall)** |
| **`dff`** | 56 | 504 | **230** (`2.19x`) | 1 ($W=1$) | `0.231 s`<br>*(wall `0.343s`)* | **`0.157 s`**<br>*(wall `0.260s`)* | `0.231 s` | **`0.157 s`**<br>*(wall `0.260s`)* | **`1.47x` tran**<br>**(`1.32x` wall)** | **`1.47x`** (`-n` / `-p -n`) |

### 8.6 Key Takeaways from Tier 4
1. **Solves the $P = 1$ Monolithic Bottleneck on Production Superconducting Netlists (`r5cal`, `r10f1n`, `xadd4`):**
   - On production circuits without PTL partitions (`r5cal`, `r10f1n`, `xadd4`), partition-level OpenMP (`-p`) has only $P = 1\text{–}2$ blocks and previously achieved `1.00x` speedup over Serial MNA. The Tier 4 Schur-Complement Nodal Admittance solver (`-n` / `-p -n`) shrinks the matrix by ~`2.0x` and accelerates **`r5cal` by `2.47x` (`218.69 s -> 88.49 s`)**, **`r10f1n` by `2.14x` (`79.05 s -> 37.01 s`)**, and **`xadd4` by `1.59x` (`25.44 s -> 15.95 s`)** on a single CPU core!
2. **`3.36x–4.37x` CPU Nodal Speedup & `1.98x` Faster GPU `cuSolverRF` on Large 2D JJ Grids (`14kJJ–64kJJ`):**
   - Eliminating the 5 branch rows per JJ cell on 2D grids shrinks `Mono-2D-Grid-64kJJ` from $392,707$ rows to $65,537$ nodal rows (`5.99x` smaller) and cuts $\text{nnz}(L+U)$ from $5.41\text{M}$ to $3.94\text{M}$, fitting the factors into L3 cache on CPU (**`3.36x–4.37x` speedup**) and cutting `cuSolverRF` refactorization + solve time on GPU nearly in half (**`1.98x` faster than Tier 3 `-g 3`** on `64kJJ` with dynamic `RTYPE=1` switching).
3. **`3.61x` OpenMP Speedup on Short-Window PTL Circuits (`adder`) & `7.53x` on `coupler`:**
   - With NUMA-aware thread capping (`16` threads) and tuned parallel thresholds, `adder` ($P = 583$, $W = 3$) accelerates from `84.24 s` to **`23.36 s` (`3.61x` speedup)** and `coupler` ($P = 464$, $W = 420$) accelerates from `35.44 s` to **`4.70 s` (`7.53x` speedup)**.

---

## 9. Phase 5 Enhancements (Items #1–#7): Series $R+L$ Elimination, Per-Partition Nodal PTL Solver (`-p -n`), Short-PTL Coarsening, Streaming Binary/Event I/O (`-e`, `-E`), Local Rollback (`.option rollback`), & Batched Sweeps (`.sweep`, `-S`)

Building on Tiers 1–4, **Phase 5** implements 7 synergistic algorithmic and architectural enhancements documented in [`ENHANCEMENT_PLAN.md`](../ENHANCEMENT_PLAN.md) and [`docs/usage_and_benchmarks.md`](usage_and_benchmarks.md):

| Rank | Phase 5 Enhancement | Architectural Summary & Measured Impact on BNL `exxact` |
| :---: | :--- | :--- |
| **1** | **Series $R + L$ Internal Node Elimination in Nodal Solver (`-n`)** | Eliminates degree-2 internal nodes between series $R$ and uncoupled $L$ damping/bias chains into composite super-edges (`RLSuperEdge`, $g_{\text{eff}} = 1 / (\tilde{Z}_R + \tilde{Z}_L)$), shrinking $A_{\text{red}}$ to **`8,565` eqns (`3.71x`)** on `xadd4`, **`9,870` (`3.93x`)** on `coupler`, **`9,195` (`5.16x`)** on `adder`, **`28,093` (`2.29x`)** on `r5cal`, and **`39,976` (`2.25x`)** on `r10f1n` (**2.3x–3.5x 1-thread speedup**). |
| **2** | **Per-Partition Nodal Reduction Inside Windowed PTL Solver (`-p -n`)** | Applies Nodal + series $R+L$ reduction inside each PTL-bounded partition (`NodalPartition`) and decouples shared Dirichlet `VS`/`PS` supply rails, multiplying `-p` and `-n` gains on `adder` (`47,425 -> 9,195` eqns) and `coupler` (`38,826 -> 9,870` eqns across 464 tiny blocks of $\le 30$ eqns). |
| **3** | **Short-PTL Coarsening & Sub-Step Lumped $\pi$-Replacement (`.option lumpptl`)** | Automatically replaces lossless transmission lines with $T_d < 0.5\Delta t$ (`1 fs` stubs in `r10forig`) with exact matched $L$-$C$ $\pi$-equivalents (**`r10forig`: `1,904 s` (`v2.6.5`) / `257 s` (`v2.7`) $\to$ `3.22 s` wall / `2.66 s` solve — `591.3x` faster than `v2.6.5`**) and coarsens 0-JJ PTL-via islands + $k < 4$ PTLs (`adder`: $P = 583 \to 126, W = 3 \to 5$). |
| **4** | **Direct In-Loop Trace Recording & Streaming `.npy`/`.bin` Output** | Drops unused JJ `variableIndex_` trace vectors, folds exclusive `V(n1, n2)` node-difference pairs (`traceType = 4`), parallelizes NUMA trace allocation, and streams `.npy`/`.bin`/`.rawb` via a 64×8 L1 cache-tiled transpose without allocating `Output::traces`: cuts `adder` peak RAM from **`26.5 GB` (`v2.6.5`) / `8.04 GB` to `1.72 GB` (`15.4x` lower)** and accelerates **`adder -p -n` to `7.20 s` wall (`75.3x` vs `v2.6.5`)** and **`coupler -p -n` to `4.08 s` wall (`38.2x` vs `v2.6.5`)**. |
| **5** | **Built-In SFQ Switching Event & Pulse Metric Extractor (`-e`, `-E`)** | Detects $(2k-1)\pi$ SFQ phase crossings with sub-step linear interpolation and optional pulse metrics (`-E`: `t_peak`, `v_peak`, `fwhm`, `dt_pi`, `area`, `time=peak`) in zero-allocation Events-Only Mode (`-e` without `-o`): cuts **`adder -p -n -e` to `4.63 s` wall / `195 MB` RSS (`605 KB` output — `5,250x` smaller, `117.1x` faster vs `v2.6.5`)** and **`coupler -p -n -e` to `2.58 s` wall / `216 MB` RSS (`60.5x` faster vs `v2.6.5`)**. |
| **6** | **In-Memory Rolling Checkpoints for Local Step-Halving Rollback (`.option rollback`)** | Maintains a double-buffered 128-step rolling checkpoint (`ckpt_[2]`, `<2 MB`) in both MNA and Nodal (`-n`), rolling back only $32\text{–}256$ base steps with in-place BDF2 history rescaling ($h_{\text{old}} \to h_{\text{new}}$) and $M = 2^k$ sub-stepping instead of restarting from $t = 0$: delivers **`2.02x–3.93x` speedup** across Phase & Voltage modes with `0.0000 rad` final phase error. |
| **7** | **Batched Deterministic Parameter Sweeps & Batched Nodal Solver (`.sweep`, `-S`, `-p -n`)** | Extends the batched engine (`Simulation::trans_sim_mc_nodal`, `Simulation::trans_sim_mc`, `CudaPartitionSolver`) with 1D/2D deterministic `.param`, `@BIAS`/`@JJ`/`@IND`/`@RES`/`@CAP`/`@SRC`/`@GLOBAL`, and wildcard sweeps plus Batched Nodal reduction (`-p -n`): achieves **`33.3x–51.8x` sweep speedup** and **up to `83.7x` Monte Carlo speedup** vs 64 sequential MNA runs (`1.46x–1.84x` faster than Batched MNA) with `0.000e+00`–`1.000e-14` parity across MNA, Nodal, and CUDA GPU and automatic SFQ operating margin detection. |


