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

### Tier 4: Explicit / Semi-Implicit Nodal Formulation on GPU
- **Target Workload:** Million-junction RCSJ meshes where Norton inductor companion stamping and node-diagonal capacitance/conductance reduction turn the timestep update into an explicit or Jacobi-preconditioned SPD Sparse Matrix-Vector multiply (`SpMV`) at $>1\text{ TB/s}$ GPU HBM bandwidth.

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
