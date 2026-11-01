# JoSIM–SPICE Superconducting / Semiconducting Co-Simulation Plan

## 1. Architectural Overview

To enable co-simulation of superconducting circuits (**JoSIM**: RSFQ, AQFP, dc-SQUIDs, SNSPDs, `nTron`/`hTron` cryotrons) and semiconducting circuits (**ngspice**, **Cadence Spectre / APS / Spectre X** on the BNL cluster, **LTspice**, and **Xyce**), JoSIM uses a **4-layer generic co-simulation architecture**:

1. **Layer 1 — Core Co-Simulation Stepping, Norton Jacobian & Auto-Rollback Engine (`libjosim_c`):**
   - Exposes `josim_cosim_add_port`, `josim_cosim_init`, `josim_cosim_step_to`, `josim_cosim_commit`, `josim_cosim_rollback`, `josim_cosim_run_window`, and `josim_cosim_finalize` in [`include/JoSIM/josim_c_api.h`](include/JoSIM/josim_c_api.h).
   - Computes the exact driving-point Norton conductance $G_{\text{eq}} = \partial I_p / \partial V_p$ (or Thevenin resistance $R_{\text{eq}} = \partial V_p / \partial I_p$) from KLU solves on JoSIM's MNA matrix $A$, enabling quadratic Newton–Raphson convergence inside SPICE.
   - Maintains a two-slot checkpoint ring (`ckpt_committed`, `ckpt_eval`) backed by [`Simulation::save_mna_checkpoint`](src/Simulation.cpp) and [`Simulation::restore_mna_checkpoint`](src/Simulation.cpp), automatically handling same-time NR iterations ($t_{\text{trial}} == t_{\text{eval}}$), accepted steps ($t_{\text{trial}} > t_{\text{eval}}$), and LTE step rejections/rollbacks ($t_{\text{trial}} < t_{\text{eval}}$).
2. **Layer 2 — Generic IPC Socket Daemon (`josim-cosim-server`) & Python Coordinator (`pyjosim.cosim`):**
   - Provides out-of-process UNIX domain socket (`AF_UNIX`) and TCP socket (`AF_INET`) transport so commercial simulators like Cadence Spectre on the BNL cluster can co-simulate with JoSIM without `libstdc++` / OpenMP / CUDA shared-library symbol conflicts.
   - Supports both lockstep Norton/Thevenin coupling and Branin transmission-line wave windowing ($Z_0, \tau_d$).
3. **Layer 3A — Cadence Spectre Verilog-A + DPI-C / VPI Adapter (`cosim/spectre/`):**
   - Reusable Verilog-A module (`josim_cosim_port.va`) + DPI-C shared library (`libjosim_spectre_dpi.so`) that stamps `I(p,n) <+ i_hist + g_eq * V(p,n);` at each Spectre time point and automatically triggers rollback when `$abstime` moves backward during Spectre LTE step rejection.
4. **Layer 3B — `ngspice` & `LTspice` Adapters (`pyjosim.cosim`):**
   - `ngspice` integration via `libngspice` (`ngSpice_Init_Sync` / `GetVSRCData` / `GetISRCData` callbacks) and CLI wave-window mode.
   - `LTspice` integration via batch (`-b`) execution, binary `.raw` waveform parsing, and Branin/Thevenin PWL waveform relaxation.

---

## 2. Four-Step Implementation & Benchmark Plan

| Step | Component | Status | Benchmark Suite |
| :---: | :--- | :---: | :--- |
| **Step 1** | **Layer 1: `libjosim_c` Co-Simulation Stepping, Norton Jacobian & Auto-Rollback Engine** | **Completed** | `B1_JTL_NORTON_SPLIT`, `B2_CMOS_TO_SFQ_ROLLBACK`, `B3_NTRON_TO_CMOS_LNA`, `B4_BRANIN_WAVE_DUPLEX` ([`test/test_cosim_step1.py`](test/test_cosim_step1.py)) |
| **Step 2** | **Layer 2: `josim-cosim-server` IPC Daemon, `josim_cosim.cm` XSPICE Code Model & `josim_cosim.py` Coordinator (Tested on `ngspice-45.2` & `LTspice.exe`)** | **Completed** | `NG1_SPLIT_JTL_XSPICE`, `NG2_CMOS_TO_SFQ_DFF`, `NG3_NTRON_TO_CMOS_LNA`, `LT1_CMOS_BRANIN_TRANSCEIVER`, `LT2_NTRON_TO_LTSPICE_LNA` ([`test/test_cosim_simulators_and_step2.py`](test/test_cosim_simulators_and_step2.py)) |
| **Step 3** | **Layer 3A: Cadence Spectre Verilog-A + `CDS_VA_DPI` / DPI-C Adapter (`cosim/spectre/`, Tested Live on Cadence Spectre 25.1 on `asic2.inst.bnl.gov`)** | **Completed** | `SP1_DPI_C_INPROC_AND_SOCKET`, `SP2A_SPLIT_JTL_SPECTRE`, `SP2B_NTRON_LNA_SPECTRE`, `CMP1_NGSPICE_OPENVAF_SCALING`, `CMP2_SPECTRE_100JJ_SCALING`, `CMP3_PHYSICS_AND_CLUSTER_AUDIT` ([`test/test_cosim_step3_and_veriloga_comparison.py`](test/test_cosim_step3_and_veriloga_comparison.py)) |
| **Step 4** | **Layer 3B: Multi-Window / Multi-Instance & End-to-End Verification** | Planned | Unified CLI & end-to-end cryo-hybrid system benchmarks |

---

## 3. Step 1 Benchmark Results (`test/test_cosim_step1.py`)

| Benchmark | Description | Key Verified Metrics |
| :--- | :--- | :--- |
| **`B1_JTL_NORTON_SPLIT`** | 4-JJ JTL split across its middle coupling inductor (`L2A` + `L2B`) into two `JoSIMSession` instances coupled at every step via `PORT_VOLTAGE_IN_CURRENT_OUT` (Norton $G_{\text{eq}}$) and `PORT_CURRENT_IN_VOLTAGE_OUT` (Thevenin $R_{\text{eq}}$) | Interface residual $7.27 \times 10^{-18}\text{ V}$, `V(nport)` RMS error $3.79 \times 10^{-14}$, phase error $\le 2.44 \times 10^{-13}\text{ rad}$, SFQ timing diff $0.00\text{ fs}$ |
| **`B2_CMOS_TO_SFQ_ROLLBACK`** | Nonlinear Cryo-CMOS FET buffer driving a JoSIM JTL + clocked RSFQ storage loop (`B_SET`, `L_STOR`, `B_RO`) via Newton–Raphson with exact KLU $G_{\text{eq}}$, stressed with 92 adaptive LTE rollbacks (explicit + implicit non-monotonic time) | 92 rollbacks (46 explicit LTE step rejections), max NR iterations = 4, $T_{\text{SET}} = 27.01\text{ ps}$, $T_{\text{RO}} = 47.55\text{ ps}$, rollback state drift = $0.00$ |
| **`B3_NTRON_TO_CMOS_LNA`** | SFQ-triggered 3-terminal electrothermal `nTron` (`N1`) driving a nonlinear capacitive/active Cryo-CMOS LNA in both Voltage (`-a 0`) and Phase (`-a 1`) modes with rollbacks across hotspot nucleation and recovery | $V(\text{nch})_{\text{peak}} = 4.41\text{ mV}$, LNA $V_{\text{out,peak}} = 40.49\text{ mV}$, rollback drift = $0.00$, Voltage vs. Phase RMS diff = $3.30 \times 10^{-14}$ |
| **`B4_BRANIN_WAVE_DUPLEX`** | Full-duplex windowed Branin wave relaxation (`PORT_WAVE_BRANIN` + `cosim_run_window`) across a $Z_0 = 50\,\Omega, \tau_d = 20\text{ ps}$ interconnect vs. monolithic `TLINE` reference | `V(ntx)` RMS error $5.86 \times 10^{-14}$, `V(nrx)` RMS error $5.03 \times 10^{-14}$, `P(BRX)` max error $7.53 \times 10^{-12}\text{ rad}$, 4-pulse burst timing diff $0.00\text{ fs}$ |

---

## 4. Step 2 & Real SPICE Simulator Benchmark Results (`test/test_cosim_simulators_and_step2.py`)

Tested against the actual SPICE simulators installed on the machine (**`ngspice-45.2`** with our compiled `josim_cosim.cm` XSPICE code model, and **ADI `LTspice.exe`** in batch `-b` double-precision `.raw` mode) across both **Step 1 (in-process `libjosim_c.so`)** and **Step 2 (`josim-cosim-server` over `AF_UNIX` and `AF_INET` `TCP_NODELAY` sockets)**:

| Benchmark | Simulator(s) & Transport Modes | Description | Key Verified Metrics |
| :--- | :--- | :--- | :--- |
| **`NG1_SPLIT_JTL_XSPICE`** | `ngspice-45.2` + `josim_cosim.cm` (Step 1 in-proc, Step 2 UNIX socket, Step 2 TCP socket) | 4-JJ JTL split across an intermediate LC filter simulated in `ngspice-45.2` (`A_tx` and `A_rx` `josim_cosim` instances) vs. monolithic JoSIM reference | `v(mid)` NRMSE vs. monolithic JoSIM = $0.338\%$, `B4` timing diff = $0.973\text{ fs}$, Step 1 vs. UNIX vs. TCP max diff = $0.0$ |
| **`NG2_CMOS_TO_SFQ_DFF`** | `ngspice-45.2` + `josim_cosim.cm` (Step 1 in-proc & Step 2 UNIX socket) | 4K Cryo-CMOS DATA & CLOCK MOSFET drivers (`Mp`, `Mn` inverter + `Mn_sf` source follower) driving a 2-port (`%gd` + `%hd`) JoSIM JTL + RSFQ storage loop (`B_SET`, `L_STOR`, `B_RO`) with adaptive timestepping (`tran 0.8p 65p`) | $T_{\text{SET}} = 31.91\text{ ps}$, $T_{\text{RO}} = 45.06\text{ ps}$, stored circulating current $I_{\text{hold}} = 51.4\,\mu\text{A}$, 6 automatic `ngspice` LTE rollbacks handled with zero duplicate events, Step 1 vs. Step 2 diff = $0.0$ |
| **`NG3_NTRON_TO_CMOS_LNA`** | `ngspice-45.2` + `josim_cosim.cm` (Step 1 in-proc & Step 2 TCP socket) | JoSIM SFQ-triggered 3-terminal `nTron` (`N1`) driving a 2-stage Cryo-CMOS Common-Source + Source-Follower LNA in `ngspice-45.2` | `nTron` drain pulse $V(\text{nd})_{\text{peak}} = 6.89\text{ mV}$, Stage-1 swing = $86.5\text{ mV}$, LNA output swing = $70.01\text{ mV}$ ($10.16\times$ voltage gain), Step 1 vs. TCP diff = $0.0$ |
| **`LT1_CMOS_BRANIN_TRANSCEIVER`** | ADI `LTspice.exe -b` $\leftrightarrow$ JoSIM (Step 1 in-proc & Step 2 TCP socket) | Full-duplex Branin wave relaxation (`PORT_WAVE_BRANIN` + `cosim_run_window`) over dual $50\,\Omega, 20\text{ ps}$ cryogenic coax lines between `LTspice.exe` Cryo-CMOS TX/RX and a JoSIM SFQ + `nTron` core | Quiet RX window before $2\tau_d = 40\text{ ps}$ ($1.11 \times 10^{-9}\text{ mV}$), amplified return wave after $44\text{ ps}$ = $56.88\text{ mV}$, Step 1 vs. TCP diff = $0.0$ |
| **`LT2_NTRON_TO_LTSPICE_LNA`** | ADI `LTspice.exe -b` vs. `ngspice-45.2` (Step 1 in-proc & Step 2 UNIX socket) | JoSIM `nTron` coupled to `LTspice.exe` 2-stage MOSFET LNA via Robin/Thevenin PWL waveform relaxation, cross-validated against `ngspice-45.2` lockstep XSPICE | Converges to $\Delta V = 57.1\,\mu\text{V}$ in 3 iterations; `LTspice.exe` vs. `ngspice-45.2` cross-simulator NRMSE = $0.557\%$ on `V(nd)` and $0.303\%$ on `V(vout)`, Step 1 vs. UNIX diff = $0.0$ |

---

## 5. Step 3: Cadence Spectre 25.1 `CDS_VA_DPI` / DPI-C Adapter (`cosim/spectre/`)

Implemented in [`include/JoSIM/josim_spectre_dpi.h`](include/JoSIM/josim_spectre_dpi.h), [`cosim/spectre/josim_cosim_dpi.c`](cosim/spectre/josim_cosim_dpi.c), [`cosim/spectre/josim_cosim_port.va`](cosim/spectre/josim_cosim_port.va), and [`cosim/spectre/run_spectre_cosim.sh`](cosim/spectre/run_spectre_cosim.sh), and verified directly on **Cadence Spectre 25.1 (`25.1.0.156.isr2 64bit`)** on `asic2.inst.bnl.gov`:

- **Native Cadence Spectre `CDS_VA_DPI` Support:** Standalone Spectre requires `import "CDS_VA_DPI" function ...;` inside the Verilog-A `module` scope and `ahdl_include "libjosim_spectre_dpi.so" -cds_va_dpi` in the `.scs` netlist (while Xcelium AMS uses `import "DPI-C"`, selected automatically via `` `ifdef INCA ``).
- **Exact Symbolic AD Jacobian Preservation:** In Verilog-A, foreign C functions have zero symbolic automatic differentiation (AD) derivative ($\partial / \partial V = 0$). By returning `i_hist = I_port(V) - G_eq * V` directly from C via `josim_dpi_eval_ihist`, the Verilog-A statement `I(p, n) <+ i_hist + g_eq * V(p, n);` stamps both the exact branch current $I_{\text{port}}(V)$ and the exact analytical KLU driving-point conductance $\partial I / \partial V = G_{\text{eq}}$ into Spectre's MNA Jacobian without AD cancellation.
- **Idempotent DC + Transient Lifecycle:** Spectre evaluates `@(initial_step)` on every Newton–Raphson iteration of DC operating-point (`dcOp dc`) and transient $t=0$ initialization; `josim_dpi_open` caches active handles by `(endpoint, netlist, port_nodes, port_modes, phase_mode)` and resets cleanly at $t=0$.

| Benchmark | Simulator / Environment | Description | Key Verified Metrics |
| :--- | :--- | :--- | :--- |
| **`SP1_DPI_C_INPROC_AND_SOCKET`** | `libjosim_spectre_dpi.so` (In-proc `dlopen`, UNIX socket, TCP socket) | Split 4-JJ JTL (`B1`) + 2-Port Cryo-CMOS $\to$ RSFQ DFF (`B2`) with 5 injected `$abstime` LTE rollbacks | `B1` NRMSE vs. monolithic JoSIM = $0.0214\%$, In-proc vs. UNIX vs. TCP max diff = $0.0$, 5 LTE rollbacks with zero state drift |
| **`SP2A_SPLIT_JTL_SPECTRE`** | Live **Cadence Spectre 25.1** on `asic2.inst.bnl.gov` | Split 4-JJ JTL: 2-JJ `seeqc_jj3.va` TX in Spectre $\to$ `L23` $\to$ 2-JJ JoSIM RX via `josim_cosim_port_norton` | `0 errors`, 880 accepted steps, `ph2` final = $13.3431\text{ rad}$ vs. monolithic JoSIM $13.3418\text{ rad}$ (error $0.0013\text{ rad} = 0.0095\%$) |
| **`SP2B_NTRON_LNA_SPECTRE`** | Live **Cadence Spectre 25.1** on `asic2.inst.bnl.gov` | JoSIM electrothermal `nTron` driving a 4K Cryo-CMOS Common-Source LNA in Spectre with full `dcOp dc` (`skipdc=no`) | `dcOp dc` converged in 2 iterations (`0 errors`), `nTron` hotspot peak = $6.941\text{ mV}$, Cryo-CMOS LNA drain swing = $88.52\text{ mV}$ ($12.75\times$ gain) |

---

## 6. Performance & Physics Comparison: Partitioned JoSIM Co-Simulation vs. Monolithic Verilog-A (`seeqc_jj3.va`)

We benchmarked our partitioned JoSIM co-simulation architecture against the alternative of embedding a Verilog-A RCSJ Josephson junction model ([`seeqc_jj3.va`](../spec2si-seeqc/models/veriloga/seeqc_jj3.va) from `spec2si-seeqc`) directly inside SPICE/Spectre across three tiers:

### 6.1 Tier 1: Local `ngspice-45.2` + `openvaf 23.5.0` (`seeqc_jj3.osdi` vs. `josim_cosim.cm`)
Hybrid circuit: **32 Cryo-CMOS MOSFETs** (clocked 16-stage inverter/buffer tree) + **$N_{\text{JJ}}$-stage RSFQ JTL chain** ($90\text{ ps}$ transient, 2 SFQ pulses):

| $N_{\text{JJ}}$ | $N_{\text{MOSFET}}$ | Monolithic `seeqc_jj3.osdi` Wall (s) | Partitioned `josim_cosim.cm` Wall (s) | **Speedup** | `seeqc_jj3.osdi` SPICE Iters / Pts | `josim_cosim.cm` SPICE Iters / Pts | Extra SPICE Phase Nodes (`seeqc_jj3` vs. Co-Sim) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **16** | 32 | $0.0677\text{ s}$ | $0.0200\text{ s}$ | **$3.38\times$** | 1,577 / 767 | 1,082 / 503 | $+16$ vs. $0$ |
| **40** | 32 | $0.1379\text{ s}$ | $0.0357\text{ s}$ | **$3.86\times$** | 1,569 / 767 | 1,082 / 503 | $+40$ vs. $0$ |
| **80** | 32 | $0.2483\text{ s}$ | $0.0596\text{ s}$ | **$4.16\times$** | 1,562 / 767 | 1,082 / 503 | $+80$ vs. $0$ |
| **160** | 32 | $4.3583\text{ s}$ | $0.1069\text{ s}$ | **$40.75\times$** | 1,562 / 767 | 1,082 / 503 | $+160$ vs. $0$ |

### 6.2 Tier 2: Live Cadence Spectre 25.1 on `asic2.inst.bnl.gov` (`CMP2`: 32 MOSFETs + 100 JJs, $150\text{ ps}$)

| Approach in Cadence Spectre 25.1 | `dcOp dc` Convergence | Accepted Spectre Steps | Intrinsic Transient CPU Time | **Transient CPU Speedup** |
| :--- | :---: | :---: | :---: | :---: |
| **Monolithic `seeqc_jj3.va` (100 JJs + 32 MOSFETs in Spectre)** | Fails (`skipdc=yes` required) | 2,072 | $0.2383\text{ s}$ | $1.00\times$ (baseline) |
| **Partitioned `libjosim_spectre_dpi.so` + `josim-cosim-server`** | **Converges (`skipdc=no`)** | **772** ($2.68\times$ fewer) | **$0.0365\text{ s}$** | **$6.53\times$ faster** |

### 6.3 Tier 3: Empirical BNL Cluster RSFQ Macro & Chip Dataset (`spec2si-seeqc`)

| Circuit / Benchmark Suite | Junctions & T-Lines | JoSIM Runtime (s) | Spectre + `seeqc_jj3.va` Runtime (s) | **JoSIM Speedup** | Accuracy / Convergence Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **9 `v6` RSFQ Macros Total** (`DCSFQ`, `DROC`, `FA`, `LA`, `MERGE`, `MERGE_SPLIT`, `SPLIT`, `SPLIT2`, `SPLIT3`) | 9 routed PTL macros | **$12.15\text{ s}$** | $62.61\text{ s}$ (at $0.05\text{ ps}$) | **$5.15\times$** ($>9\times$ at $0.025\text{ ps}$) | At `0.05 ps`, Spectre `seeqc_jj3.va` fails the $0.02\text{ rad}$ phase gate on `MERGE` ($0.031\text{ rad}$) and `MERGE_SPLIT` ($0.028\text{ rad}$), requiring `maxstep=0.025 ps` |
| **Routed 4-Bit Adder Chip (`adder-r3`)** | 1,490 JJs, 629 PTLs ($33\text{ ns}$) | **$1,251\text{ s}$** ($20.8\text{ min}$) | $5,251\text{ s}$ ($87.5\text{ min}$) | **$4.20\times$** | Even at $4.2\times$ longer runtime, Spectre (`maxstep=0.1 ps`) accumulates $1.5\text{ ps}$ carry-chain drift ($0.088\text{ rad}$ vs. $0.02\text{ rad}$ threshold) |

### 6.4 Architectural & Physics Comparison Summary

| Dimension | Monolithic Verilog-A (`seeqc_jj3.va` in Spectre/SPICE) | Partitioned JoSIM Co-Simulation (`libjosim_spectre_dpi.so` / `josim_cosim.cm`) |
| :--- | :--- | :--- |
| **Global Timestep Clamping** | Every junction calls `$bound_step(2*M_PI*tsfactor*PHI0_2PI/vj)`, forcing the **entire BSIM4 Cryo-CMOS matrix** to be re-evaluated and LU-refactored at sub-picosecond steps ($20\text{–}50\text{ fs}$) | Spectre steps Cryo-CMOS at its natural analog/digital timestep while JoSIM substeps the superconducting partition on its pre-factored KLU matrix |
| **MNA Matrix Size & `cmin` Sensitivity** | Adds an `electrical ph` node per JJ with synthetic capacitor `PHI0_2PI = 0.3291 fF`. **Setting standard CMOS `cmin=0.1f` in Spectre adds $0.1\text{ fF}$ in parallel with `ph`, corrupting $\Phi_0$ by $+30.4\%$!** | Zero extra internal nodes in Spectre; Josephson phase lives inside JoSIM's state vector, completely immune to Spectre `cmin`/`gmin` options |
| **DC Operating Point (`dcOp dc`)** | No DC resistance path on `ph` and indeterminate DC phase; **requires `skipdc=yes`**, breaking standard analog Cryo-CMOS DC bias initialization | Converges cleanly in 2 NR iterations in standard `dcOp dc` (`skipdc=no`) via JoSIM's $t=0$ Norton companion conductance |
| **Device Physics Coverage** | Constant linear resistor `rtype=0` (`rqp`) only; lacks subgap piecewise-nonlinear $I\text{–}V$ (`rtype=1`), thermal noise, phase sources, and `nTron`/`hTron` nanowires | Full JoSIM physics: `rtype=1` nonlinear subgap/gap $I\text{–}V$ ($R_0, R_n, V_g$), thermal Johnson–Nyquist noise, mutual inductors, lossy T-lines, and electrothermal `nTron`/`hTron` |

