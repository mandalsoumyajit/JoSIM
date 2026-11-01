# JoSIM–SPICE Co-Simulation Adapters & Example Decks

This directory contains the native simulator adapters and ready-to-run hybrid superconducting/semiconducting co-simulation example decks for **Cadence Spectre**, **`ngspice`**, and **ADI `LTspice`**.

## Directory Layout

```text
cosim/
├── ngspice/
│   ├── josim_xspice.c                      # Native ngspice XSPICE code model (josim_cosim.cm)
│   └── examples/
│       ├── cmos_to_sfq_dff.sp              # 4K Cryo-CMOS DATA/CLOCK drivers -> 2-port RSFQ DFF
│       ├── ntron_to_cmos_lna.sp            # SFQ-triggered electrothermal nTron -> 2-stage Cryo-CMOS LNA
│       └── multi_instance_transceiver.sp   # 2 concurrent JoSIM XSPICE partitions + pad filter + Cryo-CMOS LNA
├── spectre/
│   ├── josim_cosim_dpi.c                   # Cadence Spectre CDS_VA_DPI C bridge (libjosim_spectre_dpi.so)
│   ├── josim_cosim_port.va                 # Verilog-A boundary port modules (1-port Norton/Thevenin & 2-/3-/4-/6-/8-port)
│   └── examples/
│       ├── sfq_dff.cir                     # JoSIM 2-port RSFQ D-Flip-Flop (DRO) subcircuit
│       ├── ntron_driver.cir                # JoSIM 1-port 2-JJ SFQ + electrothermal nTron driver subcircuit
│       ├── sfq_tx.cir                      # JoSIM Partition 1: 1-port 2-JJ SFQ transmitter
│       ├── ntron_rx_2p.cir                 # JoSIM Partition 2: 2-port SFQ receiver + electrothermal nTron driver
│       ├── cryo_cmos_sfq.scs               # Spectre deck: 4K Cryo-CMOS drivers -> sfq_dff.cir
│       ├── ntron_lna.scs                   # Spectre deck: ntron_driver.cir -> 4K Cryo-CMOS LNA
│       └── multi_partition_transceiver.scs # Spectre deck: sfq_tx.cir -> pad filter -> ntron_rx_2p.cir -> LNA
└── ltspice/
    └── examples/
        ├── sfq_ntron_duplex.cir            # JoSIM 2-port full-duplex SFQ + nTron core for Branin wave coupling
        ├── ltspice_branin_transceiver.cir  # LTspice template for multi-window Branin coax transceiver
        ├── branin_duplex_job.json          # Unified CLI job config for Multi-Window Branin Wave Relaxation
        ├── ltspice_pwl_lna.cir             # LTspice template for damped Robin/Thevenin PWL relaxation
        └── pwl_lna_job.json                # Unified CLI job config for Damped PWL Waveform Relaxation
```

## Quick Start

### 1. Build Co-Simulation Libraries & Server

```bash
cmake -B /tmp/josim-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/josim-build -j
```

This produces:
- `/tmp/josim-build/libjosim_c.so` (In-memory C API shared library)
- `/tmp/josim-build/josim-cosim-server` (Multiplexed `poll()` UNIX/TCP socket daemon)
- `/tmp/josim-build/josim_cosim.cm` (`ngspice` XSPICE code model)
- `/tmp/josim-build/libjosim_spectre_dpi.so` (Cadence Spectre `CDS_VA_DPI` shared library)

### 2. Run `ngspice` XSPICE Co-Simulation Examples

```bash
LD_LIBRARY_PATH=/tmp/josim-build ngspice -b cosim/ngspice/examples/cmos_to_sfq_dff.sp
LD_LIBRARY_PATH=/tmp/josim-build ngspice -b cosim/ngspice/examples/ntron_to_cmos_lna.sp
LD_LIBRARY_PATH=/tmp/josim-build ngspice -b cosim/ngspice/examples/multi_instance_transceiver.sp
```

To run with an out-of-process multiplexed `josim-cosim-server` daemon instead of in-process `libjosim_c.so`, add `server_addr="unix:/tmp/josim.sock"` (or `server_addr="tcp:127.0.0.1:19540"`) to the `.model ... josim_cosim(...)` card.

### 3. Run ADI `LTspice` Branin Wave & PWL Relaxation Examples

```bash
# Multi-Window Causal Branin Wave Relaxation (4 windows of W = 15 ps across two 50-Ohm coax links)
python3 scripts/josim_cosim.py --config cosim/ltspice/examples/branin_duplex_job.json

# Damped Robin/Thevenin PWL Waveform Relaxation (converges to < 0.2 uV in 2 iterations)
python3 scripts/josim_cosim.py --config cosim/ltspice/examples/pwl_lna_job.json
```

### 4. Run Cadence Spectre Co-Simulation Examples

```bash
# Start multiplexed JoSIM co-simulation server (supports 1 or more concurrent Spectre instances)
/tmp/josim-build/josim-cosim-server --unix /tmp/josim_spectre.sock --max-clients 2 &

# Run Spectre with CDS_VA_DPI
export JOSIM_COSIM_ENDPOINT="unix:/tmp/josim_spectre.sock"
export LD_LIBRARY_PATH="/tmp/josim-build:$LD_LIBRARY_PATH"
cd cosim/spectre/examples
spectre -64 +aps multi_partition_transceiver.scs
```

See [`docs/cosimulation.md`](../docs/cosimulation.md) for the complete architecture, API reference, and full listings of all example decks.
