#!/usr/bin/env python3
"""
Benchmark & Numerical Verification Suite for JoSIM Tier 3:
Fully GPU-Resident Monolithic Solver (cuSolverRF + Zero-PCIe Fused Device Kernels, -g 3)
and Unified GPU Auto-Dispatcher (-g) on BNL 'exxact' Workstation (2x Xeon Gold 6338 + NVIDIA RTX A6000).
"""
import csv
import os
import re
import subprocess
import tempfile
import time

JOSIM_BIN = "/u/home/smandal/.cache/josim-gpu-build/josim-cli"
TRAN_RE = re.compile(r"Transient simulation time:\s*([0-9.eE+-]+)\s*s")


def gen_2d_jj_grid(rows: int, cols: int, rtype: int = 1, nsteps: int = 1000) -> str:
    lines = [
        f"* Monolithic 2D Josephson Junction Array ({rows}x{cols} = {rows*cols} JJs, 0 PTLs)",
        f".model jjmod jj(rtype={rtype}, vg=2.8mV, cap=0.07pF, r0=160ohm, rn=16ohm, icrit=0.1mA)",
        "Vsrc 1000000 0 pwl(0 0 1p 0 2p 0.5mV 20p 0.5mV)",
    ]
    def node_id(r: int, c: int) -> int:
        return r * cols + c + 1

    b_idx = 1
    l_idx = 1
    r_idx = 1
    for r in range(rows):
        for c in range(cols):
            n = node_id(r, c)
            lines.append(f"B{b_idx} {n} 0 jjmod area=1.0")
            lines.append(f"Rsh{b_idx} {n} 0 8ohm")
            b_idx += 1
            if c == 0:
                lines.append(f"Rin{r} 1000000 {n} 10ohm")
            if c + 1 < cols:
                n_right = node_id(r, c + 1)
                lines.append(f"Lh{l_idx} {n} {n_right} 2.0pH")
                l_idx += 1
            if r + 1 < rows:
                n_down = node_id(r + 1, c)
                lines.append(f"Lv{l_idx} {n} {n_down} 2.0pH")
                l_idx += 1
            if c == cols - 1:
                lines.append(f"Rout{r_idx} {n} 0 5ohm")
                r_idx += 1

    tstop = nsteps * 0.05
    lines.append(f".tran 0.05p {tstop:.2f}p 0p 1.0p")
    lines.append(".print DEVV B1")
    lines.append(f".print DEVV B{rows*cols}")
    lines.append(".end")
    return "\n".join(lines) + "\n"


def gen_monolithic_rsfq_network(n_stages: int = 3000, nsteps: int = 1000) -> str:
    lines = [
        f"* Monolithic RSFQ Logic Pipeline ({n_stages} stages, {4*n_stages} JJs, 0 PTLs)",
        ".model jjmod jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160ohm, rn=16ohm, icrit=0.1mA)",
        "Iin 0 1 pwl(0 0 1p 0 2p 0.18mA 4p 0 20p 0 21p 0.18mA 23p 0)",
        "Vbias 9000000 0 pwl(0 0 0.5p 2.5mV 1000p 2.5mV)",
    ]
    node = 1
    b_idx = 1
    l_idx = 1
    r_idx = 1
    k_idx = 1
    for s in range(n_stages):
        n_a = node
        n_b = node + 1
        n_c = node + 2
        n_d = node + 3
        n_next = node + 4
        lines.append(f"Rb{r_idx} 9000000 {n_a} 14ohm")
        r_idx += 1
        lines.append(f"Rb{r_idx} 9000000 {n_c} 14ohm")
        r_idx += 1
        for n_j in (n_a, n_b, n_c, n_d):
            lines.append(f"B{b_idx} {n_j} 0 jjmod area=1.0")
            lines.append(f"Rsh{b_idx} {n_j} 0 7ohm")
            b_idx += 1
        l1 = l_idx
        l2 = l_idx + 1
        l3 = l_idx + 2
        l4 = l_idx + 3
        l_idx += 4
        lines.append(f"L{l1} {n_a} {n_b} 1.8pH")
        lines.append(f"L{l2} {n_b} {n_c} 1.8pH")
        lines.append(f"L{l3} {n_c} {n_d} 1.8pH")
        lines.append(f"L{l4} {n_d} {n_next} 2.2pH")
        lines.append(f"K{k_idx} L{l1} L{l3} 0.25")
        k_idx += 1
        node = n_next
    lines.append(f"Rload {node} 0 5ohm")
    tstop = nsteps * 0.05
    lines.append(f".tran 0.05p {tstop:.2f}p 0p 1.0p")
    lines.append(".print DEVV B1")
    lines.append(f".print DEVV B{b_idx-1}")
    lines.append(".end")
    return "\n".join(lines) + "\n"


def gen_monolithic_rlk_mesh(n_ind: int = 3500, group_size: int = 15, nsteps: int = 1000) -> str:
    lines = [
        f"* Monolithic Coupled RLK Mesh ({n_ind} inductors, group_size={group_size}, 0 PTLs)",
        ".model jjmod jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160ohm, rn=16ohm, icrit=0.1mA)",
        "Vin 1 0 sin(0 1.0mV 20GHz 0 0)",
    ]
    for i in range(1, n_ind + 1):
        n_in = i
        n_mid = n_ind + 2 + 2 * i
        n_out = i + 1
        lines.append(f"R{i} {n_in} {n_mid} 2.0ohm")
        lines.append(f"L{i} {n_mid} {n_out} 3.0pH")
        lines.append(f"C{i} {n_out} 0 0.02pF")
        if i % 5 == 0:
            lines.append(f"B{i} {n_out} 0 jjmod area=1.0")
    lines.append(f"Rterm {n_ind + 1} 0 5.0ohm")
    k_idx = 1
    for g_start in range(1, n_ind + 1, group_size):
        g_end = min(n_ind + 1, g_start + group_size)
        for a in range(g_start, g_end):
            for b in range(a + 1, g_end):
                lines.append(f"K{k_idx} L{a} L{b} 0.05")
                k_idx += 1
    tstop = nsteps * 0.05
    lines.append(f".tran 0.05p {tstop:.2f}p 0p 1.0p")
    lines.append(".print NODEV 2")
    lines.append(f".print NODEV {n_ind + 1}")
    lines.append(".end")
    return "\n".join(lines) + "\n"


def gen_ptl_jtl_network(n_blocks: int = 256, jjs_per_block: int = 30, nsteps: int = 1000) -> str:
    lines = [
        f"* Multi-Block PTL-Coupled JTL Network ({n_blocks} blocks x {jjs_per_block} JJs)",
        ".model jjmod jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160ohm, rn=16ohm, icrit=0.1mA)",
        "Itrig 0 1 pwl(0 0 1p 0 2p 0.18mA 4p 0)",
    ]
    node = 1
    b_idx = 1
    l_idx = 1
    r_idx = 1
    t_idx = 1
    for blk in range(n_blocks):
        for j in range(jjs_per_block):
            n_cur = node
            lines.append(f"Ib{b_idx} 0 {n_cur} pwl(0 0 0.5p 0.07mA 1000p 0.07mA)")
            lines.append(f"B{b_idx} {n_cur} 0 jjmod area=1.0")
            lines.append(f"Rsh{b_idx} {n_cur} 0 7ohm")
            b_idx += 1
            if j + 1 < jjs_per_block:
                n_next = node + 1
                lines.append(f"L{l_idx} {n_cur} {n_next} 2.0pH")
                l_idx += 1
                node = n_next
        if blk + 1 < n_blocks:
            n_tx_out = node + 1
            lines.append(f"T{t_idx} {node} 0 {n_tx_out} 0 td=1.0p z0=5ohm")
            t_idx += 1
            lines.append(f"Rt{r_idx} {n_tx_out} 0 5ohm")
            r_idx += 1
            node = n_tx_out
        else:
            lines.append(f"Rload {node} 0 5ohm")
            node += 1
    tstop = nsteps * 0.05
    lines.append(f".tran 0.05p {tstop:.2f}p 0p 1.0p")
    lines.append(".print DEVV B1")
    lines.append(f".print DEVV B{b_idx-1}")
    lines.append(".end")
    return "\n".join(lines) + "\n"


def run_josim(cir_path: str, extra_args: list, out_path: str = None):
    cmd = [JOSIM_BIN, "-m", "-V", "1"] + extra_args
    if out_path:
        cmd += ["-o", out_path]
    cmd.append(cir_path)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    wall = time.perf_counter() - t0
    m = TRAN_RE.search(proc.stdout)
    tran_s = float(m.group(1)) if m else wall
    return tran_s, wall, proc.stdout


def max_csv_diff(csv_a: str, csv_b: str) -> float:
    with open(csv_a, "r") as fa, open(csv_b, "r") as fb:
        ra = list(csv.reader(fa))
        rb = list(csv.reader(fb))
    assert len(ra) == len(rb) and len(ra) > 1, f"Row count mismatch: {len(ra)} vs {len(rb)}"
    max_d = 0.0
    for row_a, row_b in zip(ra[1:], rb[1:]):
        for va, vb in zip(row_a[1:], row_b[1:]):
            d = abs(float(va) - float(vb))
            if d > max_d:
                max_d = d
    return max_d


def main():
    print("=" * 124)
    print("JoSIM Tier 3: Fully GPU-Resident Monolithic Solver (cuSolverRF, -g 3) & Auto-Dispatcher (-g) Benchmark Suite")
    print("=" * 124)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Numerical Verification across Phase (-a 1) and Voltage (-a 0), RTYPE=1, K, and PTLs
        print("\n[Phase 1] Numerical Accuracy Verification (Serial 1T vs Tier 3 cuSolverRF -g 3):")
        verify_cases = [
            ("Verify-2D-Grid-Phase",  gen_2d_jj_grid(30, 30, rtype=1, nsteps=400), ["-a", "1"]),
            ("Verify-2D-Grid-Volt",   gen_2d_jj_grid(30, 30, rtype=1, nsteps=400), ["-a", "0"]),
            ("Verify-RSFQ-K-Phase",   gen_monolithic_rsfq_network(200, nsteps=400), ["-a", "1"]),
            ("Verify-RSFQ-K-Volt",    gen_monolithic_rsfq_network(200, nsteps=400), ["-a", "0"]),
            ("Verify-RLK-Mesh-Phase", gen_monolithic_rlk_mesh(300, 10, nsteps=400), ["-a", "1"]),
            ("Verify-RLK-Mesh-Volt",  gen_monolithic_rlk_mesh(300, 10, nsteps=400), ["-a", "0"]),
            ("Verify-PTL-JTL-Phase",  gen_ptl_jtl_network(16, 15, nsteps=400), ["-a", "1"]),
            ("Verify-PTL-JTL-Volt",   gen_ptl_jtl_network(16, 15, nsteps=400), ["-a", "0"]),
        ]
        for name, netlist, anal_args in verify_cases:
            cir_f = os.path.join(tmpdir, f"{name}.cir")
            out_ser = os.path.join(tmpdir, f"{name}_ser.csv")
            out_rf = os.path.join(tmpdir, f"{name}_rf.csv")
            with open(cir_f, "w") as f:
                f.write(netlist)
            _, _, out1 = run_josim(cir_f, anal_args, out_ser)
            _, _, out3 = run_josim(cir_f, anal_args + ["-g", "3"], out_rf)
            diff = max_csv_diff(out_ser, out_rf)
            init_line = [ln for ln in out3.splitlines() if "Initialized CUDA" in ln]
            init_msg = init_line[0] if init_line else "OK"
            print(f"  {name:<24} | max |Serial - GPU_Tier3| = {diff:.3e} | {init_msg}")
            assert diff < 1e-11, f"Accuracy check failed for {name}: {diff}"

        # 2. Performance Benchmark across 2D Meshes, 1D Monolithic Chains, and Multi-Block PTL Circuits
        print("\n[Phase 2] Transient Solver Performance Comparison (1,000 steps):")
        bench_cases = [
            ("Mono-2D-Grid-14kJJ-Phase", gen_2d_jj_grid(120, 120, rtype=1, nsteps=1000), ["-a", "1"], True),
            ("Mono-2D-Grid-14kJJ-Volt",  gen_2d_jj_grid(120, 120, rtype=1, nsteps=1000), ["-a", "0"], True),
            ("Mono-2D-Grid-25kJJ-Phase", gen_2d_jj_grid(160, 160, rtype=1, nsteps=1000), ["-a", "1"], True),
            ("Mono-2D-Grid-25kJJ-NR",    gen_2d_jj_grid(160, 160, rtype=0, nsteps=1000), ["-a", "1"], True),
            ("Mono-2D-Grid-64kJJ-Phase", gen_2d_jj_grid(256, 256, rtype=1, nsteps=1000), ["-a", "1"], True),
            ("Mono-2D-Grid-64kJJ-NR",    gen_2d_jj_grid(256, 256, rtype=0, nsteps=1000), ["-a", "1"], True),
            ("Mono-RSFQ-12kJJ-Phase",    gen_monolithic_rsfq_network(3000, nsteps=1000), ["-a", "1"], False),
            ("Mono-RLK-49kK-Phase",      gen_monolithic_rlk_mesh(3500, 15, nsteps=1000), ["-a", "1"], False),
            ("Mono-RLK-49kK-Volt",       gen_monolithic_rlk_mesh(3500, 15, nsteps=1000), ["-a", "0"], False),
            ("PTL-256x30JTL-Phase",      gen_ptl_jtl_network(256, 30, nsteps=1000),      ["-a", "1"], False),
            ("PTL-512x30JTL-Phase",      gen_ptl_jtl_network(512, 30, nsteps=1000),      ["-a", "1"], False),
            ("PTL-800x30JTL-Phase",      gen_ptl_jtl_network(800, 30, nsteps=1000),      ["-a", "1"], False),
        ]

        header = (
            f"{'Circuit':<26} | {'Serial (s)':>10} | {'OMP -p (s)':>10} | "
            f"{'GPU -g 2 (s)':>12} | {'GPU -g 3 (s)':>12} | {'GPU -g (s)':>12} | "
            f"{'-g 2 Spd':>8} | {'-g 3 Spd':>8} | {'-g Auto':>8}"
        )
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for name, netlist, anal_args, run_g3 in bench_cases:
            cir_f = os.path.join(tmpdir, f"{name}.cir")
            with open(cir_f, "w") as f:
                f.write(netlist)

            t_ser, _, _ = run_josim(cir_f, anal_args)
            t_omp, _, _ = run_josim(cir_f, anal_args + ["-p"])
            t_g2, _, _  = run_josim(cir_f, anal_args + ["-g", "2"])
            t_g1, _, _  = run_josim(cir_f, anal_args + ["-g"])
            if run_g3:
                t_g3, _, _ = run_josim(cir_f, anal_args + ["-g", "3"])
                g3_str = f"{t_g3:12.4f}"
                s_g3_str = f"{t_ser / t_g3:7.2f}x"
            else:
                g3_str = f"{'N/A (1D)':>12}"
                s_g3_str = f"{'N/A':>8}"

            s_g2 = t_ser / t_g2
            s_g1 = t_ser / t_g1
            print(
                f"{name:<26} | {t_ser:10.4f} | {t_omp:10.4f} | "
                f"{t_g2:12.4f} | {g3_str} | {t_g1:12.4f} | "
                f"{s_g2:7.2f}x | {s_g3_str} | {s_g1:7.2f}x"
            )
        print("-" * len(header))


if __name__ == "__main__":
    main()
