#!/usr/bin/env python3
"""
Verification and Benchmark Suite for JoSIM Tier 1 Windowed PTL Block-Parallel Solver
(Multi-Core OpenMP CPU -p and NVIDIA CUDA GPU -g).
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import numpy as np


def read_josim_bin(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"JOSIMBIN", f"Invalid magic: {magic}"
        version, flags, n_traces, n_steps = struct.unpack("<IIQQ", f.read(24))
        names = []
        for _ in range(n_traces):
            (slen,) = struct.unpack("<I", f.read(4))
            names.append(f.read(slen).decode("utf-8"))
            f.read(1)  # type char
        data = np.frombuffer(f.read(n_traces * n_steps * 8), dtype=np.float64).reshape(
            (n_traces, n_steps)
        )
    return {name: data[i] for i, name in enumerate(names)}


def generate_ptl_tile_circuit(
    path,
    n_tiles=64,
    jjs_per_tile=32,
    rtype=1,
    analysis="phase",
    tstep="0.25p",
    tstop="10000p",
    prstart="0",
    prstep="10p",
    td="10p",
    interp=False,
):
    lines = [
        f"* PTL-Coupled Multi-Tile SFQ Benchmark ({n_tiles} tiles, {n_tiles * jjs_per_tile} JJs)",
        f".model jjmod jj(rtype={rtype}, vg=2.8m, cap=0.07p, r0=160, rn=16, icrit=0.1m)",
    ]
    lines.append(f".tran {tstep} {tstop} {prstart} {prstep}")

    lines.append("Vclk 1 0 pwl(0 0 2p 0 3p 0.6m 5p 0 100p 0)")

    for t in range(n_tiles):
        base = (t + 1) * 1000
        # Input termination and launch into tile t
        if t == 0:
            lines.append(f"Rin_0 1 {base + 1} 5")
        else:
            lines.append(f"Rterm_{t} {base + 1} 0 10")

        # JTL chain with mutual inductance inside tile t
        for j in range(jjs_per_tile):
            node = base + 1 + j
            next_node = base + 1 + j + 1
            lines.append(f"B_{t}_{j} {node} 0 jjmod area=1.0")
            lines.append(f"Rsh_{t}_{j} {node} 0 4.0")
            lines.append(f"Ibias_{t}_{j} 0 {node} pwl(0 0 1p 0.075m)")
            lines.append(f"L_{t}_{j} {node} {next_node} 2.0p")
            if j > 0:
                lines.append(f"K_{t}_{j} L_{t}_{j-1} L_{t}_{j} 0.05")

        out_node = base + 1 + jjs_per_tile
        lines.append(f"Rout_{t} {out_node} 0 10")

        # PTL connecting tile t output to tile (t + 1) input
        if t + 1 < n_tiles:
            next_base = (t + 2) * 1000
            interp_str = " interp=1" if interp else ""
            lines.append(
                f"T_{t} {out_node} 0 {next_base + 1} 0 td={td} z0=10{interp_str}"
            )

    # Print representative traces from first, middle, and last tiles
    mid = n_tiles // 2
    last = n_tiles - 1
    lines.append(f".print V(B_0_0) P(B_0_0)")
    lines.append(f".print V(B_{mid}_{jjs_per_tile//2}) P(B_{mid}_{jjs_per_tile//2})")
    lines.append(f".print V(B_{last}_{jjs_per_tile-1}) P(B_{last}_{jjs_per_tile-1})")
    lines.append(".end")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_josim(josim_bin, cir_path, out_path, extra_args=None, omp_threads=None):
    env = os.environ.copy()
    if omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_threads)
    cmd = [josim_bin, "-m", "-V", "1"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(["-o", out_path, cir_path])
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    dt = time.perf_counter() - t0
    return dt, proc.stdout


def verify_equivalence(josim_bin, work_dir):
    print("==========================================================================")
    print(" [1/2] Numerical Equivalence Verification: Serial vs OpenMP (-p) vs GPU (-g)")
    print("==========================================================================")
    test_cases = [
        ("Phase mode + RTYPE=1 + Mutual L + Decimation", "phase", 1, "10p", False),
        ("Phase mode + RTYPE=0 + Fractional PTL (INTERP=1)", "phase", 0, "10.35p", True),
        ("Voltage mode + RTYPE=1 + Mutual L + Decimation", "voltage", 1, "10p", False),
    ]
    for label, analysis, rtype, td, interp in test_cases:
        cir = os.path.join(work_dir, f"verify_{analysis}_r{rtype}.cir")
        generate_ptl_tile_circuit(
            cir,
            n_tiles=32,
            jjs_per_tile=16,
            rtype=rtype,
            analysis=analysis,
            tstep="0.25p",
            tstop="400p",
            prstart="10p",
            prstep="2.5p",
            td=td,
            interp=interp,
        )
        out_ser = os.path.join(work_dir, "ser.bin")
        out_omp = os.path.join(work_dir, "omp.bin")
        out_gpu = os.path.join(work_dir, "gpu.bin")

        base_args = ["-a", "0"] if analysis == "voltage" else []
        run_josim(josim_bin, cir, out_ser, extra_args=base_args, omp_threads=1)
        run_josim(josim_bin, cir, out_omp, extra_args=base_args + ["-p"], omp_threads=8)
        _, gpu_stdout = run_josim(josim_bin, cir, out_gpu, extra_args=base_args + ["-g"], omp_threads=1)
        assert "Initialized CUDA GPU PTL Block-Parallel Solver" in gpu_stdout, (
            f"GPU solver was not activated! Output:\n{gpu_stdout}"
        )

        tr_ser = read_josim_bin(out_ser)
        tr_omp = read_josim_bin(out_omp)
        tr_gpu = read_josim_bin(out_gpu)

        max_omp_err = 0.0
        max_gpu_err = 0.0
        for k in tr_ser:
            max_omp_err = max(max_omp_err, float(np.max(np.abs(tr_ser[k] - tr_omp[k]))))
            max_gpu_err = max(max_gpu_err, float(np.max(np.abs(tr_ser[k] - tr_gpu[k]))))

        print(f"  * {label}:")
        print(f"      max |Serial - OpenMP(-p)| = {max_omp_err:.3e}")
        print(f"      max |Serial - GPU(-g)|    = {max_gpu_err:.3e}")
        assert max_omp_err < 1e-11, f"OpenMP mismatch: {max_omp_err}"
        assert max_gpu_err < 1e-11, f"GPU mismatch: {max_gpu_err}"
    print("  >>> ALL EQUIVALENCE TESTS PASSED (MACHINE PRECISION AGREEMENT) <<<\n")


def parse_sim_time(stdout, fallback):
    m = re.search(r"Transient simulation time:\s*([0-9.eE+-]+)\s*s", stdout)
    return float(m.group(1)) if m else fallback


def run_benchmarks(josim_bin, work_dir):
    print("==========================================================================")
    print(" [2/2] Tier 1 Windowed PTL Solver Scaling Benchmark (RTX A6000 & EPYC 96T)")
    print("==========================================================================")
    configs = [
        ("PTL-Tile-64 (2k JJs)", 64, 32, "0.25p", "10000p", "25p"),
        ("PTL-Tile-256 (8k JJs)", 256, 32, "0.25p", "10000p", "25p"),
        ("PTL-Tile-1024 (32k JJs)", 1024, 32, "0.25p", "5000p", "25p"),
        ("PTL-Tile-4096 (131k JJs)", 4096, 32, "0.25p", "2500p", "25p"),
    ]
    print("\n--- A. Transient Solver Runtime (trans_sim) ---")
    print(
        f"{'Design':<26} | {'Eqns':>8} | {'Steps':>6} | {'Serial (1T)':>11} | {'OMP (8T)':>10} | {'OMP (32T)':>10} | {'GPU (A6000)':>11} | {'OMP Speedup':>11} | {'GPU Speedup':>11}"
    )
    print("-" * 130)
    wall_rows = []

    for name, n_tiles, jjs_per_tile, tstep, tstop, td in configs:
        cir = os.path.join(work_dir, f"bench_{n_tiles}.cir")
        generate_ptl_tile_circuit(
            cir,
            n_tiles=n_tiles,
            jjs_per_tile=jjs_per_tile,
            rtype=1,
            analysis="phase",
            tstep=tstep,
            tstop=tstop,
            prstart="0",
            prstep="25p",
            td=td,
            interp=False,
        )
        out_ser = os.path.join(work_dir, f"b_{n_tiles}_ser.bin")
        out_omp8 = os.path.join(work_dir, f"b_{n_tiles}_omp8.bin")
        out_omp32 = os.path.join(work_dir, f"b_{n_tiles}_omp32.bin")
        out_gpu = os.path.join(work_dir, f"b_{n_tiles}_gpu.bin")

        w_ser, stdout_ser = run_josim(josim_bin, cir, out_ser, extra_args=[], omp_threads=1)
        w_omp8, stdout_omp8 = run_josim(josim_bin, cir, out_omp8, extra_args=["-p"], omp_threads=8)
        w_omp32, stdout_omp32 = run_josim(josim_bin, cir, out_omp32, extra_args=["-p"], omp_threads=32)
        w_gpu, stdout_gpu = run_josim(josim_bin, cir, out_gpu, extra_args=["-g"], omp_threads=32)

        t_ser = parse_sim_time(stdout_ser, w_ser)
        t_omp8 = parse_sim_time(stdout_omp8, w_omp8)
        t_omp32 = parse_sim_time(stdout_omp32, w_omp32)
        t_gpu = parse_sim_time(stdout_gpu, w_gpu)

        m = re.search(r"Partitioned MNA matrix \((\d+) equations\)", stdout_ser)
        eqns = int(m.group(1)) if m else 0
        steps = int(round(float(tstop[:-1]) / float(tstep[:-1])))

        # Verify output agreement on benchmark circuit
        tr_ser = read_josim_bin(out_ser)
        tr_gpu = read_josim_bin(out_gpu)
        max_diff = max(float(np.max(np.abs(tr_ser[k] - tr_gpu[k]))) for k in tr_ser)
        assert max_diff < 1e-10, f"Benchmark GPU diff too large: {max_diff}"

        best_omp = min(t_omp8, t_omp32)
        omp_sp = t_ser / best_omp
        gpu_sp = t_ser / t_gpu
        print(
            f"{name:<26} | {eqns:>8d} | {steps:>6d} | {t_ser:>10.2f}s | {t_omp8:>9.2f}s | {t_omp32:>9.2f}s | {t_gpu:>10.2f}s | {omp_sp:>10.2f}x | {gpu_sp:>10.2f}x"
        )
        wall_rows.append((name, eqns, steps, w_ser, w_omp8, w_omp32, w_gpu))

    print("\n--- B. Total End-to-End Wall-Clock Time (Parse + Matrix + Factorize + Sim + Output) ---")
    print(
        f"{'Design':<26} | {'Eqns':>8} | {'Steps':>6} | {'Serial (1T)':>11} | {'OMP (8T)':>10} | {'OMP (32T)':>10} | {'GPU (A6000)':>11} | {'OMP Speedup':>11} | {'GPU Speedup':>11}"
    )
    print("-" * 130)
    for name, eqns, steps, w_ser, w_omp8, w_omp32, w_gpu in wall_rows:
        best_omp = min(w_omp8, w_omp32)
        print(
            f"{name:<26} | {eqns:>8d} | {steps:>6d} | {w_ser:>10.2f}s | {w_omp8:>9.2f}s | {w_omp32:>9.2f}s | {w_gpu:>10.2f}s | {w_ser/best_omp:>10.2f}x | {w_ser/w_gpu:>10.2f}x"
        )


if __name__ == "__main__":
    josim_bin = sys.argv[1] if len(sys.argv) > 1 else "/u/home/smandal/.cache/josim-gpu-build/josim-cli"
    work_dir = "/u/home/smandal/.cache/josim-gpu-bench"
    os.makedirs(work_dir, exist_ok=True)
    verify_equivalence(josim_bin, work_dir)
    run_benchmarks(josim_bin, work_dir)
