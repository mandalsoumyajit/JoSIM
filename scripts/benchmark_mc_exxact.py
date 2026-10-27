#!/usr/bin/env python3
"""
Tier 2 Batched Monte Carlo (.spread) Robustness Benchmark Suite for JoSIM on BNL exxact
(Dual AMD EPYC 7413 48C/96T + NVIDIA RTX A6000 GPU).

Benchmarks:
1. Monolithic 0-PTL Superconducting Circuits (P = 1):
   - ex_jtl_basic (MIT-LL JTL, 0 PTLs, P = 1) @ B = 256, 1024
   - Mono-RSFQ-Cell-Macro (128 JJs, 64 K, 64 C, 0 PTLs, P = 1, N = 642 eqs) @ B = 256, 1024
   - Mono-RLK-SQUID-Macro (96 JJs, 191 K, 48 C, 0 PTLs, P = 1, N = 578 eqs) @ B = 256, 1024
2. MIT-LL Standard-Cell & Multi-Gate Superconducting Logic Circuits:
   - ex_mitll_xor_wr (MIT-LL RSFQ XOR gate) @ B = 256, 1024
   - ex_mitll_dff_wr (MIT-LL RSFQ DFF gate) @ B = 256, 1024
   - ex_ksa4bit (4-bit Kogge-Stone RSFQ Adder, 695 JJs, 3,275 eqs) @ B = 64, 256
   - RSFQ-Pipeline-16B (16-block RSFQ ALU pipeline, 512 JJs, 4,732 eqs) @ B = 64, 256
   - RSFQ-Pipeline-64B (64-block RSFQ ALU pipeline, 2,048 JJs, 18,940 eqs) @ B = 64, 256

Verifies:
- Bitwise / machine-precision agreement between Batched Serial (1T), Batched OpenMP (-p),
  and Batched CUDA GPU (-g) across all ensemble statistics (Mean, STD, MIN, MAX).
- Transient solver and total wall-clock speedups vs. Batched Serial (1T) and
  traditional Sequential Process-Loop Monte Carlo (B separate invocations).
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from josim_io import load_josim_bin

JOSIM_BIN = Path("/u/home/smandal/.cache/josim-gpu-build/josim-cli")
WORK_DIR = Path("/tmp/josim_mc_bench")
WORK_DIR.mkdir(parents=True, exist_ok=True)


def generate_mono_rsfq_cell_macro(path: Path) -> None:
    """Monolithic 0-PTL 128-JJ coupled RSFQ logic macro (P = 1 connected component, N = 642 eqs)."""
    n_stages = 64
    lines = [
        "* Monolithic 0-PTL 128-JJ Coupled RSFQ Cell Macro (P = 1)",
        ".spread l=0.04 r=0.05 c=0.04 b=0.04",
        ".model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=160, rn=16, icrit=0.12m)",
        "VIN in_0 0 pwl(0 0 10p 0 12p 0.65m 14p 0 80p 0 82p 0.65m 84p 0 160p 0 162p 0.65m 164p 0)",
    ]
    prev = "in_0"
    for s in range(n_stages):
        n_a = f"na_{s}"
        n_b = f"nb_{s}"
        n_out = f"nout_{s}"
        lines.append(f"LA_{s} {prev} {n_a} 1.8p")
        lines.append(f"LB_{s} {n_a} {n_b} 2.1p")
        lines.append(f"LC_{s} {n_b} {n_out} 1.9p")
        lines.append(f"KM_{s} LA_{s} LB_{s} 0.18")
        lines.append(f"B1_{s} {n_a} 0 jjmod area=1.8")
        lines.append(f"RS1_{s} {n_a} 0 4.5")
        lines.append(f"B2_{s} {n_out} 0 jjmod area=2.1")
        lines.append(f"RS2_{s} {n_out} 0 4.2")
        lines.append(f"CS_{s} {n_b} 0 0.05p")
        lines.append(f"IB_{s} 0 {n_b} pwl(0 0 5p 260u)")
        prev = n_out
    lines.append(f"RLOAD {prev} 0 4.0")
    lines.append(".tran 0.2p 400p 0 1p")
    lines.append(f".print V(na_0) V(nout_{n_stages//2}) V(nout_{n_stages-1}) P(B2_{n_stages-1})")
    lines.append(".end")
    path.write_text("\n".join(lines) + "\n")


def generate_mono_rlk_squid_macro(path: Path) -> None:
    """Monolithic 0-PTL 96-JJ magnetically coupled SQUID array with 191 mutual inductances (P = 1, N = 578 eqs)."""
    n_cells = 48
    lines = [
        "* Monolithic 0-PTL 96-JJ Coupled SQUID Sensor Array (P = 1)",
        ".spread l=0.04 r=0.05 c=0.04 b=0.04",
        ".model jjsq jj(rtype=1, vg=2.8m, cap=0.065p, r0=180, rn=18, icrit=0.11m)",
        "IMOD mod_in 0 sin(0 160u 4G 2p 0)",
        "RMOD mod_in 0 50.0",
    ]
    prev_mod = "mod_in"
    prev_sq = "0"
    for c in range(n_cells):
        m_node = f"m_{c}"
        s_top = f"st_{c}"
        s_L = f"sl_{c}"
        s_R = f"sr_{c}"
        lines.append(f"LMOD_{c} {prev_mod} {m_node} 2.2p")
        lines.append(f"LS1_{c} {s_top} {s_L} 1.9p")
        lines.append(f"LS2_{c} {s_top} {s_R} 1.9p")
        lines.append(f"KM1_{c} LMOD_{c} LS1_{c} 0.30")
        lines.append(f"KM2_{c} LMOD_{c} LS2_{c} -0.30")
        lines.append(f"KSS_{c} LS1_{c} LS2_{c} 0.12")
        if c > 0:
            lines.append(f"KADJ_{c} LMOD_{c} LMOD_{c-1} 0.08")
        lines.append(f"BL_{c} {s_L} {prev_sq} jjsq area=1.6")
        lines.append(f"RL_{c} {s_L} {prev_sq} 5.0")
        lines.append(f"BR_{c} {s_R} {prev_sq} jjsq area=1.6")
        lines.append(f"RR_{c} {s_R} {prev_sq} 5.0")
        lines.append(f"CP_{c} {s_top} {prev_sq} 0.06p")
        lines.append(f"IB_{c} {prev_sq} {s_top} pwl(0 0 4p 210u)")
        prev_mod = m_node
        prev_sq = s_top
    lines.append(f"RTERM {prev_mod} 0 10.0")
    lines.append(".tran 0.2p 400p 0 1p")
    lines.append(f".print V(st_0) V(st_{n_cells//2}) V(st_{n_cells-1}) I(LMOD_0)")
    lines.append(".end")
    path.write_text("\n".join(lines) + "\n")


def generate_rsfq_pipeline(path: Path, n_blocks: int) -> None:
    """Multi-block RSFQ ALU pipeline with 32 JJs/block, M, C, R, L, B."""
    cells_per_block = 8
    lines = [
        f"* {n_blocks}-Block RSFQ ALU Pipeline Macro for Monte Carlo Yield Analysis",
        ".spread l=0.04 r=0.05 c=0.04 b=0.04",
        ".model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=160, rn=16, icrit=0.12m)",
        "VA in_a_0 0 pwl(0 0 10p 0 12p 0.65m 14p 0 70p 0 72p 0.65m 74p 0)",
        "VB in_b_0 0 pwl(0 0 18p 0 20p 0.65m 22p 0 70p 0 72p 0.65m 74p 0)",
    ]
    for b in range(n_blocks):
        pa = f"in_a_{b}"
        pb = f"in_b_{b}"
        for c in range(cells_per_block):
            p = f"b{b}_c{c}"
            lines.append(f"LA1_{p} {pa} {p}_a1 1.6p")
            lines.append(f"LA2_{p} {p}_a1 {p}_am 2.0p")
            lines.append(f"KA_{p} LA1_{p} LA2_{p} 0.16")
            lines.append(f"BA_{p} {p}_a1 0 jjmod area=1.8")
            lines.append(f"RA_{p} {p}_a1 0 4.5")
            lines.append(f"LB1_{p} {pb} {p}_b1 1.6p")
            lines.append(f"LB2_{p} {p}_b1 {p}_bm 2.0p")
            lines.append(f"KB_{p} LB1_{p} LB2_{p} 0.16")
            lines.append(f"BB_{p} {p}_b1 0 jjmod area=1.8")
            lines.append(f"RB_{p} {p}_b1 0 4.5")
            lines.append(f"KAB_{p} LA2_{p} LB2_{p} 0.08")
            lines.append(f"CC_{p} {p}_am {p}_bm 0.04p")
            lines.append(f"BC1_{p} {p}_am 0 jjmod area=2.1")
            lines.append(f"RC1_{p} {p}_am 0 4.0")
            lines.append(f"BC2_{p} {p}_bm 0 jjmod area=2.1")
            lines.append(f"RC2_{p} {p}_bm 0 4.0")
            lines.append(f"LAO_{p} {p}_am {p}_ao 1.8p")
            lines.append(f"LBO_{p} {p}_bm {p}_bo 1.8p")
            lines.append(f"IBA_{p} 0 {p}_a1 pwl(0 0 5p 240u)")
            lines.append(f"IBB_{p} 0 {p}_b1 pwl(0 0 5p 240u)")
            pa = f"{p}_ao"
            pb = f"{p}_bo"
        if b < n_blocks - 1:
            lines.append(f"RTA1_{b} {pa} txa_{b} 2.0")
            lines.append(f"TA_{b} txa_{b} 0 in_a_{b+1} 0 td=6p z0=10")
            lines.append(f"RTA2_{b} in_a_{b+1} 0 10.0")
            lines.append(f"RTB1_{b} {pb} txb_{b} 2.0")
            lines.append(f"TB_{b} txb_{b} 0 in_b_{b+1} 0 td=6p z0=10")
            lines.append(f"RTB2_{b} in_b_{b+1} 0 10.0")
        else:
            lines.append(f"RLA {pa} 0 4.0")
            lines.append(f"RLB {pb} 0 4.0")
    lines.append(".tran 0.25p 350p 0 1p")
    lines.append(f".print V(b0_c0_ao) V(b{n_blocks//2}_c0_ao) V(b{n_blocks-1}_c{cells_per_block-1}_ao)")
    lines.append(".end")
    path.write_text("\n".join(lines) + "\n")


def prepare_existing_netlist(src: Path, dst: Path) -> None:
    """Inject .spread and strip non-transient .iv commands on an existing test netlist."""
    out_lines = [".spread l=0.04 r=0.05 c=0.04 b=0.04"]
    for line in src.read_text().splitlines():
        if line.strip().lower().startswith(".iv"):
            continue
        out_lines.append(line)
    dst.write_text("\n".join(out_lines) + "\n")


def run_josim(cir: Path, out_bin: Path, extra_args: list[str], env: dict) -> tuple[float, float, str]:
    if out_bin.exists():
        out_bin.unlink()
    cmd = [str(JOSIM_BIN), "-V", "1", "-m"] + extra_args + ["-o", str(out_bin), str(cir)]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    wall = time.perf_counter() - t0
    m = re.search(r"Transient simulation time:\s*([0-9.eE+-]+)\s*s", res.stdout)
    trans = float(m.group(1)) if m else wall
    return wall, trans, res.stdout


def main() -> None:
    mono_rsfq = WORK_DIR / "mono_rsfq_cell_macro.cir"
    mono_squid = WORK_DIR / "mono_rlk_squid_macro.cir"
    pipe_16b = WORK_DIR / "rsfq_pipeline_16b.cir"
    pipe_64b = WORK_DIR / "rsfq_pipeline_64b.cir"
    jtl_cir = WORK_DIR / "ex_jtl_basic_mc.cir"
    xor_cir = WORK_DIR / "ex_mitll_xor_mc.cir"
    dff_cir = WORK_DIR / "ex_mitll_dff_mc.cir"
    ksa_cir = WORK_DIR / "ex_ksa4bit_mc.cir"

    generate_mono_rsfq_cell_macro(mono_rsfq)
    generate_mono_rlk_squid_macro(mono_squid)
    generate_rsfq_pipeline(pipe_16b, 16)
    generate_rsfq_pipeline(pipe_64b, 64)
    prepare_existing_netlist(REPO_DIR / "test" / "ex_jtl_basic.cir", jtl_cir)
    prepare_existing_netlist(REPO_DIR / "test" / "ex_mitll_xor_wr.cir", xor_cir)
    prepare_existing_netlist(REPO_DIR / "test" / "ex_mitll_dff_wr.cir", dff_cir)
    prepare_existing_netlist(REPO_DIR / "test" / "ex_ksa4bit.cir", ksa_cir)

    cases = [
        ("ex_jtl_basic (Mono 0-PTL, P=1)", jtl_cir, 256),
        ("ex_jtl_basic (Mono 0-PTL, P=1)", jtl_cir, 1024),
        ("ex_mitll_xor_wr (MIT-LL XOR)", xor_cir, 256),
        ("ex_mitll_xor_wr (MIT-LL XOR)", xor_cir, 1024),
        ("ex_mitll_dff_wr (MIT-LL DFF)", dff_cir, 256),
        ("ex_mitll_dff_wr (MIT-LL DFF)", dff_cir, 1024),
        ("Mono-RSFQ-Cell-Macro (128 JJs, 0-PTL, P=1)", mono_rsfq, 256),
        ("Mono-RSFQ-Cell-Macro (128 JJs, 0-PTL, P=1)", mono_rsfq, 1024),
        ("Mono-RLK-SQUID-Macro (96 JJs, 191 K, 0-PTL)", mono_squid, 256),
        ("Mono-RLK-SQUID-Macro (96 JJs, 191 K, 0-PTL)", mono_squid, 1024),
        ("ex_ksa4bit (4-bit KSA Adder, 695 JJs)", ksa_cir, 64),
        ("ex_ksa4bit (4-bit KSA Adder, 695 JJs)", ksa_cir, 256),
        ("RSFQ-Pipeline-16B (512 JJs, 16 Blocks)", pipe_16b, 64),
        ("RSFQ-Pipeline-16B (512 JJs, 16 Blocks)", pipe_16b, 256),
        ("RSFQ-Pipeline-64B (2,048 JJs, 64 Blocks)", pipe_64b, 64),
        ("RSFQ-Pipeline-64B (2,048 JJs, 64 Blocks)", pipe_64b, 256),
    ]

    base_env = os.environ.copy()
    base_env["JOSIM_SEED"] = "42"
    base_env["CUDA_VISIBLE_DEVICES"] = "0"

    env_1t = base_env.copy()
    env_1t["OMP_NUM_THREADS"] = "1"

    env_omp = base_env.copy()
    env_omp["OMP_NUM_THREADS"] = "48"
    env_omp["OMP_PROC_BIND"] = "close"
    env_omp["OMP_PLACES"] = "cores"

    results = []
    print("=" * 136)
    print(
        f"{'Circuit':<44} | {'B':>4} | {'1T Trans':>8} | {'OMP Trans':>9} | {'GPU Trans':>9} | "
        f"{'OMP Spd':>7} | {'GPU Spd':>7} | {'Wall Spd':>8} | {'Rel Diff':>9}"
    )
    print("-" * 136)

    single_cache = {}
    for label, cir, batch_b in cases:
        bin_1 = WORK_DIR / "single_1t.bin"
        bin_ser = WORK_DIR / "mc_ser.bin"
        bin_omp = WORK_DIR / "mc_omp.bin"
        bin_gpu = WORK_DIR / "mc_gpu.bin"

        # Single-run baseline (cached per circuit)
        if str(cir) not in single_cache:
            single_cache[str(cir)] = run_josim(cir, bin_1, [], env_1t)
        w1, t1, out1 = single_cache[str(cir)]
        seq_loop_wall_est = w1 * batch_b

        # Batched Serial (1T)
        w_ser, t_ser, out_ser = run_josim(cir, bin_ser, ["-b", str(batch_b)], env_1t)
        # Batched OpenMP (-p, 48C)
        w_omp, t_omp, out_omp = run_josim(cir, bin_omp, ["-p", "-b", str(batch_b)], env_omp)
        # Batched CUDA GPU (-g, RTX A6000)
        w_gpu, t_gpu, out_gpu = run_josim(cir, bin_gpu, ["-g", "-b", str(batch_b)], env_omp)

        d_ser = load_josim_bin(str(bin_ser))
        d_omp = load_josim_bin(str(bin_omp))
        d_gpu = load_josim_bin(str(bin_gpu))

        max_rel_omp = 0.0
        max_rel_gpu = 0.0
        for k in d_ser:
            denom = max(1.0, float(np.max(np.abs(d_ser[k]))))
            max_rel_omp = max(max_rel_omp, float(np.max(np.abs(d_ser[k] - d_omp[k]))) / denom)
            max_rel_gpu = max(max_rel_gpu, float(np.max(np.abs(d_ser[k] - d_gpu[k]))) / denom)

        omp_wall_spd = w_ser / w_omp
        gpu_wall_spd = w_ser / w_gpu
        omp_trans_spd = t_ser / t_omp
        gpu_trans_spd = t_ser / t_gpu
        gpu_vs_loop_spd = seq_loop_wall_est / w_gpu

        print(
            f"{label:<44} | {batch_b:>4} | {t_ser:>7.3f}s | {t_omp:>8.3f}s | {t_gpu:>8.3f}s | "
            f"{omp_trans_spd:>6.2f}x | {gpu_trans_spd:>6.2f}x | {gpu_wall_spd:>7.2f}x | {max_rel_gpu:>9.2e}"
        )

        assert max_rel_omp < 1e-12, f"OMP mismatch on {label}: {max_rel_omp}"
        assert max_rel_gpu < 1e-12, f"GPU mismatch on {label}: {max_rel_gpu}"

        results.append(
            {
                "label": label,
                "batch_b": batch_b,
                "single_wall_s": w1,
                "seq_loop_wall_est_s": seq_loop_wall_est,
                "ser_wall_s": w_ser,
                "ser_trans_s": t_ser,
                "omp_wall_s": w_omp,
                "omp_trans_s": t_omp,
                "gpu_wall_s": w_gpu,
                "gpu_trans_s": t_gpu,
                "omp_wall_speedup": omp_wall_spd,
                "gpu_wall_speedup": gpu_wall_spd,
                "omp_trans_speedup": omp_trans_spd,
                "gpu_trans_speedup": gpu_trans_spd,
                "gpu_vs_seq_loop_speedup": gpu_vs_loop_spd,
                "max_rel_omp": max_rel_omp,
                "max_rel_gpu": max_rel_gpu,
            }
        )

    print("=" * 136)
    out_json = WORK_DIR / "mc_benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved JSON summary to {out_json}")


if __name__ == "__main__":
    main()
