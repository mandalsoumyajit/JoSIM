#!/usr/bin/env bash
set -euo pipefail

OPT_BUILD="$HOME/.cache/josim-build"
BASE_BIN="$HOME/.cache/josim-baseline-cli"
SRC_DIR="/mnt/c/dev/JoSIM"

# Ensure latest build is up to date
bash "$SRC_DIR/scripts/build_wsl.sh" "$OPT_BUILD" "$SRC_DIR" > /dev/null

python3 - << 'PY'
import os
import subprocess
import time
import tempfile
import shutil

base_bin = os.path.expanduser("~/.cache/josim-baseline-cli")
opt_bin  = os.path.expanduser("~/.cache/josim-build/josim-cli")
test_dir = "/mnt/c/dev/JoSIM/test"
tmp_dir  = "/tmp/josim_med_large_bench"
shutil.rmtree(tmp_dir, ignore_errors=True)
os.makedirs(tmp_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: run command, measure min wall time (s) and peak child RSS (MB)
# ---------------------------------------------------------------------------
def run_bench(cmd, runs=3):
    best_t = float("inf")
    peak_rss_kb = 0
    wrapper = (
        "import resource, subprocess, sys, time\n"
        "t0 = time.perf_counter()\n"
        "subprocess.run(sys.argv[1:], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "dt = time.perf_counter() - t0\n"
        "ru = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss\n"
        "print(f'{dt:.6f} {ru}')\n"
    )
    for _ in range(runs):
        out = subprocess.check_output(["python3", "-c", wrapper] + cmd, cwd=tmp_dir, text=True).strip()
        dt_str, rss_str = out.split()
        dt = float(dt_str)
        rss = int(rss_str)
        if dt < best_t:
            best_t = dt
        if rss > peak_rss_kb:
            peak_rss_kb = rss
    return best_t, peak_rss_kb / 1024.0

# ---------------------------------------------------------------------------
# Generate Medium & Large Synthetic Superconducting IC Designs
# ---------------------------------------------------------------------------

# 1. Medium Extracted SFQ Macro: 2,500 Inductors + 6,000 Mutual Couplings (K) + 500 JJs
med_ext_cir = os.path.join(tmp_dir, "med_extracted_2500L_6000K.cir")
med_ext_kcut_cir = os.path.join(tmp_dir, "med_extracted_2500L_6000K_kcut.cir")
def gen_extracted_cir(path, with_kcutoff=False):
    with open(path, "w") as f:
        f.write("* Medium 3D-Extracted SFQ Macro (2,500 L, 6,000 K, 500 JJ)\n")
        if with_kcutoff:
            f.write(".option kcutoff=0.01\n")
        f.write(".model jmitll jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160, rN=16, icrit=0.1mA)\n")
        f.write("VIN 1 0 pwl(0 0 5p 0 7p 0.8m 9p 0)\n")
        f.write("RIN 1 2 2\n")
        n_jj = 500
        # 5 inductors per JJ stage = 2,500 inductors total
        l_names = []
        for s in range(1, n_jj + 1):
            n_a = (s - 1) * 4 + 2
            n_b = n_a + 1
            n_c = n_a + 2
            n_d = n_a + 3
            n_out = n_a + 4
            n_j = 100000 + s
            n_sh = 200000 + s
            f.write(f"L1_{s} {n_a} {n_b} 0.5p\n")
            f.write(f"L2_{s} {n_b} {n_c} 0.5p\n")
            f.write(f"L3_{s} {n_c} {n_d} 0.5p\n")
            f.write(f"L4_{s} {n_d} {n_out} 0.5p\n")
            f.write(f"LP_{s} {n_j} 0 0.08p\n")
            l_names.extend([f"L1_{s}", f"L2_{s}", f"L3_{s}", f"L4_{s}", f"LP_{s}"])
            f.write(f"B_{s} {n_out} {n_j} jmitll area=2.0\n")
            f.write(f"RS_{s} {n_out} {n_sh} 5.0\n")
            f.write(f"IB_{s} 0 {n_out} pwl(0 0 3p 140u)\n")
        f.write(f"ROUT {n_jj * 4 + 2} 0 2\n")
        # 6,000 mutual couplings: 1,000 strong (k=0.05) + 5,000 weak geometric parasitics (k=0.001)
        k_idx = 1
        for i in range(0, 2000, 2):
            f.write(f"K_{k_idx} {l_names[i]} {l_names[i+1]} 0.05\n")
            k_idx += 1
        for span in [3, 5, 7, 9, 11]:
            for i in range(1000):
                f.write(f"K_{k_idx} {l_names[i]} {l_names[i + span]} 0.001\n")
                k_idx += 1
        f.write(".tran 0.25p 40p 0 0.25p\n")
        for p in range(1, 33):
            f.write(f".print DEVV B_{p * 15}\n")
        f.write(".end\n")

gen_extracted_cir(med_ext_cir, with_kcutoff=False)
gen_extracted_cir(med_ext_kcut_cir, with_kcutoff=True)

# 2. Large 10,000-JJ JTL Pipeline (~70,000 devices, ~90,000 MNA equations, 256 output traces)
large_10k_cir = os.path.join(tmp_dir, "large_10k_jtl.cir")
with open(large_10k_cir, "w") as f:
    f.write("* Large 10,000-JJ JTL Pipeline (256 saved traces)\n")
    f.write(".model jmitll jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160, rN=16, icrit=0.1mA)\n")
    f.write("VIN 1 0 pwl(0 0 10p 0 12p 600u 14p 0)\n")
    f.write("RIN 1 2 2\n")
    n_stages = 10000
    for s in range(1, n_stages + 1):
        n_in = s + 1
        n_out = s + 2
        n_jj = 100000 + s
        n_sh = 200000 + s
        f.write(f"L_{s} {n_in} {n_out} 2.0p\n")
        f.write(f"B_{s} {n_out} {n_jj} jmitll area=2.0\n")
        f.write(f"LP_{s} {n_jj} 0 0.08p\n")
        f.write(f"RS_{s} {n_out} {n_sh} 5.0\n")
        f.write(f"LS_{s} {n_sh} 0 0.08p\n")
        f.write(f"IB_{s} 0 {n_out} pwl(0 0 5p 140u)\n")
    f.write(f"ROUT {n_stages + 2} 0 2\n")
    f.write(".tran 0.25p 60p 0 0.25p\n")
    for p in range(1, 257):
        idx = p * (n_stages // 256)
        f.write(f".print DEVV B_{idx}\n")
    f.write(".end\n")

# 3. Large 64-Block PTL-Coupled Multi-Tile SFQ Processor Fabric (3,072 JJs, 512 PTLs, 2,500 steps)
large_ptl_cir = os.path.join(tmp_dir, "large_64blk_ptl_fabric.cir")
with open(large_ptl_cir, "w") as f:
    f.write("* Large 64-Block PTL-Coupled SFQ Multi-Tile Fabric (3,072 JJs, 512 PTLs)\n")
    f.write(".model jmitll jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160, rN=16, icrit=0.1mA)\n")
    f.write("VIN in_0 0 pwl(0 0 5p 0 7p 0.8m 9p 0 25p 0 27p 0.8m 29p 0)\n")
    n_blocks = 64
    stages_per_block = 48  # 64 * 48 = 3,072 JJs
    for blk in range(n_blocks):
        prev = f"in_{blk}"
        if blk == 0:
            f.write(f"RIN_{blk} {prev} n_{blk}_0 5\n")
            prev = f"n_{blk}_0"
        for s in range(stages_per_block):
            nxt = f"n_{blk}_{s+1}"
            bnode = f"jb_{blk}_{s}"
            f.write(f"L_{blk}_{s} {prev} {nxt} 2.0p\n")
            f.write(f"B_{blk}_{s} {nxt} {bnode} jmitll area=2.0\n")
            f.write(f"LP_{blk}_{s} {bnode} 0 0.1p\n")
            f.write(f"RS_{blk}_{s} {nxt} 0 5.0\n")
            f.write(f"IB_{blk}_{s} 0 {nxt} pwl(0 0 3p 0.14m)\n")
            prev = nxt
        if blk < n_blocks - 1:
            # 8 parallel PTL lanes between tiles -> 63 * 8 + 8 = 512 PTLs
            f.write(f"RTM1_{blk} {prev} txin_{blk} 10\n")
            f.write(f"T_DATA_{blk} txin_{blk} 0 in_{blk+1} 0 td=6p z0=10\n")
            f.write(f"RTM2_{blk} in_{blk+1} 0 10\n")
            for lane in range(1, 8):
                f.write(f"T_AUX{lane}_{blk} n_{blk}_{lane*5} 0 n_{blk+1}_{lane*5} 0 td=8p z0=20\n")
        else:
            for lane in range(8):
                f.write(f"T_OUT{lane} n_{blk}_{lane*5+1} 0 out_{lane} 0 td=6p z0=10\n")
                f.write(f"ROUT_{lane} out_{lane} 0 10\n")
    f.write(".tran 0.25p 625p 0 0.25p DST\n")
    for blk in range(0, n_blocks, 4):
        f.write(f".print P(B_{blk}_0) V(n_{blk}_1)\n")
    f.write(".end\n")

# 4. Large Windowed/Decimated Long-Transient Circuit (2,000 JJs, 4,000 steps, PRSTART=800p, PRSTEP=2p)
large_decim_cir = os.path.join(tmp_dir, "large_2k_jj_windowed.cir")
with open(large_decim_cir, "w") as f:
    f.write("* Large 2,000-JJ Circuit with Long Conditioning Window (PRSTART=800p, PRSTEP=2p)\n")
    f.write(".model jmitll jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160, rN=16, icrit=0.1mA)\n")
    f.write("VIN 1 0 sin(0 0.8m 20G 5p 0)\n")
    f.write("RIN 1 2 2\n")
    n_stages = 2000
    for s in range(1, n_stages + 1):
        n_in = s + 1
        n_out = s + 2
        n_jj = 100000 + s
        f.write(f"L_{s} {n_in} {n_out} 2.0p\n")
        f.write(f"B_{s} {n_out} {n_jj} jmitll area=2.0\n")
        f.write(f"LP_{s} {n_jj} 0 0.08p\n")
        f.write(f"RS_{s} {n_out} 0 5.0\n")
        f.write(f"IB_{s} 0 {n_out} pwl(0 0 5p 140u)\n")
    f.write(f"ROUT {n_stages + 2} 0 2\n")
    f.write(".tran 0.25p 1000p 800p 2.0p DST\n")
    for s in range(1, n_stages + 1, 4):
        f.write(f".print V({s+1}) P(B_{s})\n")
    f.write(".end\n")

# 5. Very Large Waveform Export (2,500 JJs, 1,250 saved traces, 2,000 steps = 2.5M output points)
vlarge_io_cir = os.path.join(tmp_dir, "vlarge_2500jj_1250traces.cir")
with open(vlarge_io_cir, "w") as f:
    f.write("* Very Large 2,500-JJ Circuit with 1,250 Output Traces x 2,000 Steps\n")
    f.write(".model jmitll jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160, rN=16, icrit=0.1mA)\n")
    f.write("VIN 1 0 pwl(0 0 10p 0 12p 600u 14p 0)\nRIN 1 2 2\n")
    for s in range(1, 2501):
        f.write(f"L_{s} {s+1} {s+2} 2.0p\nB_{s} {s+2} {100000+s} jmitll area=2.0\nLP_{s} {100000+s} 0 0.08p\nRS_{s} {s+2} 0 5.0\nIB_{s} 0 {s+2} pwl(0 0 5p 140u)\n")
    f.write("ROUT 2502 0 2\n.tran 0.25p 500p 0 0.25p DST\n")
    for s in range(1, 2501, 2):
        f.write(f".print V({s+2})\n")
    f.write(".end\n")

# 6. Very Large PTL Interconnect Network (512 PTLs over 10,000 steps)
vlarge_ptl_cir = os.path.join(tmp_dir, "vlarge_512ptl_10ksteps.cir")
with open(vlarge_ptl_cir, "w") as f:
    f.write("* 512 PTL Interconnect Network over 10,000 Timesteps\n")
    f.write("VIN 1 0 sin(0 1m 10G 0 0)\nR1 1 2 10\n")
    for k in range(1, 513):
        f.write(f"T_{k} 2 0 {100+k} 0 td=5p z0=10\nR_{k} {100+k} 0 10\n")
    f.write(".tran 0.25p 2500p 0 0.25p DST\n.print V(2) V(101) V(612)\n.end\n")

# ---------------------------------------------------------------------------
# Run Benchmark Suite
# ---------------------------------------------------------------------------
suite = [
    ("ex_ksa4bit (Med, 4-bit Adder)",       os.path.join(test_dir, "ex_ksa4bit.cir"),       os.path.join(test_dir, "ex_ksa4bit.cir"),       3),
    ("ex_jtl_string (Med, JTL Chain)",      os.path.join(test_dir, "ex_jtl_string.cir"),    os.path.join(test_dir, "ex_jtl_string.cir"),    3),
    ("ex_gen_pp_ipht (Med, IPHT Logic)",    os.path.join(test_dir, "ex_gen_pp_ipht.cir"),   os.path.join(test_dir, "ex_gen_pp_ipht.cir"),   3),
    ("med_ext_6k_K (Med, 2.5k L + 6k K)",   med_ext_cir,                                    med_ext_kcut_cir,                               2),
    ("large_10k_jtl (Large, 10k JJs)",      large_10k_cir,                                  large_10k_cir,                                  2),
    ("large_64blk_ptl (Large, 64 PTL Blk)", large_ptl_cir,                                  large_ptl_cir,                                  2),
    ("large_2k_windowed (Large, 4k steps)", large_decim_cir,                                large_decim_cir,                                2),
    ("vlarge_io_2.5Mpts (2.5k JJ, 1250tr)", vlarge_io_cir,                                  vlarge_io_cir,                                  1),
    ("vlarge_512ptl_10k (512 PTL, 10k st)", vlarge_ptl_cir,                                 vlarge_ptl_cir,                                 1),
]

print("\n=== WALL-CLOCK RUNTIME BENCHMARK (SECONDS & SPEEDUP VS UPSTREAM CSV) ===")
hdr = f"{'Design':<35} | {'Base CSV':>9} | {'Enh 1T CSV':>10} | {'Enh -p CSV':>10} | {'Enh -p BIN':>10} | {'Speedup CSV':>11} | {'Speedup BIN':>11}"
print(hdr)
print("-" * len(hdr))

rss_rows = []
for label, base_cir, opt_cir, n_runs in suite:
    out_base = os.path.join(tmp_dir, "out_base.csv")
    out_opt1 = os.path.join(tmp_dir, "out_opt1.csv")
    out_optp = os.path.join(tmp_dir, "out_optp.csv")
    out_bin  = os.path.join(tmp_dir, "out_optp.bin")

    t_base, rss_base = run_bench([base_bin, "-m", "-o", out_base, base_cir], runs=n_runs)
    t_opt1, rss_opt1 = run_bench([opt_bin,  "-m", "-o", out_opt1, base_cir], runs=n_runs)
    t_optp, rss_optp = run_bench([opt_bin,  "-m", "-p", "-o", out_optp, opt_cir], runs=n_runs)
    t_bin,  rss_bin  = run_bench([opt_bin,  "-m", "-p", "-o", out_bin,  opt_cir], runs=n_runs)

    sp_csv = t_base / t_optp
    sp_bin = t_base / t_bin
    print(f"{label:<35} | {t_base:>8.3f}s | {t_opt1:>9.3f}s | {t_optp:>9.3f}s | {t_bin:>9.3f}s | {sp_csv:>10.2f}x | {sp_bin:>10.2f}x")
    rss_rows.append((label, rss_base, rss_bin, rss_base / max(rss_bin, 0.1)))

print("\n=== PEAK MEMORY (RSS) BENCHMARK ===")
hdr_m = f"{'Design':<35} | {'Upstream Peak RSS':>18} | {'Enhanced Peak RSS':>18} | {'RAM Reduction':>14}"
print(hdr_m)
print("-" * len(hdr_m))
for label, r_base, r_opt, ratio in rss_rows:
    print(f"{label:<35} | {r_base:>15.1f} MB | {r_opt:>15.1f} MB | {ratio:>13.2f}x")
PY
