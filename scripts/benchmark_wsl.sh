#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.venvs/smartpix/bin:/opt/venv/bin:$PATH"

OPT_BUILD="$HOME/.cache/josim-build"
BASE_BIN="$HOME/.cache/josim-baseline-cli"
SRC_DIR="/mnt/c/dev/JoSIM"

# Build optimized JoSIM first
bash "$SRC_DIR/scripts/build_wsl.sh" "$OPT_BUILD" "$SRC_DIR"

python3 - << 'EOF'
import subprocess
import time
import os

base_bin = os.path.expanduser("~/.cache/josim-baseline-cli")
opt_bin = os.path.expanduser("~/.cache/josim-build/josim-cli")
test_dir = "/mnt/c/dev/JoSIM/test"

# Generate a synthetic large 10,000-JJ JTL benchmark circuit in /tmp/synth_10k_jtl.cir
synth_path = "/tmp/synth_10k_jtl.cir"
with open(synth_path, "w") as f:
    f.write("* Synthetic 10,000-JJ JTL Chain\n")
    f.write(".model jmitll jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160, rN=16, icrit=0.1mA)\n")
    f.write("VIN 1 0 pwl(0 0 10p 0 12p 600u 14p 0)\n")
    f.write("RIN 1 2 2\n")
    n_stages = 10000
    for s in range(1, n_stages + 1):
        n_in = s + 1
        n_out = s + 2
        n_jj = 100000 + s
        n_sh = 200000 + s
        n_ib = 300000 + s
        f.write(f"L_{s} {n_in} {n_out} 2.0p\n")
        f.write(f"B_{s} {n_out} {n_jj} jmitll area=2.0\n")
        f.write(f"LP_{s} {n_jj} 0 0.08p\n")
        f.write(f"RS_{s} {n_out} {n_sh} 5.0\n")
        f.write(f"LS_{s} {n_sh} 0 0.08p\n")
        f.write(f"IB_{s} 0 {n_out} pwl(0 0 5p 140u)\n")
    f.write(f"ROUT {n_stages + 2} 0 2\n")
    f.write(".tran 0.25p 50p 0 0.25p\n")
    for p in range(1, 33):
        idx = p * (n_stages // 32)
        f.write(f".print DEVV B_{idx}\n")
    f.write(".end\n")

circuits = [
    (os.path.join(test_dir, "ex_gen_pp_ipht.cir"), "ex_gen_pp_ipht.cir", "-m"),
    (os.path.join(test_dir, "ex_ksa4bit.cir"), "ex_ksa4bit.cir", "-m"),
    (os.path.join(test_dir, "ex_jtl_string.cir"), "ex_jtl_string.cir", "-m"),
    (os.path.join(test_dir, "ex_jtl_basic.cir"), "ex_jtl_basic.cir", "-m"),
    (synth_path, "synth_10k_jtl.cir", "-m"),
]

print(f"\n{'Circuit':<22} | {'Baseline (s)':<12} | {'Opt 1T (s)':<12} | {'Opt -p (s)':<12} | {'Speedup (1T)':<12} | {'Speedup (-p)':<12} | {'Equivalence'}")
print("-" * 112)

for cir_path, cir_name, flag in circuits:
    out_base = f"/tmp/out_base_{cir_name}.csv"
    out_opt1 = f"/tmp/out_opt1_{cir_name}.csv"
    out_optp = f"/tmp/out_optp_{cir_name}.csv"

    def bench(cmd, runs=3):
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            subprocess.run(cmd, cwd="/tmp", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            times.append(time.perf_counter() - t0)
        return min(times)

    t_base = bench([base_bin, flag, "-o", out_base, cir_path])
    t_opt1 = bench([opt_bin, flag, "-o", out_opt1, cir_path])
    t_optp = bench([opt_bin, flag, "-p", "-o", out_optp, cir_path])

    with open(out_base, "rb") as f1, open(out_opt1, "rb") as f2, open(out_optp, "rb") as f3:
        b_base = f1.read()
        b_opt1 = f2.read()
        b_optp = f3.read()

    if b_base == b_opt1 and b_base == b_optp:
        diff_status = "BIT-IDENTICAL"
    else:
        diff_status = f"DIFF (1T:{b_base==b_opt1}, -p:{b_base==b_optp})"

    print(f"{cir_name:<22} | {t_base:<12.4f} | {t_opt1:<12.4f} | {t_optp:<12.4f} | {t_base/t_opt1:<11.2f}x | {t_base/t_optp:<11.2f}x | {diff_status}")
EOF
