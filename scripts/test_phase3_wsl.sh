#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.venvs/smartpix/bin:$PATH"
REPO_DIR="/mnt/c/dev/JoSIM"
JOSIM_BIN="$HOME/.cache/josim-build/josim-cli"
cmake --build "$HOME/.cache/josim-build" -j"$(nproc)" >/dev/null
TMP_DIR="$(mktemp -d /tmp/josim-phase3.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

python3 - "$REPO_DIR" "$JOSIM_BIN" "$TMP_DIR" <<'PY'
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

repo_dir = Path(sys.argv[1])
josim_bin = Path(sys.argv[2])
tmp_dir = Path(sys.argv[3])
sys.path.insert(0, str(repo_dir / "scripts"))
import josim_io

print("=== 1. Binary Output (.bin, .npy, .rawb) vs .csv Verification & Speed Benchmark ===")
# Benchmark on ex_ksa4bit.cir saving all node traces (~2,000 columns x 2,000 rows = ~32 MB binary / ~52 MB CSV)
deck_src = (repo_dir / "test" / "ex_ksa4bit.cir").read_text()
lines = []
for ln in deck_src.splitlines():
    u = ln.strip().upper()
    if u.startswith(".FILE") or u.startswith(".PRINT") or u.startswith(".PLOT"):
        continue
    if u.startswith(".TRAN"):
        lines.append(".tran 0.25p 500p 0 0.25p")
        continue
    lines.append(ln)
deck_path = tmp_dir / "bench_io.cir"
deck_path.write_text("\n".join(lines) + "\n")

formats = [
    ("CSV (.csv)", tmp_dir / "out.csv"),
    ("NumPy (.npy)", tmp_dir / "out.npy"),
    ("Binary (.bin)", tmp_dir / "out.bin"),
    ("SPICE RawBin (.rawb)", tmp_dir / "out.rawb"),
]

timings = {}
sizes = {}
for label, out_path in formats:
    t0 = time.perf_counter()
    subprocess.run(
        [str(josim_bin), "-m", "-p", "-o", str(out_path), str(deck_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    timings[label] = time.perf_counter() - t0
    sizes[label] = out_path.stat().st_size

# Measure pure Python load time + verify exact float64 match between .bin and .npy
t0 = time.perf_counter()
csv_data = josim_io.load_josim(tmp_dir / "out.csv")
t_load_csv = time.perf_counter() - t0

t0 = time.perf_counter()
npy_data = josim_io.load_josim(tmp_dir / "out.npy")
t_load_npy = time.perf_counter() - t0

t0 = time.perf_counter()
bin_data = josim_io.load_josim(tmp_dir / "out.bin")
t_load_bin = time.perf_counter() - t0

assert list(bin_data.keys()) == list(npy_data.keys()) == list(csv_data.keys()), "Column names mismatch!"
for k in bin_data:
    assert np.array_equal(bin_data[k], npy_data[k]), f"Binary mismatch between .bin and .npy on {k}"
    max_csv_diff = np.max(np.abs(bin_data[k] - csv_data[k]))
    assert max_csv_diff < 1e-4, f"Unexpected difference vs CSV on {k}: {max_csv_diff}"

print(f"Dataset: {len(bin_data)} columns x {len(bin_data['time'])} rows ({len(bin_data) * len(bin_data['time']):,} float64 values)")
print(f"{'Format':<22} | {'Sim+Write (s)':<14} | {'File Size (MB)':<15} | {'Python Load (ms)':<16}")
print("-" * 76)
print(f"{'CSV (.csv)':<22} | {timings['CSV (.csv)']:<14.4f} | {sizes['CSV (.csv)']/(1024*1024):<15.2f} | {t_load_csv*1e3:<16.2f}")
print(f"{'NumPy (.npy)':<22} | {timings['NumPy (.npy)']:<14.4f} | {sizes['NumPy (.npy)']/(1024*1024):<15.2f} | {t_load_npy*1e3:<16.2f}")
print(f"{'Binary (.bin)':<22} | {timings['Binary (.bin)']:<14.4f} | {sizes['Binary (.bin)']/(1024*1024):<15.2f} | {t_load_bin*1e3:<16.2f}")
print(f"{'SPICE RawBin (.rawb)':<22} | {timings['SPICE RawBin (.rawb)']:<14.4f} | {sizes['SPICE RawBin (.rawb)']/(1024*1024):<15.2f} | {'N/A':<16}")
print(f"-> End-to-end Sim+Write speedup (.bin vs .csv): {timings['CSV (.csv)'] / timings['Binary (.bin)']:.2f}x faster")
print(f"-> Python load speedup (.bin vs .csv): {t_load_csv / max(t_load_bin, 1e-9):.1f}x faster; .bin == .npy bit-exact (IEEE-754 float64)!\n")

print("=== 2. Fractional Transmission Line Delay Interpolation (.OPTION TXINTERP) ===")
# A pulse launched down a 1.75 ps matched 5-ohm transmission line simulated at tstep=0.1ps vs tstep=0.05ps
tx_template = """* Fractional TX delay test
{opt}
Vin 1 0 PWL(0 0 1p 0 2p 1m 3p 0)
Rin 1 2 5
T1 2 0 3 0 TD=1.75p Z0=5
Rload 3 0 5
.tran {tstep} 10p
.print DEVV Rload
.end
"""
for mode_label, opt_line in [("Default (rounded TD/h)", ""), ("Fractional (.OPTION TXINTERP)", ".option txinterp")]:
    d_coarse = tmp_dir / "tx_coarse.cir"
    d_fine = tmp_dir / "tx_fine.cir"
    b_coarse = tmp_dir / "tx_coarse.bin"
    b_fine = tmp_dir / "tx_fine.bin"
    d_coarse.write_text(tx_template.format(opt=opt_line, tstep="0.1p"))
    d_fine.write_text(tx_template.format(opt=opt_line, tstep="0.05p"))
    subprocess.run([str(josim_bin), "-a", "0", "-m", "-o", str(b_coarse), str(d_coarse)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([str(josim_bin), "-a", "0", "-m", "-o", str(b_fine), str(d_fine)], check=True, stdout=subprocess.DEVNULL)
    wc = josim_io.load_josim_bin(b_coarse)["V(RLOAD)"]
    wf = josim_io.load_josim_bin(b_fine)["V(RLOAD)"][::2]
    n = min(len(wc), len(wf))
    err = np.max(np.abs(wc[:n] - wf[:n]))
    print(f"  {mode_label:<32}: max |V_0.1ps - V_0.05ps| = {err:.6e} V")

print("\n=== 3. Predictor-Corrector JJ Iteration (.OPTION CORRECTOR) ===")
jj_template = """* JJ switching predictor vs corrector
{opt}
.model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=16, rn=1.6, icrit=0.1m)
Ib 0 1 PWL(0 0 5p 0.08m 10p 0.15m 30p 0.15m)
B1 1 0 jjmod area=1
Rshunt 1 0 2.0
.tran {tstep} 40p
.print DEVP B1
.end
"""
ref_deck = tmp_dir / "jj_ref.cir"
ref_bin = tmp_dir / "jj_ref.bin"
ref_deck.write_text(jj_template.format(opt=".option corrector=2", tstep="0.01p"))
subprocess.run([str(josim_bin), "-m", "-o", str(ref_bin), str(ref_deck)], check=True, stdout=subprocess.DEVNULL)
phi_ref = josim_io.load_josim_bin(ref_bin)["P(B1)"][::20]

for label, opt_line in [("Predictor only (default)", ""), ("Predictor-Corrector (.OPTION CORRECTOR=2)", ".option corrector=2")]:
    d = tmp_dir / "jj_test.cir"
    b = tmp_dir / "jj_test.bin"
    d.write_text(jj_template.format(opt=opt_line, tstep="0.2p"))
    subprocess.run([str(josim_bin), "-m", "-o", str(b), str(d)], check=True, stdout=subprocess.DEVNULL)
    phi = josim_io.load_josim_bin(b)["P(B1)"]
    n = min(len(phi), len(phi_ref))
    err = np.max(np.abs(phi[:n] - phi_ref[:n]))
    print(f"  {label:<42}: max |P_0.2ps - P_ref_0.01ps| = {err:.6e} rad")

print("\n=== 4. State Checkpointing (.SAVESTATE / .LOADSTATE) ===")
state_file = tmp_dir / "bias.state"
save_deck = tmp_dir / "save_state.cir"
load_deck = tmp_dir / "load_state.cir"
save_bin = tmp_dir / "save_out.bin"
load_bin = tmp_dir / "load_out.bin"

save_deck.write_text(f"""* Save biased state at end of conditioning
.model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=16, rn=1.6, icrit=0.1m)
Ib 0 1 PWL(0 0 10p 0.07m 50p 0.07m)
L1 1 2 2p
B1 2 0 jjmod area=1
.savestate {state_file}
.tran 0.1p 50p
.print DEVP B1
.end
""")
load_deck.write_text(f"""* Load biased state at t=0
.model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=16, rn=1.6, icrit=0.1m)
Ib 0 1 PWL(0 0.07m 50p 0.07m)
L1 1 2 2p
B1 2 0 jjmod area=1
.loadstate {state_file}
.tran 0.1p 10p
.print DEVP B1
.end
""")
subprocess.run([str(josim_bin), "-m", "-o", str(save_bin), str(save_deck)], check=True, stdout=subprocess.DEVNULL)
subprocess.run([str(josim_bin), "-m", "-o", str(load_bin), str(load_deck)], check=True, stdout=subprocess.DEVNULL)
p_end = josim_io.load_josim_bin(save_bin)["P(B1)"][-1]
p_start = josim_io.load_josim_bin(load_bin)["P(B1)"][0]
print(f"  Saved final phase P(B1) = {p_end:.8f} rad; Loaded initial phase P(B1) = {p_start:.8f} rad (diff = {abs(p_end - p_start):.3e} rad)")
assert abs(p_end - p_start) < 1e-6, "Loaded state did not match saved state!"
print("\nALL PHASE 3 VERIFICATION CHECKS PASSED!")
PY
