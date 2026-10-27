#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="/home/smandal/.cache/josim-build"
REPO_DIR="/mnt/c/dev/JoSIM"
JOSIM_BIN="${BUILD_DIR}/josim-cli"
TMP_DIR="/tmp/josim_phase4_test"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

echo "=== [1/4] Building JoSIM in WSL ==="
bash "${REPO_DIR}/scripts/build_wsl.sh"

echo ""
echo "=== [2/4] Testing TX Ring Buffers & Mid-Flight Checkpointing (.savestate/.loadstate) ==="
cat > "${TMP_DIR}/tx_full.cir" << 'EOF'
* Transmission line mid-flight pulse test (full 100ps run)
V1 1 0 pulse(0 1m 5p 1p 1p 5p 100p)
R1 1 2 50
T1 2 0 3 0 td=25p z0=50
R2 3 0 50
.tran 0.5p 80p 0 0.5p DST
.print V(2) V(3) I(T1)
.savestate /tmp/josim_phase4_test/tx_mid.state 20p
.end
EOF

cat > "${TMP_DIR}/tx_resume.cir" << 'EOF'
* Transmission line resume at 20ps while pulse is mid-flight inside T1 (td=25ps)
V1 1 0 dc 0
R1 1 2 50
T1 2 0 3 0 td=25p z0=50
R2 3 0 50
.loadstate /tmp/josim_phase4_test/tx_mid.state
.tran 0.5p 59.5p 0 0.5p DST
.print V(2) V(3) I(T1)
.end
EOF

"${JOSIM_BIN}" -m -o "${TMP_DIR}/tx_full.bin" "${TMP_DIR}/tx_full.cir"
"${JOSIM_BIN}" -m -o "${TMP_DIR}/tx_resume.bin" "${TMP_DIR}/tx_resume.cir"

python3 - << 'PY'
import sys
import numpy as np
sys.path.insert(0, "/mnt/c/dev/JoSIM/scripts")
from josim_io import load_josim_bin

full = load_josim_bin("/tmp/josim_phase4_test/tx_full.bin")
res  = load_josim_bin("/tmp/josim_phase4_test/tx_resume.bin")
# Map each output point to its exact simulation step index (to avoid Output::write_output prstep float rounding skips)
step_full = np.rint(full["time"] / 0.5e-12).astype(int)
step_res  = np.rint(res["time"] / 0.5e-12).astype(int) + 41
common_steps, idx_f, idx_r = np.intersect1d(step_full, step_res, return_indices=True)
# Compare after the 2-step Output::write_output BDF2 derivative warmup (idx_r >= 2)
mask = idx_r >= 2
idx_f = idx_f[mask]
idx_r = idx_r[mask]
v3_full = full["V(3)"][idx_f]
v3_res  = res["V(3)"][idx_r]
i1_full = full["I(T1)"][idx_f]
i1_res  = res["I(T1)"][idx_r]
max_err_v3 = np.max(np.abs(v3_full - v3_res))
max_err_i1 = np.max(np.abs(i1_full - i1_res))
print(f"Mid-flight TX pulse peak V(3) = {np.max(v3_res):.6e} V, max |V3_full - V3_resume| = {max_err_v3:.3e} V, max |I1_full - I1_resume| = {max_err_i1:.3e} A ({len(idx_r)} steps)")
assert np.max(v3_res) > 1e-4, "Pulse did not arrive after checkpoint!"
assert max_err_v3 < 1e-12, f"Mid-flight TX V(3) checkpoint mismatch: {max_err_v3}"
assert max_err_i1 < 1e-12, f"Mid-flight TX I(T1) checkpoint mismatch: {max_err_i1}"
PY

echo ""
echo "=== [3/4] Testing In-Loop Print Windowing & Decimation (PRSTART / PRSTEP) ==="
cat > "${TMP_DIR}/decim_all.cir" << 'EOF'
* Full output reference
V1 1 0 sin(0 1m 10G 0 0)
R1 1 2 5
L1 2 3 2p
B1 3 0 jjmod
.model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=160, rn=16, icrit=0.2m)
.tran 0.25p 200p 0 0.25p
.print V(2) V(B1) P(B1) I(L1)
.end
EOF

cat > "${TMP_DIR}/decim_sub.cir" << 'EOF'
* Windowed + decimated output (PRSTART=100p, PRSTEP=2p)
V1 1 0 sin(0 1m 10G 0 0)
R1 1 2 5
L1 2 3 2p
B1 3 0 jjmod
.model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=160, rn=16, icrit=0.2m)
.tran 0.25p 200p 100p 2p
.print V(2) V(B1) P(B1) I(L1)
.end
EOF

"${JOSIM_BIN}" -m -o "${TMP_DIR}/decim_all.bin" "${TMP_DIR}/decim_all.cir"
"${JOSIM_BIN}" -m -o "${TMP_DIR}/decim_sub.bin" "${TMP_DIR}/decim_sub.cir"

python3 - << 'PY'
import sys
import numpy as np
sys.path.insert(0, "/mnt/c/dev/JoSIM/scripts")
from josim_io import load_josim_bin

d_all = load_josim_bin("/tmp/josim_phase4_test/decim_all.bin")
d_sub = load_josim_bin("/tmp/josim_phase4_test/decim_sub.bin")
# Match timestamps from d_sub in d_all
t_all = d_all["time"]
t_sub = d_sub["time"]
idx = np.searchsorted(t_all, t_sub)
assert np.allclose(t_all[idx], t_sub, atol=1e-18), "Time axis mismatch"
for k in ["V(2)", "V(B1)", "P(B1)", "I(L1)"]:
    diff = np.max(np.abs(d_all[k][idx] - d_sub[k]))
    print(f"Decimated trace {k}: {len(t_sub)} pts vs {len(t_all)} full pts, max diff = {diff:.3e}")
    assert diff == 0.0, f"Bitwise mismatch on {k}: {diff}"
PY

echo ""
echo "=== [4/4] Testing Block-Partitioned MNA Solver Across PTLs & .option kcutoff ==="
python3 - << 'PY'
# Generate a 16-block PTL-coupled JTL pipeline with parasitic mutual inductances
lines = ["* 16-Block PTL-Coupled JTL Pipeline with Mutual Inductances"]
lines.append(".option kcutoff=0.01")
lines.append(".model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=160, rn=16, icrit=0.1m)")
lines.append("VIN in0 0 pwl(0 0 10p 0 12p 0.8m 14p 0)")

n_blocks = 16
stages_per_block = 8
for b in range(n_blocks):
    prev_node = f"in{b}"
    for s in range(stages_per_block):
        n_mid = f"n_{b}_{s}_m"
        n_jj  = f"n_{b}_{s}_j"
        ibias = f"IB_{b}_{s}"
        lines.append(f"L1_{b}_{s} {prev_node} {n_mid} 2p")
        lines.append(f"L2_{b}_{s} {n_mid} {n_jj} 2p")
        lines.append(f"B_{b}_{s} {n_jj} 0 jjmod area=2.0")
        lines.append(f"RB_{b}_{s} {n_jj} 0 5.0")
        lines.append(f"{ibias} 0 {n_mid} pwl(0 0 5p 140u)")
        # Strong local coupling + weak parasitic coupling (< 0.01 cutoff)
        lines.append(f"Kloc_{b}_{s} L1_{b}_{s} L2_{b}_{s} 0.05")
        if s > 0:
            lines.append(f"Kweak_{b}_{s} L1_{b}_{s} L1_{b}_{s-1} 0.002")
        prev_node = n_jj
    if b < n_blocks - 1:
        next_in = f"in{b+1}"
        lines.append(f"Rtx1_{b} {prev_node} txin_{b} 2.0")
        lines.append(f"T_{b} txin_{b} 0 {next_in} 0 td=5p z0=10")
        lines.append(f"Rtx2_{b} {next_in} 0 10.0")
    else:
        lines.append(f"ROUT {prev_node} 0 5.0")

lines.append(".tran 0.25p 400p 0 0.25p")
lines.append(f".print P(B_0_0) P(B_{n_blocks-1}_{stages_per_block-1})")
lines.append(".end")
with open("/tmp/josim_phase4_test/ptl_blocks.cir", "w") as f:
    f.write("\n".join(lines) + "\n")
PY

"${JOSIM_BIN}" -V 1 -o "${TMP_DIR}/ptl_serial.bin" "${TMP_DIR}/ptl_blocks.cir" | tee "${TMP_DIR}/ptl_verb.log"
"${JOSIM_BIN}" -p -m -o "${TMP_DIR}/ptl_parallel.bin" "${TMP_DIR}/ptl_blocks.cir"

grep -q "Partitioned MNA matrix" "${TMP_DIR}/ptl_verb.log"
grep -q "Pruned" "${TMP_DIR}/ptl_verb.log"

python3 - << 'PY'
import sys
import numpy as np
sys.path.insert(0, "/mnt/c/dev/JoSIM/scripts")
from josim_io import load_josim_bin

s = load_josim_bin("/tmp/josim_phase4_test/ptl_serial.bin")
p = load_josim_bin("/tmp/josim_phase4_test/ptl_parallel.bin")
for k in s:
    diff = np.max(np.abs(s[k] - p[k]))
    print(f"Partitioned Serial vs OpenMP Parallel trace {k}: max diff = {diff:.3e}")
    assert diff < 1e-12, f"Mismatch between serial and parallel block KLU solve on {k}: {diff}"
PY

echo ""
echo "=== [5/5] Testing Tier 2 Batched Monte Carlo (.spread) Solver (-b 32 Serial vs OpenMP) ==="
cat > "${TMP_DIR}/mc_test.cir" << 'EOF'
* Tier 2 Batched Monte Carlo JTL test (32 runs)
.spread l=0.10 r=0.10 c=0.10 b=0.05 runs=32
.model jjmod jj(rtype=1, vg=2.8m, cap=0.07p, r0=160, rn=16, icrit=0.12m)
VIN 1 0 pwl(0 0 10p 0 12p 0.7m 14p 0)
L1 1 2 2p
B1 2 0 jjmod area=2.0
R1 2 0 4.5
L2 2 3 2p
B2 3 0 jjmod area=2.0
R2 3 0 4.5
C2 3 0 0.05p
IB1 0 2 pwl(0 0 5p 280u)
IB2 0 3 pwl(0 0 5p 280u)
.tran 0.25p 100p 0 0.5p
.print V(2) V(3) P(B2)
.end
EOF

JOSIM_SEED=42 "${JOSIM_BIN}" -m -o "${TMP_DIR}/mc_ser.bin" "${TMP_DIR}/mc_test.cir"
JOSIM_SEED=42 "${JOSIM_BIN}" -p -m -o "${TMP_DIR}/mc_omp.bin" "${TMP_DIR}/mc_test.cir"

python3 - << 'PY'
import sys
import numpy as np
sys.path.insert(0, "/mnt/c/dev/JoSIM/scripts")
from josim_io import load_josim_bin

s = load_josim_bin("/tmp/josim_phase4_test/mc_ser.bin")
p = load_josim_bin("/tmp/josim_phase4_test/mc_omp.bin")
assert "V(3)_STD" in s and "V(3)_MIN" in s and "V(3)_MAX" in s, f"Missing MC ensemble columns: {list(s.keys())}"
assert np.max(s["V(3)_STD"]) > 1e-6, "Expected non-zero standard deviation across 32 MC runs!"
for k in s:
    diff = np.max(np.abs(s[k] - p[k]))
    print(f"Batched MC (B=32) Serial vs OpenMP trace {k}: max diff = {diff:.3e}")
    assert diff < 1e-13, f"Batched MC mismatch on {k}: {diff}"
PY

echo ""
echo "=== ALL PHASE 4 & TIER 2 MONTE CARLO TESTS PASSED SUCCESSFULLY ==="
