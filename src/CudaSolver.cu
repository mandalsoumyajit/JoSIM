// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)
#ifdef JOSIM_CUDA

#include "JoSIM/CudaSolver.hpp"
#include "JoSIM/Constants.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

namespace JoSIM {

namespace {

#define JOSIM_CUDA_CHECK(call)                                                                     \
    do {                                                                                           \
        cudaError_t err__ = (call);                                                                \
        if (err__ != cudaSuccess) {                                                                \
            std::cerr << "JoSIM CUDA Error at " << __FILE__ << ":" << __LINE__ << " - "            \
                      << cudaGetErrorString(err__) << std::endl;                                   \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

template <typename T>
bool alloc_and_upload(T** d_ptr, const std::vector<T>& h_vec) {
    if (h_vec.empty()) {
        *d_ptr = nullptr;
        return true;
    }
    const size_t bytes = h_vec.size() * sizeof(T);
    JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(d_ptr), bytes));
    JOSIM_CUDA_CHECK(cudaMemcpy(*d_ptr, h_vec.data(), bytes, cudaMemcpyHostToDevice));
    return true;
}

template <typename T>
void safe_cuda_free(T*& d_ptr) {
    if (d_ptr != nullptr) {
        cudaFree(d_ptr);
        d_ptr = nullptr;
    }
}

__device__ __forceinline__ double fetch_delayed_val(const GpuTxHistEntry* __restrict__ d_txHist,
                                                     const GpuTxLine& tx,
                                                     int64_t          stepIdx,
                                                     int32_t          which) {
    const int64_t delayedStep = tx.loadedSteps + stepIdx - tx.k;
    if (delayedStep < 0) { return 0.0; }
    const GpuTxHistEntry e = d_txHist[tx.histOffset + (delayedStep % tx.histSize)];
    double               v = (which == 0) ? e.v1 : ((which == 1) ? e.v2 : ((which == 2) ? e.i1 : e.i2));
    if (tx.fracDelay > 0.0) {
        double vPrev = 0.0;
        if (delayedStep >= 1) {
            const GpuTxHistEntry ep = d_txHist[tx.histOffset + ((delayedStep - 1) % tx.histSize)];
            vPrev = (which == 0) ? ep.v1 : ((which == 1) ? ep.v2 : ((which == 2) ? ep.i1 : ep.i2));
        }
        v = (1.0 - tx.fracDelay) * v + tx.fracDelay * vPrev;
    }
    return v;
}

// Precompute remote transmission line delayed history terms for all steps s in [0, w)
__global__ void precompute_tx_remote_kernel(int32_t               numTxLines,
                                            int64_t               i0,
                                            int32_t               w,
                                            double                stepSize,
                                            int32_t               analysisType,
                                            const GpuTxLine*      d_txLines,
                                            const GpuTxHistEntry* d_txHist,
                                            double*               d_txRemote1,
                                            double*               d_txRemote2) {
    const int32_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t total = numTxLines * w;
    if (tid >= total) { return; }

    const int32_t   tIdx = tid / w;
    const int32_t   s    = tid % w;
    const int64_t   i    = i0 + s;
    const GpuTxLine tx   = d_txLines[tIdx];
    const double    Z0   = tx.z0;

    if (analysisType == 0) {
        const double v1k         = fetch_delayed_val(d_txHist, tx, i, 0);
        const double v2k         = fetch_delayed_val(d_txHist, tx, i, 1);
        const double i1k         = fetch_delayed_val(d_txHist, tx, i, 2);
        const double i2k         = fetch_delayed_val(d_txHist, tx, i, 3);
        d_txRemote1[tIdx * w + s] = Z0 * i2k + v2k;
        d_txRemote2[tIdx * w + s] = Z0 * i1k + v1k;
    } else {
        const double c2          = (2.0 * stepSize * Z0) / (3.0 * Constants::SIGMA);
        const double p1k         = fetch_delayed_val(d_txHist, tx, i, 0);
        const double p2k         = fetch_delayed_val(d_txHist, tx, i, 1);
        const double p1k1        = fetch_delayed_val(d_txHist, tx, i - 1, 0);
        const double p2k1        = fetch_delayed_val(d_txHist, tx, i - 1, 1);
        const double p1k2        = fetch_delayed_val(d_txHist, tx, i - 2, 0);
        const double p2k2        = fetch_delayed_val(d_txHist, tx, i - 2, 1);
        const double i1k         = fetch_delayed_val(d_txHist, tx, i, 2);
        const double i2k         = fetch_delayed_val(d_txHist, tx, i, 3);
        d_txRemote1[tIdx * w + s] = c2 * i2k + p2k - (4.0 / 3.0) * p2k1 + (1.0 / 3.0) * p2k2;
        d_txRemote2[tIdx * w + s] = c2 * i1k + p1k - (4.0 / 3.0) * p1k1 + (1.0 / 3.0) * p1k2;
    }
}

// Warp-per-partition kernel with on-chip shared memory LU + working vectors and register-cached devices
__global__ void advance_partitions_window_kernel(int32_t             nParts,
                                                 int32_t             maxPartN,
                                                 int32_t             maxPartLuNnz,
                                                 int32_t             shDoublesPerWarp,
                                                 int32_t             useShared,
                                                 int64_t             i0,
                                                 int32_t             w,
                                                 int32_t             numSources,
                                                 int32_t             numKeptSteps,
                                                 int64_t             simSize,
                                                 double              stepSize,
                                                 int32_t             analysisType,
                                                 const GpuPartDesc*  d_parts,
                                                 const int32_t*      d_P,
                                                 const int32_t*      d_Q,
                                                 const int32_t*      d_luRp,
                                                 const int32_t*      d_luDiag,
                                                 const int32_t*      d_luCi,
                                                 const int32_t*      d_luSrcNz,
                                                 double*             d_luVal,
                                                 double*             d_partNz,
                                                 double*             d_xLocal,
                                                 double*             d_bLocal,
                                                 double*             d_yLocal,
                                                 GpuJJ*              d_jjs,
                                                 GpuRes*             d_ress,
                                                 GpuInd*             d_inds,
                                                 const GpuMut*       d_muts,
                                                 GpuCap*             d_caps,
                                                 GpuVS*              d_vss,
                                                 const GpuPS*        d_pss,
                                                 const GpuCS*        d_css,
                                                 GpuCCVS*            d_ccvss,
                                                 GpuVCCS*            d_vccss,
                                                 GpuTxHalf*          d_txHalves,
                                                 const GpuTxLine*    d_txLines,
                                                 GpuTxHistEntry*     d_txHist,
                                                 const double*       d_txRemote1,
                                                 const double*       d_txRemote2,
                                                 const double*       d_srcWindow,
                                                 const GpuRelTrace*  d_relTraces,
                                                 const int32_t*      d_keptSlot,
                                                 double*             d_traceOut,
                                                 int32_t*            d_needsTR) {
    extern __shared__ double s_mem[];

    const int32_t warpInBlock   = threadIdx.x >> 5;
    const int32_t lane          = threadIdx.x & 31;
    const int32_t warpsPerBlock = blockDim.x >> 5;
    const int32_t p             = blockIdx.x * warpsPerBlock + warpInBlock;
    if (p >= nParts) { return; }

    const GpuPartDesc part   = d_parts[p];
    const int32_t     n      = part.n;
    const int32_t     rowOff = part.rowStart;
    const int32_t     rpOff  = part.rpStart;
    const int32_t     luOff  = part.luStart;

    double*  s_base   = (useShared > 0) ? (s_mem + warpInBlock * shDoublesPerWarp) : nullptr;
    double*  sx       = (useShared > 0) ? s_base : nullptr;
    double*  sb       = (useShared > 0) ? (sx + maxPartN) : nullptr;
    double*  sy       = (useShared > 0) ? (sb + maxPartN) : nullptr;
    double*  s_luVal  = (useShared == 2) ? (sy + maxPartN) : nullptr;
    int32_t* s_luCi   = (useShared == 2) ? reinterpret_cast<int32_t*>(s_luVal + maxPartLuNnz) : nullptr;
    int32_t* s_luRp   = (useShared == 2) ? (s_luCi + maxPartLuNnz) : nullptr;
    int32_t* s_luDiag = (useShared == 2) ? (s_luRp + maxPartN + 1) : nullptr;
    int32_t* s_P      = (useShared == 2) ? (s_luDiag + maxPartN) : nullptr;
    int32_t* s_Q      = (useShared == 2) ? (s_P + maxPartN) : nullptr;

    double*        xPtr      = (useShared > 0) ? (sx - rowOff) : d_xLocal;
    double*        bPtr      = (useShared > 0) ? (sb - rowOff) : d_bLocal;
    double*        yPtr      = (useShared > 0) ? (sy - rowOff) : d_yLocal;
    double*        luValPtr  = (useShared == 2) ? s_luVal : (d_luVal + luOff);
    const int32_t* luCiPtr   = (useShared == 2) ? s_luCi : (d_luCi + luOff);
    const int32_t* luRpPtr   = (useShared == 2) ? s_luRp : (d_luRp + rpOff);
    const int32_t* luDiagPtr = (useShared == 2) ? s_luDiag : (d_luDiag + rowOff);
    const int32_t* pPtr      = (useShared == 2) ? s_P : (d_P + rowOff);
    const int32_t* qPtr      = (useShared == 2) ? s_Q : (d_Q + rowOff);

    if (useShared > 0) {
        for (int32_t r = lane; r < n; r += 32) { sx[r] = d_xLocal[rowOff + r]; }
    }
    if (useShared == 2) {
        for (int32_t r = lane; r < n; r += 32) {
            s_luRp[r]   = d_luRp[rpOff + r];
            s_luDiag[r] = d_luDiag[rowOff + r];
            s_P[r]      = d_P[rowOff + r];
            s_Q[r]      = d_Q[rowOff + r];
        }
        if (lane == 0) { s_luRp[n] = d_luRp[rpOff + n]; }
        for (int32_t idx = lane; idx < part.luNnz; idx += 32) {
            s_luVal[idx] = d_luVal[luOff + idx];
            s_luCi[idx]  = d_luCi[luOff + idx];
        }
    }
    if (useShared > 0) { __syncwarp(); }

    // Cache scalar JJ state in registers when partition has <= 32 JJs (1 JJ per warp lane)
    const bool regJJMode = (part.jjCount <= 32);
    const bool hasRegJJ  = regJJMode && (lane < part.jjCount);
    int32_t    j_posIdx = -1, j_negIdx = -1, j_curIdx = -1, j_varIdx = -1, j_nzIdx = -1;
    int32_t    j_rtype = 1, j_tDep = 0, j_cprLen = 1;
    double     j_cpr0 = 1.0, j_ic = 0.0, j_c = 0.0, j_r0 = 0.0, j_rn = 0.0, j_phiOff = 0.0;
    double     j_t = 0.0, j_d = 0.0, j_del = 0.0, j_lowerB = 0.0, j_upperB = 0.0, j_gLarge = 0.0;
    double     j_negInvSubgap = 0.0, j_negInvTrans = 0.0, j_negInvNormal = 0.0;
    double     j_pn1 = 0.0, j_pn2 = 0.0, j_pn3 = 0.0, j_pn4 = 0.0, j_phi0 = 0.0;
    double     j_vn1 = 0.0, j_vn2 = 0.0, j_vn3 = 0.0, j_vn4 = 0.0, j_vn5 = 0.0, j_vn6 = 0.0;
    double     j_it = 0.0, j_curNzVal = 0.0, j_cTerm = 0.0;

    if (hasRegJJ) {
        const GpuJJ jj = d_jjs[part.jjStart + lane];
        j_posIdx       = jj.posIdx;
        j_negIdx       = jj.negIdx;
        j_curIdx       = jj.curIdx;
        j_varIdx       = jj.varIdx;
        j_nzIdx        = jj.nzIdx;
        j_rtype        = jj.rtype;
        j_tDep         = jj.tDep;
        j_cprLen       = jj.cprLen;
        j_cpr0         = jj.cpr[0];
        j_ic           = jj.ic;
        j_c            = jj.c;
        j_r0           = jj.r0;
        j_rn           = jj.rn;
        j_phiOff       = jj.phiOff;
        j_t            = jj.t;
        j_d            = jj.d;
        j_del          = jj.del;
        j_lowerB       = jj.lowerB;
        j_upperB       = jj.upperB;
        j_gLarge       = jj.gLarge;
        j_negInvSubgap = jj.negInvSubgap;
        j_negInvTrans  = jj.negInvTrans;
        j_negInvNormal = jj.negInvNormal;
        j_pn1          = jj.pn1;
        j_pn2          = jj.pn2;
        j_pn3          = jj.pn3;
        j_pn4          = jj.pn4;
        j_phi0         = jj.phi0;
        j_vn1          = jj.vn1;
        j_vn2          = jj.vn2;
        j_vn3          = jj.vn3;
        j_vn4          = jj.vn4;
        j_vn5          = jj.vn5;
        j_vn6          = jj.vn6;
        j_it           = jj.it;
        j_curNzVal     = jj.curNzVal;
        j_cTerm        = jj.cTerm;
    }

    // Cache up to 2 resistors per lane in registers when analysisType == 1 and part.resCount <= 64
    const bool regResMode = (analysisType == 1 && part.resCount <= 64);
    const bool hasRes0    = regResMode && (lane < part.resCount);
    const bool hasRes1    = regResMode && (lane + 32 < part.resCount);
    int32_t    r0_pos = -1, r0_neg = -1, r0_cur = -1;
    double     r0_pn1 = 0.0, r0_pn2 = 0.0;
    int32_t    r1_pos = -1, r1_neg = -1, r1_cur = -1;
    double     r1_pn1 = 0.0, r1_pn2 = 0.0;
    if (hasRes0) {
        const GpuRes r = d_ress[part.resStart + lane];
        r0_pos = r.posIdx; r0_neg = r.negIdx; r0_cur = r.curIdx; r0_pn1 = r.pn1; r0_pn2 = r.pn2;
    }
    if (hasRes1) {
        const GpuRes r = d_ress[part.resStart + lane + 32];
        r1_pos = r.posIdx; r1_neg = r.negIdx; r1_cur = r.curIdx; r1_pn1 = r.pn1; r1_pn2 = r.pn2;
    }

    // Cache PTL port halves in registers when part.txHalfCount <= 32
    const bool regTxMode = (part.txHalfCount <= 32);
    const bool hasRegTx  = regTxMode && (lane < part.txHalfCount);
    int32_t    tx_idx = 0, tx_isPort2 = 0, tx_pos = -1, tx_neg = -1, tx_cur = -1;
    double     tx_n1 = 0.0, tx_n2 = 0.0;
    if (hasRegTx) {
        const GpuTxHalf txh = d_txHalves[part.txHalfStart + lane];
        tx_idx     = txh.txIdx;
        tx_isPort2 = txh.isPort2;
        tx_pos     = txh.posIdx;
        tx_neg     = txh.negIdx;
        tx_cur     = txh.curIdx;
        tx_n1      = txh.n1;
        tx_n2      = txh.n2;
    }

    bool anyRefactoredLU = false;

    for (int32_t s = 0; s < w; ++s) {
        if (*d_needsTR != 0) { return; }
        const int64_t i          = i0 + s;
        const bool    checkPhase = (static_cast<double>(i) / static_cast<double>(simSize) > 0.01);

        // 1. Clear partition RHS vector b
        for (int32_t r = lane; r < n; r += 32) { bPtr[rowOff + r] = 0.0; }
        __syncwarp();

        // 2. Handle JJs in parallel across the 32 warp lanes
        bool laneNeedsLU = false;
        bool laneNeedsTR = false;
        if (regJJMode) {
            if (hasRegJJ) {
                if (j_posIdx >= 0 && j_negIdx < 0) {
                    j_pn1 = xPtr[j_posIdx];
                } else if (j_posIdx < 0 && j_negIdx >= 0) {
                    j_pn1 = -xPtr[j_negIdx];
                } else if (j_posIdx >= 0 && j_negIdx >= 0) {
                    j_pn1 = xPtr[j_posIdx] - xPtr[j_negIdx];
                } else {
                    j_pn1 = 0.0;
                }
                if (i > 0) {
                    if (analysisType == 0) {
                        j_vn1 = j_pn1;
                        j_pn1 = xPtr[j_varIdx];
                    } else {
                        j_vn1 = xPtr[j_varIdx];
                    }
                }
                const double v0 = (5.0 / 2.0) * j_vn1 - 2.0 * j_vn2 + (1.0 / 2.0) * j_vn3;
                j_phi0          = (4.0 / 3.0) * j_pn1 - (1.0 / 3.0) * j_pn2
                         + ((1.0 / Constants::SIGMA) * ((2.0 * stepSize) / 3.0)) * v0;
                if (checkPhase && fabs(j_phi0 - j_pn1) > (0.20 * 2.0 * Constants::PI)) {
                    laneNeedsTR = true;
                }
                if (analysisType == 0) {
                    bPtr[j_varIdx] = Constants::SIGMA
                                     * (-(2.0 / stepSize) * j_pn1 + (1.0 / (2.0 * stepSize)) * j_pn2);
                } else {
                    bPtr[j_varIdx] = (4.0 / 3.0) * j_pn1 - (1.0 / 3.0) * j_pn2;
                }
                j_pn4 = j_pn3;
                j_pn3 = j_pn2;
                j_pn2 = j_pn1;
                j_vn6 = j_vn5;
                j_vn5 = j_vn4;
                j_vn4 = j_vn3;
                j_vn3 = j_vn2;

                if (j_rtype == 1) {
                    bool   updated = false;
                    double absV    = fabs(v0);
                    if (absV < j_lowerB) {
                        j_it = 0.0;
                        if (j_curNzVal != j_negInvSubgap) {
                            j_curNzVal = j_negInvSubgap;
                            updated    = true;
                        }
                    } else if (absV < j_upperB) {
                        j_it = j_lowerB * ((1.0 / j_r0) - j_gLarge);
                        if (v0 < 0.0) { j_it = -j_it; }
                        if (j_curNzVal != j_negInvTrans) {
                            j_curNzVal = j_negInvTrans;
                            updated    = true;
                        }
                    } else {
                        j_it = 0.0;
                        if (j_curNzVal != j_negInvNormal) {
                            j_curNzVal = j_negInvNormal;
                            updated    = true;
                        }
                    }
                    if (updated) {
                        d_partNz[j_nzIdx] = j_curNzVal;
                        laneNeedsLU       = true;
                    }
                }

                double ic_sin_phi = 0.0;
                if (j_cprLen == 1) {
                    ic_sin_phi += j_ic * (j_cpr0 * sin(1.0 * (j_phi0 - j_phiOff)));
                } else {
                    const GpuJJ& gjj = d_jjs[part.jjStart + lane];
                    for (int32_t harm = 0; harm < j_cprLen; ++harm) {
                        ic_sin_phi += j_ic * (gjj.cpr[harm] * sin((harm + 1) * (j_phi0 - j_phiOff)));
                    }
                }
                j_cTerm = -(((2.0 * j_c) / stepSize) * j_vn1) + ((j_c / (2.0 * stepSize)) * j_vn2);
                if (!j_tDep) {
                    bPtr[j_curIdx] = j_curNzVal
                                     * (ic_sin_phi - (((2.0 * j_c) / stepSize) * j_vn1)
                                        + ((j_c / (2.0 * stepSize)) * j_vn2) + j_it);
                } else {
                    const GpuJJ& gjj           = d_jjs[part.jjStart + lane];
                    double       sin2_half_phi = 0.0;
                    for (int32_t harm = 0; harm < j_cprLen; ++harm) {
                        sin2_half_phi += gjj.cpr[harm] * sin((harm + 1) * (j_phi0 - j_phiOff) / 2.0);
                    }
                    sin2_half_phi  = sin2_half_phi * sin2_half_phi;
                    double sin_phi = 0.0;
                    for (int32_t harm = 0; harm < j_cprLen; ++harm) {
                        sin_phi += gjj.cpr[harm] * sin((harm + 1) * (j_phi0 - j_phiOff));
                    }
                    double sqrt_part = sqrt(1.0 - j_d * sin2_half_phi);
                    bPtr[j_curIdx]   = j_curNzVal
                                     * ((((Constants::PI * j_del) / (2.0 * Constants::EV * j_rn))
                                         * (sin_phi / sqrt_part)
                                         * tanh(j_del / (2.0 * Constants::BOLTZMANN * j_t) * sqrt_part))
                                        - (((2.0 * j_c) / stepSize) * j_vn1)
                                        + ((j_c / (2.0 * stepSize)) * j_vn2) + j_it);
                }
                j_vn2 = j_vn1;
            }
        } else {
            for (int32_t jIdx = part.jjStart + lane; jIdx < part.jjStart + part.jjCount; jIdx += 32) {
                GpuJJ& jj = d_jjs[jIdx];
                if (jj.posIdx >= 0 && jj.negIdx < 0) {
                    jj.pn1 = xPtr[jj.posIdx];
                } else if (jj.posIdx < 0 && jj.negIdx >= 0) {
                    jj.pn1 = -xPtr[jj.negIdx];
                } else if (jj.posIdx >= 0 && jj.negIdx >= 0) {
                    jj.pn1 = xPtr[jj.posIdx] - xPtr[jj.negIdx];
                } else {
                    jj.pn1 = 0.0;
                }
                if (i > 0) {
                    if (analysisType == 0) {
                        jj.vn1 = jj.pn1;
                        jj.pn1 = xPtr[jj.varIdx];
                    } else {
                        jj.vn1 = xPtr[jj.varIdx];
                    }
                }
                const double v0 = (5.0 / 2.0) * jj.vn1 - 2.0 * jj.vn2 + (1.0 / 2.0) * jj.vn3;
                jj.phi0         = (4.0 / 3.0) * jj.pn1 - (1.0 / 3.0) * jj.pn2
                          + ((1.0 / Constants::SIGMA) * ((2.0 * stepSize) / 3.0)) * v0;
                if (checkPhase && fabs(jj.phi0 - jj.pn1) > (0.20 * 2.0 * Constants::PI)) {
                    laneNeedsTR = true;
                }
                if (analysisType == 0) {
                    bPtr[jj.varIdx] = Constants::SIGMA
                                      * (-(2.0 / stepSize) * jj.pn1 + (1.0 / (2.0 * stepSize)) * jj.pn2);
                } else {
                    bPtr[jj.varIdx] = (4.0 / 3.0) * jj.pn1 - (1.0 / 3.0) * jj.pn2;
                }
                jj.pn4 = jj.pn3;
                jj.pn3 = jj.pn2;
                jj.pn2 = jj.pn1;
                jj.vn6 = jj.vn5;
                jj.vn5 = jj.vn4;
                jj.vn4 = jj.vn3;
                jj.vn3 = jj.vn2;

                if (jj.rtype == 1) {
                    bool   updated = false;
                    double absV    = fabs(v0);
                    if (absV < jj.lowerB) {
                        jj.it = 0.0;
                        if (jj.curNzVal != jj.negInvSubgap) {
                            jj.curNzVal = jj.negInvSubgap;
                            updated     = true;
                        }
                    } else if (absV < jj.upperB) {
                        jj.it = jj.lowerB * ((1.0 / jj.r0) - jj.gLarge);
                        if (v0 < 0.0) { jj.it = -jj.it; }
                        if (jj.curNzVal != jj.negInvTrans) {
                            jj.curNzVal = jj.negInvTrans;
                            updated     = true;
                        }
                    } else {
                        jj.it = 0.0;
                        if (jj.curNzVal != jj.negInvNormal) {
                            jj.curNzVal = jj.negInvNormal;
                            updated     = true;
                        }
                    }
                    if (updated) {
                        d_partNz[jj.nzIdx] = jj.curNzVal;
                        laneNeedsLU        = true;
                    }
                }

                double ic_sin_phi = 0.0;
                if (jj.cprLen == 1) {
                    ic_sin_phi += jj.ic * (jj.cpr[0] * sin(1.0 * (jj.phi0 - jj.phiOff)));
                } else {
                    for (int32_t harm = 0; harm < jj.cprLen; ++harm) {
                        ic_sin_phi += jj.ic * (jj.cpr[harm] * sin((harm + 1) * (jj.phi0 - jj.phiOff)));
                    }
                }
                jj.cTerm = -(((2.0 * jj.c) / stepSize) * jj.vn1) + ((jj.c / (2.0 * stepSize)) * jj.vn2);
                if (!jj.tDep) {
                    bPtr[jj.curIdx] = jj.curNzVal
                                      * (ic_sin_phi - (((2.0 * jj.c) / stepSize) * jj.vn1)
                                         + ((jj.c / (2.0 * stepSize)) * jj.vn2) + jj.it);
                } else {
                    double sin2_half_phi = 0.0;
                    for (int32_t harm = 0; harm < jj.cprLen; ++harm) {
                        sin2_half_phi += jj.cpr[harm] * sin((harm + 1) * (jj.phi0 - jj.phiOff) / 2.0);
                    }
                    sin2_half_phi  = sin2_half_phi * sin2_half_phi;
                    double sin_phi = 0.0;
                    for (int32_t harm = 0; harm < jj.cprLen; ++harm) {
                        sin_phi += jj.cpr[harm] * sin((harm + 1) * (jj.phi0 - jj.phiOff));
                    }
                    double sqrt_part = sqrt(1.0 - jj.d * sin2_half_phi);
                    bPtr[jj.curIdx]  = jj.curNzVal
                                      * ((((Constants::PI * jj.del) / (2.0 * Constants::EV * jj.rn))
                                          * (sin_phi / sqrt_part)
                                          * tanh(jj.del / (2.0 * Constants::BOLTZMANN * jj.t) * sqrt_part))
                                         - (((2.0 * jj.c) / stepSize) * jj.vn1)
                                         + ((jj.c / (2.0 * stepSize)) * jj.vn2) + jj.it);
                }
                jj.vn2 = jj.vn1;
            }
        }

        if (__ballot_sync(0xffffffffu, laneNeedsTR) != 0u) {
            if (lane == 0) { atomicExch(d_needsTR, 1); }
            return;
        }

        // 3. Refactor partition LU in-place if any RTYPE=1 JJ in this partition transitioned
        if (__ballot_sync(0xffffffffu, laneNeedsLU) != 0u) {
            anyRefactoredLU = true;
            for (int32_t idx = lane; idx < part.luNnz; idx += 32) {
                const int32_t src = d_luSrcNz[luOff + idx];
                luValPtr[idx]     = (src >= 0) ? d_partNz[src] : 0.0;
            }
            __syncwarp();
            if (lane == 0) {
                for (int32_t r = 0; r < n; ++r) {
                    const int32_t rStart = luRpPtr[r];
                    const int32_t rDiag  = luDiagPtr[r];
                    for (int32_t idxL = rStart; idxL < rDiag; ++idxL) {
                        const int32_t k     = luCiPtr[idxL];
                        const int32_t kDiag = luDiagPtr[k];
                        const int32_t kEnd  = luRpPtr[k + 1];
                        const double  mult  = luValPtr[idxL] / luValPtr[kDiag];
                        luValPtr[idxL]      = mult;
                        int32_t pI          = idxL + 1;
                        for (int32_t idxU = kDiag + 1; idxU < kEnd; ++idxU) {
                            const int32_t colU = luCiPtr[idxU];
                            while (luCiPtr[pI] < colU) { ++pI; }
                            luValPtr[pI] -= mult * luValPtr[idxU];
                        }
                    }
                }
            }
            __syncwarp();
        }

        // 4. Handle Current Sources
        for (int32_t cIdx = part.csStart + lane; cIdx < part.csStart + part.csCount; cIdx += 32) {
            const GpuCS  cs     = d_css[cIdx];
            const double srcVal = d_srcWindow[(s + 2) * numSources + cs.srcIdx];
            if (cs.posIdx >= 0) { atomicAdd(&bPtr[cs.posIdx], -srcVal); }
            if (cs.negIdx >= 0) { atomicAdd(&bPtr[cs.negIdx], srcVal); }
        }

        // 5. Handle Resistors (Phase mode only, no thermal noise)
        if (analysisType == 1) {
            if (regResMode) {
                if (hasRes0) {
                    if (r0_pos >= 0 && r0_neg < 0) {
                        r0_pn1 = xPtr[r0_pos];
                    } else if (r0_pos < 0 && r0_neg >= 0) {
                        r0_pn1 = -xPtr[r0_neg];
                    } else if (r0_pos >= 0 && r0_neg >= 0) {
                        r0_pn1 = xPtr[r0_pos] - xPtr[r0_neg];
                    } else {
                        r0_pn1 = 0.0;
                    }
                    bPtr[r0_cur] = (4.0 / 3.0) * r0_pn1 - (1.0 / 3.0) * r0_pn2;
                    r0_pn2       = r0_pn1;
                }
                if (hasRes1) {
                    if (r1_pos >= 0 && r1_neg < 0) {
                        r1_pn1 = xPtr[r1_pos];
                    } else if (r1_pos < 0 && r1_neg >= 0) {
                        r1_pn1 = -xPtr[r1_neg];
                    } else if (r1_pos >= 0 && r1_neg >= 0) {
                        r1_pn1 = xPtr[r1_pos] - xPtr[r1_neg];
                    } else {
                        r1_pn1 = 0.0;
                    }
                    bPtr[r1_cur] = (4.0 / 3.0) * r1_pn1 - (1.0 / 3.0) * r1_pn2;
                    r1_pn2       = r1_pn1;
                }
            } else {
                for (int32_t rIdx = part.resStart + lane; rIdx < part.resStart + part.resCount; rIdx += 32) {
                    GpuRes& res = d_ress[rIdx];
                    if (res.posIdx >= 0 && res.negIdx < 0) {
                        res.pn1 = xPtr[res.posIdx];
                    } else if (res.posIdx < 0 && res.negIdx >= 0) {
                        res.pn1 = -xPtr[res.negIdx];
                    } else if (res.posIdx >= 0 && res.negIdx >= 0) {
                        res.pn1 = xPtr[res.posIdx] - xPtr[res.negIdx];
                    } else {
                        res.pn1 = 0.0;
                    }
                    bPtr[res.curIdx] = (4.0 / 3.0) * res.pn1 - (1.0 / 3.0) * res.pn2;
                    res.pn2          = res.pn1;
                }
            }
        }

        // 6. Handle Inductors (Voltage mode only)
        if (analysisType == 0 && part.indCount > 0) {
            const double h = stepSize;
            for (int32_t lIdx = part.indStart + lane; lIdx < part.indStart + part.indCount; lIdx += 32) {
                GpuInd& ind  = d_inds[lIdx];
                double  bVal = -(2.0 * ind.L / h) * xPtr[ind.curIdx] + ((ind.L / (2.0 * h)) * ind.In2);
                for (int32_t mIdx = ind.mutStart; mIdx < ind.mutStart + ind.mutCount; ++mIdx) {
                    const GpuMut mut  = d_muts[mIdx];
                    const double mIn2 = (mut.targetIndIdx < lIdx) ? xPtr[mut.targetCurIdx]
                                                                  : d_inds[mut.targetIndIdx].In2;
                    bVal += (-((2.0 * mut.M) / h) * xPtr[mut.targetCurIdx] + (mut.M / (2.0 * h)) * mIn2);
                }
                bPtr[ind.curIdx] = bVal;
            }
            __syncwarp();
            for (int32_t lIdx = part.indStart + lane; lIdx < part.indStart + part.indCount; lIdx += 32) {
                GpuInd& ind = d_inds[lIdx];
                ind.In2     = xPtr[ind.curIdx];
            }
        }

        // 7. Handle Capacitors
        for (int32_t cIdx = part.capStart + lane; cIdx < part.capStart + part.capCount; cIdx += 32) {
            GpuCap& cap = d_caps[cIdx];
            if (cap.posIdx >= 0 && cap.negIdx < 0) {
                cap.pn1 = xPtr[cap.posIdx];
            } else if (cap.posIdx < 0 && cap.negIdx >= 0) {
                cap.pn1 = -xPtr[cap.negIdx];
            } else if (cap.posIdx >= 0 && cap.negIdx >= 0) {
                cap.pn1 = xPtr[cap.posIdx] - xPtr[cap.negIdx];
            } else {
                cap.pn1 = 0.0;
            }
            if (analysisType == 0) {
                bPtr[cap.curIdx] = (4.0 / 3.0) * cap.pn1 - (1.0 / 3.0) * cap.pn2;
            } else {
                bPtr[cap.curIdx] = (8.0 / 3.0) * cap.pn1 - (22.0 / 9.0) * cap.pn2
                                   + (8.0 / 9.0) * cap.pn3 - (1.0 / 9.0) * cap.pn4;
                cap.pn4 = cap.pn3;
                cap.pn3 = cap.pn2;
            }
            cap.pn2 = cap.pn1;
        }

        // 8. Handle Voltage Sources
        for (int32_t vIdx = part.vsStart + lane; vIdx < part.vsStart + part.vsCount; vIdx += 32) {
            GpuVS&       vs     = d_vss[vIdx];
            const double srcVal = d_srcWindow[(s + 2) * numSources + vs.srcIdx];
            if (analysisType == 0) {
                bPtr[vs.curIdx] = srcVal;
            } else {
                if (vs.posIdx >= 0 && vs.negIdx < 0) {
                    vs.pn1 = xPtr[vs.posIdx];
                } else if (vs.posIdx < 0 && vs.negIdx >= 0) {
                    vs.pn1 = -xPtr[vs.negIdx];
                } else if (vs.posIdx >= 0 && vs.negIdx >= 0) {
                    vs.pn1 = xPtr[vs.posIdx] - xPtr[vs.negIdx];
                } else {
                    vs.pn1 = 0.0;
                }
                bPtr[vs.curIdx] = ((2.0 * stepSize) / (3.0 * Constants::SIGMA)) * srcVal
                                  + (4.0 / 3.0) * vs.pn1 - (1.0 / 3.0) * vs.pn2;
                vs.pn2 = vs.pn1;
            }
        }

        // 9. Handle Phase Sources
        for (int32_t pIdx = part.psStart + lane; pIdx < part.psStart + part.psCount; pIdx += 32) {
            const GpuPS  ps = d_pss[pIdx];
            const double s0 = d_srcWindow[(s + 2) * numSources + ps.srcIdx];
            if (analysisType == 1) {
                bPtr[ps.curIdx] = s0;
            } else {
                if (i == 0) {
                    bPtr[ps.curIdx] = (Constants::SIGMA / stepSize) * ((3.0 / 2.0) * s0);
                } else if (i == 1) {
                    const double s1 = d_srcWindow[(s + 1) * numSources + ps.srcIdx];
                    bPtr[ps.curIdx] = (Constants::SIGMA / stepSize) * ((3.0 / 2.0) * s0 - 2.0 * s1);
                } else {
                    const double s1 = d_srcWindow[(s + 1) * numSources + ps.srcIdx];
                    const double s2 = d_srcWindow[s * numSources + ps.srcIdx];
                    bPtr[ps.curIdx] = (Constants::SIGMA / stepSize)
                                      * ((3.0 / 2.0) * s0 - 2.0 * s1 + 0.5 * s2);
                }
            }
        }

        // 10. Handle CCVS & VCCS (Phase mode only)
        if (analysisType == 1) {
            for (int32_t idx = part.ccvsStart + lane; idx < part.ccvsStart + part.ccvsCount; idx += 32) {
                GpuCCVS& ccvs = d_ccvss[idx];
                if (ccvs.posIdx >= 0 && ccvs.negIdx < 0) {
                    ccvs.pn1 = xPtr[ccvs.posIdx];
                } else if (ccvs.posIdx < 0 && ccvs.negIdx >= 0) {
                    ccvs.pn1 = -xPtr[ccvs.negIdx];
                } else if (ccvs.posIdx >= 0 && ccvs.negIdx >= 0) {
                    ccvs.pn1 = xPtr[ccvs.posIdx] - xPtr[ccvs.negIdx];
                } else {
                    ccvs.pn1 = 0.0;
                }
                bPtr[ccvs.curIdx] = (4.0 / 3.0) * ccvs.pn1 - (1.0 / 3.0) * ccvs.pn2;
                ccvs.pn2          = ccvs.pn1;
            }
            for (int32_t idx = part.vccsStart + lane; idx < part.vccsStart + part.vccsCount; idx += 32) {
                GpuVCCS& vccs = d_vccss[idx];
                if (vccs.posIdx2 >= 0 && vccs.negIdx2 < 0) {
                    vccs.pn1 = xPtr[vccs.posIdx2];
                } else if (vccs.posIdx2 < 0 && vccs.negIdx2 >= 0) {
                    vccs.pn1 = -xPtr[vccs.negIdx2];
                } else if (vccs.posIdx2 >= 0 && vccs.negIdx2 >= 0) {
                    vccs.pn1 = xPtr[vccs.posIdx2] - xPtr[vccs.negIdx2];
                } else {
                    vccs.pn1 = 0.0;
                }
                bPtr[vccs.curIdx] = (4.0 / 3.0) * vccs.pn1 - (1.0 / 3.0) * vccs.pn2;
                vccs.pn2          = vccs.pn1;
            }
        }

        // 11. Handle Transmission Line Port Halves (RHS setup before solve)
        if (regTxMode) {
            if (hasRegTx) {
                const double rem = (tx_isPort2 == 0) ? d_txRemote1[tx_idx * w + s]
                                                     : d_txRemote2[tx_idx * w + s];
                if (analysisType == 0) {
                    bPtr[tx_cur] = rem;
                } else {
                    bPtr[tx_cur] = (4.0 / 3.0) * tx_n1 - (1.0 / 3.0) * tx_n2 + rem;
                }
            }
        } else {
            for (int32_t tIdx = part.txHalfStart + lane; tIdx < part.txHalfStart + part.txHalfCount; tIdx += 32) {
                const GpuTxHalf txh = d_txHalves[tIdx];
                const double    rem = (txh.isPort2 == 0) ? d_txRemote1[txh.txIdx * w + s]
                                                         : d_txRemote2[txh.txIdx * w + s];
                if (analysisType == 0) {
                    bPtr[txh.curIdx] = rem;
                } else {
                    bPtr[txh.curIdx] = (4.0 / 3.0) * txh.n1 - (1.0 / 3.0) * txh.n2 + rem;
                }
            }
        }
        __syncwarp();

        // 12 & 13. Forward and Backward substitution in shared memory
        if (lane == 0) {
            for (int32_t r = 0; r < n; ++r) {
                double        sum    = bPtr[rowOff + pPtr[r]];
                const int32_t rStart = luRpPtr[r];
                const int32_t rDiag  = luDiagPtr[r];
                for (int32_t idx = rStart; idx < rDiag; ++idx) {
                    sum -= luValPtr[idx] * yPtr[rowOff + luCiPtr[idx]];
                }
                yPtr[rowOff + r] = sum;
            }
            for (int32_t r = n - 1; r >= 0; --r) {
                double        sum   = yPtr[rowOff + r];
                const int32_t rDiag = luDiagPtr[r];
                const int32_t rEnd  = luRpPtr[r + 1];
                for (int32_t idx = rDiag + 1; idx < rEnd; ++idx) {
                    sum -= luValPtr[idx] * yPtr[rowOff + luCiPtr[idx]];
                }
                const double zr          = sum / luValPtr[rDiag];
                yPtr[rowOff + r]         = zr;
                xPtr[rowOff + qPtr[r]]   = zr;
            }
        }
        __syncwarp();

        // 14. Record completed step i into transmission line circular history and shift local port state
        if (regTxMode) {
            if (hasRegTx) {
                const GpuTxLine tx = d_txLines[tx_idx];
                double          v  = 0.0;
                if (tx_pos >= 0 && tx_neg < 0) {
                    v = xPtr[tx_pos];
                } else if (tx_pos < 0 && tx_neg >= 0) {
                    v = -xPtr[tx_neg];
                } else if (tx_pos >= 0 && tx_neg >= 0) {
                    v = xPtr[tx_pos] - xPtr[tx_neg];
                }
                const double  cur  = xPtr[tx_cur];
                const int64_t slot = tx.histOffset + ((tx.loadedSteps + i) % tx.histSize);
                if (tx_isPort2 == 0) {
                    d_txHist[slot].v1 = v;
                    d_txHist[slot].i1 = cur;
                } else {
                    d_txHist[slot].v2 = v;
                    d_txHist[slot].i2 = cur;
                }
                if (analysisType == 1) {
                    tx_n2 = tx_n1;
                    tx_n1 = v;
                }
            }
        } else {
            for (int32_t tIdx = part.txHalfStart + lane; tIdx < part.txHalfStart + part.txHalfCount; tIdx += 32) {
                GpuTxHalf&      txh = d_txHalves[tIdx];
                const GpuTxLine tx  = d_txLines[txh.txIdx];
                double          v   = 0.0;
                if (txh.posIdx >= 0 && txh.negIdx < 0) {
                    v = xPtr[txh.posIdx];
                } else if (txh.posIdx < 0 && txh.negIdx >= 0) {
                    v = -xPtr[txh.negIdx];
                } else if (txh.posIdx >= 0 && txh.negIdx >= 0) {
                    v = xPtr[txh.posIdx] - xPtr[txh.negIdx];
                }
                const double  cur  = xPtr[txh.curIdx];
                const int64_t slot = tx.histOffset + ((tx.loadedSteps + i) % tx.histSize);
                if (txh.isPort2 == 0) {
                    d_txHist[slot].v1 = v;
                    d_txHist[slot].i1 = cur;
                } else {
                    d_txHist[slot].v2 = v;
                    d_txHist[slot].i2 = cur;
                }
                if (analysisType == 1) {
                    txh.n2 = txh.n1;
                    txh.n1 = v;
                }
            }
        }

        // 15. Store kept trace outputs for step i
        const int32_t kSlot = d_keptSlot[i];
        if (kSlot >= 0) {
            for (int32_t rIdx = part.relStart + lane; rIdx < part.relStart + part.relCount; rIdx += 32) {
                const GpuRelTrace rt = d_relTraces[rIdx];
                d_traceOut[static_cast<int64_t>(rt.traceSlot) * numKeptSteps + kSlot]
                        = xPtr[rt.localAbsIdx];
            }
        }
        __syncwarp();
    }

    // Write back register-cached device state and shared memory state at end of window
    if (hasRegJJ) {
        GpuJJ& jj   = d_jjs[part.jjStart + lane];
        jj.pn1      = j_pn1;
        jj.pn2      = j_pn2;
        jj.pn3      = j_pn3;
        jj.pn4      = j_pn4;
        jj.phi0     = j_phi0;
        jj.vn1      = j_vn1;
        jj.vn2      = j_vn2;
        jj.vn3      = j_vn3;
        jj.vn4      = j_vn4;
        jj.vn5      = j_vn5;
        jj.vn6      = j_vn6;
        jj.it       = j_it;
        jj.curNzVal = j_curNzVal;
        jj.cTerm    = j_cTerm;
    }
    if (hasRes0) {
        d_ress[part.resStart + lane].pn1 = r0_pn1;
        d_ress[part.resStart + lane].pn2 = r0_pn2;
    }
    if (hasRes1) {
        d_ress[part.resStart + lane + 32].pn1 = r1_pn1;
        d_ress[part.resStart + lane + 32].pn2 = r1_pn2;
    }
    if (hasRegTx) {
        d_txHalves[part.txHalfStart + lane].n1 = tx_n1;
        d_txHalves[part.txHalfStart + lane].n2 = tx_n2;
    }
    if (useShared > 0) {
        for (int32_t r = lane; r < n; r += 32) { d_xLocal[rowOff + r] = sx[r]; }
    }
    if (useShared == 2 && anyRefactoredLU) {
        for (int32_t idx = lane; idx < part.luNnz; idx += 32) {
            d_luVal[luOff + idx] = s_luVal[idx];
        }
    }
}

// Multi-block monolithic kernel: evaluates all JJs, Resistors, Capacitors, and Inductors (+ Mutual L)
// across all GPU SMs when the circuit is a single connected component (P = 1)
__global__ void monolithic_eval_devices_kernel(int32_t       numJJs,
                                               int32_t       numRess,
                                               int32_t       numCaps,
                                               int32_t       numInds,
                                               int64_t       i,
                                               int64_t       simSize,
                                               double        stepSize,
                                               int32_t       analysisType,
                                               const double* __restrict__ d_x,
                                               double*       __restrict__ d_b,
                                               GpuJJ*        d_jjs,
                                               GpuRes*       d_ress,
                                               GpuInd*       d_inds,
                                               const GpuMut* __restrict__ d_muts,
                                               GpuCap*       d_caps,
                                               const double* __restrict__ d_indIn2Old,
                                               double*       __restrict__ d_indIn2New,
                                               int32_t*      d_changedNzIdx,
                                               double*       d_changedNzVal,
                                               int32_t*      d_numChangedNz,
                                               int32_t*      d_needsTR) {
    const int32_t tid        = blockIdx.x * blockDim.x + threadIdx.x;
    const bool    checkPhase = (static_cast<double>(i) / static_cast<double>(simSize) > 0.01);

    // 1. Evaluate Josephson Junction tid
    if (tid < numJJs) {
        GpuJJ& jj = d_jjs[tid];
        if (jj.posIdx >= 0 && jj.negIdx < 0) {
            jj.pn1 = d_x[jj.posIdx];
        } else if (jj.posIdx < 0 && jj.negIdx >= 0) {
            jj.pn1 = -d_x[jj.negIdx];
        } else if (jj.posIdx >= 0 && jj.negIdx >= 0) {
            jj.pn1 = d_x[jj.posIdx] - d_x[jj.negIdx];
        } else {
            jj.pn1 = 0.0;
        }
        if (i > 0) {
            if (analysisType == 0) {
                jj.vn1 = jj.pn1;
                jj.pn1 = d_x[jj.varIdx];
            } else {
                jj.vn1 = d_x[jj.varIdx];
            }
        }
        const double v0 = (5.0 / 2.0) * jj.vn1 - 2.0 * jj.vn2 + (1.0 / 2.0) * jj.vn3;
        jj.phi0         = (4.0 / 3.0) * jj.pn1 - (1.0 / 3.0) * jj.pn2
                  + ((1.0 / Constants::SIGMA) * ((2.0 * stepSize) / 3.0)) * v0;
        if (checkPhase && fabs(jj.phi0 - jj.pn1) > (0.20 * 2.0 * Constants::PI)) {
            atomicExch(d_needsTR, 1);
            return;
        }
        if (analysisType == 0) {
            d_b[jj.varIdx] = Constants::SIGMA
                             * (-(2.0 / stepSize) * jj.pn1 + (1.0 / (2.0 * stepSize)) * jj.pn2);
        } else {
            d_b[jj.varIdx] = (4.0 / 3.0) * jj.pn1 - (1.0 / 3.0) * jj.pn2;
        }
        jj.pn4 = jj.pn3;
        jj.pn3 = jj.pn2;
        jj.pn2 = jj.pn1;
        jj.vn6 = jj.vn5;
        jj.vn5 = jj.vn4;
        jj.vn4 = jj.vn3;
        jj.vn3 = jj.vn2;

        if (jj.rtype == 1) {
            bool   updated = false;
            double absV    = fabs(v0);
            if (absV < jj.lowerB) {
                jj.it = 0.0;
                if (jj.curNzVal != jj.negInvSubgap) {
                    jj.curNzVal = jj.negInvSubgap;
                    updated     = true;
                }
            } else if (absV < jj.upperB) {
                jj.it = jj.lowerB * ((1.0 / jj.r0) - jj.gLarge);
                if (v0 < 0.0) { jj.it = -jj.it; }
                if (jj.curNzVal != jj.negInvTrans) {
                    jj.curNzVal = jj.negInvTrans;
                    updated     = true;
                }
            } else {
                jj.it = 0.0;
                if (jj.curNzVal != jj.negInvNormal) {
                    jj.curNzVal = jj.negInvNormal;
                    updated     = true;
                }
            }
            if (updated) {
                const int32_t slot   = atomicAdd(d_numChangedNz, 1);
                d_changedNzIdx[slot] = jj.nzIdx;
                d_changedNzVal[slot] = jj.curNzVal;
            }
        }

        double ic_sin_phi = 0.0;
        if (jj.cprLen == 1) {
            ic_sin_phi += jj.ic * (jj.cpr[0] * sin(1.0 * (jj.phi0 - jj.phiOff)));
        } else {
            for (int32_t harm = 0; harm < jj.cprLen; ++harm) {
                ic_sin_phi += jj.ic * (jj.cpr[harm] * sin((harm + 1) * (jj.phi0 - jj.phiOff)));
            }
        }
        jj.cTerm = -(((2.0 * jj.c) / stepSize) * jj.vn1) + ((jj.c / (2.0 * stepSize)) * jj.vn2);
        if (!jj.tDep) {
            d_b[jj.curIdx] = jj.curNzVal
                             * (ic_sin_phi - (((2.0 * jj.c) / stepSize) * jj.vn1)
                                + ((jj.c / (2.0 * stepSize)) * jj.vn2) + jj.it);
        } else {
            double sin2_half_phi = 0.0;
            for (int32_t harm = 0; harm < jj.cprLen; ++harm) {
                sin2_half_phi += jj.cpr[harm] * sin((harm + 1) * (jj.phi0 - jj.phiOff) / 2.0);
            }
            sin2_half_phi  = sin2_half_phi * sin2_half_phi;
            double sin_phi = 0.0;
            for (int32_t harm = 0; harm < jj.cprLen; ++harm) {
                sin_phi += jj.cpr[harm] * sin((harm + 1) * (jj.phi0 - jj.phiOff));
            }
            double sqrt_part = sqrt(1.0 - jj.d * sin2_half_phi);
            d_b[jj.curIdx]   = jj.curNzVal
                             * ((((Constants::PI * jj.del) / (2.0 * Constants::EV * jj.rn))
                                 * (sin_phi / sqrt_part)
                                 * tanh(jj.del / (2.0 * Constants::BOLTZMANN * jj.t) * sqrt_part))
                                - (((2.0 * jj.c) / stepSize) * jj.vn1)
                                + ((jj.c / (2.0 * stepSize)) * jj.vn2) + jj.it);
        }
        jj.vn2 = jj.vn1;
    }

    // 2. Evaluate Resistor tid (Phase mode only)
    if (analysisType == 1 && tid < numRess) {
        GpuRes& res = d_ress[tid];
        if (res.posIdx >= 0 && res.negIdx < 0) {
            res.pn1 = d_x[res.posIdx];
        } else if (res.posIdx < 0 && res.negIdx >= 0) {
            res.pn1 = -d_x[res.negIdx];
        } else if (res.posIdx >= 0 && res.negIdx >= 0) {
            res.pn1 = d_x[res.posIdx] - d_x[res.negIdx];
        } else {
            res.pn1 = 0.0;
        }
        d_b[res.curIdx] = (4.0 / 3.0) * res.pn1 - (1.0 / 3.0) * res.pn2;
        res.pn2         = res.pn1;
    }

    // 3. Evaluate Capacitor tid
    if (tid < numCaps) {
        GpuCap& cap = d_caps[tid];
        if (cap.posIdx >= 0 && cap.negIdx < 0) {
            cap.pn1 = d_x[cap.posIdx];
        } else if (cap.posIdx < 0 && cap.negIdx >= 0) {
            cap.pn1 = -d_x[cap.negIdx];
        } else if (cap.posIdx >= 0 && cap.negIdx >= 0) {
            cap.pn1 = d_x[cap.posIdx] - d_x[cap.negIdx];
        } else {
            cap.pn1 = 0.0;
        }
        if (analysisType == 0) {
            d_b[cap.curIdx] = (4.0 / 3.0) * cap.pn1 - (1.0 / 3.0) * cap.pn2;
        } else {
            d_b[cap.curIdx] = (8.0 / 3.0) * cap.pn1 - (22.0 / 9.0) * cap.pn2
                              + (8.0 / 9.0) * cap.pn3 - (1.0 / 9.0) * cap.pn4;
            cap.pn4 = cap.pn3;
            cap.pn3 = cap.pn2;
        }
        cap.pn2 = cap.pn1;
    }

    // 4. Evaluate Inductor tid + Mutual Inductances (Voltage mode only)
    if (analysisType == 0 && tid < numInds) {
        const GpuInd ind  = d_inds[tid];
        const double h    = stepSize;
        const double curI = d_x[ind.curIdx];
        double       bVal = -(2.0 * ind.L / h) * curI + ((ind.L / (2.0 * h)) * d_indIn2Old[tid]);
        for (int32_t mIdx = ind.mutStart; mIdx < ind.mutStart + ind.mutCount; ++mIdx) {
            const GpuMut mut  = d_muts[mIdx];
            const double mIn2 = (mut.targetIndIdx < tid) ? d_x[mut.targetCurIdx]
                                                         : d_indIn2Old[mut.targetIndIdx];
            bVal += (-((2.0 * mut.M) / h) * d_x[mut.targetCurIdx] + (mut.M / (2.0 * h)) * mIn2);
        }
        d_b[ind.curIdx]  = bVal;
        d_indIn2New[tid] = curI;
    }
}

} // namespace

CudaPartitionSolver::~CudaPartitionSolver() { free_device(); }

bool CudaPartitionSolver::initialize(int32_t                            nParts,
                                     int32_t                            totalRows,
                                     int32_t                            numSources,
                                     int32_t                            maxWindow,
                                     int32_t                            numRelTraces,
                                     int32_t                            numKeptSteps,
                                     int64_t                            simSize,
                                     double                             stepSize,
                                     int32_t                            analysisType,
                                     const std::vector<GpuPartDesc>&    parts,
                                     const std::vector<int32_t>&        P,
                                     const std::vector<int32_t>&        Q,
                                     const std::vector<int32_t>&        luRp,
                                     const std::vector<int32_t>&        luDiag,
                                     const std::vector<int32_t>&        luCi,
                                     const std::vector<int32_t>&        luSrcNz,
                                     const std::vector<double>&         luVal,
                                     const std::vector<double>&         partNz,
                                     const std::vector<double>&         xInit,
                                     const std::vector<GpuJJ>&          jjs,
                                     const std::vector<GpuRes>&         ress,
                                     const std::vector<GpuInd>&         inds,
                                     const std::vector<GpuMut>&         muts,
                                     const std::vector<GpuCap>&         caps,
                                     const std::vector<GpuVS>&          vss,
                                     const std::vector<GpuPS>&          pss,
                                     const std::vector<GpuCS>&          css,
                                     const std::vector<GpuCCVS>&        ccvss,
                                     const std::vector<GpuVCCS>&        vccss,
                                     const std::vector<GpuTxHalf>&      txHalves,
                                     const std::vector<GpuTxLine>&      txLines,
                                     const std::vector<GpuTxHistEntry>& txHist,
                                     const std::vector<GpuRelTrace>&    relTraces,
                                     const std::vector<int32_t>&        keptSlot) {
    free_device();

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) { return false; }

    nParts_       = nParts;
    totalRows_    = totalRows;
    numSources_   = numSources;
    maxWindow_    = maxWindow;
    numRelTraces_ = numRelTraces;
    numKeptSteps_ = numKeptSteps;
    numTxLines_   = static_cast<int32_t>(txLines.size());
    numInds_      = static_cast<int32_t>(inds.size());
    maxPartN_     = 0;
    maxPartLuNnz_ = 0;
    for (const auto& pt : parts) {
        maxPartN_     = std::max<int32_t>(maxPartN_, pt.n);
        maxPartLuNnz_ = std::max<int32_t>(maxPartLuNnz_, pt.luNnz);
    }
    simSize_      = simSize;
    stepSize_     = stepSize;
    analysisType_ = analysisType;

    if (!alloc_and_upload(&d_parts_, parts)) { return false; }
    if (!alloc_and_upload(&d_P_, P)) { return false; }
    if (!alloc_and_upload(&d_Q_, Q)) { return false; }
    if (!alloc_and_upload(&d_luRp_, luRp)) { return false; }
    if (!alloc_and_upload(&d_luDiag_, luDiag)) { return false; }
    if (!alloc_and_upload(&d_luCi_, luCi)) { return false; }
    if (!alloc_and_upload(&d_luSrcNz_, luSrcNz)) { return false; }
    if (!alloc_and_upload(&d_luVal_, luVal)) { return false; }
    if (!alloc_and_upload(&d_partNz_, partNz)) { return false; }
    if (!alloc_and_upload(&d_xLocal_, xInit)) { return false; }

    JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bLocal_), totalRows_ * sizeof(double)));
    JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_yLocal_), totalRows_ * sizeof(double)));

    if (!alloc_and_upload(&d_jjs_, jjs)) { return false; }
    if (!alloc_and_upload(&d_ress_, ress)) { return false; }
    if (!alloc_and_upload(&d_inds_, inds)) { return false; }
    if (!alloc_and_upload(&d_muts_, muts)) { return false; }
    if (!alloc_and_upload(&d_caps_, caps)) { return false; }
    if (!alloc_and_upload(&d_vss_, vss)) { return false; }
    if (!alloc_and_upload(&d_pss_, pss)) { return false; }
    if (!alloc_and_upload(&d_css_, css)) { return false; }
    if (!alloc_and_upload(&d_ccvss_, ccvss)) { return false; }
    if (!alloc_and_upload(&d_vccss_, vccss)) { return false; }
    if (!alloc_and_upload(&d_txHalves_, txHalves)) { return false; }
    if (!alloc_and_upload(&d_txLines_, txLines)) { return false; }
    if (!alloc_and_upload(&d_txHist_, txHist)) { return false; }
    if (!alloc_and_upload(&d_relTraces_, relTraces)) { return false; }
    if (!alloc_and_upload(&d_keptSlot_, keptSlot)) { return false; }

    if (numTxLines_ > 0 && maxWindow_ > 0) {
        const size_t remBytes = static_cast<size_t>(numTxLines_) * maxWindow_ * sizeof(double);
        JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_txRemote1_), remBytes));
        JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_txRemote2_), remBytes));
    }
    if (numSources_ > 0 && maxWindow_ > 0) {
        const size_t srcBytes = static_cast<size_t>(maxWindow_ + 2) * numSources_ * sizeof(double);
        JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_srcWindow_), srcBytes));
    }
    if (numRelTraces_ > 0 && numKeptSteps_ > 0) {
        const size_t traceBytes = static_cast<size_t>(numRelTraces_) * numKeptSteps_ * sizeof(double);
        JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_traceOut_), traceBytes));
    }
    JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_needsTR_), sizeof(int32_t)));
    JOSIM_CUDA_CHECK(cudaMemset(d_needsTR_, 0, sizeof(int32_t)));

    cudaFuncSetAttribute(
            advance_partitions_window_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    initialized_ = true;
    return true;
}

bool CudaPartitionSolver::step_window(int64_t i0, int64_t i1, const double* h_srcWindow) {
    const int32_t w = static_cast<int32_t>(i1 - i0);
    if (w <= 0) { return true; }

    if (numSources_ > 0 && h_srcWindow != nullptr) {
        const size_t srcBytes = static_cast<size_t>(w + 2) * numSources_ * sizeof(double);
        JOSIM_CUDA_CHECK(cudaMemcpy(d_srcWindow_, h_srcWindow, srcBytes, cudaMemcpyHostToDevice));
    }

    if (numTxLines_ > 0) {
        const int32_t total  = numTxLines_ * w;
        const int32_t block  = 128;
        const int32_t blocks = (total + block - 1) / block;
        precompute_tx_remote_kernel<<<blocks, block>>>(numTxLines_,
                                                       i0,
                                                       w,
                                                       stepSize_,
                                                       analysisType_,
                                                       d_txLines_,
                                                       d_txHist_,
                                                       d_txRemote1_,
                                                       d_txRemote2_);
    }

    const int32_t warpsPerBlock    = 1;
    const int32_t threadsPerBlock  = warpsPerBlock * 32;
    const int32_t numBlocks        = (nParts_ + warpsPerBlock - 1) / warpsPerBlock;
    const int32_t fullDoubles      = 3 * maxPartN_ + maxPartLuNnz_ + ((maxPartLuNnz_ + 4 * maxPartN_ + 2) / 2);
    const int32_t vecDoubles       = 3 * maxPartN_;
    const size_t  fullShmemBytes   = static_cast<size_t>(warpsPerBlock) * fullDoubles * sizeof(double);
    const size_t  vecShmemBytes    = static_cast<size_t>(warpsPerBlock) * vecDoubles * sizeof(double);
    int32_t       useShared        = 0;
    int32_t       shDoublesPerWarp = 0;
    size_t        launchShmem      = 0;
    if (fullShmemBytes <= 48 * 1024) {
        useShared        = 2;
        shDoublesPerWarp = fullDoubles;
        launchShmem      = fullShmemBytes;
    } else if (vecShmemBytes <= 48 * 1024) {
        useShared        = 1;
        shDoublesPerWarp = vecDoubles;
        launchShmem      = vecShmemBytes;
    }

    advance_partitions_window_kernel<<<numBlocks, threadsPerBlock, launchShmem>>>(nParts_,
                                                                                  maxPartN_,
                                                                                  maxPartLuNnz_,
                                                                                  shDoublesPerWarp,
                                                                                  useShared,
                                                                                  i0,
                                                                                  w,
                                                                                  numSources_,
                                                                                  numKeptSteps_,
                                                                                  simSize_,
                                                                                  stepSize_,
                                                                                  analysisType_,
                                                                                  d_parts_,
                                                                                  d_P_,
                                                                                  d_Q_,
                                                                                  d_luRp_,
                                                                                  d_luDiag_,
                                                                                  d_luCi_,
                                                                                  d_luSrcNz_,
                                                                                  d_luVal_,
                                                                                  d_partNz_,
                                                                                  d_xLocal_,
                                                                                  d_bLocal_,
                                                                                  d_yLocal_,
                                                                                  d_jjs_,
                                                                                  d_ress_,
                                                                                  d_inds_,
                                                                                  d_muts_,
                                                                                  d_caps_,
                                                                                  d_vss_,
                                                                                  d_pss_,
                                                                                  d_css_,
                                                                                  d_ccvss_,
                                                                                  d_vccss_,
                                                                                  d_txHalves_,
                                                                                  d_txLines_,
                                                                                  d_txHist_,
                                                                                  d_txRemote1_,
                                                                                  d_txRemote2_,
                                                                                  d_srcWindow_,
                                                                                  d_relTraces_,
                                                                                  d_keptSlot_,
                                                                                  d_traceOut_,
                                                                                  d_needsTR_);

    int32_t h_needsTR = 0;
    JOSIM_CUDA_CHECK(cudaMemcpy(&h_needsTR, d_needsTR_, sizeof(int32_t), cudaMemcpyDeviceToHost));
    return (h_needsTR == 0);
}

bool CudaPartitionSolver::initialize_monolithic(int32_t                    totalRows,
                                                int64_t                    simSize,
                                                double                     stepSize,
                                                int32_t                    analysisType,
                                                const std::vector<double>& xInit,
                                                const std::vector<GpuJJ>&  jjs,
                                                const std::vector<GpuRes>& ress,
                                                const std::vector<GpuInd>& inds,
                                                const std::vector<GpuMut>& muts,
                                                const std::vector<GpuCap>& caps) {
    free_device();

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) { return false; }

    nParts_       = 1;
    totalRows_    = totalRows;
    numJJs_       = static_cast<int32_t>(jjs.size());
    numRess_      = static_cast<int32_t>(ress.size());
    numInds_      = static_cast<int32_t>(inds.size());
    numCaps_      = static_cast<int32_t>(caps.size());
    simSize_      = simSize;
    stepSize_     = stepSize;
    analysisType_ = analysisType;

    if (!alloc_and_upload(&d_xLocal_, xInit)) { return false; }
    JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bLocal_), totalRows_ * sizeof(double)));

    if (!alloc_and_upload(&d_jjs_, jjs)) { return false; }
    if (!alloc_and_upload(&d_ress_, ress)) { return false; }
    if (!alloc_and_upload(&d_inds_, inds)) { return false; }
    if (!alloc_and_upload(&d_muts_, muts)) { return false; }
    if (!alloc_and_upload(&d_caps_, caps)) { return false; }

    if (numInds_ > 0) {
        std::vector<double> indIn2(numInds_, 0.0);
        for (int32_t idx = 0; idx < numInds_; ++idx) { indIn2[idx] = inds[idx].In2; }
        if (!alloc_and_upload(&d_indIn2Old_, indIn2)) { return false; }
        JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indIn2New_), numInds_ * sizeof(double)));
    }
    if (numJJs_ > 0) {
        JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_changedNzIdx_), numJJs_ * sizeof(int32_t)));
        JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_changedNzVal_), numJJs_ * sizeof(double)));
    }
    JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_numChangedNz_), sizeof(int32_t)));
    JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_needsTR_), sizeof(int32_t)));
    JOSIM_CUDA_CHECK(cudaMemset(d_needsTR_, 0, sizeof(int32_t)));

    initialized_ = true;
    return true;
}

bool CudaPartitionSolver::step_monolithic_devices(
        int64_t                                 i,
        const double*                           h_x,
        double*                                 h_b,
        std::vector<std::pair<int32_t, double>>& changedNz) {
    changedNz.clear();
    if (i > 0) {
        JOSIM_CUDA_CHECK(
                cudaMemcpy(d_xLocal_, h_x, totalRows_ * sizeof(double), cudaMemcpyHostToDevice));
    }
    JOSIM_CUDA_CHECK(cudaMemset(d_bLocal_, 0, totalRows_ * sizeof(double)));
    JOSIM_CUDA_CHECK(cudaMemset(d_numChangedNz_, 0, sizeof(int32_t)));

    const int32_t nResWork = (analysisType_ == 1) ? numRess_ : 0;
    const int32_t nIndWork = (analysisType_ == 0) ? numInds_ : 0;
    const int32_t maxItems = std::max(std::max(numJJs_, nResWork), std::max(numCaps_, nIndWork));
    if (maxItems > 0) {
        const int32_t block  = 256;
        const int32_t blocks = (maxItems + block - 1) / block;
        monolithic_eval_devices_kernel<<<blocks, block>>>(numJJs_,
                                                          numRess_,
                                                          numCaps_,
                                                          numInds_,
                                                          i,
                                                          simSize_,
                                                          stepSize_,
                                                          analysisType_,
                                                          d_xLocal_,
                                                          d_bLocal_,
                                                          d_jjs_,
                                                          d_ress_,
                                                          d_inds_,
                                                          d_muts_,
                                                          d_caps_,
                                                          d_indIn2Old_,
                                                          d_indIn2New_,
                                                          d_changedNzIdx_,
                                                          d_changedNzVal_,
                                                          d_numChangedNz_,
                                                          d_needsTR_);
        if (analysisType_ == 0 && numInds_ > 0) { std::swap(d_indIn2Old_, d_indIn2New_); }
    }

    int32_t h_needsTR = 0;
    JOSIM_CUDA_CHECK(cudaMemcpy(&h_needsTR, d_needsTR_, sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (h_needsTR != 0) { return false; }

    int32_t h_numChanged = 0;
    JOSIM_CUDA_CHECK(cudaMemcpy(&h_numChanged, d_numChangedNz_, sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (h_numChanged > 0) {
        std::vector<int32_t> h_idx(h_numChanged);
        std::vector<double>  h_val(h_numChanged);
        JOSIM_CUDA_CHECK(cudaMemcpy(
                h_idx.data(), d_changedNzIdx_, h_numChanged * sizeof(int32_t), cudaMemcpyDeviceToHost));
        JOSIM_CUDA_CHECK(cudaMemcpy(
                h_val.data(), d_changedNzVal_, h_numChanged * sizeof(double), cudaMemcpyDeviceToHost));
        changedNz.reserve(h_numChanged);
        for (int32_t k = 0; k < h_numChanged; ++k) { changedNz.emplace_back(h_idx[k], h_val[k]); }
    }

    JOSIM_CUDA_CHECK(cudaMemcpy(h_b, d_bLocal_, totalRows_ * sizeof(double), cudaMemcpyDeviceToHost));
    return true;
}

void CudaPartitionSolver::download_results(std::vector<double>& h_traceOut,
                                           std::vector<double>& h_xFinal) {
    if (!initialized_) { return; }
    if (numRelTraces_ > 0 && numKeptSteps_ > 0 && d_traceOut_ != nullptr) {
        const size_t count = static_cast<size_t>(numRelTraces_) * numKeptSteps_;
        h_traceOut.resize(count);
        cudaMemcpy(h_traceOut.data(), d_traceOut_, count * sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (totalRows_ > 0 && d_xLocal_ != nullptr) {
        h_xFinal.resize(totalRows_);
        cudaMemcpy(h_xFinal.data(), d_xLocal_, totalRows_ * sizeof(double), cudaMemcpyDeviceToHost);
    }
}

void CudaPartitionSolver::free_device() {
    safe_cuda_free(d_parts_);
    safe_cuda_free(d_P_);
    safe_cuda_free(d_Q_);
    safe_cuda_free(d_luRp_);
    safe_cuda_free(d_luDiag_);
    safe_cuda_free(d_luCi_);
    safe_cuda_free(d_luSrcNz_);
    safe_cuda_free(d_luVal_);
    safe_cuda_free(d_partNz_);
    safe_cuda_free(d_xLocal_);
    safe_cuda_free(d_bLocal_);
    safe_cuda_free(d_yLocal_);
    safe_cuda_free(d_indIn2Old_);
    safe_cuda_free(d_indIn2New_);
    safe_cuda_free(d_changedNzIdx_);
    safe_cuda_free(d_changedNzVal_);
    safe_cuda_free(d_numChangedNz_);
    safe_cuda_free(d_jjs_);
    safe_cuda_free(d_ress_);
    safe_cuda_free(d_inds_);
    safe_cuda_free(d_muts_);
    safe_cuda_free(d_caps_);
    safe_cuda_free(d_vss_);
    safe_cuda_free(d_pss_);
    safe_cuda_free(d_css_);
    safe_cuda_free(d_ccvss_);
    safe_cuda_free(d_vccss_);
    safe_cuda_free(d_txHalves_);
    safe_cuda_free(d_txLines_);
    safe_cuda_free(d_txHist_);
    safe_cuda_free(d_txRemote1_);
    safe_cuda_free(d_txRemote2_);
    safe_cuda_free(d_srcWindow_);
    safe_cuda_free(d_relTraces_);
    safe_cuda_free(d_keptSlot_);
    safe_cuda_free(d_traceOut_);
    safe_cuda_free(d_needsTR_);
    if (h_xPinned_ != nullptr) {
        cudaFreeHost(h_xPinned_);
        h_xPinned_ = nullptr;
    }
    if (h_bPinned_ != nullptr) {
        cudaFreeHost(h_bPinned_);
        h_bPinned_ = nullptr;
    }
    initialized_ = false;
}

} // namespace JoSIM

#endif // JOSIM_CUDA
