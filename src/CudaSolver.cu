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

// Thread-per-trial coalesced Monte Carlo kernel:
// Grid: dim3((B + blockDim.x - 1) / blockDim.x, nBaseParts), Block: warpsPerBlock * 32 (warpsPerBlock in {1, 2, 4})
// All warps in a block solve trials for the same base partition p = blockIdx.y and share a single copy of the
// symbolic CSR structure (s_luRp, s_luDiag, s_P, s_Q, s_luCi, s_luSrcNz) in shared memory, while each warp indexes
// its own bank-conflict-free shared memory vector slice (s_x, s_b, s_y).
__global__ void advance_mc_coalesced_kernel(int32_t             B,
                                            int32_t             nBaseParts,
                                            int32_t             nBaseTraces,
                                            int32_t             nBaseTxLines,
                                            int32_t             useShmemSym,
                                            int32_t             useShmemVec,
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
    extern __shared__ double s_mem_mc[];

    const int32_t tid           = threadIdx.x;
    const int32_t warpInBlock   = tid >> 5;
    const int32_t lane          = tid & 31;
    const int32_t warpsPerBlock = blockDim.x >> 5;
    const int32_t b             = blockIdx.x * blockDim.x + tid;
    const int32_t p             = blockIdx.y;
    if (p >= nBaseParts) { return; }

    const GpuPartDesc part   = d_parts[p];
    const int32_t     n      = part.n;
    const int32_t     rowOff = part.rowStart;
    const int32_t     rpOff  = part.rpStart;
    const int32_t     luOff  = part.luStart;

    const int32_t vecDoublesPerWarp = 32 * 3 * n;
    double*       s_warpBase        = useShmemVec ? (s_mem_mc + warpInBlock * vecDoublesPerWarp) : nullptr;
    double*       s_x               = useShmemVec ? s_warpBase : nullptr;
    double*       s_b               = useShmemVec ? (s_x + 32 * n) : nullptr;
    double*       s_y               = useShmemVec ? (s_b + 32 * n) : nullptr;
    int32_t*      s_sym             = reinterpret_cast<int32_t*>(
            useShmemVec ? (s_mem_mc + warpsPerBlock * vecDoublesPerWarp) : s_mem_mc);

    int32_t* s_luRp    = useShmemSym ? s_sym : nullptr;
    int32_t* s_luDiag  = useShmemSym ? (s_luRp + n + 1) : nullptr;
    int32_t* s_P       = useShmemSym ? (s_luDiag + n) : nullptr;
    int32_t* s_Q       = useShmemSym ? (s_P + n) : nullptr;
    int32_t* s_luCi    = useShmemSym ? (s_Q + n) : nullptr;
    int32_t* s_luSrcNz = useShmemSym ? (s_luCi + part.luNnz) : nullptr;

    if (useShmemSym) {
        for (int32_t r = tid; r < n; r += blockDim.x) {
            s_luRp[r]   = d_luRp[rpOff + r];
            s_luDiag[r] = d_luDiag[rowOff + r];
            s_P[r]      = d_P[rowOff + r];
            s_Q[r]      = d_Q[rowOff + r];
        }
        if (tid == 0) { s_luRp[n] = d_luRp[rpOff + n]; }
        for (int32_t idx = tid; idx < part.luNnz; idx += blockDim.x) {
            s_luCi[idx]    = d_luCi[luOff + idx];
            s_luSrcNz[idx] = d_luSrcNz[luOff + idx];
        }
        __syncthreads();
    }

    if (b >= B) { return; }

    if (useShmemVec) {
        for (int32_t r = 0; r < n; ++r) { s_x[r * 32 + lane] = d_xLocal[(rowOff + r) * B + b]; }
    }

    auto x_at = [&](int32_t absR) -> double& {
        return useShmemVec ? s_x[(absR - rowOff) * 32 + lane] : d_xLocal[absR * B + b];
    };
    auto b_at = [&](int32_t absR) -> double& {
        return useShmemVec ? s_b[(absR - rowOff) * 32 + lane] : d_bLocal[absR * B + b];
    };
    auto y_at = [&](int32_t locR) -> double& {
        return useShmemVec ? s_y[locR * 32 + lane] : d_yLocal[(rowOff + locR) * B + b];
    };

    const int32_t* luRpPtr    = useShmemSym ? s_luRp : (d_luRp + rpOff);
    const int32_t* luDiagPtr  = useShmemSym ? s_luDiag : (d_luDiag + rowOff);
    const int32_t* pPtr       = useShmemSym ? s_P : (d_P + rowOff);
    const int32_t* qPtr       = useShmemSym ? s_Q : (d_Q + rowOff);
    const int32_t* luCiPtr    = useShmemSym ? s_luCi : (d_luCi + luOff);
    const int32_t* luSrcNzPtr = useShmemSym ? s_luSrcNz : (d_luSrcNz + luOff);

    const int32_t txLineOffB = b * nBaseTxLines;

    for (int32_t s = 0; s < w; ++s) {
        const int64_t i          = i0 + s;
        const bool    checkPhase = (static_cast<double>(i) / static_cast<double>(simSize) > 0.01);

        for (int32_t r = 0; r < n; ++r) { b_at(rowOff + r) = 0.0; }

        bool partNeedsLU = false;
        for (int32_t k = 0; k < part.jjCount; ++k) {
            const int32_t jjSlot = (part.jjStart + k) * B + b;
            GpuJJ         jj     = d_jjs[jjSlot];
            const double  vPos   = (jj.posIdx >= 0) ? x_at(jj.posIdx) : 0.0;
            const double  vNeg   = (jj.negIdx >= 0) ? x_at(jj.negIdx) : 0.0;
            jj.pn1               = vPos - vNeg;
            if (i > 0) {
                if (analysisType == 0) {
                    jj.vn1 = jj.pn1;
                    jj.pn1 = x_at(jj.varIdx);
                } else {
                    jj.vn1 = x_at(jj.varIdx);
                }
            }
            const double v0 = 2.5 * jj.vn1 - 2.0 * jj.vn2 + 0.5 * jj.vn3;
            jj.phi0         = (4.0 / 3.0) * jj.pn1 - (1.0 / 3.0) * jj.pn2
                      + ((2.0 * stepSize) / (3.0 * Constants::SIGMA)) * v0;
            if (checkPhase && fabs(jj.phi0 - jj.pn1) > (0.20 * 2.0 * Constants::PI)) {
                atomicExch(d_needsTR, 1);
                return;
            }
            if (analysisType == 0) {
                b_at(jj.varIdx) = Constants::SIGMA * (-(2.0 / stepSize) * jj.pn1 + (1.0 / (2.0 * stepSize)) * jj.pn2);
            } else {
                b_at(jj.varIdx) = (4.0 / 3.0) * jj.pn1 - (1.0 / 3.0) * jj.pn2;
            }
            jj.pn2 = jj.pn1;
            jj.vn3 = jj.vn2;

            if (jj.rtype == 1) {
                bool         updated = false;
                const double absV    = fabs(v0);
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
                    d_partNz[jj.nzIdx * B + b] = jj.curNzVal;
                    partNeedsLU                = true;
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
            if (!jj.tDep) {
                b_at(jj.curIdx) = jj.curNzVal
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
                const double sqrt_part = sqrt(1.0 - jj.d * sin2_half_phi);
                b_at(jj.curIdx)
                        = jj.curNzVal
                          * ((((Constants::PI * jj.del) / (2.0 * Constants::EV * jj.rn)) * (sin_phi / sqrt_part)
                              * tanh(jj.del / (2.0 * Constants::BOLTZMANN * jj.t) * sqrt_part))
                             - (((2.0 * jj.c) / stepSize) * jj.vn1) + ((jj.c / (2.0 * stepSize)) * jj.vn2) + jj.it);
            }
            jj.vn2        = jj.vn1;
            d_jjs[jjSlot] = jj;
        }

        if (partNeedsLU) {
            for (int32_t idx = 0; idx < part.luNnz; ++idx) {
                const int32_t src             = luSrcNzPtr[idx];
                d_luVal[(luOff + idx) * B + b] = (src >= 0) ? d_partNz[src * B + b] : 0.0;
            }
            for (int32_t r = 0; r < n; ++r) {
                const int32_t rStart = luRpPtr[r];
                const int32_t rDiag  = luDiagPtr[r];
                for (int32_t idxL = rStart; idxL < rDiag; ++idxL) {
                    const int32_t colK  = luCiPtr[idxL];
                    const int32_t kDiag = luDiagPtr[colK];
                    const int32_t kEndU = luRpPtr[colK + 1];
                    const double  piv   = d_luVal[(luOff + kDiag) * B + b];
                    const double  mult  = d_luVal[(luOff + idxL) * B + b] / piv;
                    d_luVal[(luOff + idxL) * B + b] = mult;
                    int32_t pI = idxL + 1;
                    for (int32_t idxU = kDiag + 1; idxU < kEndU; ++idxU) {
                        const int32_t colU = luCiPtr[idxU];
                        while (luCiPtr[pI] < colU) { ++pI; }
                        d_luVal[(luOff + pI) * B + b] -= mult * d_luVal[(luOff + idxU) * B + b];
                    }
                }
            }
        }

        // Current Sources
        for (int32_t k = 0; k < part.csCount; ++k) {
            const GpuCS  cs     = d_css[(part.csStart + k) * B + b];
            const double srcVal = d_srcWindow[(s + 2) * numSources + cs.srcIdx];
            if (cs.posIdx >= 0) { b_at(cs.posIdx) -= srcVal; }
            if (cs.negIdx >= 0) { b_at(cs.negIdx) += srcVal; }
        }

        // Resistors (Phase mode)
        if (analysisType == 1) {
            for (int32_t k = 0; k < part.resCount; ++k) {
                const int32_t rSlot = (part.resStart + k) * B + b;
                GpuRes        res   = d_ress[rSlot];
                const double  vPos  = (res.posIdx >= 0) ? x_at(res.posIdx) : 0.0;
                const double  vNeg  = (res.negIdx >= 0) ? x_at(res.negIdx) : 0.0;
                res.pn1             = vPos - vNeg;
                b_at(res.curIdx)    = (4.0 / 3.0) * res.pn1 - (1.0 / 3.0) * res.pn2;
                res.pn2             = res.pn1;
                d_ress[rSlot]       = res;
            }
        }

        // Inductors (Voltage mode)
        if (analysisType == 0) {
            for (int32_t k = 0; k < part.indCount; ++k) {
                const GpuInd ind  = d_inds[(part.indStart + k) * B + b];
                const double iCur = x_at(ind.curIdx);
                double       rhs  = -(2.0 * ind.L / stepSize) * iCur + (ind.L / (2.0 * stepSize)) * ind.In2;
                for (int32_t m = 0; m < ind.mutCount; ++m) {
                    const GpuMut mut  = d_muts[(ind.mutStart + m) * B + b];
                    const GpuInd tInd = d_inds[mut.targetIndIdx * B + b];
                    rhs += -((2.0 * mut.M) / stepSize) * x_at(mut.targetCurIdx)
                           + (mut.M / (2.0 * stepSize)) * tInd.In2;
                }
                b_at(ind.curIdx) = rhs;
            }
            for (int32_t k = 0; k < part.indCount; ++k) {
                const int32_t lSlot = (part.indStart + k) * B + b;
                GpuInd        ind   = d_inds[lSlot];
                ind.In2             = x_at(ind.curIdx);
                d_inds[lSlot]       = ind;
            }
        }

        // Capacitors
        for (int32_t k = 0; k < part.capCount; ++k) {
            const int32_t cSlot = (part.capStart + k) * B + b;
            GpuCap        cap   = d_caps[cSlot];
            const double  vPos  = (cap.posIdx >= 0) ? x_at(cap.posIdx) : 0.0;
            const double  vNeg  = (cap.negIdx >= 0) ? x_at(cap.negIdx) : 0.0;
            cap.pn1             = vPos - vNeg;
            if (analysisType == 0) {
                b_at(cap.curIdx) = (4.0 / 3.0) * cap.pn1 - (1.0 / 3.0) * cap.pn2;
            } else {
                b_at(cap.curIdx)
                        = (8.0 / 3.0) * cap.pn1 - (22.0 / 9.0) * cap.pn2 + (8.0 / 9.0) * cap.pn3
                          - (1.0 / 9.0) * cap.pn4;
                cap.pn4 = cap.pn3;
                cap.pn3 = cap.pn2;
            }
            cap.pn2       = cap.pn1;
            d_caps[cSlot] = cap;
        }

        // Voltage Sources
        for (int32_t k = 0; k < part.vsCount; ++k) {
            const int32_t vSlot  = (part.vsStart + k) * B + b;
            GpuVS         vs     = d_vss[vSlot];
            const double  srcVal = d_srcWindow[(s + 2) * numSources + vs.srcIdx];
            if (analysisType == 0) {
                b_at(vs.curIdx) = srcVal;
            } else {
                const double vPos = (vs.posIdx >= 0) ? x_at(vs.posIdx) : 0.0;
                const double vNeg = (vs.negIdx >= 0) ? x_at(vs.negIdx) : 0.0;
                vs.pn1            = vPos - vNeg;
                b_at(vs.curIdx)
                        = ((2.0 * stepSize) / (3.0 * Constants::SIGMA)) * srcVal + (4.0 / 3.0) * vs.pn1
                          - (1.0 / 3.0) * vs.pn2;
                vs.pn2       = vs.pn1;
                d_vss[vSlot] = vs;
            }
        }

        // Phase Sources
        for (int32_t k = 0; k < part.psCount; ++k) {
            const GpuPS  ps = d_pss[(part.psStart + k) * B + b];
            const double s0 = d_srcWindow[(s + 2) * numSources + ps.srcIdx];
            if (analysisType == 1) {
                b_at(ps.curIdx) = s0;
            } else {
                const double s1 = d_srcWindow[(s + 1) * numSources + ps.srcIdx];
                const double s2 = d_srcWindow[s * numSources + ps.srcIdx];
                if (i == 0) {
                    b_at(ps.curIdx) = (Constants::SIGMA / stepSize) * (1.5 * s0);
                } else if (i == 1) {
                    b_at(ps.curIdx) = (Constants::SIGMA / stepSize) * (1.5 * s0 - 2.0 * s1);
                } else {
                    b_at(ps.curIdx) = (Constants::SIGMA / stepSize) * (1.5 * s0 - 2.0 * s1 + 0.5 * s2);
                }
            }
        }

        // CCVS & VCCS (Phase mode)
        if (analysisType == 1) {
            for (int32_t k = 0; k < part.ccvsCount; ++k) {
                const int32_t slot = (part.ccvsStart + k) * B + b;
                GpuCCVS       ccvs = d_ccvss[slot];
                const double  vPos = (ccvs.posIdx >= 0) ? x_at(ccvs.posIdx) : 0.0;
                const double  vNeg = (ccvs.negIdx >= 0) ? x_at(ccvs.negIdx) : 0.0;
                ccvs.pn1           = vPos - vNeg;
                b_at(ccvs.curIdx)  = (4.0 / 3.0) * ccvs.pn1 - (1.0 / 3.0) * ccvs.pn2;
                ccvs.pn2           = ccvs.pn1;
                d_ccvss[slot]      = ccvs;
            }
            for (int32_t k = 0; k < part.vccsCount; ++k) {
                const int32_t slot = (part.vccsStart + k) * B + b;
                GpuVCCS       vccs = d_vccss[slot];
                const double  vPos = (vccs.posIdx2 >= 0) ? x_at(vccs.posIdx2) : 0.0;
                const double  vNeg = (vccs.negIdx2 >= 0) ? x_at(vccs.negIdx2) : 0.0;
                vccs.pn1           = vPos - vNeg;
                b_at(vccs.curIdx)  = (4.0 / 3.0) * vccs.pn1 - (1.0 / 3.0) * vccs.pn2;
                vccs.pn2           = vccs.pn1;
                d_vccss[slot]      = vccs;
            }
        }

        // Transmission Line Port Halves
        for (int32_t k = 0; k < part.txHalfCount; ++k) {
            const GpuTxHalf txh       = d_txHalves[(part.txHalfStart + k) * B + b];
            const int32_t   globalTx  = txLineOffB + txh.txIdx;
            const double    rem
                    = (txh.isPort2 == 0) ? d_txRemote1[globalTx * w + s] : d_txRemote2[globalTx * w + s];
            if (analysisType == 0) {
                b_at(txh.curIdx) = rem;
            } else {
                b_at(txh.curIdx) = (4.0 / 3.0) * txh.n1 - (1.0 / 3.0) * txh.n2 + rem;
            }
        }

        // Coalesced Forward substitution: L * y = P * b
        for (int32_t r = 0; r < n; ++r) {
            double        sum    = b_at(rowOff + pPtr[r]);
            const int32_t rStart = luRpPtr[r];
            const int32_t rDiag  = luDiagPtr[r];
            for (int32_t idx = rStart; idx < rDiag; ++idx) {
                sum -= d_luVal[(luOff + idx) * B + b] * y_at(luCiPtr[idx]);
            }
            y_at(r) = sum;
        }
        // Coalesced Backward substitution: U * z = y, x[Q[r]] = z[r]
        for (int32_t r = n - 1; r >= 0; --r) {
            double        sum   = y_at(r);
            const int32_t rDiag = luDiagPtr[r];
            const int32_t rEnd  = luRpPtr[r + 1];
            for (int32_t idx = rDiag + 1; idx < rEnd; ++idx) {
                sum -= d_luVal[(luOff + idx) * B + b] * y_at(luCiPtr[idx]);
            }
            const double z         = sum / d_luVal[(luOff + rDiag) * B + b];
            y_at(r)                = z;
            x_at(rowOff + qPtr[r]) = z;
        }

        // Update transmission line circular history and local port phase history
        for (int32_t k = 0; k < part.txHalfCount; ++k) {
            const int32_t   slotHalf = (part.txHalfStart + k) * B + b;
            GpuTxHalf       txh      = d_txHalves[slotHalf];
            const GpuTxLine tx       = d_txLines[txLineOffB + txh.txIdx];
            const double    vPos     = (txh.posIdx >= 0) ? x_at(txh.posIdx) : 0.0;
            const double    vNeg     = (txh.negIdx >= 0) ? x_at(txh.negIdx) : 0.0;
            const double    vPort    = vPos - vNeg;
            const double    iPort    = x_at(txh.curIdx);
            const int32_t   hSlot    = tx.histOffset + static_cast<int32_t>((tx.loadedSteps + i) % tx.histSize);
            if (txh.isPort2 == 0) {
                d_txHist[hSlot].v1 = vPort;
                d_txHist[hSlot].i1 = iPort;
            } else {
                d_txHist[hSlot].v2 = vPort;
                d_txHist[hSlot].i2 = iPort;
            }
            if (analysisType == 1) {
                txh.n2               = txh.n1;
                txh.n1               = vPort;
                d_txHalves[slotHalf] = txh;
            }
        }

        // Record kept trace outputs
        const int32_t kSlot = d_keptSlot[i];
        if (kSlot >= 0) {
            for (int32_t k = 0; k < part.relCount; ++k) {
                const GpuRelTrace rt        = d_relTraces[part.relStart + k];
                const int32_t     traceSlot = b * nBaseTraces + rt.traceSlot;
                d_traceOut[static_cast<size_t>(traceSlot) * numKeptSteps + kSlot] = x_at(rt.localAbsIdx);
            }
        }
    }

    if (useShmemVec) {
        for (int32_t r = 0; r < n; ++r) { d_xLocal[(rowOff + r) * B + b] = s_x[r * 32 + lane]; }
    }
}

} // namespace

CudaPartitionSolver::~CudaPartitionSolver() { free_device(); }

void CudaPartitionSolver::warmup_device() {
    static bool warmed = false;
    if (!warmed) {
        (void) cudaFree(nullptr);
        warmed = true;
    }
}

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
                                     const std::vector<int32_t>&        keptSlot,
                                     int32_t                            mcRuns) {
    free_device();

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) { return false; }

    mcRuns_       = std::max<int32_t>(1, mcRuns);
    nParts_       = nParts;
    totalRows_    = totalRows;
    numSources_   = numSources;
    maxWindow_    = maxWindow;
    numRelTraces_ = numRelTraces;
    numKeptSteps_ = numKeptSteps;
    numTxLines_   = static_cast<int32_t>(txLines.size());
    numInds_      = static_cast<int32_t>(inds.size());
    nBaseParts_   = nParts / mcRuns_;
    nBaseRows_    = totalRows / mcRuns_;
    nBaseTraces_  = numRelTraces / mcRuns_;
    nBaseTxLines_ = numTxLines_ / mcRuns_;
    maxPartN_     = 0;
    maxPartLuNnz_ = 0;
    for (int32_t p = 0; p < nBaseParts_; ++p) {
        maxPartN_     = std::max<int32_t>(maxPartN_, parts[p].n);
        maxPartLuNnz_ = std::max<int32_t>(maxPartLuNnz_, parts[p].luNnz);
    }
    simSize_      = simSize;
    stepSize_     = stepSize;
    analysisType_ = analysisType;

    if (mcRuns_ > 1) {
        const int32_t B           = mcRuns_;
        const size_t  nBaseLu     = luVal.size() / static_cast<size_t>(B);
        const size_t  nBasePartNz = partNz.size() / static_cast<size_t>(B);
        const size_t  nBaseRp     = luRp.size() / static_cast<size_t>(B);

        std::vector<GpuPartDesc> baseParts(parts.begin(), parts.begin() + nBaseParts_);
        std::vector<int32_t>     baseP(P.begin(), P.begin() + nBaseRows_);
        std::vector<int32_t>     baseQ(Q.begin(), Q.begin() + nBaseRows_);
        std::vector<int32_t>     baseLuRp(luRp.begin(), luRp.begin() + nBaseRp);
        std::vector<int32_t>     baseLuDiag(luDiag.begin(), luDiag.begin() + nBaseRows_);
        std::vector<int32_t>     baseLuCi(luCi.begin(), luCi.begin() + nBaseLu);
        std::vector<int32_t>     baseLuSrcNz(luSrcNz.begin(), luSrcNz.begin() + nBaseLu);
        std::vector<GpuRelTrace> baseRelTraces(relTraces.begin(), relTraces.begin() + nBaseTraces_);

        if (!alloc_and_upload(&d_parts_, baseParts)) { return false; }
        if (!alloc_and_upload(&d_P_, baseP)) { return false; }
        if (!alloc_and_upload(&d_Q_, baseQ)) { return false; }
        if (!alloc_and_upload(&d_luRp_, baseLuRp)) { return false; }
        if (!alloc_and_upload(&d_luDiag_, baseLuDiag)) { return false; }
        if (!alloc_and_upload(&d_luCi_, baseLuCi)) { return false; }
        if (!alloc_and_upload(&d_luSrcNz_, baseLuSrcNz)) { return false; }
        if (!alloc_and_upload(&d_relTraces_, baseRelTraces)) { return false; }

        auto transpose_dbl = [&](const std::vector<double>& src, size_t nBase) -> std::vector<double> {
            std::vector<double> dst(src.size());
            for (int32_t b = 0; b < B; ++b) {
                for (size_t idx = 0; idx < nBase; ++idx) {
                    dst[idx * B + b] = src[static_cast<size_t>(b) * nBase + idx];
                }
            }
            return dst;
        };

        if (!alloc_and_upload(&d_luVal_, transpose_dbl(luVal, nBaseLu))) { return false; }
        if (!alloc_and_upload(&d_partNz_, transpose_dbl(partNz, nBasePartNz))) { return false; }
        if (!alloc_and_upload(&d_xLocal_, transpose_dbl(xInit, static_cast<size_t>(nBaseRows_)))) { return false; }

        JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bLocal_), totalRows_ * sizeof(double)));
        JOSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_yLocal_), totalRows_ * sizeof(double)));

        const size_t nBaseJJs  = jjs.size() / static_cast<size_t>(B);
        const size_t nBaseRess = ress.size() / static_cast<size_t>(B);
        const size_t nBaseInds = inds.size() / static_cast<size_t>(B);
        const size_t nBaseMuts = muts.size() / static_cast<size_t>(B);
        const size_t nBaseCaps = caps.size() / static_cast<size_t>(B);
        const size_t nBaseVSs  = vss.size() / static_cast<size_t>(B);
        const size_t nBasePSs  = pss.size() / static_cast<size_t>(B);
        const size_t nBaseCSs  = css.size() / static_cast<size_t>(B);
        const size_t nBaseCCVS = ccvss.size() / static_cast<size_t>(B);
        const size_t nBaseVCCS = vccss.size() / static_cast<size_t>(B);
        const size_t nBaseTxH  = txHalves.size() / static_cast<size_t>(B);

        std::vector<GpuJJ> jjsT(jjs.size());
        for (int32_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < nBaseJJs; ++k) {
                GpuJJ gjj  = jjs[static_cast<size_t>(b) * nBaseJJs + k];
                gjj.posIdx = jjs[k].posIdx;
                gjj.negIdx = jjs[k].negIdx;
                gjj.curIdx = jjs[k].curIdx;
                gjj.varIdx = jjs[k].varIdx;
                gjj.nzIdx  = jjs[k].nzIdx;
                jjsT[k * B + b] = gjj;
            }
        }
        std::vector<GpuRes> ressT(ress.size());
        for (int32_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < nBaseRess; ++k) { ressT[k * B + b] = ress[k]; }
        }
        std::vector<GpuInd> indsT(inds.size());
        for (int32_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < nBaseInds; ++k) {
                GpuInd gind   = inds[static_cast<size_t>(b) * nBaseInds + k];
                gind.curIdx   = inds[k].curIdx;
                gind.mutStart = inds[k].mutStart;
                indsT[k * B + b] = gind;
            }
        }
        std::vector<GpuMut> mutsT(muts.size());
        for (int32_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < nBaseMuts; ++k) {
                GpuMut gmut       = muts[static_cast<size_t>(b) * nBaseMuts + k];
                gmut.targetIndIdx = muts[k].targetIndIdx;
                gmut.targetCurIdx = muts[k].targetCurIdx;
                mutsT[k * B + b] = gmut;
            }
        }
        std::vector<GpuCap> capsT(caps.size());
        for (int32_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < nBaseCaps; ++k) { capsT[k * B + b] = caps[k]; }
        }
        std::vector<GpuVS> vssT(vss.size());
        for (int32_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < nBaseVSs; ++k) { vssT[k * B + b] = vss[k]; }
        }
        std::vector<GpuPS> pssT(pss.size());
        for (int32_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < nBasePSs; ++k) { pssT[k * B + b] = pss[k]; }
        }
        std::vector<GpuCS> cssT(css.size());
        for (int32_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < nBaseCSs; ++k) { cssT[k * B + b] = css[k]; }
        }
        std::vector<GpuCCVS> ccvssT(ccvss.size());
        for (int32_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < nBaseCCVS; ++k) { ccvssT[k * B + b] = ccvss[k]; }
        }
        std::vector<GpuVCCS> vccssT(vccss.size());
        for (int32_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < nBaseVCCS; ++k) { vccssT[k * B + b] = vccss[k]; }
        }
        std::vector<GpuTxHalf> txHalvesT(txHalves.size());
        for (int32_t b = 0; b < B; ++b) {
            for (size_t k = 0; k < nBaseTxH; ++k) { txHalvesT[k * B + b] = txHalves[k]; }
        }

        if (!alloc_and_upload(&d_jjs_, jjsT)) { return false; }
        if (!alloc_and_upload(&d_ress_, ressT)) { return false; }
        if (!alloc_and_upload(&d_inds_, indsT)) { return false; }
        if (!alloc_and_upload(&d_muts_, mutsT)) { return false; }
        if (!alloc_and_upload(&d_caps_, capsT)) { return false; }
        if (!alloc_and_upload(&d_vss_, vssT)) { return false; }
        if (!alloc_and_upload(&d_pss_, pssT)) { return false; }
        if (!alloc_and_upload(&d_css_, cssT)) { return false; }
        if (!alloc_and_upload(&d_ccvss_, ccvssT)) { return false; }
        if (!alloc_and_upload(&d_vccss_, vccssT)) { return false; }
        if (!alloc_and_upload(&d_txHalves_, txHalvesT)) { return false; }
        if (!alloc_and_upload(&d_txLines_, txLines)) { return false; }
        if (!alloc_and_upload(&d_txHist_, txHist)) { return false; }
        if (!alloc_and_upload(&d_keptSlot_, keptSlot)) { return false; }
    } else {
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
    }

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
    numSMs_        = 1;
    warpSize_      = 32;
    maxShmemBytes_ = 48 * 1024;
    maxShmemPerSM_ = 48 * 1024;
    int dev        = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) {
        int attrVal = 0;
        if (cudaDeviceGetAttribute(&attrVal, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && attrVal > 0) {
            numSMs_ = attrVal;
        }
        if (cudaDeviceGetAttribute(&attrVal, cudaDevAttrWarpSize, dev) == cudaSuccess && attrVal > 0) {
            warpSize_ = attrVal;
        }
        if (cudaDeviceGetAttribute(&attrVal, cudaDevAttrMaxSharedMemoryPerBlock, dev) == cudaSuccess && attrVal > 0) {
            maxShmemBytes_ = static_cast<size_t>(attrVal);
        }
        if (cudaDeviceGetAttribute(&attrVal, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) == cudaSuccess
            && attrVal > static_cast<int>(maxShmemBytes_)) {
            maxShmemBytes_ = static_cast<size_t>(attrVal);
        }
        if (cudaDeviceGetAttribute(&attrVal, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) == cudaSuccess
            && attrVal > 0) {
            maxShmemPerSM_ = static_cast<size_t>(attrVal);
        }
    }
    const int32_t fullDoubles    = 3 * maxPartN_ + maxPartLuNnz_ + ((maxPartLuNnz_ + 4 * maxPartN_ + 2) / 2);
    const size_t  fullShmemBytes = static_cast<size_t>(fullDoubles) * sizeof(double);
    const size_t  vecShmemBytes  = static_cast<size_t>(3 * maxPartN_) * sizeof(double);
    if (fullShmemBytes > 48 * 1024 && fullShmemBytes <= maxShmemBytes_) {
        cudaFuncSetAttribute(advance_partitions_window_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(fullShmemBytes));
    } else if (vecShmemBytes > 48 * 1024 && vecShmemBytes <= maxShmemBytes_) {
        cudaFuncSetAttribute(advance_partitions_window_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(vecShmemBytes));
    }

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

    if (mcRuns_ > 1) {
        const size_t symBytes
                = static_cast<size_t>(4 * maxPartN_ + 2 + 2 * maxPartLuNnz_) * sizeof(int32_t);
        const size_t vecBytesPerWarp
                = static_cast<size_t>(warpSize_ * 3 * maxPartN_) * sizeof(double);
        const int32_t trialWarps = (mcRuns_ + warpSize_ - 1) / warpSize_;

        int32_t warpsPerBlock = 1;
        for (int32_t cand : {4, 2, 1}) {
            if (trialWarps < cand) { continue; }
            const int32_t candBlocks
                    = ((trialWarps + cand - 1) / cand) * nBaseParts_;
            const size_t candShmem = static_cast<size_t>(cand) * vecBytesPerWarp + symBytes;
            if (candBlocks >= 2 * numSMs_ && candShmem <= maxShmemBytes_
                && candShmem * 2 <= maxShmemPerSM_) {
                warpsPerBlock = cand;
                break;
            }
        }
        const int32_t threadsPerBlock = warpsPerBlock * warpSize_;
        const size_t  vecBytes        = static_cast<size_t>(warpsPerBlock) * vecBytesPerWarp;
        int32_t       useShmemSym     = 0;
        int32_t       useShmemVec     = 0;
        size_t        launchShmem     = 0;
        if (vecBytes + symBytes <= maxShmemBytes_) {
            useShmemVec = 1;
            useShmemSym = 1;
            launchShmem = vecBytes + symBytes;
        } else if (symBytes <= maxShmemBytes_) {
            useShmemSym = 1;
            launchShmem = symBytes;
        }
        cudaFuncSetAttribute(
                advance_mc_coalesced_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (launchShmem > 48 * 1024) {
            cudaFuncSetAttribute(advance_mc_coalesced_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(launchShmem));
        }
        dim3 grid((mcRuns_ + threadsPerBlock - 1) / threadsPerBlock, nBaseParts_);
        advance_mc_coalesced_kernel<<<grid, threadsPerBlock, launchShmem>>>(mcRuns_,
                                                                            nBaseParts_,
                                                                            nBaseTraces_,
                                                                            nBaseTxLines_,
                                                                            useShmemSym,
                                                                            useShmemVec,
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

    const int32_t fullDoubles = 3 * maxPartN_ + maxPartLuNnz_ + ((maxPartLuNnz_ + 4 * maxPartN_ + 2) / 2);
    const int32_t vecDoubles  = 3 * maxPartN_;
    int32_t       warpsPerBlock = 1;
    for (int32_t cand : {4, 2, 1}) {
        const int32_t candBlocks = (nParts_ + cand - 1) / cand;
        const size_t  candShmem  = static_cast<size_t>(cand) * fullDoubles * sizeof(double);
        if (candBlocks >= 4 * numSMs_ && candShmem <= maxShmemBytes_ && candShmem * 2 <= maxShmemPerSM_) {
            warpsPerBlock = cand;
            break;
        }
    }
    const int32_t threadsPerBlock  = warpsPerBlock * warpSize_;
    const int32_t numBlocks        = (nParts_ + warpsPerBlock - 1) / warpsPerBlock;
    const size_t  fullShmemBytes   = static_cast<size_t>(warpsPerBlock) * fullDoubles * sizeof(double);
    const size_t  vecShmemBytes    = static_cast<size_t>(warpsPerBlock) * vecDoubles * sizeof(double);
    int32_t       useShared        = 0;
    int32_t       shDoublesPerWarp = 0;
    size_t        launchShmem      = 0;
    if (fullShmemBytes <= maxShmemBytes_) {
        useShared        = 2;
        shDoublesPerWarp = fullDoubles;
        launchShmem      = fullShmemBytes;
    } else if (vecShmemBytes <= maxShmemBytes_) {
        useShared        = 1;
        shDoublesPerWarp = vecDoubles;
        launchShmem      = vecShmemBytes;
    }
    if (launchShmem > 48 * 1024) {
        cudaFuncSetAttribute(advance_partitions_window_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(launchShmem));
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
        if (mcRuns_ > 1) {
            std::vector<double> xT(totalRows_);
            cudaMemcpy(xT.data(), d_xLocal_, totalRows_ * sizeof(double), cudaMemcpyDeviceToHost);
            h_xFinal.resize(totalRows_);
            for (int32_t b = 0; b < mcRuns_; ++b) {
                for (int32_t r = 0; r < nBaseRows_; ++r) {
                    h_xFinal[b * nBaseRows_ + r] = xT[r * mcRuns_ + b];
                }
            }
        } else {
            h_xFinal.resize(totalRows_);
            cudaMemcpy(h_xFinal.data(), d_xLocal_, totalRows_ * sizeof(double), cudaMemcpyDeviceToHost);
        }
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
