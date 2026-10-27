// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)
#ifndef JOSIM_CUDASOLVER_HPP
#define JOSIM_CUDASOLVER_HPP

#include <cstdint>
#include <vector>

namespace JoSIM {

struct GpuPartDesc {
    int32_t n        = 0;
    int32_t rowStart = 0; // offset in d_xLocal, d_bLocal, d_yLocal, d_P, d_Q, d_luDiag
    int32_t rpStart  = 0; // offset in d_luRp (length n + 1)
    int32_t luStart  = 0; // offset in d_luCi, d_luSrcNz, d_luVal
    int32_t luNnz    = 0;
    int32_t nzStart  = 0; // offset in d_partNz
    int32_t nzCount  = 0;

    int32_t jjStart = 0, jjCount = 0;
    int32_t resStart = 0, resCount = 0;
    int32_t indStart = 0, indCount = 0;
    int32_t capStart = 0, capCount = 0;
    int32_t vsStart = 0, vsCount = 0;
    int32_t psStart = 0, psCount = 0;
    int32_t csStart = 0, csCount = 0;
    int32_t ccvsStart = 0, ccvsCount = 0;
    int32_t vccsStart = 0, vccsCount = 0;
    int32_t txHalfStart = 0, txHalfCount = 0;
    int32_t relStart = 0, relCount = 0;
};

struct GpuJJ {
    int32_t posIdx = -1; // absolute index in d_xLocal (-1 if GND)
    int32_t negIdx = -1;
    int32_t curIdx = -1; // absolute index in d_bLocal / d_xLocal
    int32_t varIdx = -1;
    int32_t nzIdx  = -1; // absolute index in d_partNz
    int32_t rtype  = 0;
    int32_t tDep   = 0;
    int32_t cprLen = 1;
    double  cpr[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double  ic     = 0.0;
    double  c      = 0.0;
    double  r0     = 0.0;
    double  rn     = 0.0;
    double  phiOff = 0.0;
    double  t      = 0.0;
    double  d      = 0.0;
    double  del    = 0.0;
    double  lowerB = 0.0, upperB = 0.0, gLarge = 0.0;
    double  negInvSubgap = 0.0, negInvTrans = 0.0, negInvNormal = 0.0;
    double  pn1 = 0.0, pn2 = 0.0, pn3 = 0.0, pn4 = 0.0, phi0 = 0.0;
    double  vn1 = 0.0, vn2 = 0.0, vn3 = 0.0, vn4 = 0.0, vn5 = 0.0, vn6 = 0.0;
    double  it = 0.0, cTerm = 0.0, curNzVal = 0.0;
};

struct GpuRes {
    int32_t posIdx = -1, negIdx = -1, curIdx = -1;
    double  pn1 = 0.0, pn2 = 0.0;
};

struct GpuInd {
    int32_t curIdx   = -1;
    int32_t mutStart = 0, mutCount = 0;
    double  L   = 0.0;
    double  In2 = 0.0, In3 = 0.0, In4 = 0.0;
};

struct GpuMut {
    int32_t targetIndIdx = -1; // index in d_inds (for In2prev)
    int32_t targetCurIdx = -1; // absolute index in d_xLocal
    double  M            = 0.0;
};

struct GpuCap {
    int32_t posIdx = -1, negIdx = -1, curIdx = -1;
    double  pn1 = 0.0, pn2 = 0.0, pn3 = 0.0, pn4 = 0.0;
};

struct GpuVS {
    int32_t posIdx = -1, negIdx = -1, curIdx = -1, srcIdx = -1;
    double  pn1 = 0.0, pn2 = 0.0;
};

struct GpuPS {
    int32_t curIdx = -1, srcIdx = -1;
};

struct GpuCS {
    int32_t posIdx = -1, negIdx = -1, srcIdx = -1;
};

struct GpuCCVS {
    int32_t posIdx = -1, negIdx = -1, curIdx = -1;
    double  pn1 = 0.0, pn2 = 0.0;
};

struct GpuVCCS {
    int32_t posIdx2 = -1, negIdx2 = -1, curIdx = -1;
    double  pn1 = 0.0, pn2 = 0.0;
};

struct GpuTxHalf {
    int32_t txIdx   = -1;
    int32_t isPort2 = 0;
    int32_t posIdx = -1, negIdx = -1, curIdx = -1;
    double  n1 = 0.0, n2 = 0.0;
};

struct GpuTxHistEntry {
    double v1 = 0.0;
    double v2 = 0.0;
    double i1 = 0.0;
    double i2 = 0.0;
};

struct GpuTxLine {
    double  z0          = 0.0;
    int64_t k           = 1;
    double  fracDelay   = 0.0;
    int64_t loadedSteps = 0;
    int32_t histOffset  = 0;
    int32_t histSize    = 0;
};

struct GpuRelTrace {
    int32_t localAbsIdx = 0; // absolute index in d_xLocal
    int32_t traceSlot   = 0; // 0 .. numRelTraces - 1
};

class CudaPartitionSolver {
  public:
    CudaPartitionSolver() = default;
    ~CudaPartitionSolver();

    CudaPartitionSolver(const CudaPartitionSolver&)            = delete;
    CudaPartitionSolver& operator=(const CudaPartitionSolver&) = delete;

    bool initialize(int32_t                            nParts,
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
                    const std::vector<int32_t>&        keptSlot);

    // Execute window [i0, i1) on GPU; returns false if any JJ triggered step halving (needsTR)
    bool step_window(int64_t i0, int64_t i1, const double* h_srcWindow);

    // Download completed trace table (numRelTraces * numKeptSteps) and final xLocal to host
    void download_results(std::vector<double>& h_traceOut, std::vector<double>& h_xFinal);

    void free_device();

  private:
    bool    initialized_  = false;
    int32_t nParts_       = 0;
    int32_t totalRows_    = 0;
    int32_t numSources_   = 0;
    int32_t maxWindow_    = 0;
    int32_t numRelTraces_ = 0;
    int32_t numKeptSteps_ = 0;
    int32_t numTxLines_   = 0;
    int32_t numInds_      = 0;
    int32_t maxPartN_     = 0;
    int32_t maxPartLuNnz_ = 0;
    int64_t simSize_      = 0;
    double  stepSize_     = 0.0;
    int32_t analysisType_ = 1; // 0 = Voltage, 1 = Phase

    GpuPartDesc*    d_parts_     = nullptr;
    int32_t*        d_P_         = nullptr;
    int32_t*        d_Q_         = nullptr;
    int32_t*        d_luRp_      = nullptr;
    int32_t*        d_luDiag_    = nullptr;
    int32_t*        d_luCi_      = nullptr;
    int32_t*        d_luSrcNz_   = nullptr;
    double*         d_luVal_     = nullptr;
    double*         d_partNz_    = nullptr;
    double*         d_xLocal_    = nullptr;
    double*         d_bLocal_    = nullptr;
    double*         d_yLocal_    = nullptr;
    double*         d_indIn2Old_ = nullptr;

    GpuJJ*          d_jjs_       = nullptr;
    GpuRes*         d_ress_      = nullptr;
    GpuInd*         d_inds_      = nullptr;
    GpuMut*         d_muts_      = nullptr;
    GpuCap*         d_caps_      = nullptr;
    GpuVS*          d_vss_       = nullptr;
    GpuPS*          d_pss_       = nullptr;
    GpuCS*          d_css_       = nullptr;
    GpuCCVS*        d_ccvss_     = nullptr;
    GpuVCCS*        d_vccss_     = nullptr;
    GpuTxHalf*      d_txHalves_  = nullptr;
    GpuTxLine*      d_txLines_   = nullptr;
    GpuTxHistEntry* d_txHist_    = nullptr;
    double*         d_txRemote1_ = nullptr;
    double*         d_txRemote2_ = nullptr;
    double*         d_srcWindow_ = nullptr;
    GpuRelTrace*    d_relTraces_ = nullptr;
    int32_t*        d_keptSlot_  = nullptr;
    double*         d_traceOut_  = nullptr;
    int32_t*        d_needsTR_   = nullptr;
};

} // namespace JoSIM

#endif // JOSIM_CUDASOLVER_HPP
