// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)
#ifndef JOSIM_SIMULATION_HPP
#define JOSIM_SIMULATION_HPP

#include "JoSIM/Constants.hpp"
#include "JoSIM/CudaSolver.hpp"
#include "JoSIM/Errors.hpp"
#include "JoSIM/LUSolve.hpp"
#include "JoSIM/Matrix.hpp"
#include "JoSIM/Misc.hpp"

#include <cassert>
#include <cmath>
#include <memory>
#include <suitesparse/klu.h>

namespace JoSIM {

#define TRANSIENT      0
#define DC             1
#define AC             2
#define PHASE          3
#define NONE_SPECIFIED 4

struct SfqEvent {
    double  time     = 0.0;
    double  phase    = 0.0;
    int32_t traceId  = 0;
    int32_t polarity = 1;
    int32_t winding  = 0;
    double  tPeak    = 0.0;
    double  vPeak    = 0.0;
    double  fwhm     = 0.0;
    double  dtPi     = 0.0;
    double  area     = 0.0;
};

struct InLoopEventTracker {
    static constexpr int32_t HIST_CAP  = 2048;
    static constexpr int32_t HIST_MASK = HIST_CAP - 1;
    static constexpr int32_t LOOKBACK  = 1024;
    static constexpr int32_t LOOKAHEAD = 768;

    double                firstPhi         = 0.0;
    double                prevPhi          = 0.0;
    double                prevT            = 0.0;
    int64_t               prevK            = 0;
    int32_t               traceId          = -1;
    int32_t               sign             = 1;
    int32_t               initialized      = 0;
    int32_t               stepCount        = 0;
    int32_t               hasPending       = 0;
    int32_t               pendingEventIdx  = -1;
    int32_t               pendingCrossStep = 0;
    std::vector<double>   histT;
    std::vector<double>   histPhi;
    std::vector<SfqEvent> events;

    void init_metrics(uint32_t metricsMask) {
        if (metricsMask != 0u) {
            histT.assign(HIST_CAP, 0.0);
            histPhi.assign(HIST_CAP, 0.0);
        } else {
            histT.clear();
            histPhi.clear();
        }
    }

    void finalize_pending(int32_t lastValidStep, uint32_t metricsMask) {
        if (!hasPending) { return; }
        hasPending   = 0;
        SfqEvent& ev = events[pendingEventIdx];
        if (metricsMask == 0u || histT.empty()) { return; }

        const int32_t cStep       = pendingCrossStep;
        const int32_t oldestValid = std::max<int32_t>(0, lastValidStep - HIST_CAP + 1);
        const int32_t wStart      = std::max<int32_t>(oldestValid, cStep - LOOKBACK);
        const int32_t wEnd        = std::min<int32_t>(lastValidStep, cStep + LOOKAHEAD);
        if (wEnd <= wStart) {
            ev.tPeak = ev.time;
            if ((metricsMask & EVENT_METRIC_TIME_PEAK) != 0u) { ev.time = ev.tPeak; }
            return;
        }

        auto          hT     = [&](int32_t idx) -> double { return histT[idx & HIST_MASK]; };
        auto          hPhi   = [&](int32_t idx) -> double { return histPhi[idx & HIST_MASK]; };
        const double  pol    = static_cast<double>(ev.polarity);
        const double  thresh = ev.phase;

        int32_t iMin = std::max<int32_t>(wStart, cStep - 1);
        while (iMin > wStart) {
            const double relPrev = pol * (hPhi(iMin - 1) - thresh);
            const double relCurr = pol * (hPhi(iMin) - thresh);
            if (relCurr <= -1.75 * Constants::PI) { break; }
            if (relCurr < -0.5 * Constants::PI && (relCurr - relPrev) <= 0.0) { break; }
            --iMin;
        }

        int32_t iMax = std::min<int32_t>(wEnd, cStep);
        while (iMax < wEnd) {
            const double relCurr = pol * (hPhi(iMax) - thresh);
            const double relNext = pol * (hPhi(iMax + 1) - thresh);
            if (relCurr >= 1.75 * Constants::PI) { break; }
            if (relCurr > 0.5 * Constants::PI && (relNext - relCurr) <= 0.0) { break; }
            ++iMax;
        }

        const int32_t vStart = std::max<int32_t>(wStart + 1, iMin);
        const int32_t vEnd   = std::min<int32_t>(wEnd, iMax + 1);

        auto midT = [&](int32_t m) -> double { return 0.5 * (hT(m - 1) + hT(m)); };
        auto midV = [&](int32_t m) -> double {
            const double dt = hT(m) - hT(m - 1);
            return (dt > 1e-30) ? (Constants::SIGMA * (hPhi(m) - hPhi(m - 1)) / dt) : 0.0;
        };

        const int32_t searchLo = std::max<int32_t>(vStart, iMin + 1);
        const int32_t searchHi = std::max<int32_t>(searchLo, iMax);
        int32_t       mPeak    = searchLo;
        double        bestPolV = -1e300;
        for (int32_t m = searchLo; m <= searchHi; ++m) {
            const double pv = pol * midV(m);
            if (pv > bestPolV) {
                bestPolV = pv;
                mPeak    = m;
            }
        }

        double tPeak = midT(mPeak);
        double vPeak = midV(mPeak);
        if (mPeak > vStart && mPeak < vEnd) {
            const double ym1   = pol * midV(mPeak - 1);
            const double y0    = pol * midV(mPeak);
            const double yp1   = pol * midV(mPeak + 1);
            const double denom = ym1 - 2.0 * y0 + yp1;
            if (y0 >= ym1 && y0 >= yp1 && denom < -1e-30) {
                double delta = 0.5 * (ym1 - yp1) / denom;
                if (delta < -0.5) { delta = -0.5; }
                if (delta > 0.5) { delta = 0.5; }
                const double yPeak = y0 - 0.125 * (ym1 - yp1) * (ym1 - yp1) / denom;
                const double dtSide
                        = (delta >= 0.0) ? (midT(mPeak + 1) - midT(mPeak)) : (midT(mPeak) - midT(mPeak - 1));
                tPeak = midT(mPeak) + delta * dtSide;
                vPeak = pol * yPeak;
            }
        }
        ev.tPeak = tPeak;
        ev.vPeak = vPeak;

        const double yHalf  = 0.5 * std::abs(vPeak);
        double       tLeft  = midT(vStart);
        for (int32_t m = mPeak; m > vStart; --m) {
            const double yPrev = pol * midV(m - 1);
            const double yCurr = pol * midV(m);
            if (yPrev <= yHalf && yCurr >= yHalf) {
                const double dy   = yCurr - yPrev;
                const double frac = (std::abs(dy) > 1e-30) ? ((yHalf - yPrev) / dy) : 0.5;
                tLeft             = midT(m - 1) + frac * (midT(m) - midT(m - 1));
                break;
            }
        }
        double tRight = midT(vEnd);
        for (int32_t m = mPeak; m < vEnd; ++m) {
            const double yCurr = pol * midV(m);
            const double yNext = pol * midV(m + 1);
            if (yCurr >= yHalf && yNext <= yHalf) {
                const double dy   = yCurr - yNext;
                const double frac = (std::abs(dy) > 1e-30) ? ((yCurr - yHalf) / dy) : 0.5;
                tRight            = midT(m) + frac * (midT(m + 1) - midT(m));
                break;
            }
        }
        ev.fwhm = std::max(0.0, tRight - tLeft);

        const double thetaEnter = thresh - pol * (0.5 * Constants::PI);
        const double thetaExit  = thresh + pol * (0.5 * Constants::PI);
        double       tEnter     = hT(iMin);
        for (int32_t idx = std::min<int32_t>(wEnd, cStep); idx > wStart; --idx) {
            const double p0 = hPhi(idx - 1);
            const double p1 = hPhi(idx);
            if (pol * (p0 - thetaEnter) <= 0.0 && pol * (p1 - thetaEnter) >= 0.0) {
                const double dp   = p1 - p0;
                const double frac = (std::abs(dp) > 1e-30) ? ((thetaEnter - p0) / dp) : 0.5;
                tEnter            = hT(idx - 1) + frac * (hT(idx) - hT(idx - 1));
                break;
            }
        }
        double tExit = hT(iMax);
        for (int32_t idx = std::max<int32_t>(wStart + 1, cStep); idx <= wEnd; ++idx) {
            const double p0 = hPhi(idx - 1);
            const double p1 = hPhi(idx);
            if (pol * (p0 - thetaExit) <= 0.0 && pol * (p1 - thetaExit) >= 0.0) {
                const double dp   = p1 - p0;
                const double frac = (std::abs(dp) > 1e-30) ? ((thetaExit - p0) / dp) : 0.5;
                tExit             = hT(idx - 1) + frac * (hT(idx) - hT(idx - 1));
                break;
            }
        }
        ev.dtPi = std::max(0.0, tExit - tEnter);
        ev.area = Constants::SIGMA * (hPhi(iMax) - hPhi(iMin));

        if ((metricsMask & EVENT_METRIC_TIME_PEAK) != 0u) { ev.time = ev.tPeak; }
    }

    inline void step(double tCurr, double rawVal, uint32_t metricsMask) {
        const double  phi     = (sign >= 0) ? rawVal : -rawVal;
        const int64_t k       = static_cast<int64_t>(std::floor((phi + Constants::PI) / (2.0 * Constants::PI)));
        const int32_t curStep = stepCount++;
        if (metricsMask != 0u && !histT.empty()) {
            histT[curStep & HIST_MASK]   = tCurr;
            histPhi[curStep & HIST_MASK] = phi;
        }
        if (!initialized) {
            firstPhi    = phi;
            prevPhi     = phi;
            prevT       = tCurr;
            prevK       = k;
            initialized = 1;
            return;
        }
        if (k != prevK) {
            if (hasPending) {
                finalize_pending(std::max<int32_t>(pendingCrossStep, curStep - 1), metricsMask);
            }
            const double  dPhi = phi - prevPhi;
            const int32_t pol  = (k > prevK) ? 1 : -1;
            int64_t       curK = prevK;
            while (curK != k) {
                if (hasPending) { finalize_pending(curStep, metricsMask); }
                const int64_t nextK = curK + pol;
                const double  thresh
                        = (pol > 0) ? ((2.0 * static_cast<double>(nextK) - 1.0) * Constants::PI)
                                    : ((2.0 * static_cast<double>(curK) - 1.0) * Constants::PI);
                double frac = (std::abs(dPhi) > 1e-30) ? ((thresh - prevPhi) / dPhi) : 0.5;
                if (frac < 0.0) { frac = 0.0; }
                if (frac > 1.0) { frac = 1.0; }
                const double tCross = prevT + frac * (tCurr - prevT);
                events.push_back(SfqEvent{
                        tCross, thresh, traceId, pol, static_cast<int32_t>(nextK), tCross, 0.0, 0.0, 0.0, 0.0});
                if (metricsMask != 0u) {
                    hasPending       = 1;
                    pendingEventIdx  = static_cast<int32_t>(events.size()) - 1;
                    pendingCrossStep = curStep;
                }
                curK = nextK;
            }
            prevK = k;
        } else if (hasPending && (curStep - pendingCrossStep) >= LOOKAHEAD) {
            finalize_pending(curStep, metricsMask);
        }
        prevPhi = phi;
        prevT   = tCurr;
    }

    void finalize(uint32_t metricsMask) {
        if (hasPending && stepCount > 0) {
            finalize_pending(stepCount - 1, metricsMask);
        }
    }
};

class Results {
  public:
    std::vector<std::optional<std::vector<double>>> xVector;
    std::vector<double>                             timeAxis;
    std::vector<int64_t>                            printIndices;
    int64_t                                         mcRuns        = 1;
    int64_t                                         mcNumRows     = 0;
    int64_t                                         mcNumKeptSteps = 0;
    std::vector<int32_t>                            mcRowToSlot;
    std::vector<double>                             mcTraceOut;
};

struct KluPartition {
    int64_t              n = 0;
    std::vector<int64_t> globalRows;
    std::vector<int64_t> rp;
    std::vector<int64_t> ci;
    std::vector<double>  nz;
    std::vector<double>  xLocal;
    klu_l_symbolic*      symbolic = nullptr;
    klu_l_numeric*       numeric  = nullptr;
    klu_l_common         common{};
    bool                 needsLU = false;

    // Per-partition device index lists for windowed PTL block-parallel execution
    std::vector<int64_t> jjIndices;
    std::vector<int64_t> resIndices;
    std::vector<int64_t> indIndices;
    std::vector<int64_t> capIndices;
    std::vector<int64_t> vsIndices;
    std::vector<int64_t> psIndices;
    std::vector<int64_t> csIndices;
    std::vector<int64_t> ccvsIndices;
    std::vector<int64_t> vccsIndices;
    std::vector<int64_t> txPort1Indices;
    std::vector<int64_t> txPort2Indices;
    std::vector<int64_t> traceGlobalRows;
};

struct NodalPartition {
    int32_t              n = 0;
    std::vector<int32_t> redRows;
    std::vector<int32_t> dirLocalRows;
    std::vector<int32_t> dirGlobalIdx;
    std::vector<int64_t> rp;
    std::vector<int64_t> ci;
    std::vector<double>  nz;
    std::vector<double>  xLocal;
    std::vector<double>  bLocal;
    klu_l_symbolic*      symbolic = nullptr;
    klu_l_numeric*       numeric  = nullptr;
    klu_l_common         common{};

    std::vector<GpuJJ>       jjs;
    std::vector<int64_t>     jjOrigIdx;
    std::vector<GpuRes>      ress;
    std::vector<GpuInd>      inds;
    std::vector<GpuMut>      muts;
    std::vector<GpuCap>      caps;
    std::vector<GpuVS>       vss;
    std::vector<GpuPS>       pss;
    std::vector<GpuCS>       css;
    std::vector<GpuCCVS>     ccvss;
    std::vector<GpuVCCS>     vccss;
    std::vector<GpuTxHalf>   txHalves;
    std::vector<GpuRelTrace> relTraces;
    std::vector<int64_t>     traceGlobalRows;
};

struct NodalRLEdgeStamp {
    double                                  zR0         = 0.0;
    double                                  zL0         = 0.0;
    int32_t                                 nzIdxPP     = -1;
    int32_t                                 nzIdxQQ     = -1;
    int32_t                                 nzIdxPQ     = -1;
    int32_t                                 nzIdxQP     = -1;
    int32_t                                 nodalResIdx = -1;
    int32_t                                 nodalIndIdx = -1;
    int32_t                                 origResIdx  = -1;
    int32_t                                 origIndIdx  = -1;
    std::vector<std::pair<int32_t, double>> resParts;
    std::vector<std::pair<int32_t, double>> indParts;
    std::vector<std::pair<int32_t, double>> indLParts;
};

struct NodalCoupledIndStamp {
    int32_t origIndIdx  = -1;
    int32_t nodalIndIdx = -1;
    int32_t diagCsrIdx  = -1;
    double  diagBaseVal = 0.0;
};

struct NodalCoupledMutStamp {
    int32_t origIndIdx1 = -1;
    int32_t origIndIdx2 = -1;
    int32_t nodalMutIdx = -1;
    int32_t csrIdx      = -1;
    double  baseVal     = 0.0;
};

struct TrialDeviceScales {
    std::vector<double>                       indScale;
    std::vector<double>                       resScale;
    std::vector<double>                       capScale;
    std::vector<double>                       jjScale;
    std::vector<double>                       vsScale;
    std::vector<double>                       psScale;
    std::vector<double>                       csScale;
    std::vector<std::pair<int32_t, Function>> srcOverrides;
};

struct Nodal2TermStamp {
    double  g0      = 0.0;
    int32_t nzIdxPP = -1;
    int32_t nzIdxQQ = -1;
    int32_t nzIdxPQ = -1;
    int32_t nzIdxQP = -1;
};

struct NodalRelTraceRef {
    int32_t devKind = 0; // 0 = none, 1 = res, 2 = ind, 3 = cap, 4 = txHalf
    int32_t devIdx  = -1;
};

struct MnaJJState {
    int64_t state     = 0;
    double  pn1       = 0.0, pn2 = 0.0, pn3 = 0.0, pn4 = 0.0, phi0 = 0.0;
    double  vn1       = 0.0, vn2 = 0.0, vn3 = 0.0, vn4 = 0.0, vn5 = 0.0, vn6 = 0.0;
    double  it        = 0.0, cTerm = 0.0, curNzBack = 0.0;
};

struct MnaP4State {
    double pn1 = 0.0, pn2 = 0.0, pn3 = 0.0, pn4 = 0.0;
};

struct MnaCapState {
    double pn1 = 0.0, pn2 = 0.0, pn3 = 0.0, pn4 = 0.0, pn5 = 0.0, pn6 = 0.0, pn7 = 0.0;
};

struct MnaIndState {
    double In2 = 0.0, In3 = 0.0, In4 = 0.0;
};

struct TxScalarState {
    double n1_1 = 0.0, n2_1 = 0.0, n1_2 = 0.0, n2_2 = 0.0;
    double nk_1 = 0.0, nk_2 = 0.0, nk1_1 = 0.0, nk1_2 = 0.0, nk2_1 = 0.0, nk2_2 = 0.0;
};

struct RollingCheckpoint {
    bool                            valid     = false;
    int64_t                         stepIdx   = -1;
    int64_t                         subSteps  = 1;
    double                          stepSize  = 0.0;
    size_t                          keptCount = 0;
    std::vector<double>             x;
    std::vector<double>             b;
    std::vector<double>             nodalX;
    std::vector<double>             nodalB;
    std::vector<double>             nodalNz;
    std::vector<GpuJJ>              nodalJJs;
    std::vector<GpuRes>             nodalRess;
    std::vector<GpuInd>             nodalInds;
    std::vector<GpuCap>             nodalCaps;
    std::vector<GpuVS>              nodalVSs;
    std::vector<GpuCCVS>            nodalCCVSs;
    std::vector<GpuVCCS>            nodalVCCSs;
    std::vector<GpuTxHalf>          nodalTxHalves;
    std::vector<MnaJJState>         mnaJJs;
    std::vector<MnaP4State>         mnaRess;
    std::vector<MnaIndState>        mnaInds;
    std::vector<MnaCapState>        mnaCaps;
    std::vector<MnaP4State>         mnaVSs;
    std::vector<MnaP4State>         mnaCCVSs;
    std::vector<MnaP4State>         mnaVCCSs;
    std::vector<TxScalarState>                 txScalars;
    std::vector<TransmissionLine::TxHistEntry> txHistFlat;
    std::vector<InLoopEventTracker>            eventTrackers;
};

class Simulation {
  private:
    bool                SLU = false;
    Input*              input_ = nullptr;
    std::vector<double> x_, b_;
    int64_t             simSize_;
    JoSIM::AnalysisType atyp_;
    bool                minOut_;
    bool                parallel_                = false;
    bool                gpu_                     = false;
    int64_t             gpuMode_                 = 1;
    bool                nodal_                   = false;
    int64_t             nodalLevel_              = 0;
    bool                bypass_                  = false;
    double              bypassTol_               = 1e-6;
    bool                fixedStep_               = false;
    bool                warnStep_                = false;
    int64_t             rollbackMode_            = 1;
    int64_t             verb_                    = 0;
    int64_t             mcRuns_                  = 1;
    int64_t             correctorIter_           = 0;
    double              correctorTol_            = 1e-3;
    int64_t             stepHalvings_            = 0;
    std::string         offendingJJ_;
    std::optional<std::pair<std::string, std::optional<double>>> saveStateFile_;
    std::optional<std::string>                                   loadStateFile_;
    bool                hasJJThermalNoise_       = false;
    bool                hasResistorThermalNoise_ = false;
    bool                needsLU_;
    bool                needsTR_ = true;
    bool                startup_;
    double              stepSize_, baseStepSize_ = 0.0, prstep_, prstart_;
    int                 subSteps_                = 1;
    int                 subStepIdx_              = 0;
    RollingCheckpoint   ckpt_[2];

    // In-loop print windowing & decimation schedule
    bool                 useDecimation_ = false;
    std::vector<uint8_t> keepStep_;
    std::vector<int32_t> keptSlot_;
    int64_t              numKeptSteps_  = 0;

    // Contiguous homogeneous device storage for fast hot-loop iteration
    std::vector<JJ>               jjs_;
    std::vector<Inductor>         inductors_;
    std::vector<Resistor>         resistors_;
    std::vector<Capacitor>        capacitors_;
    std::vector<TransmissionLine> txlines_;
    std::vector<VoltageSource>    vsources_;
    std::vector<PhaseSource>      psources_;
    std::vector<CCVS>             ccvss_;
    std::vector<VCCS>             vccss_;
#ifdef SLU
    LUSolve lu;
#else
    int64_t                              simOK_;
    klu_l_symbolic*                      Symbolic_      = nullptr;
    klu_l_common                         Common_{};
    klu_l_numeric*                       Numeric_       = nullptr;
    bool                                 usePartitions_ = false;
    int64_t                              windowSize_    = 0;
    std::vector<KluPartition>            partitions_;
    std::vector<int64_t>                 jjPartIdx_;
    std::vector<int64_t>                 jjPartNzIdx_;
    std::vector<int64_t>                 resPartIdx_;
    std::vector<int64_t>                 resPartNzIdx_;
    std::vector<int64_t>                 capPartIdx_;
    std::vector<int64_t>                 capPartNzIdx_;
    std::vector<int64_t>                 indPartIdx_;
    std::vector<int64_t>                 indPartNzIdx_;
    std::vector<std::vector<int64_t>>    indMutPartNzIdx_;

    // Tier 4 Schur-Complement Nodal Admittance solver state
    bool                                 nodalActive_    = false;
    int32_t                              nodalNRed_      = 0;
    std::vector<int32_t>                 nodalPermToSolver_;
    std::vector<int64_t>                 nodalRp_;
    std::vector<int64_t>                 nodalCi_;
    std::vector<double>                  nodalNz_;
    std::vector<double>                  nodalX_;
    std::vector<double>                  nodalB_;
    klu_l_symbolic*                      SymbolicRed_    = nullptr;
    klu_l_numeric*                       NumericRed_     = nullptr;
    klu_l_common                         CommonRed_{};
    std::vector<GpuJJ>                   nodalJJs_;
    std::vector<GpuRes>                  nodalRess_;
    std::vector<GpuInd>                  nodalInds_;
    std::vector<GpuMut>                  nodalMuts_;
    std::vector<GpuCap>                  nodalCaps_;
    std::vector<GpuVS>                   nodalVSs_;
    std::vector<GpuPS>                   nodalPSs_;
    std::vector<GpuCS>                   nodalCSs_;
    std::vector<GpuCCVS>                 nodalCCVSs_;
    std::vector<GpuVCCS>                 nodalVCCSs_;
    std::vector<GpuTxHalf>               nodalTxHalves_;
    std::vector<GpuRelTrace>             nodalRelTraces_;
    std::vector<uint8_t>                 nodalIsDirichletRed_;
    std::vector<double>                  nodalDirichletSign_;
    std::vector<double>                  nodalFixedNz_;
    std::vector<std::pair<int32_t, double>> nodalScaleH_;
    std::vector<std::pair<int32_t, double>> nodalScaleInvH_;
    std::vector<NodalRLEdgeStamp>        nodalRLEdges_;
    std::vector<NodalCoupledIndStamp>    nodalCoupledIndStamps_;
    std::vector<NodalCoupledMutStamp>    nodalCoupledMutStamps_;
    std::vector<Nodal2TermStamp>         nodalCapStamps_;
    std::vector<Nodal2TermStamp>         nodalTxStamps_;
    std::vector<NodalRelTraceRef>        nodalRelTraceRefs_;
    bool                                 useNodalPartitions_ = false;
    std::vector<NodalPartition>          nodalPartitions_;
#ifdef JOSIM_CUDA
    std::unique_ptr<CudaPartitionSolver> cudaSolver_;
#endif
#endif

    void setup(Input& iObj, Matrix& mObj);
    void save_nodal_checkpoint(RollingCheckpoint& ckpt, int64_t stepIdx);
    void restore_nodal_checkpoint(const RollingCheckpoint& ckpt);
    void rescale_nodal_timestep(double oldH, double newH, bool hasPrevStep);
    void save_mna_checkpoint(RollingCheckpoint& ckpt, int64_t stepIdx);
    void restore_mna_checkpoint(Matrix& mObj, const RollingCheckpoint& ckpt);
    void rescale_mna_timestep(Matrix& mObj, double oldH, double newH, bool hasPrevStep);
    void init_event_trackers(const Matrix& mObj);
    void finalize_event_trackers();
    inline void record_inloop_event(int64_t globR, int32_t kSlot, double rawVal) {
        eventTrackers_[globR].step(results.timeAxis[kSlot], rawVal, eventMetrics_);
    }
    void setup_partitions(Matrix& mObj);
    bool build_nodal_system(Matrix& mObj);
    bool setup_nodal_partitions(Matrix& mObj);
    bool setup_cpu_nodal(Matrix& mObj);
    bool setup_gpu_solver(Matrix& mObj);
    bool setup_gpu_monolithic(Matrix& mObj);
    bool setup_gpu_monolithic_full(Matrix& mObj);
    bool setup_gpu_nodal(Matrix& mObj);
    void factorize_lu(Matrix& mObj);
    void refactor_lu(Matrix& mObj);
    void solve_lu(Matrix& mObj);
    void free_lu();
    void trans_sim(Matrix& mObj);
    void compute_trial_scales(Matrix& mObj, std::vector<TrialDeviceScales>& outScales);
    void trans_sim_mc(Matrix& mObj);
    void trans_sim_mc_nodal(Matrix& mObj);
    void trans_sim_nodal(Matrix& mObj);
    void trans_sim_nodal_omp_windowed(Matrix& mObj);
    void trans_sim_omp_windowed(Matrix& mObj);
    void trans_sim_gpu_windowed(Matrix& mObj);
    void trans_sim_gpu_monolithic(Matrix& mObj);
    void trans_sim_gpu_monolithic_full(Matrix& mObj);
    void trans_sim_gpu_nodal(Matrix& mObj);
    void setup_b(Matrix& mObj, int64_t i, double step, double factor = 1);
    void reduce_step(Input& iObj, Matrix& mObj);
    void corrector_jj(Matrix& mObj);
    void save_state(const std::string& filename, const Matrix& mObj, int64_t currentStep = -1) const;
    void load_state(const std::string& filename, Matrix& mObj);

    void handle_cs(Matrix& mObj, double& step, const int64_t& i);
    void handle_resistors(Matrix& mObj, double& step);
    void handle_inductors(Matrix& mObj, double factor = 1);
    void handle_capacitors(Matrix& mObj);
    void handle_jj(Matrix& mObj, int64_t& i, double& step, double factor = 1);
    void handle_vs(Matrix& mObj, const int64_t& i, double& step, double factor = 1);
    void handle_ps(Matrix& mObj, const int64_t& i, double& step, double factor = 1);
    void handle_ccvs(Matrix& mObj);
    void handle_vccs(Matrix& mObj);
    void handle_tx(Matrix& mObj, const int64_t& i, double& step, double factor = 1);

  public:
    Results                                      results;
    bool                                         eventsOnly_   = false;
    uint32_t                                     eventMetrics_ = 0u;
    std::vector<uint8_t>                         useEventTracker_;
    std::vector<InLoopEventTracker>              eventTrackers_;
    std::vector<std::vector<InLoopEventTracker>> mcEventTrackers_;

    Simulation(Input& iObj, Matrix& mObj);
};
} // namespace JoSIM
#endif // JOSIM_SIMULATION_HPP
