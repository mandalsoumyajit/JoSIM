// Copyright (c) 2025 Johannes Delport
// This code is licensed under MIT license (see LICENSE for details)

#include "JoSIM/Simulation.hpp"

#include "JoSIM/AnalysisType.hpp"
#include "JoSIM/Components.hpp"
#include "JoSIM/Constants.hpp"
#include "JoSIM/Matrix.hpp"
#include "JoSIM/Model.hpp"
#include "JoSIM/ProgressBar.hpp"
#include "JoSIM/Rng.hpp"
#include "JoSIM/Spread.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace JoSIM;

Simulation::Simulation(Input& iObj, Matrix& mObj) {
    while (needsTR_) {
        // Do generic simulation setup for given step size
        setup(iObj, mObj);
        // Do solver setup
        factorize_lu(mObj);

        // Run transient simulation
        const auto tSimStart = std::chrono::steady_clock::now();
        trans_sim(mObj);
        const double tSimSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tSimStart).count();
        if (!needsTR_ && verb_ >= 1) {
            std::cout << "JoSIM: Transient simulation time: " << tSimSec << " s\n";
        }
        // If step size is too large, reduce and try again
        if (needsTR_) { reduce_step(iObj, mObj); }

        // Do solver cleanup
        free_lu();
    }
}

void Simulation::setup(Input& iObj, Matrix& mObj) {
    // Simulation setup
    simSize_                 = iObj.transSim.simsize();
    atyp_                    = iObj.argAnal;
    minOut_                  = iObj.argMin;
    parallel_                = iObj.argPar;
    gpu_                     = iObj.argGpu;
    gpuMode_                 = iObj.argGpuMode;
#ifdef JOSIM_CUDA
    if (gpu_) { CudaPartitionSolver::warmup_device(); }
#endif
    fixedStep_               = iObj.argFixedStep;
    warnStep_                = iObj.argWarnStep;
    verb_                    = iObj.argVerb;
    mcRuns_                  = std::max<int64_t>(1, iObj.argMcRuns);
    results.mcRuns           = 1;
    results.mcNumRows        = 0;
    results.mcNumKeptSteps   = 0;
    results.mcRowToSlot.clear();
    results.mcTraceOut.clear();
    correctorIter_           = iObj.argCorrector;
    correctorTol_            = iObj.argCorrectorTol;
    saveStateFile_           = iObj.saveStateFile;
    loadStateFile_           = iObj.loadStateFile;
    offendingJJ_.clear();
    hasJJThermalNoise_       = false;
    hasResistorThermalNoise_ = false;
    needsLU_                 = false;
    needsTR_                 = false;
    stepSize_                = iObj.transSim.tstep();
    prstep_                  = iObj.transSim.prstep();
    prstart_                 = iObj.transSim.prstart();
    startup_                 = iObj.transSim.startup();

    // Populate contiguous homogeneous device arrays for cache-friendly hot loops
    jjs_.clear();
    jjs_.reserve(mObj.components.junctionIndices.size());
    for (const auto& idx : mObj.components.junctionIndices) {
        const auto& jj = std::get<JJ>(mObj.components.devices[idx]);
        if (jj.thermalNoise) { hasJJThermalNoise_ = true; }
        jjs_.push_back(jj);
    }

    resistors_.clear();
    resistors_.reserve(mObj.components.resistorIndices.size());
    for (const auto& idx : mObj.components.resistorIndices) {
        const auto& res = std::get<Resistor>(mObj.components.devices[idx]);
        if (res.thermalNoise) { hasResistorThermalNoise_ = true; }
        resistors_.push_back(res);
    }

    std::unordered_map<int64_t, int64_t> indIdxMap;
    indIdxMap.reserve(mObj.components.inductorIndices.size());
    inductors_.clear();
    inductors_.reserve(mObj.components.inductorIndices.size());
    for (int64_t i = 0; i < static_cast<int64_t>(mObj.components.inductorIndices.size()); ++i) {
        int64_t devIdx    = mObj.components.inductorIndices[i];
        indIdxMap[devIdx] = i;
        inductors_.push_back(std::get<Inductor>(mObj.components.devices[devIdx]));
    }
    for (auto& ind : inductors_) {
        for (auto& m : ind.mutualInductances_) { m.first = indIdxMap.at(m.first); }
    }

    capacitors_.clear();
    capacitors_.reserve(mObj.components.capacitorIndices.size());
    for (const auto& idx : mObj.components.capacitorIndices) {
        capacitors_.push_back(std::get<Capacitor>(mObj.components.devices[idx]));
    }

    txlines_.clear();
    txlines_.reserve(mObj.components.txIndices.size());
    for (const auto& idx : mObj.components.txIndices) {
        txlines_.push_back(std::get<TransmissionLine>(mObj.components.devices[idx]));
    }

    vsources_.clear();
    vsources_.reserve(mObj.components.vsIndices.size());
    for (const auto& idx : mObj.components.vsIndices) {
        vsources_.push_back(std::get<VoltageSource>(mObj.components.devices[idx]));
    }

    psources_.clear();
    psources_.reserve(mObj.components.psIndices.size());
    for (const auto& idx : mObj.components.psIndices) {
        psources_.push_back(std::get<PhaseSource>(mObj.components.devices[idx]));
    }

    ccvss_.clear();
    ccvss_.reserve(mObj.components.ccvsIndices.size());
    for (const auto& idx : mObj.components.ccvsIndices) {
        ccvss_.push_back(std::get<CCVS>(mObj.components.devices[idx]));
    }

    vccss_.clear();
    vccss_.reserve(mObj.components.vccsIndices.size());
    for (const auto& idx : mObj.components.vccsIndices) {
        vccss_.push_back(std::get<VCCS>(mObj.components.devices[idx]));
    }

    // Determine whether in-loop print windowing / decimation can be used
    useDecimation_ = false;
    keepStep_.clear();
    results.printIndices.clear();
    int64_t reserveSteps = simSize_;
    if (iObj.transSim.prfirwindow() <= 0.0 && (prstart_ > 0.0 || prstep_ > stepSize_ * (1.0 + 1e-9))) {
        bool needsFullHistoryPhase = false;
        bool needsBDF2Lookback     = false;
        for (const auto& tr : mObj.relevantTraces) {
            const bool isJJ = (tr.deviceLabel.has_value() && tr.deviceLabel->size() > 3 && tr.deviceLabel->at(3) == 'B'
                               && tr.variableIndex.has_value());
            if (atyp_ == AnalysisType::Voltage && tr.storageType == StorageType::Phase && !isJJ) {
                needsFullHistoryPhase = true;
            }
            if (atyp_ == AnalysisType::Phase && tr.storageType == StorageType::Voltage && !isJJ) {
                needsBDF2Lookback = true;
            }
        }
        if (!needsFullHistoryPhase && simSize_ > 0) {
            std::vector<int64_t> printSteps;
            int64_t              firstStep = -1;
            for (int64_t i = 0; i < simSize_; ++i) {
                if (i * stepSize_ >= prstart_) {
                    firstStep = i;
                    printSteps.push_back(i);
                    break;
                }
            }
            if (firstStep != -1) {
                double nextPrintPoint = prstart_ + prstep_;
                for (int64_t i = firstStep; i < simSize_; ++i) {
                    if (i * stepSize_ >= nextPrintPoint) {
                        printSteps.push_back(i);
                        nextPrintPoint += prstep_;
                    }
                }
            }
            keepStep_.assign(static_cast<size_t>(simSize_), 0);
            for (int64_t p : printSteps) {
                keepStep_[p] = 2; // 2 = print point (and kept)
                if (needsBDF2Lookback) {
                    if (p >= 1 && keepStep_[p - 1] == 0) { keepStep_[p - 1] = 1; }
                    if (p >= 2 && keepStep_[p - 2] == 0) { keepStep_[p - 2] = 1; }
                }
            }
            int64_t keptCount = 0;
            results.printIndices.reserve(printSteps.size());
            for (int64_t i = 0; i < simSize_; ++i) {
                if (keepStep_[i] != 0) {
                    if (keepStep_[i] == 2) { results.printIndices.push_back(keptCount); }
                    ++keptCount;
                }
            }
            if (keptCount < simSize_) {
                useDecimation_ = true;
                reserveSteps   = keptCount;
            } else {
                keepStep_.clear();
                results.printIndices.clear();
            }
        }
    }

    numKeptSteps_ = reserveSteps;
    keptSlot_.assign(static_cast<size_t>(simSize_), -1);
    if (useDecimation_) {
        int32_t slot = 0;
        for (int64_t i = 0; i < simSize_; ++i) {
            if (keepStep_[i] != 0) { keptSlot_[i] = slot++; }
        }
    } else {
        for (int64_t i = 0; i < simSize_; ++i) { keptSlot_[i] = static_cast<int32_t>(i); }
    }

    b_.assign(mObj.rp.size(), 0.0);
    x_.assign(mObj.rp.size(), 0.0);
    if (!mObj.relevantTraces.empty()) {
        results.xVector.resize(mObj.branchIndex);
        for (const auto& i : mObj.relevantIndices) {
            results.xVector.at(i).emplace();
            results.xVector.at(i)->reserve(reserveSteps);
        }
    } else {
        results.xVector.resize(mObj.branchIndex, std::vector<double>(0));
        for (auto& v : results.xVector) { v->reserve(reserveSteps); }
    }

    setup_partitions(mObj);
}

void Simulation::setup_partitions(Matrix& mObj) {
#ifndef SLU
    usePartitions_ = false;
    windowSize_    = 0;
    partitions_.clear();
    jjPartIdx_.clear();
    jjPartNzIdx_.clear();
    resPartIdx_.clear();
    resPartNzIdx_.clear();
    capPartIdx_.clear();
    capPartNzIdx_.clear();
    indPartIdx_.clear();
    indPartNzIdx_.clear();
    indMutPartNzIdx_.clear();
#ifdef JOSIM_CUDA
    cudaSolver_.reset();
#endif
    const int64_t N = static_cast<int64_t>(mObj.rp.size()) - 1;
    if (gpu_ && gpuMode_ >= 2 && mcRuns_ <= 1) { return; }
    if (N < 32 && mcRuns_ <= 1) { return; }
    if (N < 4) { return; }

    std::vector<int64_t> parent(N);
    for (int64_t i = 0; i < N; ++i) { parent[i] = i; }
    auto find_root = [&](int64_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x         = parent[x];
        }
        return x;
    };
    for (int64_t r = 0; r < N; ++r) {
        const int64_t rowStart = mObj.rp[r];
        const int64_t rowEnd   = mObj.rp[r + 1];
        if (rowStart == rowEnd) { return; } // Singular empty row; let standard KLU handle error
        const int64_t rootR = find_root(r);
        for (int64_t k = rowStart; k < rowEnd; ++k) {
            const int64_t c = mObj.ci[k];
            if (c >= 0 && c < N) {
                const int64_t rootC = find_root(c);
                if (rootR != rootC) { parent[rootC] = rootR; }
            }
        }
    }
    // Ensure POSNEG current sources stay within a single partition
    for (const auto& cs : mObj.components.currentsources) {
        if (cs.indexInfo.posIndex_ && cs.indexInfo.negIndex_) {
            const int64_t r1 = find_root(cs.indexInfo.posIndex_.value());
            const int64_t r2 = find_root(cs.indexInfo.negIndex_.value());
            if (r1 != r2) { parent[r2] = r1; }
        }
    }

    std::unordered_map<int64_t, int64_t> rootToPart;
    std::vector<int64_t>                 rowPart(N);
    for (int64_t r = 0; r < N; ++r) {
        const int64_t root = find_root(r);
        auto          it   = rootToPart.find(root);
        if (it == rootToPart.end()) {
            const int64_t newPart = static_cast<int64_t>(rootToPart.size());
            rootToPart.emplace(root, newPart);
            rowPart[r] = newPart;
        } else {
            rowPart[r] = it->second;
        }
    }

    const int64_t nParts = static_cast<int64_t>(rootToPart.size());
    if (nParts < 2 && mcRuns_ <= 1 && !(gpu_ && N <= 4096)) { return; }

    partitions_.resize(nParts);
    std::vector<int64_t> localRow(N, 0);
    for (int64_t r = 0; r < N; ++r) {
        const int64_t p  = rowPart[r];
        localRow[r]      = partitions_[p].n++;
        partitions_[p].globalRows.push_back(r);
    }

    std::vector<int64_t> globalNzToPartNz(mObj.nz.size(), -1);
    for (int64_t p = 0; p < nParts; ++p) {
        auto& part = partitions_[p];
        part.rp.reserve(part.n + 1);
        part.rp.push_back(0);
        part.xLocal.assign(part.n, 0.0);
        for (int64_t locR = 0; locR < part.n; ++locR) {
            const int64_t globR    = part.globalRows[locR];
            const int64_t rowStart = mObj.rp[globR];
            const int64_t rowEnd   = mObj.rp[globR + 1];
            for (int64_t k = rowStart; k < rowEnd; ++k) {
                globalNzToPartNz[k] = static_cast<int64_t>(part.nz.size());
                part.ci.push_back(localRow[mObj.ci[k]]);
                part.nz.push_back(mObj.nz[k]);
            }
            part.rp.push_back(static_cast<int64_t>(part.nz.size()));
        }
    }

    jjPartIdx_.resize(jjs_.size(), 0);
    jjPartNzIdx_.resize(jjs_.size(), 0);
    for (size_t idx = 0; idx < jjs_.size(); ++idx) {
        const int64_t globR = jjs_[idx].indexInfo.currentIndex_.value();
        const int64_t p     = rowPart[globR];
        jjPartIdx_[idx]     = p;
        jjPartNzIdx_[idx]   = globalNzToPartNz[jjs_[idx].nzBackIndex_];
        partitions_[p].jjIndices.push_back(static_cast<int64_t>(idx));
    }
    auto find_part_nz = [&](int64_t p, int64_t globR, int64_t globC) -> int64_t {
        const auto&   part = partitions_[p];
        const int64_t locR = localRow[globR];
        const int64_t locC = localRow[globC];
        for (int64_t k = part.rp[locR]; k < part.rp[locR + 1]; ++k) {
            if (part.ci[k] == locC) { return k; }
        }
        return -1;
    };

    resPartIdx_.resize(resistors_.size(), 0);
    resPartNzIdx_.resize(resistors_.size(), -1);
    for (size_t idx = 0; idx < resistors_.size(); ++idx) {
        const int64_t globR = resistors_[idx].indexInfo.currentIndex_.value();
        const int64_t p     = rowPart[globR];
        resPartIdx_[idx]    = p;
        resPartNzIdx_[idx]  = find_part_nz(p, globR, globR);
        partitions_[p].resIndices.push_back(static_cast<int64_t>(idx));
    }

    indPartIdx_.resize(inductors_.size(), 0);
    indPartNzIdx_.resize(inductors_.size(), -1);
    indMutPartNzIdx_.resize(inductors_.size());
    for (size_t idx = 0; idx < inductors_.size(); ++idx) {
        const int64_t globR = inductors_[idx].indexInfo.currentIndex_.value();
        const int64_t p     = rowPart[globR];
        indPartIdx_[idx]    = p;
        indPartNzIdx_[idx]  = find_part_nz(p, globR, globR);
        const auto& muts    = inductors_[idx].get_mutualInductance();
        indMutPartNzIdx_[idx].resize(muts.size(), -1);
        for (size_t mIdx = 0; mIdx < muts.size(); ++mIdx) {
            const int64_t globC           = inductors_[muts[mIdx].first].indexInfo.currentIndex_.value();
            indMutPartNzIdx_[idx][mIdx]   = find_part_nz(p, globR, globC);
        }
        partitions_[p].indIndices.push_back(static_cast<int64_t>(idx));
    }

    capPartIdx_.resize(capacitors_.size(), 0);
    capPartNzIdx_.resize(capacitors_.size(), -1);
    for (size_t idx = 0; idx < capacitors_.size(); ++idx) {
        const int64_t globR = capacitors_[idx].indexInfo.currentIndex_.value();
        const int64_t p     = rowPart[globR];
        capPartIdx_[idx]    = p;
        capPartNzIdx_[idx]  = find_part_nz(p, globR, globR);
        partitions_[p].capIndices.push_back(static_cast<int64_t>(idx));
    }
    for (size_t idx = 0; idx < vsources_.size(); ++idx) {
        partitions_[rowPart[vsources_[idx].indexInfo.currentIndex_.value()]].vsIndices.push_back(
                static_cast<int64_t>(idx));
    }
    for (size_t idx = 0; idx < psources_.size(); ++idx) {
        partitions_[rowPart[psources_[idx].indexInfo.currentIndex_.value()]].psIndices.push_back(
                static_cast<int64_t>(idx));
    }
    for (size_t idx = 0; idx < mObj.components.currentsources.size(); ++idx) {
        const auto& cs = mObj.components.currentsources[idx];
        int64_t     p  = 0;
        if (cs.indexInfo.posIndex_) {
            p = rowPart[cs.indexInfo.posIndex_.value()];
        } else if (cs.indexInfo.negIndex_) {
            p = rowPart[cs.indexInfo.negIndex_.value()];
        }
        partitions_[p].csIndices.push_back(static_cast<int64_t>(idx));
    }
    for (size_t idx = 0; idx < ccvss_.size(); ++idx) {
        partitions_[rowPart[ccvss_[idx].indexInfo.currentIndex_.value()]].ccvsIndices.push_back(
                static_cast<int64_t>(idx));
    }
    for (size_t idx = 0; idx < vccss_.size(); ++idx) {
        partitions_[rowPart[vccss_[idx].indexInfo.currentIndex_.value()]].vccsIndices.push_back(
                static_cast<int64_t>(idx));
    }
    int64_t kMin = txlines_.empty() ? 8192 : 1024;
    for (size_t idx = 0; idx < txlines_.size(); ++idx) {
        const auto& tx = txlines_[idx];
        kMin           = std::min<int64_t>(kMin, tx.timestepDelay_);
        partitions_[rowPart[tx.indexInfo.currentIndex_.value()]].txPort1Indices.push_back(
                static_cast<int64_t>(idx));
        partitions_[rowPart[tx.currentIndex2_]].txPort2Indices.push_back(static_cast<int64_t>(idx));
    }
    windowSize_ = (kMin >= 1) ? std::min<int64_t>(kMin, std::max<int64_t>(1, simSize_)) : 0;

    if (!mObj.relevantTraces.empty()) {
        for (const auto& globR : mObj.relevantIndices) {
            if (globR >= 0 && globR < N) {
                partitions_[rowPart[globR]].traceGlobalRows.push_back(globR);
            }
        }
    } else {
        for (int64_t globR = 0; globR < static_cast<int64_t>(results.xVector.size()) && globR < N; ++globR) {
            if (results.xVector[globR].has_value()) {
                partitions_[rowPart[globR]].traceGlobalRows.push_back(globR);
            }
        }
    }

    usePartitions_ = true;
    if (verb_ >= 1) {
        std::cout << "JoSIM: Partitioned MNA matrix (" << N << " equations) into " << nParts
                  << " independent KLU blocks (PTL lookahead window W = " << windowSize_ << " steps)\n";
    }
#else
    (void) mObj;
#endif
}

bool Simulation::setup_gpu_solver(Matrix& mObj) {
#if defined(JOSIM_CUDA) && !defined(SLU)
    if (!usePartitions_ || windowSize_ < 1 || hasJJThermalNoise_ || hasResistorThermalNoise_
        || correctorIter_ > 0 || saveStateFile_.has_value()) {
        return false;
    }

    const int64_t nParts = static_cast<int64_t>(partitions_.size());
    const int64_t N      = static_cast<int64_t>(mObj.rp.size()) - 1;

    std::vector<int32_t> globalToAbsLocal(N, 0);
    int32_t              rowOffset = 0;
    for (int64_t p = 0; p < nParts; ++p) {
        const auto& part = partitions_[p];
        for (int64_t locR = 0; locR < part.n; ++locR) {
            globalToAbsLocal[part.globalRows[locR]] = rowOffset + static_cast<int32_t>(locR);
        }
        rowOffset += static_cast<int32_t>(part.n);
    }

    std::vector<GpuPartDesc>    gpuParts(nParts);
    std::vector<int32_t>        P;
    std::vector<int32_t>        Q;
    std::vector<int32_t>        luRp;
    std::vector<int32_t>        luDiag;
    std::vector<int32_t>        luCi;
    std::vector<int32_t>        luSrcNz;
    std::vector<double>         luVal;
    std::vector<double>         partNz;
    std::vector<double>         xInit(rowOffset, 0.0);
    for (int64_t r = 0; r < N; ++r) { xInit[globalToAbsLocal[r]] = x_[r]; }

    P.reserve(rowOffset);
    Q.reserve(rowOffset);
    luDiag.reserve(rowOffset);
    luRp.reserve(rowOffset + nParts);

    std::vector<GpuJJ>          gpuJJs;
    std::vector<GpuRes>         gpuRess;
    std::vector<GpuInd>         gpuInds;
    std::vector<GpuMut>         gpuMuts;
    std::vector<GpuCap>         gpuCaps;
    std::vector<GpuVS>          gpuVSs;
    std::vector<GpuPS>          gpuPSs;
    std::vector<GpuCS>          gpuCSs;
    std::vector<GpuCCVS>        gpuCCVSs;
    std::vector<GpuVCCS>        gpuVCCSs;
    std::vector<GpuTxHalf>      gpuTxHalves;
    std::vector<GpuRelTrace>    gpuRelTraces;

    gpuJJs.reserve(jjs_.size());
    gpuRess.reserve(resistors_.size());
    gpuInds.reserve(inductors_.size());
    gpuCaps.reserve(capacitors_.size());
    gpuVSs.reserve(vsources_.size());
    gpuPSs.reserve(psources_.size());
    gpuCSs.reserve(mObj.components.currentsources.size());
    gpuCCVSs.reserve(ccvss_.size());
    gpuVCCSs.reserve(vccss_.size());
    gpuTxHalves.reserve(txlines_.size() * 2);

    std::vector<int32_t> globalIndToGpuInd(inductors_.size(), -1);
    {
        int32_t gIndCounter = 0;
        for (int64_t p = 0; p < nParts; ++p) {
            for (int64_t idx : partitions_[p].indIndices) {
                globalIndToGpuInd[idx] = gIndCounter++;
            }
        }
    }

    int32_t curRowOff = 0;
    for (int64_t p = 0; p < nParts; ++p) {
        const auto& part = partitions_[p];
        if (!part.symbolic || !part.numeric || !part.symbolic->Q || !part.numeric->Pnum) {
            return false;
        }
        const int32_t n      = static_cast<int32_t>(part.n);
        GpuPartDesc&  desc   = gpuParts[p];
        desc.n               = n;
        desc.rowStart        = curRowOff;
        desc.rpStart         = static_cast<int32_t>(luRp.size());
        desc.luStart         = static_cast<int32_t>(luCi.size());
        desc.nzStart         = static_cast<int32_t>(partNz.size());
        desc.nzCount         = static_cast<int32_t>(part.nz.size());

        partNz.insert(partNz.end(), part.nz.begin(), part.nz.end());

        std::vector<int32_t> invQ(n, 0);
        for (int32_t j = 0; j < n; ++j) {
            const int32_t pRow = static_cast<int32_t>(part.symbolic->Q[j]);
            const int32_t qCol = static_cast<int32_t>(part.numeric->Pnum[j]);
            P.push_back(pRow);
            Q.push_back(qCol);
            invQ[qCol] = j;
        }

        // Build static CSR (L_p + U_p) pattern and initial numeric values for P_p * A_p * Q_p
        for (int32_t r = 0; r < n; ++r) {
            luRp.push_back(static_cast<int32_t>(luCi.size()) - desc.luStart);
            const int32_t origR  = P[curRowOff + r];
            const int64_t kStart = part.rp[origR];
            const int64_t kEnd   = part.rp[origR + 1];

            // Collect original nonzeros in permuted column order
            std::vector<std::pair<int32_t, int32_t>> origCols;
            origCols.reserve(static_cast<size_t>(kEnd - kStart + 4));
            for (int64_t k = kStart; k < kEnd; ++k) {
                origCols.emplace_back(invQ[part.ci[k]], desc.nzStart + static_cast<int32_t>(k));
            }
            std::sort(origCols.begin(), origCols.end());
            // Deduplicate if any parallel static elements stamped the same (row, col)
            if (origCols.size() > 1) {
                size_t writePos = 0;
                for (size_t readPos = 1; readPos < origCols.size(); ++readPos) {
                    if (origCols[readPos].first == origCols[writePos].first) {
                        partNz[origCols[writePos].second] += partNz[origCols[readPos].second];
                    } else {
                        origCols[++writePos] = origCols[readPos];
                    }
                }
                origCols.resize(writePos + 1);
            }

            // Symbolic fill-in closure across k < r
            std::vector<std::pair<int32_t, int32_t>> rowEntries = origCols;
            // Ensure diagonal entry r exists symbolically
            {
                auto itDiag = std::lower_bound(
                        rowEntries.begin(), rowEntries.end(), std::make_pair(r, -1));
                if (itDiag == rowEntries.end() || itDiag->first != r) {
                    rowEntries.insert(itDiag, std::make_pair(r, -1));
                }
            }
            for (size_t idx = 0; idx < rowEntries.size(); ++idx) {
                const int32_t colK = rowEntries[idx].first;
                if (colK >= r) { break; }
                const int32_t kDiag = luDiag[curRowOff + colK];
                const int32_t kEnd  = luRp[desc.rpStart + colK + 1];
                for (int32_t idxU = kDiag + 1; idxU < kEnd; ++idxU) {
                    const int32_t colU = luCi[desc.luStart + idxU];
                    auto          it   = std::lower_bound(
                            rowEntries.begin(), rowEntries.end(), std::make_pair(colU, -1));
                    if (it == rowEntries.end() || it->first != colU) {
                        const size_t curIdx = idx;
                        rowEntries.insert(it, std::make_pair(colU, -1));
                        idx = curIdx;
                    }
                }
            }

            const int32_t rStart = static_cast<int32_t>(luCi.size()) - desc.luStart;
            int32_t       rDiag  = -1;
            for (const auto& kv : rowEntries) {
                const int32_t localIdx = static_cast<int32_t>(luCi.size()) - desc.luStart;
                if (kv.first == r) { rDiag = localIdx; }
                luCi.push_back(kv.first);
                luSrcNz.push_back(kv.second);
                luVal.push_back((kv.second >= 0) ? partNz[kv.second] : 0.0);
            }
            if (rDiag < 0) { return false; }
            luDiag.push_back(rDiag);
            const int32_t rEnd = static_cast<int32_t>(luCi.size()) - desc.luStart;

            // Numeric sparse IKJ elimination for row r
            for (int32_t idxL = rStart; idxL < rDiag; ++idxL) {
                const int32_t colK  = luCi[desc.luStart + idxL];
                const int32_t kDiag = luDiag[curRowOff + colK];
                const int32_t kEndU = luRp[desc.rpStart + colK + 1];
                const double  piv   = luVal[desc.luStart + kDiag];
                if (piv == 0.0) { return false; }
                const double mult           = luVal[desc.luStart + idxL] / piv;
                luVal[desc.luStart + idxL]  = mult;
                int32_t pI                  = idxL + 1;
                for (int32_t idxU = kDiag + 1; idxU < kEndU; ++idxU) {
                    const int32_t colU = luCi[desc.luStart + idxU];
                    while (luCi[desc.luStart + pI] < colU) { ++pI; }
                    luVal[desc.luStart + pI] -= mult * luVal[desc.luStart + idxU];
                }
            }
            if (luVal[desc.luStart + rDiag] == 0.0) { return false; }
        }
        luRp.push_back(static_cast<int32_t>(luCi.size()) - desc.luStart);
        desc.luNnz = static_cast<int32_t>(luCi.size()) - desc.luStart;

        // Populate device structs for partition p
        desc.jjStart = static_cast<int32_t>(gpuJJs.size());
        desc.jjCount = static_cast<int32_t>(part.jjIndices.size());
        for (int64_t jIdx : part.jjIndices) {
            const auto& jj   = jjs_[jIdx];
            const auto& m    = jj.model_;
            GpuJJ       gjj{};
            gjj.posIdx       = jj.indexInfo.posIndex_ ? globalToAbsLocal[jj.indexInfo.posIndex_.value()] : -1;
            gjj.negIdx       = jj.indexInfo.negIndex_ ? globalToAbsLocal[jj.indexInfo.negIndex_.value()] : -1;
            gjj.curIdx       = globalToAbsLocal[jj.indexInfo.currentIndex_.value()];
            gjj.varIdx       = globalToAbsLocal[jj.variableIndex_];
            gjj.nzIdx        = desc.nzStart + static_cast<int32_t>(jjPartNzIdx_[jIdx]);
            gjj.rtype        = m.rtype();
            gjj.tDep         = m.tDep() ? 1 : 0;
            const auto& cpr  = m.cpr();
            gjj.cprLen       = std::min<int32_t>(8, static_cast<int32_t>(cpr.size()));
            for (int32_t h = 0; h < gjj.cprLen; ++h) { gjj.cpr[h] = cpr[h]; }
            gjj.ic           = m.ic();
            gjj.c            = m.c();
            gjj.r0           = m.r0();
            gjj.rn           = m.rn();
            gjj.phiOff       = m.phiOff();
            gjj.t            = m.t();
            gjj.d            = m.d();
            gjj.del          = jj.del_;
            gjj.lowerB       = jj.lowerB_;
            gjj.upperB       = jj.upperB_;
            gjj.gLarge       = jj.gLarge_;
            gjj.negInvSubgap = jj.negInvSubgap_;
            gjj.negInvTrans  = jj.negInvTrans_;
            gjj.negInvNormal = jj.negInvNormal_;
            gjj.pn1          = jj.pn1_;
            gjj.pn2          = jj.pn2_;
            gjj.pn3          = jj.pn3_;
            gjj.pn4          = jj.pn4_;
            gjj.phi0         = jj.phi0_;
            gjj.vn1          = jj.vn1_;
            gjj.vn2          = jj.vn2_;
            gjj.vn3          = jj.vn3_;
            gjj.vn4          = jj.vn4_;
            gjj.vn5          = jj.vn5_;
            gjj.vn6          = jj.vn6_;
            gjj.it           = jj.it_;
            gjj.curNzVal     = jj.matrixInfo.nonZeros_.back();
            gpuJJs.push_back(gjj);
        }

        desc.resStart = static_cast<int32_t>(gpuRess.size());
        desc.resCount = static_cast<int32_t>(part.resIndices.size());
        for (int64_t rIdx : part.resIndices) {
            const auto& res  = resistors_[rIdx];
            GpuRes      gres{};
            gres.posIdx      = res.indexInfo.posIndex_ ? globalToAbsLocal[res.indexInfo.posIndex_.value()] : -1;
            gres.negIdx      = res.indexInfo.negIndex_ ? globalToAbsLocal[res.indexInfo.negIndex_.value()] : -1;
            gres.curIdx      = globalToAbsLocal[res.indexInfo.currentIndex_.value()];
            gres.pn1         = res.pn1_;
            gres.pn2         = res.pn2_;
            gpuRess.push_back(gres);
        }

        desc.indStart = static_cast<int32_t>(gpuInds.size());
        desc.indCount = static_cast<int32_t>(part.indIndices.size());
        for (int64_t lIdx : part.indIndices) {
            const auto& ind  = inductors_[lIdx];
            GpuInd      gind{};
            gind.curIdx      = globalToAbsLocal[ind.indexInfo.currentIndex_.value()];
            gind.L           = ind.netlistInfo.value_;
            gind.In2         = ind.In2_;
            gind.In3         = ind.In3_;
            gind.In4         = ind.In4_;
            gind.mutStart    = static_cast<int32_t>(gpuMuts.size());
            gind.mutCount    = static_cast<int32_t>(ind.get_mutualInductance().size());
            for (const auto& m : ind.get_mutualInductance()) {
                GpuMut gmut{};
                gmut.targetIndIdx = globalIndToGpuInd[m.first];
                gmut.targetCurIdx = globalToAbsLocal[inductors_[m.first].indexInfo.currentIndex_.value()];
                gmut.M            = m.second;
                gpuMuts.push_back(gmut);
            }
            gpuInds.push_back(gind);
        }

        desc.capStart = static_cast<int32_t>(gpuCaps.size());
        desc.capCount = static_cast<int32_t>(part.capIndices.size());
        for (int64_t cIdx : part.capIndices) {
            const auto& cap  = capacitors_[cIdx];
            GpuCap      gcap{};
            gcap.posIdx      = cap.indexInfo.posIndex_ ? globalToAbsLocal[cap.indexInfo.posIndex_.value()] : -1;
            gcap.negIdx      = cap.indexInfo.negIndex_ ? globalToAbsLocal[cap.indexInfo.negIndex_.value()] : -1;
            gcap.curIdx      = globalToAbsLocal[cap.indexInfo.currentIndex_.value()];
            gcap.pn1         = cap.pn1_;
            gcap.pn2         = cap.pn2_;
            gcap.pn3         = cap.pn3_;
            gcap.pn4         = cap.pn4_;
            gpuCaps.push_back(gcap);
        }

        desc.vsStart = static_cast<int32_t>(gpuVSs.size());
        desc.vsCount = static_cast<int32_t>(part.vsIndices.size());
        for (int64_t vIdx : part.vsIndices) {
            const auto& vs  = vsources_[vIdx];
            GpuVS       gvs{};
            gvs.posIdx      = vs.indexInfo.posIndex_ ? globalToAbsLocal[vs.indexInfo.posIndex_.value()] : -1;
            gvs.negIdx      = vs.indexInfo.negIndex_ ? globalToAbsLocal[vs.indexInfo.negIndex_.value()] : -1;
            gvs.curIdx      = globalToAbsLocal[vs.indexInfo.currentIndex_.value()];
            gvs.srcIdx      = static_cast<int32_t>(vs.sourceIndex_);
            gvs.pn1         = vs.pn1_;
            gvs.pn2         = vs.pn2_;
            gpuVSs.push_back(gvs);
        }

        desc.psStart = static_cast<int32_t>(gpuPSs.size());
        desc.psCount = static_cast<int32_t>(part.psIndices.size());
        for (int64_t pIdx : part.psIndices) {
            const auto& ps  = psources_[pIdx];
            GpuPS       gps{};
            gps.curIdx      = globalToAbsLocal[ps.indexInfo.currentIndex_.value()];
            gps.srcIdx      = static_cast<int32_t>(ps.sourceIndex_);
            gpuPSs.push_back(gps);
        }

        desc.csStart = static_cast<int32_t>(gpuCSs.size());
        desc.csCount = static_cast<int32_t>(part.csIndices.size());
        for (int64_t cIdx : part.csIndices) {
            const auto& cs  = mObj.components.currentsources[cIdx];
            GpuCS       gcs{};
            gcs.posIdx      = cs.indexInfo.posIndex_ ? globalToAbsLocal[cs.indexInfo.posIndex_.value()] : -1;
            gcs.negIdx      = cs.indexInfo.negIndex_ ? globalToAbsLocal[cs.indexInfo.negIndex_.value()] : -1;
            gcs.srcIdx      = static_cast<int32_t>(cs.sourceIndex_);
            gpuCSs.push_back(gcs);
        }

        desc.ccvsStart = static_cast<int32_t>(gpuCCVSs.size());
        desc.ccvsCount = static_cast<int32_t>(part.ccvsIndices.size());
        for (int64_t idx : part.ccvsIndices) {
            const auto& ccvs  = ccvss_[idx];
            GpuCCVS     gccvs{};
            gccvs.posIdx      = ccvs.indexInfo.posIndex_ ? globalToAbsLocal[ccvs.indexInfo.posIndex_.value()] : -1;
            gccvs.negIdx      = ccvs.indexInfo.negIndex_ ? globalToAbsLocal[ccvs.indexInfo.negIndex_.value()] : -1;
            gccvs.curIdx      = globalToAbsLocal[ccvs.indexInfo.currentIndex_.value()];
            gpuCCVSs.push_back(gccvs);
        }

        desc.vccsStart = static_cast<int32_t>(gpuVCCSs.size());
        desc.vccsCount = static_cast<int32_t>(part.vccsIndices.size());
        for (int64_t idx : part.vccsIndices) {
            const auto& vccs  = vccss_[idx];
            GpuVCCS     gvccs{};
            gvccs.posIdx2     = vccs.posIndex2_ ? globalToAbsLocal[vccs.posIndex2_.value()] : -1;
            gvccs.negIdx2     = vccs.negIndex2_ ? globalToAbsLocal[vccs.negIndex2_.value()] : -1;
            gvccs.curIdx      = globalToAbsLocal[vccs.indexInfo.currentIndex_.value()];
            gpuVCCSs.push_back(gvccs);
        }

        desc.txHalfStart = static_cast<int32_t>(gpuTxHalves.size());
        for (int64_t tIdx : part.txPort1Indices) {
            const auto& tx   = txlines_[tIdx];
            GpuTxHalf   gtxh{};
            gtxh.txIdx       = static_cast<int32_t>(tIdx);
            gtxh.isPort2     = 0;
            gtxh.posIdx      = tx.indexInfo.posIndex_ ? globalToAbsLocal[tx.indexInfo.posIndex_.value()] : -1;
            gtxh.negIdx      = tx.indexInfo.negIndex_ ? globalToAbsLocal[tx.indexInfo.negIndex_.value()] : -1;
            gtxh.curIdx      = globalToAbsLocal[tx.indexInfo.currentIndex_.value()];
            gtxh.n1          = tx.n1_1_;
            gtxh.n2          = tx.n2_1_;
            gpuTxHalves.push_back(gtxh);
        }
        for (int64_t tIdx : part.txPort2Indices) {
            const auto& tx   = txlines_[tIdx];
            GpuTxHalf   gtxh{};
            gtxh.txIdx       = static_cast<int32_t>(tIdx);
            gtxh.isPort2     = 1;
            gtxh.posIdx      = tx.posIndex2_ ? globalToAbsLocal[tx.posIndex2_.value()] : -1;
            gtxh.negIdx      = tx.negIndex2_ ? globalToAbsLocal[tx.negIndex2_.value()] : -1;
            gtxh.curIdx      = globalToAbsLocal[tx.currentIndex2_];
            gtxh.n1          = tx.n1_2_;
            gtxh.n2          = tx.n2_2_;
            gpuTxHalves.push_back(gtxh);
        }
        desc.txHalfCount = static_cast<int32_t>(gpuTxHalves.size()) - desc.txHalfStart;

        desc.relStart = static_cast<int32_t>(gpuRelTraces.size());
        desc.relCount = static_cast<int32_t>(part.traceGlobalRows.size());
        for (int64_t globR : part.traceGlobalRows) {
            GpuRelTrace grt{};
            grt.localAbsIdx = globalToAbsLocal[globR];
            grt.traceSlot   = static_cast<int32_t>(gpuRelTraces.size());
            gpuRelTraces.push_back(grt);
        }

        curRowOff += n;
    }

    std::vector<GpuTxLine>      gpuTxLines(txlines_.size());
    std::vector<GpuTxHistEntry> gpuTxHist;
    int32_t                     histOff = 0;
    for (size_t idx = 0; idx < txlines_.size(); ++idx) {
        const auto& tx       = txlines_[idx];
        GpuTxLine&  gtx      = gpuTxLines[idx];
        gtx.z0               = tx.netlistInfo.value_;
        gtx.k                = tx.timestepDelay_;
        gtx.fracDelay        = tx.fracDelay_;
        gtx.loadedSteps      = tx.loadedSteps_;
        gtx.histOffset       = histOff;
        gtx.histSize         = static_cast<int32_t>(tx.histSize_);
        histOff             += gtx.histSize;
        for (const auto& h : tx.hist_) {
            gpuTxHist.push_back({h.v1, h.v2, h.i1, h.i2});
        }
    }

    cudaSolver_ = std::make_unique<CudaPartitionSolver>();
    const bool ok = cudaSolver_->initialize(static_cast<int32_t>(nParts),
                                            curRowOff,
                                            static_cast<int32_t>(mObj.sourcegen.size()),
                                            static_cast<int32_t>(windowSize_),
                                            static_cast<int32_t>(gpuRelTraces.size()),
                                            static_cast<int32_t>(numKeptSteps_),
                                            simSize_,
                                            stepSize_,
                                            (atyp_ == AnalysisType::Voltage) ? 0 : 1,
                                            gpuParts,
                                            P,
                                            Q,
                                            luRp,
                                            luDiag,
                                            luCi,
                                            luSrcNz,
                                            luVal,
                                            partNz,
                                            xInit,
                                            gpuJJs,
                                            gpuRess,
                                            gpuInds,
                                            gpuMuts,
                                            gpuCaps,
                                            gpuVSs,
                                            gpuPSs,
                                            gpuCSs,
                                            gpuCCVSs,
                                            gpuVCCSs,
                                            gpuTxHalves,
                                            gpuTxLines,
                                            gpuTxHist,
                                            gpuRelTraces,
                                            keptSlot_);
    if (!ok) {
        cudaSolver_.reset();
        return false;
    }
    if (verb_ >= 1) {
        std::cout << "JoSIM: Initialized CUDA GPU PTL Block-Parallel Solver (" << nParts << " partitions, "
                  << luCi.size() << " LU nonzeros, window W = " << windowSize_ << ")\n";
    }
    return true;
#else
    (void) mObj;
    return false;
#endif
}

bool Simulation::setup_gpu_monolithic(Matrix& mObj) {
#if defined(JOSIM_CUDA) && !defined(SLU)
    if (usePartitions_ || hasJJThermalNoise_ || hasResistorThermalNoise_ || correctorIter_ > 0
        || saveStateFile_.has_value()) {
        return false;
    }
    const int64_t N = static_cast<int64_t>(mObj.rp.size()) - 1;
    if (N < 32) { return false; }

    std::vector<GpuJJ>  gpuJJs(jjs_.size());
    std::vector<GpuRes> gpuRess(resistors_.size());
    std::vector<GpuInd> gpuInds(inductors_.size());
    std::vector<GpuMut> gpuMuts;
    std::vector<GpuCap> gpuCaps(capacitors_.size());

    for (size_t idx = 0; idx < jjs_.size(); ++idx) {
        const auto& jj   = jjs_[idx];
        const auto& m    = jj.model_;
        GpuJJ&      gjj  = gpuJJs[idx];
        gjj.posIdx       = jj.indexInfo.posIndex_ ? static_cast<int32_t>(jj.indexInfo.posIndex_.value()) : -1;
        gjj.negIdx       = jj.indexInfo.negIndex_ ? static_cast<int32_t>(jj.indexInfo.negIndex_.value()) : -1;
        gjj.curIdx       = static_cast<int32_t>(jj.indexInfo.currentIndex_.value());
        gjj.varIdx       = static_cast<int32_t>(jj.variableIndex_);
        gjj.nzIdx        = static_cast<int32_t>(jj.nzBackIndex_);
        gjj.rtype        = m.rtype();
        gjj.tDep         = m.tDep() ? 1 : 0;
        const auto& cpr  = m.cpr();
        gjj.cprLen       = std::min<int32_t>(8, static_cast<int32_t>(cpr.size()));
        for (int32_t h = 0; h < gjj.cprLen; ++h) { gjj.cpr[h] = cpr[h]; }
        gjj.ic           = m.ic();
        gjj.c            = m.c();
        gjj.r0           = m.r0();
        gjj.rn           = m.rn();
        gjj.phiOff       = m.phiOff();
        gjj.t            = m.t();
        gjj.d            = m.d();
        gjj.del          = jj.del_;
        gjj.lowerB       = jj.lowerB_;
        gjj.upperB       = jj.upperB_;
        gjj.gLarge       = jj.gLarge_;
        gjj.negInvSubgap = jj.negInvSubgap_;
        gjj.negInvTrans  = jj.negInvTrans_;
        gjj.negInvNormal = jj.negInvNormal_;
        gjj.pn1          = jj.pn1_;
        gjj.pn2          = jj.pn2_;
        gjj.pn3          = jj.pn3_;
        gjj.pn4          = jj.pn4_;
        gjj.phi0         = jj.phi0_;
        gjj.vn1          = jj.vn1_;
        gjj.vn2          = jj.vn2_;
        gjj.vn3          = jj.vn3_;
        gjj.vn4          = jj.vn4_;
        gjj.vn5          = jj.vn5_;
        gjj.vn6          = jj.vn6_;
        gjj.it           = jj.it_;
        gjj.curNzVal     = jj.matrixInfo.nonZeros_.back();
        gjj.cTerm        = jj.cTerm_;
    }

    for (size_t idx = 0; idx < resistors_.size(); ++idx) {
        const auto& res  = resistors_[idx];
        GpuRes&     gres = gpuRess[idx];
        gres.posIdx      = res.indexInfo.posIndex_ ? static_cast<int32_t>(res.indexInfo.posIndex_.value()) : -1;
        gres.negIdx      = res.indexInfo.negIndex_ ? static_cast<int32_t>(res.indexInfo.negIndex_.value()) : -1;
        gres.curIdx      = static_cast<int32_t>(res.indexInfo.currentIndex_.value());
        gres.pn1         = res.pn1_;
        gres.pn2         = res.pn2_;
    }

    for (size_t idx = 0; idx < inductors_.size(); ++idx) {
        const auto& ind  = inductors_[idx];
        GpuInd&     gind = gpuInds[idx];
        gind.curIdx      = static_cast<int32_t>(ind.indexInfo.currentIndex_.value());
        gind.mutStart    = static_cast<int32_t>(gpuMuts.size());
        gind.L           = ind.netlistInfo.value_;
        gind.In2         = ind.In2_;
        gind.In3         = ind.In3_;
        gind.In4         = ind.In4_;
        for (const auto& m : ind.get_mutualInductance()) {
            GpuMut gmut{};
            gmut.targetIndIdx = static_cast<int32_t>(m.first);
            gmut.targetCurIdx = static_cast<int32_t>(inductors_[m.first].indexInfo.currentIndex_.value());
            gmut.M            = m.second;
            gpuMuts.push_back(gmut);
        }
        gind.mutCount = static_cast<int32_t>(gpuMuts.size()) - gind.mutStart;
    }

    for (size_t idx = 0; idx < capacitors_.size(); ++idx) {
        const auto& cap  = capacitors_[idx];
        GpuCap&     gcap = gpuCaps[idx];
        gcap.posIdx      = cap.indexInfo.posIndex_ ? static_cast<int32_t>(cap.indexInfo.posIndex_.value()) : -1;
        gcap.negIdx      = cap.indexInfo.negIndex_ ? static_cast<int32_t>(cap.indexInfo.negIndex_.value()) : -1;
        gcap.curIdx      = static_cast<int32_t>(cap.indexInfo.currentIndex_.value());
        gcap.pn1         = cap.pn1_;
        gcap.pn2         = cap.pn2_;
        gcap.pn3         = cap.pn3_;
        gcap.pn4         = cap.pn4_;
    }

    std::vector<double> xInit(x_.begin(), x_.begin() + N);
    cudaSolver_   = std::make_unique<CudaPartitionSolver>();
    const bool ok = cudaSolver_->initialize_monolithic(static_cast<int32_t>(N),
                                                       simSize_,
                                                       stepSize_,
                                                       (atyp_ == AnalysisType::Voltage) ? 0 : 1,
                                                       xInit,
                                                       gpuJJs,
                                                       gpuRess,
                                                       gpuInds,
                                                       gpuMuts,
                                                       gpuCaps);
    if (!ok) {
        cudaSolver_.reset();
        return false;
    }
    if (verb_ >= 1) {
        std::cout << "JoSIM: Initialized CUDA GPU Monolithic Device Solver (" << jjs_.size() << " JJs, "
                  << inductors_.size() << " inductors, " << gpuMuts.size() << " mutual terms)\n";
    }
    return true;
#else
    (void) mObj;
    return false;
#endif
}

bool Simulation::setup_gpu_monolithic_full(Matrix& mObj) {
#if defined(JOSIM_CUDA) && !defined(SLU)
    if (usePartitions_ || hasJJThermalNoise_ || hasResistorThermalNoise_ || correctorIter_ > 0
        || saveStateFile_.has_value()) {
        return false;
    }
    const int64_t N = static_cast<int64_t>(mObj.rp.size()) - 1;
    if (N < 4 || !Symbolic_ || !Numeric_ || !Symbolic_->Q || !Numeric_->Pnum || Common_.btf != 0 || Common_.scale != 0
        || Numeric_->Offp[N] != 0) {
        return false;
    }

    const int64_t lnz = Numeric_->lnz;
    const int64_t unz = Numeric_->unz;
    if (lnz <= 0 || unz <= 0) { return false; }

    std::vector<int64_t> Lp64(N + 1), Li64(lnz), Up64(N + 1), Ui64(unz);
    std::vector<int64_t> Fp64(N + 1), Fi64(1), Pnum64(N), Q64(N), R64(N + 1);
    std::vector<double>  Lx(lnz), Ux(unz), Fx(1), Rs(N);
    if (!klu_l_extract(Numeric_,
                       Symbolic_,
                       Lp64.data(),
                       Li64.data(),
                       Lx.data(),
                       Up64.data(),
                       Ui64.data(),
                       Ux.data(),
                       Fp64.data(),
                       Fi64.data(),
                       Fx.data(),
                       Pnum64.data(),
                       Q64.data(),
                       Rs.data(),
                       R64.data(),
                       &Common_)) {
        return false;
    }

    // Permutations for CSR matrix A: (P * A * Q) = L_csr * U_csr
    // where P[r] = Symbolic_->Q[r] and Q[c] = Numeric_->Pnum[c]
    std::vector<int32_t> P(N), Q(N);
    for (int64_t i = 0; i < N; ++i) {
        P[i] = static_cast<int32_t>(Q64[i]);
        Q[i] = static_cast<int32_t>(Pnum64[i]);
    }

    // Build sorted CSR factors L_csr (unit diagonal) and U_csr (non-unit diagonal)
    // From KLU: (P_klu * A^T * Q_klu)^T = U_klu^T * L_klu^T = (U_klu^T * D^{-1}) * (D * L_klu^T)
    std::vector<double>  diagD(N, 1.0);
    std::vector<int32_t> csrRowPtrL(N + 1, 0), csrColIndL(unz, 0);
    std::vector<double>  csrValL(unz, 0.0);
    std::vector<std::pair<int32_t, double>> rowBuf;
    for (int64_t r = 0; r < N; ++r) {
        const int64_t kStart = Up64[r];
        const int64_t kEnd   = Up64[r + 1];
        csrRowPtrL[r]        = static_cast<int32_t>(kStart);
        rowBuf.clear();
        for (int64_t k = kStart; k < kEnd; ++k) {
            rowBuf.emplace_back(static_cast<int32_t>(Ui64[k]), Ux[k]);
        }
        std::sort(rowBuf.begin(), rowBuf.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        if (rowBuf.empty() || rowBuf.back().first != static_cast<int32_t>(r) || rowBuf.back().second == 0.0) {
            return false;
        }
        diagD[r] = rowBuf.back().second;
        for (size_t idx = 0; idx < rowBuf.size(); ++idx) {
            const int32_t col        = rowBuf[idx].first;
            csrColIndL[kStart + idx] = col;
            csrValL[kStart + idx]    = (col == static_cast<int32_t>(r)) ? 1.0 : (rowBuf[idx].second / diagD[col]);
        }
    }
    csrRowPtrL[N] = static_cast<int32_t>(unz);

    std::vector<int32_t> csrRowPtrU(N + 1, 0), csrColIndU(lnz, 0);
    std::vector<double>  csrValU(lnz, 0.0);
    for (int64_t r = 0; r < N; ++r) {
        const int64_t kStart = Lp64[r];
        const int64_t kEnd   = Lp64[r + 1];
        csrRowPtrU[r]        = static_cast<int32_t>(kStart);
        rowBuf.clear();
        for (int64_t k = kStart; k < kEnd; ++k) {
            rowBuf.emplace_back(static_cast<int32_t>(Li64[k]), Lx[k]);
        }
        std::sort(rowBuf.begin(), rowBuf.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        if (rowBuf.empty() || rowBuf.front().first != static_cast<int32_t>(r)) { return false; }
        const double dR = diagD[r];
        for (size_t idx = 0; idx < rowBuf.size(); ++idx) {
            csrColIndU[kStart + idx] = rowBuf[idx].first;
            csrValU[kStart + idx]    = rowBuf[idx].second * dR;
        }
    }
    csrRowPtrU[N] = static_cast<int32_t>(lnz);

    // Deduplicate and sort CSR matrix A while mapping original nonzero indices to deduplicated CSR indices
    std::vector<int32_t> csrRowPtrA(N + 1, 0);
    std::vector<int32_t> csrColIndA;
    std::vector<double>  csrValA;
    csrColIndA.reserve(mObj.ci.size());
    csrValA.reserve(mObj.nz.size());
    std::vector<int32_t> origNzToDedupNz(mObj.nz.size(), -1);
    std::vector<std::pair<int32_t, int64_t>> colOrigPairs;
    for (int64_t r = 0; r < N; ++r) {
        csrRowPtrA[r]        = static_cast<int32_t>(csrColIndA.size());
        const int64_t kStart = mObj.rp[r];
        const int64_t kEnd   = mObj.rp[r + 1];
        colOrigPairs.clear();
        for (int64_t k = kStart; k < kEnd; ++k) {
            colOrigPairs.emplace_back(static_cast<int32_t>(mObj.ci[k]), k);
        }
        std::sort(colOrigPairs.begin(), colOrigPairs.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        for (size_t idx = 0; idx < colOrigPairs.size(); ++idx) {
            const int32_t col   = colOrigPairs[idx].first;
            const int64_t origK = colOrigPairs[idx].second;
            if (!csrColIndA.empty() && static_cast<int32_t>(csrColIndA.size()) > csrRowPtrA[r]
                && csrColIndA.back() == col) {
                csrValA.back()        += mObj.nz[origK];
                origNzToDedupNz[origK] = static_cast<int32_t>(csrValA.size()) - 1;
            } else {
                origNzToDedupNz[origK] = static_cast<int32_t>(csrValA.size());
                csrColIndA.push_back(col);
                csrValA.push_back(mObj.nz[origK]);
            }
        }
    }
    csrRowPtrA[N] = static_cast<int32_t>(csrColIndA.size());

    // Populate device arrays
    std::vector<GpuJJ>     gpuJJs(jjs_.size());
    std::vector<GpuRes>    gpuRess(resistors_.size());
    std::vector<GpuInd>    gpuInds(inductors_.size());
    std::vector<GpuMut>    gpuMuts;
    std::vector<GpuCap>    gpuCaps(capacitors_.size());
    std::vector<GpuVS>     gpuVSs(vsources_.size());
    std::vector<GpuPS>     gpuPSs(psources_.size());
    std::vector<GpuCS>     gpuCSs(mObj.components.currentsources.size());
    std::vector<GpuCCVS>   gpuCCVSs(ccvss_.size());
    std::vector<GpuVCCS>   gpuVCCSs(vccss_.size());
    std::vector<GpuTxHalf> gpuTxHalves;

    for (size_t idx = 0; idx < jjs_.size(); ++idx) {
        const auto& jj   = jjs_[idx];
        const auto& m    = jj.model_;
        GpuJJ&      gjj  = gpuJJs[idx];
        gjj.posIdx       = jj.indexInfo.posIndex_ ? static_cast<int32_t>(jj.indexInfo.posIndex_.value()) : -1;
        gjj.negIdx       = jj.indexInfo.negIndex_ ? static_cast<int32_t>(jj.indexInfo.negIndex_.value()) : -1;
        gjj.curIdx       = static_cast<int32_t>(jj.indexInfo.currentIndex_.value());
        gjj.varIdx       = static_cast<int32_t>(jj.variableIndex_);
        gjj.nzIdx        = origNzToDedupNz[jj.nzBackIndex_];
        gjj.rtype        = m.rtype();
        gjj.tDep         = m.tDep() ? 1 : 0;
        const auto& cpr  = m.cpr();
        gjj.cprLen       = std::min<int32_t>(8, static_cast<int32_t>(cpr.size()));
        for (int32_t h = 0; h < gjj.cprLen; ++h) { gjj.cpr[h] = cpr[h]; }
        gjj.ic           = m.ic();
        gjj.c            = m.c();
        gjj.r0           = m.r0();
        gjj.rn           = m.rn();
        gjj.phiOff       = m.phiOff();
        gjj.t            = m.t();
        gjj.d            = m.d();
        gjj.del          = jj.del_;
        gjj.lowerB       = jj.lowerB_;
        gjj.upperB       = jj.upperB_;
        gjj.gLarge       = jj.gLarge_;
        gjj.negInvSubgap = jj.negInvSubgap_;
        gjj.negInvTrans  = jj.negInvTrans_;
        gjj.negInvNormal = jj.negInvNormal_;
        gjj.pn1          = jj.pn1_;
        gjj.pn2          = jj.pn2_;
        gjj.pn3          = jj.pn3_;
        gjj.pn4          = jj.pn4_;
        gjj.phi0         = jj.phi0_;
        gjj.vn1          = jj.vn1_;
        gjj.vn2          = jj.vn2_;
        gjj.vn3          = jj.vn3_;
        gjj.vn4          = jj.vn4_;
        gjj.vn5          = jj.vn5_;
        gjj.vn6          = jj.vn6_;
        gjj.it           = jj.it_;
        gjj.curNzVal     = jj.matrixInfo.nonZeros_.back();
        gjj.cTerm        = jj.cTerm_;
    }

    for (size_t idx = 0; idx < resistors_.size(); ++idx) {
        const auto& res  = resistors_[idx];
        GpuRes&     gres = gpuRess[idx];
        gres.posIdx      = res.indexInfo.posIndex_ ? static_cast<int32_t>(res.indexInfo.posIndex_.value()) : -1;
        gres.negIdx      = res.indexInfo.negIndex_ ? static_cast<int32_t>(res.indexInfo.negIndex_.value()) : -1;
        gres.curIdx      = static_cast<int32_t>(res.indexInfo.currentIndex_.value());
        gres.pn1         = res.pn1_;
        gres.pn2         = res.pn2_;
    }

    for (size_t idx = 0; idx < inductors_.size(); ++idx) {
        const auto& ind  = inductors_[idx];
        GpuInd&     gind = gpuInds[idx];
        gind.curIdx      = static_cast<int32_t>(ind.indexInfo.currentIndex_.value());
        gind.mutStart    = static_cast<int32_t>(gpuMuts.size());
        gind.L           = ind.netlistInfo.value_;
        gind.In2         = ind.In2_;
        gind.In3         = ind.In3_;
        gind.In4         = ind.In4_;
        for (const auto& m : ind.get_mutualInductance()) {
            GpuMut gmut{};
            gmut.targetIndIdx = static_cast<int32_t>(m.first);
            gmut.targetCurIdx = static_cast<int32_t>(inductors_[m.first].indexInfo.currentIndex_.value());
            gmut.M            = m.second;
            gpuMuts.push_back(gmut);
        }
        gind.mutCount = static_cast<int32_t>(gpuMuts.size()) - gind.mutStart;
    }

    for (size_t idx = 0; idx < capacitors_.size(); ++idx) {
        const auto& cap  = capacitors_[idx];
        GpuCap&     gcap = gpuCaps[idx];
        gcap.posIdx      = cap.indexInfo.posIndex_ ? static_cast<int32_t>(cap.indexInfo.posIndex_.value()) : -1;
        gcap.negIdx      = cap.indexInfo.negIndex_ ? static_cast<int32_t>(cap.indexInfo.negIndex_.value()) : -1;
        gcap.curIdx      = static_cast<int32_t>(cap.indexInfo.currentIndex_.value());
        gcap.pn1         = cap.pn1_;
        gcap.pn2         = cap.pn2_;
        gcap.pn3         = cap.pn3_;
        gcap.pn4         = cap.pn4_;
    }

    for (size_t idx = 0; idx < vsources_.size(); ++idx) {
        const auto& vs  = vsources_[idx];
        GpuVS&      gvs = gpuVSs[idx];
        gvs.posIdx      = vs.indexInfo.posIndex_ ? static_cast<int32_t>(vs.indexInfo.posIndex_.value()) : -1;
        gvs.negIdx      = vs.indexInfo.negIndex_ ? static_cast<int32_t>(vs.indexInfo.negIndex_.value()) : -1;
        gvs.curIdx      = static_cast<int32_t>(vs.indexInfo.currentIndex_.value());
        gvs.srcIdx      = static_cast<int32_t>(vs.sourceIndex_);
        gvs.pn1         = vs.pn1_;
        gvs.pn2         = vs.pn2_;
    }

    for (size_t idx = 0; idx < psources_.size(); ++idx) {
        const auto& ps  = psources_[idx];
        GpuPS&      gps = gpuPSs[idx];
        gps.curIdx      = static_cast<int32_t>(ps.indexInfo.currentIndex_.value());
        gps.srcIdx      = static_cast<int32_t>(ps.sourceIndex_);
    }

    for (size_t idx = 0; idx < mObj.components.currentsources.size(); ++idx) {
        const auto& cs  = mObj.components.currentsources[idx];
        GpuCS&      gcs = gpuCSs[idx];
        gcs.posIdx      = cs.indexInfo.posIndex_ ? static_cast<int32_t>(cs.indexInfo.posIndex_.value()) : -1;
        gcs.negIdx      = cs.indexInfo.negIndex_ ? static_cast<int32_t>(cs.indexInfo.negIndex_.value()) : -1;
        gcs.srcIdx      = static_cast<int32_t>(cs.sourceIndex_);
    }

    for (size_t idx = 0; idx < ccvss_.size(); ++idx) {
        const auto& ccvs  = ccvss_[idx];
        GpuCCVS&    gccvs = gpuCCVSs[idx];
        gccvs.posIdx      = ccvs.indexInfo.posIndex_ ? static_cast<int32_t>(ccvs.indexInfo.posIndex_.value()) : -1;
        gccvs.negIdx      = ccvs.indexInfo.negIndex_ ? static_cast<int32_t>(ccvs.indexInfo.negIndex_.value()) : -1;
        gccvs.curIdx      = static_cast<int32_t>(ccvs.indexInfo.currentIndex_.value());
    }

    for (size_t idx = 0; idx < vccss_.size(); ++idx) {
        const auto& vccs  = vccss_[idx];
        GpuVCCS&    gvccs = gpuVCCSs[idx];
        gvccs.posIdx2     = vccs.posIndex2_ ? static_cast<int32_t>(vccs.posIndex2_.value()) : -1;
        gvccs.negIdx2     = vccs.negIndex2_ ? static_cast<int32_t>(vccs.negIndex2_.value()) : -1;
        gvccs.curIdx      = static_cast<int32_t>(vccs.indexInfo.currentIndex_.value());
    }

    std::vector<GpuTxLine>      gpuTxLines(txlines_.size());
    std::vector<GpuTxHistEntry> gpuTxHist;
    int32_t                     histOff = 0;
    int64_t                     kMin    = simSize_ > 0 ? simSize_ : 1;
    for (size_t idx = 0; idx < txlines_.size(); ++idx) {
        const auto& tx   = txlines_[idx];
        GpuTxLine&  gtx  = gpuTxLines[idx];
        gtx.z0           = tx.netlistInfo.value_;
        gtx.k            = tx.timestepDelay_;
        gtx.fracDelay    = tx.fracDelay_;
        gtx.loadedSteps  = tx.loadedSteps_;
        gtx.histOffset   = histOff;
        gtx.histSize     = static_cast<int32_t>(tx.histSize_);
        histOff         += gtx.histSize;
        if (tx.timestepDelay_ < kMin) { kMin = tx.timestepDelay_; }
        for (const auto& h : tx.hist_) { gpuTxHist.push_back({h.v1, h.v2, h.i1, h.i2}); }

        GpuTxHalf p1{};
        p1.txIdx   = static_cast<int32_t>(idx);
        p1.isPort2 = 0;
        p1.posIdx  = tx.indexInfo.posIndex_ ? static_cast<int32_t>(tx.indexInfo.posIndex_.value()) : -1;
        p1.negIdx  = tx.indexInfo.negIndex_ ? static_cast<int32_t>(tx.indexInfo.negIndex_.value()) : -1;
        p1.curIdx  = static_cast<int32_t>(tx.indexInfo.currentIndex_.value());
        p1.n1      = tx.n1_1_;
        p1.n2      = tx.n2_1_;
        gpuTxHalves.push_back(p1);

        GpuTxHalf p2{};
        p2.txIdx   = static_cast<int32_t>(idx);
        p2.isPort2 = 1;
        p2.posIdx  = tx.posIndex2_ ? static_cast<int32_t>(tx.posIndex2_.value()) : -1;
        p2.negIdx  = tx.negIndex2_ ? static_cast<int32_t>(tx.negIndex2_.value()) : -1;
        p2.curIdx  = static_cast<int32_t>(tx.currentIndex2_);
        p2.n1      = tx.n1_2_;
        p2.n2      = tx.n2_2_;
        gpuTxHalves.push_back(p2);
    }

    if (txlines_.empty()) {
        windowSize_ = std::max<int64_t>(1, std::min<int64_t>(simSize_, 1024));
    } else {
        windowSize_ = std::max<int64_t>(1, std::min<int64_t>(simSize_, std::min<int64_t>(kMin, 256)));
    }

    std::vector<GpuRelTrace> gpuRelTraces;
    if (!mObj.relevantTraces.empty()) {
        for (int64_t globR : mObj.relevantIndices) {
            if (globR >= 0 && globR < N) {
                GpuRelTrace grt{};
                grt.localAbsIdx = static_cast<int32_t>(globR);
                grt.traceSlot   = static_cast<int32_t>(gpuRelTraces.size());
                gpuRelTraces.push_back(grt);
            }
        }
    } else {
        for (int64_t globR = 0; globR < N; ++globR) {
            GpuRelTrace grt{};
            grt.localAbsIdx = static_cast<int32_t>(globR);
            grt.traceSlot   = static_cast<int32_t>(gpuRelTraces.size());
            gpuRelTraces.push_back(grt);
        }
    }

    std::vector<double> xInit(x_.begin(), x_.begin() + N);
    cudaSolver_   = std::make_unique<CudaPartitionSolver>();
    const bool ok = cudaSolver_->initialize_monolithic_full(static_cast<int32_t>(N),
                                                            static_cast<int32_t>(mObj.sourcegen.size()),
                                                            static_cast<int32_t>(windowSize_),
                                                            static_cast<int32_t>(gpuRelTraces.size()),
                                                            static_cast<int32_t>(numKeptSteps_),
                                                            simSize_,
                                                            stepSize_,
                                                            (atyp_ == AnalysisType::Voltage) ? 0 : 1,
                                                            csrRowPtrA,
                                                            csrColIndA,
                                                            csrValA,
                                                            csrRowPtrL,
                                                            csrColIndL,
                                                            csrValL,
                                                            csrRowPtrU,
                                                            csrColIndU,
                                                            csrValU,
                                                            P,
                                                            Q,
                                                            xInit,
                                                            gpuJJs,
                                                            gpuRess,
                                                            gpuInds,
                                                            gpuMuts,
                                                            gpuCaps,
                                                            gpuVSs,
                                                            gpuPSs,
                                                            gpuCSs,
                                                            gpuCCVSs,
                                                            gpuVCCSs,
                                                            gpuTxHalves,
                                                            gpuTxLines,
                                                            gpuTxHist,
                                                            gpuRelTraces,
                                                            keptSlot_);
    if (!ok) {
        cudaSolver_.reset();
        return false;
    }
    if (verb_ >= 1) {
        std::cout << "JoSIM: Initialized CUDA Tier 3 Fully GPU-Resident Monolithic Solver (cuSolverRF, N = " << N
                  << ", nnz(L+U) = " << (lnz + unz) << ", window W = " << windowSize_ << ")\n";
    }
    return true;
#else
    (void) mObj;
    return false;
#endif
}

void Simulation::factorize_lu(Matrix& mObj) {
#ifdef SLU
    lu.create_matrix(mObj.rp.size() - 1, mObj.nz, mObj.ci, mObj.rp);
    lu.factorize();
#else
    if (usePartitions_) {
        const int64_t nParts = static_cast<int64_t>(partitions_.size());
#ifdef _OPENMP
        if ((parallel_ || gpu_) && nParts >= 4 && b_.size() >= 8192) {
            const int nThreads = std::min<int>(32, std::min<int>(omp_get_max_threads(), static_cast<int>(nParts)));
#pragma omp parallel for schedule(static) num_threads(nThreads)
            for (int64_t p = 0; p < nParts; ++p) {
                auto&   part = partitions_[p];
                int64_t ok   = klu_l_defaults(&part.common);
                assert(ok);
                (void) ok;
                if (gpu_ || mcRuns_ > 1) {
                    part.common.btf   = 0;
                    part.common.scale = 0;
                }
                part.symbolic = klu_l_analyze(part.n, &part.rp.front(), &part.ci.front(), &part.common);
                part.numeric  = klu_l_factor(
                        &part.rp.front(), &part.ci.front(), &part.nz.front(), part.symbolic, &part.common);
            }
            return;
        }
#endif
        for (int64_t p = 0; p < nParts; ++p) {
            auto&   part = partitions_[p];
            int64_t ok   = klu_l_defaults(&part.common);
            assert(ok);
            (void) ok;
            if (gpu_ || mcRuns_ > 1) {
                part.common.btf   = 0;
                part.common.scale = 0;
            }
            part.symbolic = klu_l_analyze(part.n, &part.rp.front(), &part.ci.front(), &part.common);
            part.numeric
                    = klu_l_factor(&part.rp.front(), &part.ci.front(), &part.nz.front(), part.symbolic, &part.common);
        }
    } else {
        simOK_ = klu_l_defaults(&Common_);
        assert(simOK_);
        if (gpu_ && (gpuMode_ == 3 || gpuMode_ == 1)) {
            Common_.btf   = 0;
            Common_.scale = 0;
        }
        Symbolic_ = klu_l_analyze(mObj.rp.size() - 1, &mObj.rp.front(), &mObj.ci.front(), &Common_);
        Numeric_  = klu_l_factor(&mObj.rp.front(), &mObj.ci.front(), &mObj.nz.front(), Symbolic_, &Common_);
        if (gpu_ && (gpuMode_ == 3 || gpuMode_ == 1) && (!Numeric_ || Common_.status != KLU_OK)) {
            if (Numeric_) { klu_l_free_numeric(&Numeric_, &Common_); }
            if (Symbolic_) { klu_l_free_symbolic(&Symbolic_, &Common_); }
            klu_l_defaults(&Common_);
            Symbolic_ = klu_l_analyze(mObj.rp.size() - 1, &mObj.rp.front(), &mObj.ci.front(), &Common_);
            Numeric_  = klu_l_factor(&mObj.rp.front(), &mObj.ci.front(), &mObj.nz.front(), Symbolic_, &Common_);
        }
    }
#endif
}

void Simulation::refactor_lu(Matrix& mObj) {
#ifdef SLU
    lu.factorize(true);
#else
    if (usePartitions_) {
        const int64_t nParts = static_cast<int64_t>(partitions_.size());
#ifdef _OPENMP
        if ((parallel_ || gpu_) && nParts >= 4 && b_.size() >= 8192) {
            const int nThreads = std::min<int>(32, std::min<int>(omp_get_max_threads(), static_cast<int>(nParts)));
#pragma omp parallel for schedule(static) num_threads(nThreads)
            for (int64_t p = 0; p < nParts; ++p) {
                auto& part = partitions_[p];
                if (!part.needsLU) { continue; }
                int64_t ok = klu_l_refactor(&part.rp.front(),
                                            &part.ci.front(),
                                            &part.nz.front(),
                                            part.symbolic,
                                            part.numeric,
                                            &part.common);
                if (!ok) {
                    klu_l_free_numeric(&part.numeric, &part.common);
                    part.numeric = klu_l_factor(
                            &part.rp.front(), &part.ci.front(), &part.nz.front(), part.symbolic, &part.common);
                }
                part.needsLU = false;
            }
            return;
        }
#endif
        for (int64_t p = 0; p < nParts; ++p) {
            auto& part = partitions_[p];
            if (!part.needsLU) { continue; }
            int64_t ok = klu_l_refactor(
                    &part.rp.front(), &part.ci.front(), &part.nz.front(), part.symbolic, part.numeric, &part.common);
            if (!ok) {
                klu_l_free_numeric(&part.numeric, &part.common);
                part.numeric = klu_l_factor(
                        &part.rp.front(), &part.ci.front(), &part.nz.front(), part.symbolic, &part.common);
            }
            part.needsLU = false;
        }
    } else {
        simOK_ = klu_l_refactor(&mObj.rp.front(), &mObj.ci.front(), &mObj.nz.front(), Symbolic_, Numeric_, &Common_);
        if (!simOK_) {
            klu_l_free_numeric(&Numeric_, &Common_);
            Numeric_ = klu_l_factor(&mObj.rp.front(), &mObj.ci.front(), &mObj.nz.front(), Symbolic_, &Common_);
        }
    }
#endif
}

void Simulation::solve_lu(Matrix& mObj) {
#ifdef SLU
    std::copy(b_.begin(), b_.end(), x_.begin());
    lu.solve(x_);
#else
    if (usePartitions_) {
        const int64_t nParts = static_cast<int64_t>(partitions_.size());
        bool          anyErr = false;
#ifdef _OPENMP
        if ((parallel_ || gpu_) && nParts >= 4 && b_.size() >= 8192) {
            const int nThreads = std::min<int>(32, std::min<int>(omp_get_max_threads(), static_cast<int>(nParts)));
#pragma omp parallel for schedule(static) num_threads(nThreads) reduction(|| : anyErr)
            for (int64_t p = 0; p < nParts; ++p) {
                auto& part = partitions_[p];
                for (int64_t i = 0; i < part.n; ++i) { part.xLocal[i] = b_[part.globalRows[i]]; }
                int64_t ok = klu_l_tsolve(part.symbolic, part.numeric, part.n, 1, part.xLocal.data(), &part.common);
                if (!ok) { anyErr = true; }
                for (int64_t i = 0; i < part.n; ++i) { x_[part.globalRows[i]] = part.xLocal[i]; }
            }
            if (anyErr) { Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR); }
            return;
        }
#endif
        for (int64_t p = 0; p < nParts; ++p) {
            auto& part = partitions_[p];
            for (int64_t i = 0; i < part.n; ++i) { part.xLocal[i] = b_[part.globalRows[i]]; }
            int64_t ok = klu_l_tsolve(part.symbolic, part.numeric, part.n, 1, part.xLocal.data(), &part.common);
            if (!ok) { Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR); }
            for (int64_t i = 0; i < part.n; ++i) { x_[part.globalRows[i]] = part.xLocal[i]; }
        }
    } else {
        std::copy(b_.begin(), b_.end(), x_.begin());
        simOK_ = klu_l_tsolve(Symbolic_, Numeric_, mObj.rp.size() - 1, 1, &x_.front(), &Common_);
        if (!simOK_) { Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR); }
    }
#endif
}

void Simulation::free_lu() {
#ifdef SLU
    lu.free();
#else
#ifdef JOSIM_CUDA
    cudaSolver_.reset();
#endif
    if (usePartitions_) {
        for (auto& part : partitions_) {
            if (part.symbolic) { klu_l_free_symbolic(&part.symbolic, &part.common); }
            if (part.numeric) { klu_l_free_numeric(&part.numeric, &part.common); }
        }
        partitions_.clear();
        usePartitions_ = false;
    } else {
        if (Symbolic_) { klu_l_free_symbolic(&Symbolic_, &Common_); }
        if (Numeric_) { klu_l_free_numeric(&Numeric_, &Common_); }
    }
#endif
}

void Simulation::trans_sim_mc(Matrix& mObj) {
#ifndef SLU
    const int64_t B      = mcRuns_;
    const int64_t nParts = static_cast<int64_t>(partitions_.size());
    const int64_t N      = static_cast<int64_t>(mObj.rp.size()) - 1;

    std::vector<int32_t> globalToAbsLocal(N, 0);
    int32_t              baseRowOffset = 0;
    for (int64_t p = 0; p < nParts; ++p) {
        const auto& part = partitions_[p];
        for (int64_t locR = 0; locR < part.n; ++locR) {
            globalToAbsLocal[part.globalRows[locR]] = baseRowOffset + static_cast<int32_t>(locR);
        }
        baseRowOffset += static_cast<int32_t>(part.n);
    }

    std::vector<GpuPartDesc> baseParts(nParts);
    std::vector<int32_t>     baseP;
    std::vector<int32_t>     baseQ;
    std::vector<int32_t>     baseLuRp;
    std::vector<int32_t>     baseLuDiag;
    std::vector<int32_t>     baseLuCi;
    std::vector<int32_t>     baseLuSrcNz;
    std::vector<double>      baseLuVal;
    std::vector<double>      basePartNz;
    std::vector<double>      baseXInit(baseRowOffset, 0.0);
    for (int64_t r = 0; r < N; ++r) { baseXInit[globalToAbsLocal[r]] = x_[r]; }

    baseP.reserve(baseRowOffset);
    baseQ.reserve(baseRowOffset);
    baseLuDiag.reserve(baseRowOffset);
    baseLuRp.reserve(baseRowOffset + nParts);

    std::vector<GpuJJ>       baseJJs;
    std::vector<GpuRes>      baseRess;
    std::vector<GpuInd>      baseInds;
    std::vector<GpuMut>      baseMuts;
    std::vector<GpuCap>      baseCaps;
    std::vector<GpuVS>       baseVSs;
    std::vector<GpuPS>       basePSs;
    std::vector<GpuCS>       baseCSs;
    std::vector<GpuCCVS>     baseCCVSs;
    std::vector<GpuVCCS>     baseVCCSs;
    std::vector<GpuTxHalf>   baseTxHalves;
    std::vector<GpuRelTrace> baseRelTraces;

    std::vector<int32_t> globalIndToGpuInd(inductors_.size(), -1);
    std::vector<int32_t> globalJJToGpuJJ(jjs_.size(), -1);
    std::vector<int32_t> globalResToGpuRes(resistors_.size(), -1);
    std::vector<int32_t> globalCapToGpuCap(capacitors_.size(), -1);
    {
        int32_t gIndCounter = 0;
        for (int64_t p = 0; p < nParts; ++p) {
            for (int64_t idx : partitions_[p].indIndices) { globalIndToGpuInd[idx] = gIndCounter++; }
        }
    }

    results.mcRowToSlot.assign(static_cast<size_t>(N), -1);

    int32_t curRowOff = 0;
    for (int64_t p = 0; p < nParts; ++p) {
        const auto& part = partitions_[p];
        if (!part.symbolic || !part.numeric || !part.symbolic->Q || !part.numeric->Pnum) {
            Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR);
            return;
        }
        const int32_t n    = static_cast<int32_t>(part.n);
        GpuPartDesc&  desc = baseParts[p];
        desc.n             = n;
        desc.rowStart      = curRowOff;
        desc.rpStart       = static_cast<int32_t>(baseLuRp.size());
        desc.luStart       = static_cast<int32_t>(baseLuCi.size());
        desc.nzStart       = static_cast<int32_t>(basePartNz.size());
        desc.nzCount       = static_cast<int32_t>(part.nz.size());

        basePartNz.insert(basePartNz.end(), part.nz.begin(), part.nz.end());

        std::vector<int32_t> invQ(n, 0);
        for (int32_t j = 0; j < n; ++j) {
            const int32_t pRow = static_cast<int32_t>(part.symbolic->Q[j]);
            const int32_t qCol = static_cast<int32_t>(part.numeric->Pnum[j]);
            baseP.push_back(pRow);
            baseQ.push_back(qCol);
            invQ[qCol] = j;
        }

        for (int32_t r = 0; r < n; ++r) {
            baseLuRp.push_back(static_cast<int32_t>(baseLuCi.size()) - desc.luStart);
            const int32_t origR  = baseP[curRowOff + r];
            const int64_t kStart = part.rp[origR];
            const int64_t kEnd   = part.rp[origR + 1];

            std::vector<std::pair<int32_t, int32_t>> origCols;
            origCols.reserve(static_cast<size_t>(kEnd - kStart + 4));
            for (int64_t k = kStart; k < kEnd; ++k) {
                origCols.emplace_back(invQ[part.ci[k]], desc.nzStart + static_cast<int32_t>(k));
            }
            std::sort(origCols.begin(), origCols.end());
            if (origCols.size() > 1) {
                size_t writePos = 0;
                for (size_t readPos = 1; readPos < origCols.size(); ++readPos) {
                    if (origCols[readPos].first == origCols[writePos].first) {
                        basePartNz[origCols[writePos].second] += basePartNz[origCols[readPos].second];
                    } else {
                        origCols[++writePos] = origCols[readPos];
                    }
                }
                origCols.resize(writePos + 1);
            }

            std::vector<std::pair<int32_t, int32_t>> rowEntries = origCols;
            {
                auto itDiag = std::lower_bound(rowEntries.begin(), rowEntries.end(), std::make_pair(r, -1));
                if (itDiag == rowEntries.end() || itDiag->first != r) {
                    rowEntries.insert(itDiag, std::make_pair(r, -1));
                }
            }
            for (size_t idx = 0; idx < rowEntries.size(); ++idx) {
                const int32_t colK = rowEntries[idx].first;
                if (colK >= r) { break; }
                const int32_t kDiag = baseLuDiag[curRowOff + colK];
                const int32_t kEndU = baseLuRp[desc.rpStart + colK + 1];
                for (int32_t idxU = kDiag + 1; idxU < kEndU; ++idxU) {
                    const int32_t colU = baseLuCi[desc.luStart + idxU];
                    auto          it
                            = std::lower_bound(rowEntries.begin(), rowEntries.end(), std::make_pair(colU, -1));
                    if (it == rowEntries.end() || it->first != colU) {
                        const size_t curIdx = idx;
                        rowEntries.insert(it, std::make_pair(colU, -1));
                        idx = curIdx;
                    }
                }
            }

            const int32_t rStart = static_cast<int32_t>(baseLuCi.size()) - desc.luStart;
            int32_t       rDiag  = -1;
            for (const auto& kv : rowEntries) {
                const int32_t localIdx = static_cast<int32_t>(baseLuCi.size()) - desc.luStart;
                if (kv.first == r) { rDiag = localIdx; }
                baseLuCi.push_back(kv.first);
                baseLuSrcNz.push_back(kv.second);
                baseLuVal.push_back((kv.second >= 0) ? basePartNz[kv.second] : 0.0);
            }
            if (rDiag < 0) { Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR); }
            baseLuDiag.push_back(rDiag);

            for (int32_t idxL = rStart; idxL < rDiag; ++idxL) {
                const int32_t colK  = baseLuCi[desc.luStart + idxL];
                const int32_t kDiag = baseLuDiag[curRowOff + colK];
                const int32_t kEndU = baseLuRp[desc.rpStart + colK + 1];
                const double  piv   = baseLuVal[desc.luStart + kDiag];
                if (piv == 0.0) { Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR); }
                const double mult               = baseLuVal[desc.luStart + idxL] / piv;
                baseLuVal[desc.luStart + idxL]  = mult;
                int32_t pI                      = idxL + 1;
                for (int32_t idxU = kDiag + 1; idxU < kEndU; ++idxU) {
                    const int32_t colU = baseLuCi[desc.luStart + idxU];
                    while (baseLuCi[desc.luStart + pI] < colU) { ++pI; }
                    baseLuVal[desc.luStart + pI] -= mult * baseLuVal[desc.luStart + idxU];
                }
            }
            if (baseLuVal[desc.luStart + rDiag] == 0.0) {
                Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR);
            }
        }
        baseLuRp.push_back(static_cast<int32_t>(baseLuCi.size()) - desc.luStart);
        desc.luNnz = static_cast<int32_t>(baseLuCi.size()) - desc.luStart;

        desc.jjStart = static_cast<int32_t>(baseJJs.size());
        desc.jjCount = static_cast<int32_t>(part.jjIndices.size());
        for (int64_t jIdx : part.jjIndices) {
            globalJJToGpuJJ[jIdx] = static_cast<int32_t>(baseJJs.size());
            const auto& jj        = jjs_[jIdx];
            const auto& m         = jj.model_;
            GpuJJ       gjj{};
            gjj.posIdx       = jj.indexInfo.posIndex_ ? globalToAbsLocal[jj.indexInfo.posIndex_.value()] : -1;
            gjj.negIdx       = jj.indexInfo.negIndex_ ? globalToAbsLocal[jj.indexInfo.negIndex_.value()] : -1;
            gjj.curIdx       = globalToAbsLocal[jj.indexInfo.currentIndex_.value()];
            gjj.varIdx       = globalToAbsLocal[jj.variableIndex_];
            gjj.nzIdx        = desc.nzStart + static_cast<int32_t>(jjPartNzIdx_[jIdx]);
            gjj.rtype        = m.rtype();
            gjj.tDep         = m.tDep() ? 1 : 0;
            const auto& cpr  = m.cpr();
            gjj.cprLen       = std::min<int32_t>(8, static_cast<int32_t>(cpr.size()));
            for (int32_t h = 0; h < gjj.cprLen; ++h) { gjj.cpr[h] = cpr[h]; }
            gjj.ic           = m.ic();
            gjj.c            = m.c();
            gjj.r0           = m.r0();
            gjj.rn           = m.rn();
            gjj.phiOff       = m.phiOff();
            gjj.t            = m.t();
            gjj.d            = m.d();
            gjj.del          = jj.del_;
            gjj.lowerB       = jj.lowerB_;
            gjj.upperB       = jj.upperB_;
            gjj.gLarge       = jj.gLarge_;
            gjj.negInvSubgap = jj.negInvSubgap_;
            gjj.negInvTrans  = jj.negInvTrans_;
            gjj.negInvNormal = jj.negInvNormal_;
            gjj.pn1          = jj.pn1_;
            gjj.pn2          = jj.pn2_;
            gjj.pn3          = jj.pn3_;
            gjj.pn4          = jj.pn4_;
            gjj.phi0         = jj.phi0_;
            gjj.vn1          = jj.vn1_;
            gjj.vn2          = jj.vn2_;
            gjj.vn3          = jj.vn3_;
            gjj.vn4          = jj.vn4_;
            gjj.vn5          = jj.vn5_;
            gjj.vn6          = jj.vn6_;
            gjj.it           = jj.it_;
            gjj.curNzVal     = jj.matrixInfo.nonZeros_.back();
            baseJJs.push_back(gjj);
        }

        desc.resStart = static_cast<int32_t>(baseRess.size());
        desc.resCount = static_cast<int32_t>(part.resIndices.size());
        for (int64_t rIdx : part.resIndices) {
            globalResToGpuRes[rIdx] = static_cast<int32_t>(baseRess.size());
            const auto& res         = resistors_[rIdx];
            GpuRes      gres{};
            gres.posIdx = res.indexInfo.posIndex_ ? globalToAbsLocal[res.indexInfo.posIndex_.value()] : -1;
            gres.negIdx = res.indexInfo.negIndex_ ? globalToAbsLocal[res.indexInfo.negIndex_.value()] : -1;
            gres.curIdx = globalToAbsLocal[res.indexInfo.currentIndex_.value()];
            gres.pn1    = res.pn1_;
            gres.pn2    = res.pn2_;
            baseRess.push_back(gres);
        }

        desc.indStart = static_cast<int32_t>(baseInds.size());
        desc.indCount = static_cast<int32_t>(part.indIndices.size());
        for (int64_t lIdx : part.indIndices) {
            const auto& ind = inductors_[lIdx];
            GpuInd      gind{};
            gind.curIdx   = globalToAbsLocal[ind.indexInfo.currentIndex_.value()];
            gind.L        = ind.netlistInfo.value_;
            gind.In2      = ind.In2_;
            gind.In3      = ind.In3_;
            gind.In4      = ind.In4_;
            gind.mutStart = static_cast<int32_t>(baseMuts.size());
            gind.mutCount = static_cast<int32_t>(ind.get_mutualInductance().size());
            for (const auto& m : ind.get_mutualInductance()) {
                GpuMut gmut{};
                gmut.targetIndIdx = globalIndToGpuInd[m.first];
                gmut.targetCurIdx = globalToAbsLocal[inductors_[m.first].indexInfo.currentIndex_.value()];
                gmut.M            = m.second;
                baseMuts.push_back(gmut);
            }
            baseInds.push_back(gind);
        }

        desc.capStart = static_cast<int32_t>(baseCaps.size());
        desc.capCount = static_cast<int32_t>(part.capIndices.size());
        for (int64_t cIdx : part.capIndices) {
            globalCapToGpuCap[cIdx] = static_cast<int32_t>(baseCaps.size());
            const auto& cap         = capacitors_[cIdx];
            GpuCap      gcap{};
            gcap.posIdx = cap.indexInfo.posIndex_ ? globalToAbsLocal[cap.indexInfo.posIndex_.value()] : -1;
            gcap.negIdx = cap.indexInfo.negIndex_ ? globalToAbsLocal[cap.indexInfo.negIndex_.value()] : -1;
            gcap.curIdx = globalToAbsLocal[cap.indexInfo.currentIndex_.value()];
            gcap.pn1    = cap.pn1_;
            gcap.pn2    = cap.pn2_;
            gcap.pn3    = cap.pn3_;
            gcap.pn4    = cap.pn4_;
            baseCaps.push_back(gcap);
        }

        desc.vsStart = static_cast<int32_t>(baseVSs.size());
        desc.vsCount = static_cast<int32_t>(part.vsIndices.size());
        for (int64_t vIdx : part.vsIndices) {
            const auto& vs = vsources_[vIdx];
            GpuVS       gvs{};
            gvs.posIdx = vs.indexInfo.posIndex_ ? globalToAbsLocal[vs.indexInfo.posIndex_.value()] : -1;
            gvs.negIdx = vs.indexInfo.negIndex_ ? globalToAbsLocal[vs.indexInfo.negIndex_.value()] : -1;
            gvs.curIdx = globalToAbsLocal[vs.indexInfo.currentIndex_.value()];
            gvs.srcIdx = static_cast<int32_t>(vs.sourceIndex_);
            gvs.pn1    = vs.pn1_;
            gvs.pn2    = vs.pn2_;
            baseVSs.push_back(gvs);
        }

        desc.psStart = static_cast<int32_t>(basePSs.size());
        desc.psCount = static_cast<int32_t>(part.psIndices.size());
        for (int64_t pIdx : part.psIndices) {
            const auto& ps = psources_[pIdx];
            GpuPS       gps{};
            gps.curIdx = globalToAbsLocal[ps.indexInfo.currentIndex_.value()];
            gps.srcIdx = static_cast<int32_t>(ps.sourceIndex_);
            basePSs.push_back(gps);
        }

        desc.csStart = static_cast<int32_t>(baseCSs.size());
        desc.csCount = static_cast<int32_t>(part.csIndices.size());
        for (int64_t cIdx : part.csIndices) {
            const auto& cs = mObj.components.currentsources[cIdx];
            GpuCS       gcs{};
            gcs.posIdx = cs.indexInfo.posIndex_ ? globalToAbsLocal[cs.indexInfo.posIndex_.value()] : -1;
            gcs.negIdx = cs.indexInfo.negIndex_ ? globalToAbsLocal[cs.indexInfo.negIndex_.value()] : -1;
            gcs.srcIdx = static_cast<int32_t>(cs.sourceIndex_);
            baseCSs.push_back(gcs);
        }

        desc.ccvsStart = static_cast<int32_t>(baseCCVSs.size());
        desc.ccvsCount = static_cast<int32_t>(part.ccvsIndices.size());
        for (int64_t idx : part.ccvsIndices) {
            const auto& ccvs = ccvss_[idx];
            GpuCCVS     gccvs{};
            gccvs.posIdx = ccvs.indexInfo.posIndex_ ? globalToAbsLocal[ccvs.indexInfo.posIndex_.value()] : -1;
            gccvs.negIdx = ccvs.indexInfo.negIndex_ ? globalToAbsLocal[ccvs.indexInfo.negIndex_.value()] : -1;
            gccvs.curIdx = globalToAbsLocal[ccvs.indexInfo.currentIndex_.value()];
            baseCCVSs.push_back(gccvs);
        }

        desc.vccsStart = static_cast<int32_t>(baseVCCSs.size());
        desc.vccsCount = static_cast<int32_t>(part.vccsIndices.size());
        for (int64_t idx : part.vccsIndices) {
            const auto& vccs = vccss_[idx];
            GpuVCCS     gvccs{};
            gvccs.posIdx2 = vccs.posIndex2_ ? globalToAbsLocal[vccs.posIndex2_.value()] : -1;
            gvccs.negIdx2 = vccs.negIndex2_ ? globalToAbsLocal[vccs.negIndex2_.value()] : -1;
            gvccs.curIdx  = globalToAbsLocal[vccs.indexInfo.currentIndex_.value()];
            baseVCCSs.push_back(gvccs);
        }

        desc.txHalfStart = static_cast<int32_t>(baseTxHalves.size());
        for (int64_t tIdx : part.txPort1Indices) {
            const auto& tx = txlines_[tIdx];
            GpuTxHalf   gtxh{};
            gtxh.txIdx   = static_cast<int32_t>(tIdx);
            gtxh.isPort2 = 0;
            gtxh.posIdx  = tx.indexInfo.posIndex_ ? globalToAbsLocal[tx.indexInfo.posIndex_.value()] : -1;
            gtxh.negIdx  = tx.indexInfo.negIndex_ ? globalToAbsLocal[tx.indexInfo.negIndex_.value()] : -1;
            gtxh.curIdx  = globalToAbsLocal[tx.indexInfo.currentIndex_.value()];
            gtxh.n1      = tx.n1_1_;
            gtxh.n2      = tx.n2_1_;
            baseTxHalves.push_back(gtxh);
        }
        for (int64_t tIdx : part.txPort2Indices) {
            const auto& tx = txlines_[tIdx];
            GpuTxHalf   gtxh{};
            gtxh.txIdx   = static_cast<int32_t>(tIdx);
            gtxh.isPort2 = 1;
            gtxh.posIdx  = tx.posIndex2_ ? globalToAbsLocal[tx.posIndex2_.value()] : -1;
            gtxh.negIdx  = tx.negIndex2_ ? globalToAbsLocal[tx.negIndex2_.value()] : -1;
            gtxh.curIdx  = globalToAbsLocal[tx.currentIndex2_];
            gtxh.n1      = tx.n1_2_;
            gtxh.n2      = tx.n2_2_;
            baseTxHalves.push_back(gtxh);
        }
        desc.txHalfCount = static_cast<int32_t>(baseTxHalves.size()) - desc.txHalfStart;

        desc.relStart = static_cast<int32_t>(baseRelTraces.size());
        desc.relCount = static_cast<int32_t>(part.traceGlobalRows.size());
        for (int64_t globR : part.traceGlobalRows) {
            const int32_t slot          = static_cast<int32_t>(baseRelTraces.size());
            results.mcRowToSlot[globR]  = slot;
            GpuRelTrace grt{};
            grt.localAbsIdx = globalToAbsLocal[globR];
            grt.traceSlot   = slot;
            baseRelTraces.push_back(grt);
        }

        curRowOff += n;
    }

    std::vector<GpuTxLine>      baseTxLines(txlines_.size());
    std::vector<GpuTxHistEntry> baseTxHist;
    int32_t                     baseHistOff = 0;
    for (size_t idx = 0; idx < txlines_.size(); ++idx) {
        const auto& tx      = txlines_[idx];
        GpuTxLine&  gtx     = baseTxLines[idx];
        gtx.z0              = tx.netlistInfo.value_;
        gtx.k               = tx.timestepDelay_;
        gtx.fracDelay       = tx.fracDelay_;
        gtx.loadedSteps     = tx.loadedSteps_;
        gtx.histOffset      = baseHistOff;
        gtx.histSize        = static_cast<int32_t>(tx.histSize_);
        baseHistOff        += gtx.histSize;
        for (const auto& h : tx.hist_) { baseTxHist.push_back({h.v1, h.v2, h.i1, h.i2}); }
    }

    // Replicate and perturb across all B Monte Carlo trials in parallel
    const int64_t nPartsTotal   = B * nParts;
    const int32_t nBaseRows     = baseRowOffset;
    const int64_t nBaseRp       = static_cast<int64_t>(baseLuRp.size());
    const int64_t nBaseLu       = static_cast<int64_t>(baseLuCi.size());
    const int64_t nBasePartNz   = static_cast<int64_t>(basePartNz.size());
    const int64_t nBaseJJs      = static_cast<int64_t>(baseJJs.size());
    const int64_t nBaseRess     = static_cast<int64_t>(baseRess.size());
    const int64_t nBaseInds     = static_cast<int64_t>(baseInds.size());
    const int64_t nBaseMuts     = static_cast<int64_t>(baseMuts.size());
    const int64_t nBaseCaps     = static_cast<int64_t>(baseCaps.size());
    const int64_t nBaseVSs      = static_cast<int64_t>(baseVSs.size());
    const int64_t nBasePSs      = static_cast<int64_t>(basePSs.size());
    const int64_t nBaseCSs      = static_cast<int64_t>(baseCSs.size());
    const int64_t nBaseCCVSs    = static_cast<int64_t>(baseCCVSs.size());
    const int64_t nBaseVCCSs    = static_cast<int64_t>(baseVCCSs.size());
    const int64_t nBaseTxHalves = static_cast<int64_t>(baseTxHalves.size());
    const int64_t nBaseTraces   = static_cast<int64_t>(baseRelTraces.size());
    const int64_t nBaseTxLines  = static_cast<int64_t>(baseTxLines.size());
    const int64_t nBaseTxHist   = static_cast<int64_t>(baseTxHist.size());

    std::vector<GpuPartDesc>    allParts(nPartsTotal);
    std::vector<int32_t>        allP(B * nBaseRows);
    std::vector<int32_t>        allQ(B * nBaseRows);
    std::vector<int32_t>        allLuRp(B * nBaseRp);
    std::vector<int32_t>        allLuDiag(B * nBaseRows);
    std::vector<int32_t>        allLuCi(B * nBaseLu);
    std::vector<int32_t>        allLuSrcNz(B * nBaseLu);
    std::vector<double>         allLuVal(B * nBaseLu);
    std::vector<double>         allPartNz(B * nBasePartNz);
    std::vector<double>         allXInit(B * nBaseRows);
    std::vector<GpuJJ>          allJJs(B * nBaseJJs);
    std::vector<GpuRes>         allRess(B * nBaseRess);
    std::vector<GpuInd>         allInds(B * nBaseInds);
    std::vector<GpuMut>         allMuts(B * nBaseMuts);
    std::vector<GpuCap>         allCaps(B * nBaseCaps);
    std::vector<GpuVS>          allVSs(B * nBaseVSs);
    std::vector<GpuPS>          allPSs(B * nBasePSs);
    std::vector<GpuCS>          allCSs(B * nBaseCSs);
    std::vector<GpuCCVS>        allCCVSs(B * nBaseCCVSs);
    std::vector<GpuVCCS>        allVCCSs(B * nBaseVCCSs);
    std::vector<GpuTxHalf>      allTxHalves(B * nBaseTxHalves);
    std::vector<GpuRelTrace>    allRelTraces(B * nBaseTraces);
    std::vector<GpuTxLine>      allTxLines(B * nBaseTxLines);
    std::vector<GpuTxHistEntry> allTxHist(B * nBaseTxHist);

    bool setupSingErr = false;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel_ || gpu_) reduction(|| : setupSingErr)
#endif
    for (int64_t b = 0; b < B; ++b) {
        const int32_t rowOffB    = static_cast<int32_t>(b * nBaseRows);
        const int32_t rpOffB     = static_cast<int32_t>(b * nBaseRp);
        const int32_t luOffB     = static_cast<int32_t>(b * nBaseLu);
        const int32_t nzOffB     = static_cast<int32_t>(b * nBasePartNz);
        const int32_t jjOffB     = static_cast<int32_t>(b * nBaseJJs);
        const int32_t resOffB    = static_cast<int32_t>(b * nBaseRess);
        const int32_t indOffB    = static_cast<int32_t>(b * nBaseInds);
        const int32_t mutOffB    = static_cast<int32_t>(b * nBaseMuts);
        const int32_t capOffB    = static_cast<int32_t>(b * nBaseCaps);
        const int32_t vsOffB     = static_cast<int32_t>(b * nBaseVSs);
        const int32_t psOffB     = static_cast<int32_t>(b * nBasePSs);
        const int32_t csOffB     = static_cast<int32_t>(b * nBaseCSs);
        const int32_t ccvsOffB   = static_cast<int32_t>(b * nBaseCCVSs);
        const int32_t vccsOffB   = static_cast<int32_t>(b * nBaseVCCSs);
        const int32_t txHalfOffB = static_cast<int32_t>(b * nBaseTxHalves);
        const int32_t relOffB    = static_cast<int32_t>(b * nBaseTraces);
        const int32_t txLineOffB = static_cast<int32_t>(b * nBaseTxLines);
        const int32_t txHistOffB = static_cast<int32_t>(b * nBaseTxHist);

        std::copy(baseP.begin(), baseP.end(), allP.begin() + rowOffB);
        std::copy(baseQ.begin(), baseQ.end(), allQ.begin() + rowOffB);
        std::copy(baseLuRp.begin(), baseLuRp.end(), allLuRp.begin() + rpOffB);
        std::copy(baseLuDiag.begin(), baseLuDiag.end(), allLuDiag.begin() + rowOffB);
        std::copy(baseLuCi.begin(), baseLuCi.end(), allLuCi.begin() + luOffB);
        for (int64_t idx = 0; idx < nBaseLu; ++idx) {
            const int32_t src         = baseLuSrcNz[idx];
            allLuSrcNz[luOffB + idx]  = (src >= 0) ? (src + nzOffB) : -1;
        }
        std::copy(baseLuVal.begin(), baseLuVal.end(), allLuVal.begin() + luOffB);
        std::copy(basePartNz.begin(), basePartNz.end(), allPartNz.begin() + nzOffB);
        std::copy(baseXInit.begin(), baseXInit.end(), allXInit.begin() + rowOffB);

        for (int64_t p = 0; p < nParts; ++p) {
            GpuPartDesc desc  = baseParts[p];
            desc.rowStart    += rowOffB;
            desc.rpStart     += rpOffB;
            desc.luStart     += luOffB;
            desc.nzStart     += nzOffB;
            desc.jjStart     += jjOffB;
            desc.resStart    += resOffB;
            desc.indStart    += indOffB;
            desc.capStart    += capOffB;
            desc.vsStart     += vsOffB;
            desc.psStart     += psOffB;
            desc.csStart     += csOffB;
            desc.ccvsStart   += ccvsOffB;
            desc.vccsStart   += vccsOffB;
            desc.txHalfStart += txHalfOffB;
            desc.relStart    += relOffB;
            allParts[b * nParts + p] = desc;
        }

        for (int64_t idx = 0; idx < nBaseJJs; ++idx) {
            GpuJJ gjj   = baseJJs[idx];
            if (gjj.posIdx >= 0) { gjj.posIdx += rowOffB; }
            if (gjj.negIdx >= 0) { gjj.negIdx += rowOffB; }
            gjj.curIdx += rowOffB;
            gjj.varIdx += rowOffB;
            gjj.nzIdx  += nzOffB;
            allJJs[jjOffB + idx] = gjj;
        }
        for (int64_t idx = 0; idx < nBaseRess; ++idx) {
            GpuRes gres = baseRess[idx];
            if (gres.posIdx >= 0) { gres.posIdx += rowOffB; }
            if (gres.negIdx >= 0) { gres.negIdx += rowOffB; }
            gres.curIdx += rowOffB;
            allRess[resOffB + idx] = gres;
        }
        for (int64_t idx = 0; idx < nBaseInds; ++idx) {
            GpuInd gind    = baseInds[idx];
            gind.curIdx   += rowOffB;
            gind.mutStart += mutOffB;
            allInds[indOffB + idx] = gind;
        }
        for (int64_t idx = 0; idx < nBaseMuts; ++idx) {
            GpuMut gmut        = baseMuts[idx];
            gmut.targetIndIdx += indOffB;
            gmut.targetCurIdx += rowOffB;
            allMuts[mutOffB + idx] = gmut;
        }
        for (int64_t idx = 0; idx < nBaseCaps; ++idx) {
            GpuCap gcap = baseCaps[idx];
            if (gcap.posIdx >= 0) { gcap.posIdx += rowOffB; }
            if (gcap.negIdx >= 0) { gcap.negIdx += rowOffB; }
            gcap.curIdx += rowOffB;
            allCaps[capOffB + idx] = gcap;
        }
        for (int64_t idx = 0; idx < nBaseVSs; ++idx) {
            GpuVS gvs = baseVSs[idx];
            if (gvs.posIdx >= 0) { gvs.posIdx += rowOffB; }
            if (gvs.negIdx >= 0) { gvs.negIdx += rowOffB; }
            gvs.curIdx += rowOffB;
            allVSs[vsOffB + idx] = gvs;
        }
        for (int64_t idx = 0; idx < nBasePSs; ++idx) {
            GpuPS gps   = basePSs[idx];
            gps.curIdx += rowOffB;
            allPSs[psOffB + idx] = gps;
        }
        for (int64_t idx = 0; idx < nBaseCSs; ++idx) {
            GpuCS gcs = baseCSs[idx];
            if (gcs.posIdx >= 0) { gcs.posIdx += rowOffB; }
            if (gcs.negIdx >= 0) { gcs.negIdx += rowOffB; }
            allCSs[csOffB + idx] = gcs;
        }
        for (int64_t idx = 0; idx < nBaseCCVSs; ++idx) {
            GpuCCVS gccvs = baseCCVSs[idx];
            if (gccvs.posIdx >= 0) { gccvs.posIdx += rowOffB; }
            if (gccvs.negIdx >= 0) { gccvs.negIdx += rowOffB; }
            gccvs.curIdx += rowOffB;
            allCCVSs[ccvsOffB + idx] = gccvs;
        }
        for (int64_t idx = 0; idx < nBaseVCCSs; ++idx) {
            GpuVCCS gvccs = baseVCCSs[idx];
            if (gvccs.posIdx2 >= 0) { gvccs.posIdx2 += rowOffB; }
            if (gvccs.negIdx2 >= 0) { gvccs.negIdx2 += rowOffB; }
            gvccs.curIdx += rowOffB;
            allVCCSs[vccsOffB + idx] = gvccs;
        }
        for (int64_t idx = 0; idx < nBaseTxHalves; ++idx) {
            GpuTxHalf gtxh = baseTxHalves[idx];
            gtxh.txIdx    += txLineOffB;
            if (gtxh.posIdx >= 0) { gtxh.posIdx += rowOffB; }
            if (gtxh.negIdx >= 0) { gtxh.negIdx += rowOffB; }
            gtxh.curIdx += rowOffB;
            allTxHalves[txHalfOffB + idx] = gtxh;
        }
        for (int64_t idx = 0; idx < nBaseTraces; ++idx) {
            GpuRelTrace grt  = baseRelTraces[idx];
            grt.localAbsIdx += rowOffB;
            grt.traceSlot   += relOffB;
            allRelTraces[relOffB + idx] = grt;
        }
        for (int64_t idx = 0; idx < nBaseTxLines; ++idx) {
            GpuTxLine gtx   = baseTxLines[idx];
            gtx.histOffset += txHistOffB;
            allTxLines[txLineOffB + idx] = gtx;
        }
        std::copy(baseTxHist.begin(), baseTxHist.end(), allTxHist.begin() + txHistOffB);

        if (b > 0) {
            std::mt19937_64 rng(Rng::base_seed() + static_cast<uint64_t>(b) * 0x9E3779B97F4A7C15ULL);

            std::vector<double> newIndL(inductors_.size(), 0.0);
            for (size_t lIdx = 0; lIdx < inductors_.size(); ++lIdx) {
                const auto&   ind   = inductors_[lIdx];
                const double  baseL = ind.netlistInfo.value_;
                const double  L     = mObj.spread.spread_value_rng(ind.nominalValue_, Spread::IND, ind.spread_, rng);
                newIndL[lIdx]       = L;
                const int32_t gInd  = globalIndToGpuInd[lIdx];
                if (gInd >= 0) { allInds[indOffB + gInd].L = L; }
                const int64_t p     = indPartIdx_[lIdx];
                const int64_t nzLoc = indPartNzIdx_[lIdx];
                if (nzLoc >= 0) {
                    const double baseDiag = (atyp_ == AnalysisType::Voltage) ? (-(3.0 / (2.0 * stepSize_)) * baseL)
                                                                             : (-(baseL / Constants::SIGMA));
                    const double newDiag  = (atyp_ == AnalysisType::Voltage) ? (-(3.0 / (2.0 * stepSize_)) * L)
                                                                             : (-(L / Constants::SIGMA));
                    allPartNz[nzOffB + baseParts[p].nzStart + nzLoc] += (newDiag - baseDiag);
                }
            }

            for (size_t lIdx = 0; lIdx < inductors_.size(); ++lIdx) {
                const auto&   ind   = inductors_[lIdx];
                const auto&   muts  = ind.get_mutualInductance();
                const int32_t gInd  = globalIndToGpuInd[lIdx];
                const int64_t p     = indPartIdx_[lIdx];
                const double  baseL = ind.netlistInfo.value_;
                const double  L1    = newIndL[lIdx];
                for (size_t mIdx = 0; mIdx < muts.size(); ++mIdx) {
                    const int64_t tIdx   = muts[mIdx].first;
                    const double  baseM  = muts[mIdx].second;
                    const double  baseL2 = inductors_[tIdx].netlistInfo.value_;
                    const double  L2     = newIndL[tIdx];
                    const double  denom  = baseL * baseL2;
                    const double  M      = (denom > 0.0) ? (baseM * std::sqrt((L1 * L2) / denom)) : baseM;
                    if (gInd >= 0) {
                        const int32_t mSlot = allInds[indOffB + gInd].mutStart + static_cast<int32_t>(mIdx);
                        allMuts[mSlot].M    = M;
                    }
                    const int64_t nzLoc = indMutPartNzIdx_[lIdx][mIdx];
                    if (nzLoc >= 0) {
                        const double baseOff = (atyp_ == AnalysisType::Voltage) ? (-(3.0 / (2.0 * stepSize_)) * baseM)
                                                                                : (-(baseM / Constants::SIGMA));
                        const double newOff  = (atyp_ == AnalysisType::Voltage) ? (-(3.0 / (2.0 * stepSize_)) * M)
                                                                                : (-(M / Constants::SIGMA));
                        allPartNz[nzOffB + baseParts[p].nzStart + nzLoc] += (newOff - baseOff);
                    }
                }
            }

            for (size_t rIdx = 0; rIdx < resistors_.size(); ++rIdx) {
                const auto&   res   = resistors_[rIdx];
                const double  baseR = res.netlistInfo.value_;
                const double  R     = mObj.spread.spread_value_rng(res.nominalValue_, Spread::RES, res.spread_, rng);
                const int64_t p     = resPartIdx_[rIdx];
                const int64_t nzLoc = resPartNzIdx_[rIdx];
                if (nzLoc >= 0) {
                    const double baseDiag = (atyp_ == AnalysisType::Voltage)
                                                    ? (-baseR)
                                                    : (-((2.0 * stepSize_) / (3.0 * Constants::SIGMA * baseR)));
                    const double newDiag  = (atyp_ == AnalysisType::Voltage)
                                                    ? (-R)
                                                    : (-((2.0 * stepSize_) / (3.0 * Constants::SIGMA * R)));
                    allPartNz[nzOffB + baseParts[p].nzStart + nzLoc] += (newDiag - baseDiag);
                }
            }

            for (size_t cIdx = 0; cIdx < capacitors_.size(); ++cIdx) {
                const auto&   cap   = capacitors_[cIdx];
                const double  baseC = cap.netlistInfo.value_;
                const double  C     = mObj.spread.spread_value_rng(cap.nominalValue_, Spread::CAP, cap.spread_, rng);
                const int64_t p     = capPartIdx_[cIdx];
                const int64_t nzLoc = capPartNzIdx_[cIdx];
                if (nzLoc >= 0) {
                    const double baseDiag = (atyp_ == AnalysisType::Voltage)
                                                    ? (-((2.0 * stepSize_) / (3.0 * baseC)))
                                                    : (-((4.0 * stepSize_ * stepSize_)
                                                         / (9.0 * Constants::SIGMA * baseC)));
                    const double newDiag  = (atyp_ == AnalysisType::Voltage)
                                                    ? (-((2.0 * stepSize_) / (3.0 * C)))
                                                    : (-((4.0 * stepSize_ * stepSize_)
                                                         / (9.0 * Constants::SIGMA * C)));
                    allPartNz[nzOffB + baseParts[p].nzStart + nzLoc] += (newDiag - baseDiag);
                }
            }

            for (size_t jIdx = 0; jIdx < jjs_.size(); ++jIdx) {
                const auto&   jj    = jjs_[jIdx];
                const double  area  = mObj.spread.spread_value_rng(jj.nominalArea_, Spread::JJ, jj.spread_, rng);
                const int32_t gJJ   = globalJJToGpuJJ[jIdx];
                const int64_t p     = jjPartIdx_[jIdx];
                const int64_t nzLoc = jjPartNzIdx_[jIdx];
                if (gJJ >= 0) {
                    GpuJJ&       gjj     = allJJs[jjOffB + gJJ];
                    const double baseNz  = gjj.curNzVal;
                    const double baseIc  = jj.nominalIc_.has_value() ? jj.nominalIc_.value() : jj.baseModel_.ic();
                    gjj.ic               = baseIc * area;
                    gjj.r0               = jj.baseModel_.r0() / area;
                    gjj.rn               = jj.baseModel_.rn() / area;
                    gjj.c                = jj.baseModel_.c() * area;
                    gjj.gLarge           = (jj.baseModel_.ic() / jj.del_)
                                 * ((jj.upperB_ + jj.lowerB_) / (jj.upperB_ - jj.lowerB_));
                    gjj.negInvSubgap     = -(1.0 / (((3.0 * gjj.c) / (2.0 * stepSize_)) + (1.0 / gjj.r0)));
                    gjj.negInvNormal     = -(1.0 / (((3.0 * gjj.c) / (2.0 * stepSize_)) + (1.0 / gjj.rn)));
                    gjj.negInvTrans      = -(1.0 / (((3.0 * gjj.c) / (2.0 * stepSize_)) + gjj.gLarge));
                    gjj.curNzVal         = (gjj.rtype == 1) ? gjj.negInvSubgap : gjj.negInvNormal;
                    if (nzLoc >= 0) {
                        allPartNz[nzOffB + baseParts[p].nzStart + nzLoc] += (gjj.curNzVal - baseNz);
                    }
                }
            }

            // Refactorize all partitions of Trial b using sparse IKJ elimination
            for (int64_t p = 0; p < nParts; ++p) {
                const GpuPartDesc& desc = allParts[b * nParts + p];
                for (int32_t idx = 0; idx < desc.luNnz; ++idx) {
                    const int32_t src             = allLuSrcNz[desc.luStart + idx];
                    allLuVal[desc.luStart + idx]  = (src >= 0) ? allPartNz[src] : 0.0;
                }
                for (int32_t r = 0; r < desc.n; ++r) {
                    const int32_t rStart = allLuRp[desc.rpStart + r];
                    const int32_t rDiag  = allLuDiag[desc.rowStart + r];
                    for (int32_t idxL = rStart; idxL < rDiag; ++idxL) {
                        const int32_t colK  = allLuCi[desc.luStart + idxL];
                        const int32_t kDiag = allLuDiag[desc.rowStart + colK];
                        const int32_t kEndU = allLuRp[desc.rpStart + colK + 1];
                        const double  piv   = allLuVal[desc.luStart + kDiag];
                        if (piv == 0.0) {
                            setupSingErr = true;
                            break;
                        }
                        const double mult              = allLuVal[desc.luStart + idxL] / piv;
                        allLuVal[desc.luStart + idxL]  = mult;
                        int32_t pI                     = idxL + 1;
                        for (int32_t idxU = kDiag + 1; idxU < kEndU; ++idxU) {
                            const int32_t colU = allLuCi[desc.luStart + idxU];
                            while (allLuCi[desc.luStart + pI] < colU) { ++pI; }
                            allLuVal[desc.luStart + pI] -= mult * allLuVal[desc.luStart + idxU];
                        }
                    }
                    if (setupSingErr || allLuVal[desc.luStart + rDiag] == 0.0) {
                        setupSingErr = true;
                        break;
                    }
                }
            }
        }
    }
    if (setupSingErr) { Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR); }

    results.timeAxis.clear();
    results.timeAxis.reserve(static_cast<size_t>(numKeptSteps_));
    for (int64_t i = 0; i < simSize_; ++i) {
        if (keptSlot_[i] >= 0) { results.timeAxis.push_back(i * stepSize_); }
    }

#ifdef JOSIM_CUDA
    if (gpu_) {
        cudaSolver_   = std::make_unique<CudaPartitionSolver>();
        const bool ok = cudaSolver_->initialize(static_cast<int32_t>(nPartsTotal),
                                                static_cast<int32_t>(B * nBaseRows),
                                                static_cast<int32_t>(mObj.sourcegen.size()),
                                                static_cast<int32_t>(windowSize_),
                                                static_cast<int32_t>(allRelTraces.size()),
                                                static_cast<int32_t>(numKeptSteps_),
                                                simSize_,
                                                stepSize_,
                                                (atyp_ == AnalysisType::Voltage) ? 0 : 1,
                                                allParts,
                                                allP,
                                                allQ,
                                                allLuRp,
                                                allLuDiag,
                                                allLuCi,
                                                allLuSrcNz,
                                                allLuVal,
                                                allPartNz,
                                                allXInit,
                                                allJJs,
                                                allRess,
                                                allInds,
                                                allMuts,
                                                allCaps,
                                                allVSs,
                                                allPSs,
                                                allCSs,
                                                allCCVSs,
                                                allVCCSs,
                                                allTxHalves,
                                                allTxLines,
                                                allTxHist,
                                                allRelTraces,
                                                keptSlot_,
                                                static_cast<int32_t>(B));
        if (ok) {
            if (verb_ >= 1) {
                std::cout << "JoSIM: Initialized CUDA GPU Batched Monte Carlo Solver (" << B << " trials x " << nParts
                          << " partitions = " << nPartsTotal << " blocks, window W = " << windowSize_ << ")\n";
            }
            ProgressBar bar;
            if (!minOut_) {
                bar.create_thread();
                bar.set_bar_width(30);
                bar.fill_bar_progress_with("O");
                bar.fill_bar_remainder_with(" ");
                bar.set_status_text("Simulating MC (GPU)");
                bar.set_total(static_cast<float>(simSize_));
            }
            const int64_t       numSrc = static_cast<int64_t>(mObj.sourcegen.size());
            std::vector<double> h_srcWindow(
                    static_cast<size_t>((windowSize_ + 2) * std::max<int64_t>(1, numSrc)), 0.0);
            for (int64_t i0 = 0; i0 < simSize_; i0 += windowSize_) {
                if (!minOut_) { bar.update(static_cast<float>(i0)); }
                const int64_t i1 = std::min<int64_t>(simSize_, i0 + windowSize_);
                const int64_t w  = i1 - i0;
                if (numSrc > 0) {
                    for (int64_t s = -2; s < w; ++s) {
                        const int64_t stepIdx = i0 + s;
                        const double  t       = (stepIdx >= 0) ? (stepIdx * stepSize_) : 0.0;
                        double*       row     = h_srcWindow.data() + (s + 2) * numSrc;
                        if (stepIdx >= 0) {
                            for (int64_t srcIdx = 0; srcIdx < numSrc; ++srcIdx) {
                                row[srcIdx] = mObj.sourcegen[srcIdx].value(t);
                            }
                        } else {
                            std::fill(row, row + numSrc, 0.0);
                        }
                    }
                }
                if (!cudaSolver_->step_window(i0, i1, numSrc > 0 ? h_srcWindow.data() : nullptr)) {
                    needsTR_ = true;
                    if (!minOut_) {
                        bar.complete();
                        std::cout << "\n";
                    }
                    return;
                }
            }
            std::vector<double> h_xFinal;
            cudaSolver_->download_results(results.mcTraceOut, h_xFinal);
            int64_t curRowOff0 = 0;
            for (const auto& part : partitions_) {
                for (int64_t locR = 0; locR < part.n; ++locR) {
                    x_[part.globalRows[locR]] = h_xFinal[curRowOff0 + locR];
                }
                curRowOff0 += part.n;
                for (int64_t globR : part.traceGlobalRows) {
                    const int32_t slot = results.mcRowToSlot[globR];
                    auto&         dst  = *results.xVector[globR];
                    const double* src  = results.mcTraceOut.data() + static_cast<size_t>(slot) * numKeptSteps_;
                    dst.assign(src, src + numKeptSteps_);
                }
            }
            results.mcRuns         = B;
            results.mcNumRows      = nBaseTraces;
            results.mcNumKeptSteps = numKeptSteps_;
            if (!minOut_) {
                bar.complete();
                std::cout << "\n";
            }
            return;
        }
        cudaSolver_.reset();
    }
#endif

    // Host Batched Monte Carlo Solver (Serial 1T or OpenMP -p across B trials)
    if (verb_ >= 1) {
        std::cout << "JoSIM: Running CPU Batched Monte Carlo Solver (" << B << " trials x " << nParts
                  << " partitions = " << nPartsTotal << " blocks)\n";
    }
    ProgressBar bar;
    if (!minOut_) {
        bar.create_thread();
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text(parallel_ ? "Simulating MC (OMP)" : "Simulating MC");
        bar.set_total(static_cast<float>(B));
    }

    const int64_t       numSrc = static_cast<int64_t>(mObj.sourcegen.size());
    std::vector<double> srcAll(static_cast<size_t>((simSize_ + 2) * std::max<int64_t>(1, numSrc)), 0.0);
    if (numSrc > 0) {
        for (int64_t i = 0; i < simSize_; ++i) {
            const double t   = i * stepSize_;
            double*      row = srcAll.data() + (i + 2) * numSrc;
            for (int64_t srcIdx = 0; srcIdx < numSrc; ++srcIdx) { row[srcIdx] = mObj.sourcegen[srcIdx].value(t); }
        }
    }

    results.mcTraceOut.assign(static_cast<size_t>(B * nBaseTraces * numKeptSteps_), 0.0);
    bool anyNeedsTR = false;
    bool anySingErr = false;

#ifdef _OPENMP
    const int nThreads = parallel_ ? std::min<int>(omp_get_max_threads(), static_cast<int>(B)) : 1;
#pragma omp parallel for schedule(dynamic, 1) num_threads(nThreads) if (parallel_) reduction(|| : anyNeedsTR, anySingErr)
#endif
    for (int64_t b = 0; b < B; ++b) {
        if (anyNeedsTR || anySingErr) { continue; }
        if (!minOut_ && (!parallel_ || b % 4 == 0)) { bar.update(static_cast<float>(b)); }

        const int32_t       rowOffB    = static_cast<int32_t>(b * nBaseRows);
        const int32_t       txLineOffB = static_cast<int32_t>(b * nBaseTxLines);
        double*             xPtr       = allXInit.data() - rowOffB + rowOffB; // global index -> allXInit[idx]
        std::vector<double> bTrial(nBaseRows, 0.0);
        std::vector<double> yTrial(nBaseRows, 0.0);
        double*             bPtr = bTrial.data() - rowOffB;
        double*             yPtr = yTrial.data() - rowOffB;

        std::vector<double> txRem1(static_cast<size_t>(std::max<int64_t>(1, nBaseTxLines) * windowSize_), 0.0);
        std::vector<double> txRem2(static_cast<size_t>(std::max<int64_t>(1, nBaseTxLines) * windowSize_), 0.0);

        auto fetch_delayed = [&](const GpuTxLine& tx, int64_t step, int32_t which) -> double {
            const int64_t delayed = tx.loadedSteps + step - tx.k;
            if (delayed < 0) { return 0.0; }
            const int32_t         idx = tx.histOffset + static_cast<int32_t>(delayed % tx.histSize);
            const GpuTxHistEntry& e   = allTxHist[idx];
            double                v   = (which == 0) ? e.v1 : ((which == 1) ? e.v2 : ((which == 2) ? e.i1 : e.i2));
            if (tx.fracDelay > 0.0) {
                double vPrev = 0.0;
                if (delayed >= 1) {
                    const int32_t         idxPrev = tx.histOffset + static_cast<int32_t>((delayed - 1) % tx.histSize);
                    const GpuTxHistEntry& ep      = allTxHist[idxPrev];
                    vPrev = (which == 0) ? ep.v1 : ((which == 1) ? ep.v2 : ((which == 2) ? ep.i1 : ep.i2));
                }
                v = (1.0 - tx.fracDelay) * v + tx.fracDelay * vPrev;
            }
            return v;
        };

        for (int64_t i0 = 0; i0 < simSize_; i0 += windowSize_) {
            if (anyNeedsTR || anySingErr) { break; }
            const int64_t i1 = std::min<int64_t>(simSize_, i0 + windowSize_);
            const int64_t w  = i1 - i0;

            // 1. Precompute remote transmission line delayed history terms for Trial b across [i0, i1)
            for (int64_t tLocal = 0; tLocal < nBaseTxLines; ++tLocal) {
                const GpuTxLine& tx = allTxLines[txLineOffB + tLocal];
                const double     Z0 = tx.z0;
                for (int64_t s = 0; s < w; ++s) {
                    const int64_t i = i0 + s;
                    if (atyp_ == AnalysisType::Voltage) {
                        const double v1k          = fetch_delayed(tx, i, 0);
                        const double v2k          = fetch_delayed(tx, i, 1);
                        const double i1k          = fetch_delayed(tx, i, 2);
                        const double i2k          = fetch_delayed(tx, i, 3);
                        txRem1[tLocal * w + s]    = Z0 * i2k + v2k;
                        txRem2[tLocal * w + s]    = Z0 * i1k + v1k;
                    } else {
                        const double c2           = (2.0 * stepSize_ * Z0) / (3.0 * Constants::SIGMA);
                        const double p1k          = fetch_delayed(tx, i, 0);
                        const double p2k          = fetch_delayed(tx, i, 1);
                        const double p1k1         = fetch_delayed(tx, i - 1, 0);
                        const double p2k1         = fetch_delayed(tx, i - 1, 1);
                        const double p1k2         = fetch_delayed(tx, i - 2, 0);
                        const double p2k2         = fetch_delayed(tx, i - 2, 1);
                        const double i1k          = fetch_delayed(tx, i, 2);
                        const double i2k          = fetch_delayed(tx, i, 3);
                        txRem1[tLocal * w + s]    = c2 * i2k + p2k - (4.0 / 3.0) * p2k1 + (1.0 / 3.0) * p2k2;
                        txRem2[tLocal * w + s]    = c2 * i1k + p1k - (4.0 / 3.0) * p1k1 + (1.0 / 3.0) * p1k2;
                    }
                }
            }

            // 2. Advance each partition p of Trial b across [i0, i1)
            for (int64_t p = 0; p < nParts; ++p) {
                const GpuPartDesc& part = allParts[b * nParts + p];
                for (int64_t s = 0; s < w; ++s) {
                    const int64_t i          = i0 + s;
                    const bool    checkPhase = (static_cast<double>(i) / static_cast<double>(simSize_) > 0.01);

                    for (int32_t r = 0; r < part.n; ++r) { bPtr[part.rowStart + r] = 0.0; }

                    bool partNeedsLU = false;
                    for (int32_t k = 0; k < part.jjCount; ++k) {
                        GpuJJ&       jj   = allJJs[part.jjStart + k];
                        const double vPos = (jj.posIdx >= 0) ? xPtr[jj.posIdx] : 0.0;
                        const double vNeg = (jj.negIdx >= 0) ? xPtr[jj.negIdx] : 0.0;
                        jj.pn1            = vPos - vNeg;
                        if (i > 0) {
                            if (atyp_ == AnalysisType::Voltage) {
                                jj.vn1 = jj.pn1;
                                jj.pn1 = xPtr[jj.varIdx];
                            } else {
                                jj.vn1 = xPtr[jj.varIdx];
                            }
                        }
                        const double v0 = 2.5 * jj.vn1 - 2.0 * jj.vn2 + 0.5 * jj.vn3;
                        jj.phi0         = (4.0 / 3.0) * jj.pn1 - (1.0 / 3.0) * jj.pn2
                                  + ((2.0 * stepSize_) / (3.0 * Constants::SIGMA)) * v0;
                        if (checkPhase && std::abs(jj.phi0 - jj.pn1) > (0.20 * 2.0 * Constants::PI)) {
                            anyNeedsTR = true;
                            break;
                        }
                        if (atyp_ == AnalysisType::Voltage) {
                            bPtr[jj.varIdx]
                                    = Constants::SIGMA
                                      * (-(2.0 / stepSize_) * jj.pn1 + (1.0 / (2.0 * stepSize_)) * jj.pn2);
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
                            bool         updated = false;
                            const double absV    = std::abs(v0);
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
                                allPartNz[jj.nzIdx] = jj.curNzVal;
                                partNeedsLU         = true;
                            }
                        }

                        double ic_sin_phi = 0.0;
                        if (jj.cprLen == 1) {
                            ic_sin_phi += jj.ic * (jj.cpr[0] * std::sin(1.0 * (jj.phi0 - jj.phiOff)));
                        } else {
                            for (int32_t harm = 0; harm < jj.cprLen; ++harm) {
                                ic_sin_phi += jj.ic * (jj.cpr[harm] * std::sin((harm + 1) * (jj.phi0 - jj.phiOff)));
                            }
                        }
                        if (!jj.tDep) {
                            bPtr[jj.curIdx] = jj.curNzVal
                                              * (ic_sin_phi - (((2.0 * jj.c) / stepSize_) * jj.vn1)
                                                 + ((jj.c / (2.0 * stepSize_)) * jj.vn2) + jj.it);
                        } else {
                            double sin2_half_phi = 0.0;
                            for (int32_t harm = 0; harm < jj.cprLen; ++harm) {
                                sin2_half_phi += jj.cpr[harm] * std::sin((harm + 1) * (jj.phi0 - jj.phiOff) / 2.0);
                            }
                            sin2_half_phi  = sin2_half_phi * sin2_half_phi;
                            double sin_phi = 0.0;
                            for (int32_t harm = 0; harm < jj.cprLen; ++harm) {
                                sin_phi += jj.cpr[harm] * std::sin((harm + 1) * (jj.phi0 - jj.phiOff));
                            }
                            const double sqrt_part = std::sqrt(1.0 - jj.d * sin2_half_phi);
                            bPtr[jj.curIdx]
                                    = jj.curNzVal
                                      * ((((Constants::PI * jj.del) / (2.0 * Constants::EV * jj.rn))
                                          * (sin_phi / sqrt_part)
                                          * std::tanh(jj.del / (2.0 * Constants::BOLTZMANN * jj.t) * sqrt_part))
                                         - (((2.0 * jj.c) / stepSize_) * jj.vn1)
                                         + ((jj.c / (2.0 * stepSize_)) * jj.vn2) + jj.it);
                        }
                        jj.vn2 = jj.vn1;
                    }
                    if (anyNeedsTR) { break; }

                    if (partNeedsLU) {
                        for (int32_t idx = 0; idx < part.luNnz; ++idx) {
                            const int32_t src             = allLuSrcNz[part.luStart + idx];
                            allLuVal[part.luStart + idx]  = (src >= 0) ? allPartNz[src] : 0.0;
                        }
                        for (int32_t r = 0; r < part.n; ++r) {
                            const int32_t rStart = allLuRp[part.rpStart + r];
                            const int32_t rDiag  = allLuDiag[part.rowStart + r];
                            for (int32_t idxL = rStart; idxL < rDiag; ++idxL) {
                                const int32_t colK  = allLuCi[part.luStart + idxL];
                                const int32_t kDiag = allLuDiag[part.rowStart + colK];
                                const int32_t kEndU = allLuRp[part.rpStart + colK + 1];
                                const double  piv   = allLuVal[part.luStart + kDiag];
                                if (piv == 0.0) {
                                    anySingErr = true;
                                    break;
                                }
                                const double mult              = allLuVal[part.luStart + idxL] / piv;
                                allLuVal[part.luStart + idxL]  = mult;
                                int32_t pI                     = idxL + 1;
                                for (int32_t idxU = kDiag + 1; idxU < kEndU; ++idxU) {
                                    const int32_t colU = allLuCi[part.luStart + idxU];
                                    while (allLuCi[part.luStart + pI] < colU) { ++pI; }
                                    allLuVal[part.luStart + pI] -= mult * allLuVal[part.luStart + idxU];
                                }
                            }
                        }
                    }

                    // Current Sources
                    for (int32_t k = 0; k < part.csCount; ++k) {
                        const GpuCS& cs     = allCSs[part.csStart + k];
                        const double srcVal = srcAll[(i + 2) * numSrc + cs.srcIdx];
                        if (cs.posIdx >= 0) { bPtr[cs.posIdx] -= srcVal; }
                        if (cs.negIdx >= 0) { bPtr[cs.negIdx] += srcVal; }
                    }

                    // Resistors (Phase mode)
                    if (atyp_ == AnalysisType::Phase) {
                        for (int32_t k = 0; k < part.resCount; ++k) {
                            GpuRes&      res  = allRess[part.resStart + k];
                            const double vPos = (res.posIdx >= 0) ? xPtr[res.posIdx] : 0.0;
                            const double vNeg = (res.negIdx >= 0) ? xPtr[res.negIdx] : 0.0;
                            res.pn1           = vPos - vNeg;
                            bPtr[res.curIdx]  = (4.0 / 3.0) * res.pn1 - (1.0 / 3.0) * res.pn2;
                            res.pn2           = res.pn1;
                        }
                    }

                    // Inductors (Voltage mode)
                    if (atyp_ == AnalysisType::Voltage) {
                        for (int32_t k = 0; k < part.indCount; ++k) {
                            const GpuInd& ind = allInds[part.indStart + k];
                            const double  iCur = xPtr[ind.curIdx];
                            double rhs = -(2.0 * ind.L / stepSize_) * iCur + (ind.L / (2.0 * stepSize_)) * ind.In2;
                            for (int32_t m = 0; m < ind.mutCount; ++m) {
                                const GpuMut& mut  = allMuts[ind.mutStart + m];
                                const GpuInd& tInd = allInds[mut.targetIndIdx];
                                rhs += -((2.0 * mut.M) / stepSize_) * xPtr[mut.targetCurIdx]
                                       + (mut.M / (2.0 * stepSize_)) * tInd.In2;
                            }
                            bPtr[ind.curIdx] = rhs;
                        }
                        for (int32_t k = 0; k < part.indCount; ++k) {
                            GpuInd& ind = allInds[part.indStart + k];
                            ind.In2     = xPtr[ind.curIdx];
                        }
                    }

                    // Capacitors
                    for (int32_t k = 0; k < part.capCount; ++k) {
                        GpuCap&      cap  = allCaps[part.capStart + k];
                        const double vPos = (cap.posIdx >= 0) ? xPtr[cap.posIdx] : 0.0;
                        const double vNeg = (cap.negIdx >= 0) ? xPtr[cap.negIdx] : 0.0;
                        cap.pn1           = vPos - vNeg;
                        if (atyp_ == AnalysisType::Voltage) {
                            bPtr[cap.curIdx] = (4.0 / 3.0) * cap.pn1 - (1.0 / 3.0) * cap.pn2;
                        } else {
                            bPtr[cap.curIdx] = (8.0 / 3.0) * cap.pn1 - (22.0 / 9.0) * cap.pn2 + (8.0 / 9.0) * cap.pn3
                                               - (1.0 / 9.0) * cap.pn4;
                            cap.pn4 = cap.pn3;
                            cap.pn3 = cap.pn2;
                        }
                        cap.pn2 = cap.pn1;
                    }

                    // Voltage Sources
                    for (int32_t k = 0; k < part.vsCount; ++k) {
                        GpuVS&       vs     = allVSs[part.vsStart + k];
                        const double srcVal = srcAll[(i + 2) * numSrc + vs.srcIdx];
                        if (atyp_ == AnalysisType::Voltage) {
                            bPtr[vs.curIdx] = srcVal;
                        } else {
                            const double vPos = (vs.posIdx >= 0) ? xPtr[vs.posIdx] : 0.0;
                            const double vNeg = (vs.negIdx >= 0) ? xPtr[vs.negIdx] : 0.0;
                            vs.pn1            = vPos - vNeg;
                            bPtr[vs.curIdx]   = ((2.0 * stepSize_) / (3.0 * Constants::SIGMA)) * srcVal
                                              + (4.0 / 3.0) * vs.pn1 - (1.0 / 3.0) * vs.pn2;
                            vs.pn2            = vs.pn1;
                        }
                    }

                    // Phase Sources
                    for (int32_t k = 0; k < part.psCount; ++k) {
                        const GpuPS& ps = allPSs[part.psStart + k];
                        const double s0 = srcAll[(i + 2) * numSrc + ps.srcIdx];
                        if (atyp_ == AnalysisType::Phase) {
                            bPtr[ps.curIdx] = s0;
                        } else {
                            const double s1 = srcAll[(i + 1) * numSrc + ps.srcIdx];
                            const double s2 = srcAll[i * numSrc + ps.srcIdx];
                            if (i == 0) {
                                bPtr[ps.curIdx] = (Constants::SIGMA / stepSize_) * (1.5 * s0);
                            } else if (i == 1) {
                                bPtr[ps.curIdx] = (Constants::SIGMA / stepSize_) * (1.5 * s0 - 2.0 * s1);
                            } else {
                                bPtr[ps.curIdx] = (Constants::SIGMA / stepSize_) * (1.5 * s0 - 2.0 * s1 + 0.5 * s2);
                            }
                        }
                    }

                    // CCVS & VCCS (Phase mode)
                    if (atyp_ == AnalysisType::Phase) {
                        for (int32_t k = 0; k < part.ccvsCount; ++k) {
                            GpuCCVS&     ccvs = allCCVSs[part.ccvsStart + k];
                            const double vPos = (ccvs.posIdx >= 0) ? xPtr[ccvs.posIdx] : 0.0;
                            const double vNeg = (ccvs.negIdx >= 0) ? xPtr[ccvs.negIdx] : 0.0;
                            ccvs.pn1          = vPos - vNeg;
                            bPtr[ccvs.curIdx] = (4.0 / 3.0) * ccvs.pn1 - (1.0 / 3.0) * ccvs.pn2;
                            ccvs.pn2          = ccvs.pn1;
                        }
                        for (int32_t k = 0; k < part.vccsCount; ++k) {
                            GpuVCCS&     vccs = allVCCSs[part.vccsStart + k];
                            const double vPos = (vccs.posIdx2 >= 0) ? xPtr[vccs.posIdx2] : 0.0;
                            const double vNeg = (vccs.negIdx2 >= 0) ? xPtr[vccs.negIdx2] : 0.0;
                            vccs.pn1          = vPos - vNeg;
                            bPtr[vccs.curIdx] = (4.0 / 3.0) * vccs.pn1 - (1.0 / 3.0) * vccs.pn2;
                            vccs.pn2          = vccs.pn1;
                        }
                    }

                    // Transmission Line Port Halves
                    for (int32_t k = 0; k < part.txHalfCount; ++k) {
                        const GpuTxHalf& txh    = allTxHalves[part.txHalfStart + k];
                        const int32_t    tLocal = txh.txIdx - txLineOffB;
                        const double     rem
                                = (txh.isPort2 == 0) ? txRem1[tLocal * w + s] : txRem2[tLocal * w + s];
                        if (atyp_ == AnalysisType::Voltage) {
                            bPtr[txh.curIdx] = rem;
                        } else {
                            bPtr[txh.curIdx] = (4.0 / 3.0) * txh.n1 - (1.0 / 3.0) * txh.n2 + rem;
                        }
                    }

                    // Forward substitution: L * y = P * b
                    for (int32_t r = 0; r < part.n; ++r) {
                        double        sum    = bPtr[part.rowStart + allP[part.rowStart + r]];
                        const int32_t rStart = allLuRp[part.rpStart + r];
                        const int32_t rDiag  = allLuDiag[part.rowStart + r];
                        for (int32_t idx = rStart; idx < rDiag; ++idx) {
                            sum -= allLuVal[part.luStart + idx]
                                   * yPtr[part.rowStart + allLuCi[part.luStart + idx]];
                        }
                        yPtr[part.rowStart + r] = sum;
                    }
                    // Backward substitution: U * z = y, x[Q[r]] = z[r]
                    for (int32_t r = part.n - 1; r >= 0; --r) {
                        double        sum   = yPtr[part.rowStart + r];
                        const int32_t rDiag = allLuDiag[part.rowStart + r];
                        const int32_t rEnd  = allLuRp[part.rpStart + r + 1];
                        for (int32_t idx = rDiag + 1; idx < rEnd; ++idx) {
                            sum -= allLuVal[part.luStart + idx]
                                   * yPtr[part.rowStart + allLuCi[part.luStart + idx]];
                        }
                        const double z                              = sum / allLuVal[part.luStart + rDiag];
                        yPtr[part.rowStart + r]                     = z;
                        xPtr[part.rowStart + allQ[part.rowStart + r]] = z;
                    }

                    // Update transmission line circular history and local port phase history
                    for (int32_t k = 0; k < part.txHalfCount; ++k) {
                        GpuTxHalf&       txh  = allTxHalves[part.txHalfStart + k];
                        const GpuTxLine& tx   = allTxLines[txh.txIdx];
                        const double     vPos = (txh.posIdx >= 0) ? xPtr[txh.posIdx] : 0.0;
                        const double     vNeg = (txh.negIdx >= 0) ? xPtr[txh.negIdx] : 0.0;
                        const double     vPort = vPos - vNeg;
                        const double     iPort = xPtr[txh.curIdx];
                        const int32_t    slot
                                = tx.histOffset + static_cast<int32_t>((tx.loadedSteps + i) % tx.histSize);
                        if (txh.isPort2 == 0) {
                            allTxHist[slot].v1 = vPort;
                            allTxHist[slot].i1 = iPort;
                        } else {
                            allTxHist[slot].v2 = vPort;
                            allTxHist[slot].i2 = iPort;
                        }
                        if (atyp_ == AnalysisType::Phase) {
                            txh.n2 = txh.n1;
                            txh.n1 = vPort;
                        }
                    }

                    // Record kept trace outputs
                    const int32_t kSlot = keptSlot_[i];
                    if (kSlot >= 0) {
                        for (int32_t k = 0; k < part.relCount; ++k) {
                            const GpuRelTrace& rt = allRelTraces[part.relStart + k];
                            results.mcTraceOut[static_cast<size_t>(rt.traceSlot) * numKeptSteps_ + kSlot]
                                    = xPtr[rt.localAbsIdx];
                        }
                    }
                }
            }
        }
    }

    if (anySingErr) { Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR); }
    if (anyNeedsTR) {
        needsTR_ = true;
        if (!minOut_) {
            bar.complete();
            std::cout << "\n";
        }
        return;
    }

    // Copy Trial 0 traces into xVector and final state into x_
    int64_t curRowOff0 = 0;
    for (const auto& part : partitions_) {
        for (int64_t locR = 0; locR < part.n; ++locR) {
            x_[part.globalRows[locR]] = allXInit[curRowOff0 + locR];
        }
        curRowOff0 += part.n;
        for (int64_t globR : part.traceGlobalRows) {
            const int32_t slot = results.mcRowToSlot[globR];
            auto&         dst  = *results.xVector[globR];
            const double* src  = results.mcTraceOut.data() + static_cast<size_t>(slot) * numKeptSteps_;
            dst.assign(src, src + numKeptSteps_);
        }
    }
    results.mcRuns         = B;
    results.mcNumRows      = nBaseTraces;
    results.mcNumKeptSteps = numKeptSteps_;

    if (!minOut_) {
        bar.complete();
        std::cout << "\n";
    }
#else
    (void) mObj;
#endif
}

void Simulation::trans_sim_gpu_windowed(Matrix& mObj) {
#if defined(JOSIM_CUDA) && !defined(SLU)
    results.timeAxis.clear();
    results.timeAxis.reserve(static_cast<size_t>(numKeptSteps_));
    for (int64_t i = 0; i < simSize_; ++i) {
        if (keptSlot_[i] >= 0) { results.timeAxis.push_back(i * stepSize_); }
    }

    ProgressBar bar;
    if (!minOut_) {
        bar.create_thread();
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text("Simulating (GPU)");
        bar.set_total(static_cast<float>(simSize_));
    }

    const int64_t       numSrc = static_cast<int64_t>(mObj.sourcegen.size());
    std::vector<double> h_srcWindow(static_cast<size_t>((windowSize_ + 2) * std::max<int64_t>(1, numSrc)), 0.0);

    for (int64_t i0 = 0; i0 < simSize_; i0 += windowSize_) {
        if (!minOut_) { bar.update(static_cast<float>(i0)); }
        const int64_t i1 = std::min<int64_t>(simSize_, i0 + windowSize_);
        const int64_t w  = i1 - i0;

        if (numSrc > 0) {
            for (int64_t s = -2; s < w; ++s) {
                const int64_t stepIdx = i0 + s;
                const double  t       = (stepIdx >= 0) ? (stepIdx * stepSize_) : 0.0;
                double*       row     = h_srcWindow.data() + (s + 2) * numSrc;
                if (stepIdx >= 0) {
                    for (int64_t srcIdx = 0; srcIdx < numSrc; ++srcIdx) {
                        row[srcIdx] = mObj.sourcegen[srcIdx].value(t);
                    }
                } else {
                    std::fill(row, row + numSrc, 0.0);
                }
            }
        }

        if (!cudaSolver_->step_window(i0, i1, numSrc > 0 ? h_srcWindow.data() : nullptr)) {
            needsTR_ = true;
            if (!minOut_) {
                bar.complete();
                std::cout << "\n";
            }
            return;
        }
    }

    std::vector<double> h_traceOut;
    std::vector<double> h_xFinal;
    cudaSolver_->download_results(h_traceOut, h_xFinal);

    int64_t curRowOff = 0;
    int64_t traceSlot = 0;
    for (const auto& part : partitions_) {
        for (int64_t locR = 0; locR < part.n; ++locR) {
            x_[part.globalRows[locR]] = h_xFinal[curRowOff + locR];
        }
        curRowOff += part.n;
        for (int64_t globR : part.traceGlobalRows) {
            auto&        dst = *results.xVector[globR];
            const double* src = h_traceOut.data() + traceSlot * numKeptSteps_;
            dst.assign(src, src + numKeptSteps_);
            ++traceSlot;
        }
    }

    if (!minOut_) {
        bar.complete();
        std::cout << "\n";
    }
#else
    (void) mObj;
#endif
}

void Simulation::trans_sim_gpu_monolithic(Matrix& mObj) {
#if defined(JOSIM_CUDA) && !defined(SLU)
    results.timeAxis.clear();
    results.timeAxis.reserve(useDecimation_ ? results.printIndices.size() * 3 : simSize_);
    ProgressBar bar;
    if (!minOut_) {
        bar.create_thread();
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text("Simulating (GPU Mono)");
        bar.set_total(static_cast<float>(simSize_));
    }
    const bool                             hasRelevantTraces = !mObj.relevantTraces.empty();
    std::vector<std::pair<int32_t, double>> changedNz;
    for (int64_t i = 0; i < simSize_; ++i) {
        double step = i * stepSize_;
        if (!minOut_) { bar.update(static_cast<float>(i)); }

        if (!cudaSolver_->step_monolithic_devices(i, x_.data(), b_.data(), changedNz)) {
            needsTR_ = true;
            if (!minOut_) {
                bar.complete();
                std::cout << "\n";
            }
            return;
        }
        if (!changedNz.empty()) {
            for (const auto& kv : changedNz) { mObj.nz[kv.first] = kv.second; }
            refactor_lu(mObj);
        }

        handle_cs(mObj, step, i);
        handle_vs(mObj, i, step, 1.0);
        handle_ps(mObj, i, step, 1.0);
        handle_ccvs(mObj);
        handle_vccs(mObj);
        handle_tx(mObj, i, step);

        solve_lu(mObj);

        if (!useDecimation_ || keepStep_[i] != 0) {
            if (hasRelevantTraces) {
                for (const auto& j : mObj.relevantIndices) { results.xVector[j]->emplace_back(x_[j]); }
            } else {
                for (size_t j = 0; j < results.xVector.size(); ++j) {
                    if (results.xVector[j]) { results.xVector[j]->emplace_back(x_[j]); }
                }
            }
            results.timeAxis.emplace_back(step);
        }
    }
    if (!minOut_) {
        bar.complete();
        std::cout << "\n";
    }
#else
    (void) mObj;
#endif
}

void Simulation::trans_sim_gpu_monolithic_full(Matrix& mObj) {
#if defined(JOSIM_CUDA) && !defined(SLU)
    results.timeAxis.clear();
    results.timeAxis.reserve(static_cast<size_t>(numKeptSteps_));
    for (int64_t i = 0; i < simSize_; ++i) {
        if (keptSlot_[i] >= 0) { results.timeAxis.push_back(i * stepSize_); }
    }

    ProgressBar bar;
    if (!minOut_) {
        bar.create_thread();
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text("Simulating (GPU RF)");
        bar.set_total(static_cast<float>(simSize_));
    }

    const int64_t       numSrc = static_cast<int64_t>(mObj.sourcegen.size());
    std::vector<double> h_srcWindow(static_cast<size_t>((windowSize_ + 2) * std::max<int64_t>(1, numSrc)), 0.0);

    for (int64_t i0 = 0; i0 < simSize_; i0 += windowSize_) {
        if (!minOut_) { bar.update(static_cast<float>(i0)); }
        const int64_t i1 = std::min<int64_t>(simSize_, i0 + windowSize_);
        const int64_t w  = i1 - i0;

        if (numSrc > 0) {
            for (int64_t s = -2; s < w; ++s) {
                const int64_t stepIdx = i0 + s;
                const double  t       = (stepIdx >= 0) ? (stepIdx * stepSize_) : 0.0;
                double*       row     = h_srcWindow.data() + (s + 2) * numSrc;
                if (stepIdx >= 0) {
                    for (int64_t srcIdx = 0; srcIdx < numSrc; ++srcIdx) {
                        row[srcIdx] = mObj.sourcegen[srcIdx].value(t);
                    }
                } else {
                    std::fill(row, row + numSrc, 0.0);
                }
            }
        }

        if (!cudaSolver_->step_monolithic_full_window(i0, i1, numSrc > 0 ? h_srcWindow.data() : nullptr)) {
            needsTR_ = true;
            if (!minOut_) {
                bar.complete();
                std::cout << "\n";
            }
            return;
        }
    }

    std::vector<double> h_traceOut;
    std::vector<double> h_xFinal;
    cudaSolver_->download_results(h_traceOut, h_xFinal);

    const int64_t N = static_cast<int64_t>(mObj.rp.size()) - 1;
    for (int64_t r = 0; r < N; ++r) { x_[r] = h_xFinal[r]; }

    int64_t traceSlot = 0;
    if (!mObj.relevantTraces.empty()) {
        for (int64_t globR : mObj.relevantIndices) {
            if (globR >= 0 && globR < N) {
                auto&         dst = *results.xVector[globR];
                const double* src = h_traceOut.data() + traceSlot * numKeptSteps_;
                dst.assign(src, src + numKeptSteps_);
                ++traceSlot;
            }
        }
    } else {
        for (int64_t globR = 0; globR < N; ++globR) {
            if (results.xVector[globR]) {
                auto&         dst = *results.xVector[globR];
                const double* src = h_traceOut.data() + traceSlot * numKeptSteps_;
                dst.assign(src, src + numKeptSteps_);
            }
            ++traceSlot;
        }
    }

    if (!minOut_) {
        bar.complete();
        std::cout << "\n";
    }
#else
    (void) mObj;
#endif
}

void Simulation::trans_sim_omp_windowed(Matrix& mObj) {
#ifndef SLU
    results.timeAxis.clear();
    results.timeAxis.reserve(static_cast<size_t>(numKeptSteps_));
    for (int64_t i = 0; i < simSize_; ++i) {
        if (keptSlot_[i] >= 0) { results.timeAxis.push_back(i * stepSize_); }
    }
    for (const auto& part : partitions_) {
        for (int64_t globR : part.traceGlobalRows) {
            results.xVector[globR]->resize(static_cast<size_t>(numKeptSteps_), 0.0);
        }
    }

    ProgressBar bar;
    if (!minOut_) {
        bar.create_thread();
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text("Simulating");
        bar.set_total(static_cast<float>(simSize_));
    }

    b_.assign(mObj.rp.size(), 0.0);

    const int64_t       nParts   = static_cast<int64_t>(partitions_.size());
    const int64_t       numSrc   = static_cast<int64_t>(mObj.sourcegen.size());
    const int64_t       numTx    = static_cast<int64_t>(txlines_.size());
    std::vector<double> srcWin(static_cast<size_t>((windowSize_ + 2) * std::max<int64_t>(1, numSrc)), 0.0);
    std::vector<double> txRem1(static_cast<size_t>(std::max<int64_t>(1, numTx) * windowSize_), 0.0);
    std::vector<double> txRem2(static_cast<size_t>(std::max<int64_t>(1, numTx) * windowSize_), 0.0);

#ifdef _OPENMP
    const int nThreads = std::min<int>(omp_get_max_threads(), static_cast<int>(nParts));
#endif

    for (int64_t i0 = 0; i0 < simSize_; i0 += windowSize_) {
        if (!minOut_) { bar.update(static_cast<float>(i0)); }
        const int64_t i1 = std::min<int64_t>(simSize_, i0 + windowSize_);
        const int64_t w  = i1 - i0;

        // 1. Pre-evaluate sources for the window [i0 - 2, i1)
        if (numSrc > 0) {
            for (int64_t s = -2; s < w; ++s) {
                const int64_t stepIdx = i0 + s;
                const double  t       = (stepIdx >= 0) ? (stepIdx * stepSize_) : 0.0;
                double*       row     = srcWin.data() + (s + 2) * numSrc;
                if (stepIdx >= 0) {
                    for (int64_t srcIdx = 0; srcIdx < numSrc; ++srcIdx) {
                        row[srcIdx] = mObj.sourcegen[srcIdx].value(t);
                    }
                } else {
                    std::fill(row, row + numSrc, 0.0);
                }
            }
        }

        // 2. Precompute delayed remote port contributions for all transmission lines across [i0, i1)
        if (numTx > 0) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (numTx * w >= 256)
#endif
            for (int64_t txIdx = 0; txIdx < numTx; ++txIdx) {
                const auto&   tx      = txlines_[txIdx];
                const double  Z       = tx.netlistInfo.value_;
                const int64_t k       = tx.timestepDelay_;
                const int64_t hSize   = tx.histSize_;
                const double  cZPhi   = (Z / Constants::SIGMA) * ((2.0 * stepSize_) / 3.0);
                const auto&   hist    = tx.hist_;
                const double  alpha   = tx.fracDelay_;
                const double  oma     = 1.0 - alpha;

                auto interp_v1 = [&](int64_t m) -> double {
                    if (m < 0) { return 0.0; }
                    const double s0 = hist[m % hSize].v1;
                    const double s1 = (m >= 1) ? hist[(m - 1) % hSize].v1 : 0.0;
                    return oma * s0 + alpha * s1;
                };
                auto interp_v2 = [&](int64_t m) -> double {
                    if (m < 0) { return 0.0; }
                    const double s0 = hist[m % hSize].v2;
                    const double s1 = (m >= 1) ? hist[(m - 1) % hSize].v2 : 0.0;
                    return oma * s0 + alpha * s1;
                };
                auto interp_i1 = [&](int64_t m) -> double {
                    if (m < 0) { return 0.0; }
                    const double s0 = hist[m % hSize].i1;
                    const double s1 = (m >= 1) ? hist[(m - 1) % hSize].i1 : 0.0;
                    return oma * s0 + alpha * s1;
                };
                auto interp_i2 = [&](int64_t m) -> double {
                    if (m < 0) { return 0.0; }
                    const double s0 = hist[m % hSize].i2;
                    const double s1 = (m >= 1) ? hist[(m - 1) % hSize].i2 : 0.0;
                    return oma * s0 + alpha * s1;
                };

                for (int64_t s = 0; s < w; ++s) {
                    const int64_t effStep = tx.loadedSteps_ + i0 + s;
                    double        rem1    = 0.0;
                    double        rem2    = 0.0;
                    if (effStep >= k) {
                        const int64_t m = effStep - k;
                        if (alpha > 0.0) {
                            const double nk_1 = interp_v1(m);
                            const double nk_2 = interp_v2(m);
                            const double I1nk = interp_i1(m);
                            const double I2nk = interp_i2(m);
                            if (atyp_ == AnalysisType::Voltage) {
                                rem1 = Z * I2nk + nk_2;
                                rem2 = Z * I1nk + nk_1;
                            } else {
                                rem1 = cZPhi * I2nk + nk_2 - (4.0 / 3.0) * interp_v2(m - 1)
                                       + (1.0 / 3.0) * interp_v2(m - 2);
                                rem2 = cZPhi * I1nk + nk_1 - (4.0 / 3.0) * interp_v1(m - 1)
                                       + (1.0 / 3.0) * interp_v1(m - 2);
                            }
                        } else {
                            const auto& h_k = hist[m % hSize];
                            if (atyp_ == AnalysisType::Voltage) {
                                rem1 = Z * h_k.i2 + h_k.v2;
                                rem2 = Z * h_k.i1 + h_k.v1;
                            } else if (effStep == k) {
                                rem1 = cZPhi * h_k.i2 + h_k.v2;
                                rem2 = cZPhi * h_k.i1 + h_k.v1;
                            } else if (effStep == k + 1) {
                                const auto& h_k1 = hist[(m - 1) % hSize];
                                rem1             = cZPhi * h_k.i2 + h_k.v2 - (4.0 / 3.0) * h_k1.v2;
                                rem2             = cZPhi * h_k.i1 + h_k.v1 - (4.0 / 3.0) * h_k1.v1;
                            } else {
                                const auto& h_k1 = hist[(m - 1) % hSize];
                                const auto& h_k2 = hist[(m - 2) % hSize];
                                rem1             = cZPhi * h_k.i2 + h_k.v2 - (4.0 / 3.0) * h_k1.v2
                                       + (1.0 / 3.0) * h_k2.v2;
                                rem2 = cZPhi * h_k.i1 + h_k.v1 - (4.0 / 3.0) * h_k1.v1
                                       + (1.0 / 3.0) * h_k2.v1;
                            }
                        }
                    }
                    txRem1[txIdx * w + s] = rem1;
                    txRem2[txIdx * w + s] = rem2;
                }
            }
        }

        // 3. Advance all PTL-decoupled partitions in parallel across the entire w-step window
        bool windowNeedsTR = false;
        bool windowSingErr = false;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nThreads) reduction(|| : windowNeedsTR, windowSingErr)
#endif
        for (int64_t p = 0; p < nParts; ++p) {
            auto& part = partitions_[p];
            for (int64_t s = 0; s < w; ++s) {
                if (windowNeedsTR || windowSingErr) { break; }
                const int64_t i          = i0 + s;
                const bool    checkPhase = (static_cast<double>(i) / static_cast<double>(simSize_) > 0.01);

                for (int64_t globR : part.globalRows) { b_[globR] = 0.0; }

                // JJs
                for (int64_t jIdx : part.jjIndices) {
                    auto&       temp  = jjs_[jIdx];
                    const auto& model = temp.model_;
                    if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
                        temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
                    } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                        temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
                    } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                        temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
                    } else {
                        temp.pn1_ = 0.0;
                    }
                    if (i > 0) {
                        if (atyp_ == AnalysisType::Voltage) {
                            temp.vn1_ = temp.pn1_;
                            temp.pn1_ = x_[temp.variableIndex_];
                        } else if (atyp_ == AnalysisType::Phase) {
                            temp.vn1_ = x_[temp.variableIndex_];
                        }
                    }
                    const double v0 = (5.0 / 2.0) * temp.vn1_ - 2.0 * temp.vn2_ + (1.0 / 2.0) * temp.vn3_;
                    temp.phi0_      = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_
                                 + ((1.0 / Constants::SIGMA) * ((2.0 * stepSize_) / 3.0)) * v0;
                    if (checkPhase && std::abs(temp.phi0_ - temp.pn1_) > (0.20 * 2.0 * Constants::PI)) {
                        windowNeedsTR = true;
                        break;
                    }
                    if (atyp_ == AnalysisType::Voltage) {
                        b_[temp.variableIndex_] = Constants::SIGMA
                                                  * (-(2.0 / stepSize_) * temp.pn1_
                                                     + (1.0 / (2.0 * stepSize_)) * temp.pn2_);
                    } else if (atyp_ == AnalysisType::Phase) {
                        b_[temp.variableIndex_] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
                    }
                    temp.pn4_ = temp.pn3_;
                    temp.pn3_ = temp.pn2_;
                    temp.pn2_ = temp.pn1_;
                    temp.vn6_ = temp.vn5_;
                    temp.vn5_ = temp.vn4_;
                    temp.vn4_ = temp.vn3_;
                    temp.vn3_ = temp.vn2_;
                    if (model.rtype() == 1) {
                        if (temp.update_value(v0)) {
                            const double newNz            = temp.matrixInfo.nonZeros_.back();
                            mObj.nz[temp.nzBackIndex_]    = newNz;
                            part.nz[jjPartNzIdx_[jIdx]]   = newNz;
                            part.needsLU                  = true;
                        }
                    }
                    const auto&  cpr        = model.cpr();
                    const double phiOff     = model.phiOff();
                    const double ic         = model.ic();
                    double       ic_sin_phi = 0.0;
                    if (cpr.size() == 1) {
                        ic_sin_phi += ic * (cpr[0] * std::sin(1.0 * (temp.phi0_ - phiOff)));
                    } else {
                        for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                            ic_sin_phi += ic * (cpr[harm] * std::sin((harm + 1) * (temp.phi0_ - phiOff)));
                        }
                    }
                    temp.cTerm_ = -(((2.0 * model.c()) / stepSize_) * temp.vn1_)
                                  + ((model.c() / (2.0 * stepSize_)) * temp.vn2_);
                    if (!model.tDep()) {
                        b_[temp.indexInfo.currentIndex_.value()]
                                = (temp.matrixInfo.nonZeros_.back())
                                  * (ic_sin_phi - (((2.0 * model.c()) / stepSize_) * temp.vn1_)
                                     + ((model.c() / (2.0 * stepSize_)) * temp.vn2_) + temp.it_);
                    } else {
                        double sin2_half_phi = 0.0;
                        for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                            sin2_half_phi += cpr[harm] * std::sin((harm + 1) * (temp.phi0_ - phiOff) / 2.0);
                        }
                        sin2_half_phi  = sin2_half_phi * sin2_half_phi;
                        double sin_phi = 0.0;
                        for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                            sin_phi += cpr[harm] * std::sin((harm + 1) * (temp.phi0_ - phiOff));
                        }
                        double sqrt_part = std::sqrt(1.0 - model.d() * sin2_half_phi);
                        b_[temp.indexInfo.currentIndex_.value()]
                                = (temp.matrixInfo.nonZeros_.back())
                                  * ((((Constants::PI * temp.del_) / (2.0 * Constants::EV * model.rn()))
                                      * (sin_phi / sqrt_part)
                                      * std::tanh(temp.del_ / (2.0 * Constants::BOLTZMANN * model.t()) * sqrt_part))
                                     - (((2.0 * model.c()) / stepSize_) * temp.vn1_)
                                     + ((model.c() / (2.0 * stepSize_)) * temp.vn2_) + temp.it_);
                    }
                    temp.vn2_ = temp.vn1_;
                }
                if (windowNeedsTR) { break; }

                if (part.needsLU) {
                    int64_t ok = klu_l_refactor(&part.rp.front(),
                                                &part.ci.front(),
                                                &part.nz.front(),
                                                part.symbolic,
                                                part.numeric,
                                                &part.common);
                    if (!ok) {
                        klu_l_free_numeric(&part.numeric, &part.common);
                        part.numeric = klu_l_factor(&part.rp.front(),
                                                    &part.ci.front(),
                                                    &part.nz.front(),
                                                    part.symbolic,
                                                    &part.common);
                    }
                    part.needsLU = false;
                }

                // Current Sources
                for (int64_t cIdx : part.csIndices) {
                    const auto&  cs     = mObj.components.currentsources[cIdx];
                    const double srcVal = srcWin[(s + 2) * numSrc + cs.sourceIndex_];
                    if (cs.indexInfo.nodeConfig_ == NodeConfig::POSGND) {
                        b_[cs.indexInfo.posIndex_.value()] -= srcVal;
                    } else if (cs.indexInfo.nodeConfig_ == NodeConfig::GNDNEG) {
                        b_[cs.indexInfo.negIndex_.value()] += srcVal;
                    } else if (cs.indexInfo.nodeConfig_ == NodeConfig::POSNEG) {
                        b_[cs.indexInfo.posIndex_.value()] -= srcVal;
                        b_[cs.indexInfo.negIndex_.value()] += srcVal;
                    }
                }

                // Resistors (Phase mode)
                if (atyp_ == AnalysisType::Phase) {
                    for (int64_t rIdx : part.resIndices) {
                        auto&       temp = resistors_[rIdx];
                        NodeConfig& nc   = temp.indexInfo.nodeConfig_;
                        if (nc == NodeConfig::POSGND) {
                            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
                        } else if (nc == NodeConfig::GNDNEG) {
                            temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
                        } else if (nc == NodeConfig::POSNEG) {
                            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
                        } else {
                            temp.pn1_ = 0.0;
                        }
                        b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
                        temp.pn4_                                = temp.pn3_;
                        temp.pn3_                                = temp.pn2_;
                        temp.pn2_                                = temp.pn1_;
                    }
                }

                // Inductors (Voltage mode)
                if (atyp_ == AnalysisType::Voltage) {
                    const double h = stepSize_;
                    for (int64_t lIdx : part.indIndices) {
                        auto&         temp   = inductors_[lIdx];
                        const int64_t curIdx = temp.indexInfo.currentIndex_.value();
                        b_[curIdx]           = -(2.0 * temp.netlistInfo.value_ / h) * x_[curIdx]
                                     + ((temp.netlistInfo.value_ / (2.0 * h)) * temp.In2_);
                        for (const auto& m : temp.get_mutualInductance()) {
                            Inductor& mi  = inductors_[m.first];
                            b_[curIdx]   += (-((2.0 * m.second) / h) * x_[mi.indexInfo.currentIndex_.value()]
                                           + (m.second / (2.0 * h)) * mi.In2_);
                        }
                        temp.In4_ = temp.In3_;
                        temp.In3_ = temp.In2_;
                        temp.In2_ = x_[curIdx];
                    }
                }

                // Capacitors
                for (int64_t cIdx : part.capIndices) {
                    auto& temp = capacitors_[cIdx];
                    if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
                        temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
                    } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                        temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
                    } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                        temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
                    } else {
                        temp.pn1_ = 0.0;
                    }
                    if (atyp_ == AnalysisType::Voltage) {
                        b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
                    } else if (atyp_ == AnalysisType::Phase) {
                        b_[temp.indexInfo.currentIndex_.value()] = (8.0 / 3.0) * temp.pn1_ - (22.0 / 9.0) * temp.pn2_
                                                                   + (8.0 / 9.0) * temp.pn3_ - (1.0 / 9.0) * temp.pn4_;
                        temp.pn7_ = temp.pn6_;
                        temp.pn6_ = temp.pn5_;
                        temp.pn5_ = temp.pn4_;
                        temp.pn4_ = temp.pn3_;
                        temp.pn3_ = temp.pn2_;
                    }
                    temp.pn2_ = temp.pn1_;
                }

                // Voltage Sources
                for (int64_t vIdx : part.vsIndices) {
                    auto&              temp   = vsources_[vIdx];
                    JoSIM::NodeConfig& nc     = temp.indexInfo.nodeConfig_;
                    const double       srcVal = srcWin[(s + 2) * numSrc + temp.sourceIndex_];
                    if (atyp_ == AnalysisType::Voltage) {
                        b_[temp.indexInfo.currentIndex_.value()] = srcVal;
                    } else if (atyp_ == AnalysisType::Phase) {
                        if (nc == NodeConfig::POSGND) {
                            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
                        } else if (nc == NodeConfig::GNDNEG) {
                            temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
                        } else if (nc == NodeConfig::POSNEG) {
                            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
                        } else {
                            temp.pn1_ = 0.0;
                        }
                        b_[temp.indexInfo.currentIndex_.value()]
                                = ((2.0 * stepSize_) / (3.0 * Constants::SIGMA)) * srcVal
                                  + (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
                        temp.pn4_ = temp.pn3_;
                        temp.pn3_ = temp.pn2_;
                        temp.pn2_ = temp.pn1_;
                    }
                }

                // Phase Sources
                for (int64_t pIdx : part.psIndices) {
                    auto&        temp = psources_[pIdx];
                    const double s0   = srcWin[(s + 2) * numSrc + temp.sourceIndex_];
                    if (atyp_ == AnalysisType::Phase) {
                        b_[temp.indexInfo.currentIndex_.value()] = s0;
                    } else if (atyp_ == AnalysisType::Voltage) {
                        if (i == 0) {
                            b_[temp.indexInfo.currentIndex_.value()]
                                    = (Constants::SIGMA / stepSize_) * ((3.0 / 2.0) * s0);
                        } else if (i == 1) {
                            const double s1 = srcWin[(s + 1) * numSrc + temp.sourceIndex_];
                            b_[temp.indexInfo.currentIndex_.value()]
                                    = (Constants::SIGMA / stepSize_) * ((3.0 / 2.0) * s0 - 2.0 * s1);
                        } else {
                            const double s1 = srcWin[(s + 1) * numSrc + temp.sourceIndex_];
                            const double s2 = srcWin[s * numSrc + temp.sourceIndex_];
                            b_[temp.indexInfo.currentIndex_.value()]
                                    = (Constants::SIGMA / stepSize_) * ((3.0 / 2.0) * s0 - 2.0 * s1 + 0.5 * s2);
                        }
                    }
                }

                // CCVS & VCCS (Phase mode)
                if (atyp_ == AnalysisType::Phase) {
                    for (int64_t idx : part.ccvsIndices) {
                        auto& temp = ccvss_[idx];
                        if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
                            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
                        } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                            temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
                        } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
                        } else {
                            temp.pn1_ = 0.0;
                        }
                        b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
                        temp.pn4_                                = temp.pn3_;
                        temp.pn3_                                = temp.pn2_;
                        temp.pn2_                                = temp.pn1_;
                    }
                    for (int64_t idx : part.vccsIndices) {
                        auto& temp = vccss_[idx];
                        if (temp.posIndex2_ && !temp.negIndex2_) {
                            temp.pn1_ = x_[temp.posIndex2_.value()];
                        } else if (!temp.posIndex2_ && temp.negIndex2_) {
                            temp.pn1_ = -x_[temp.negIndex2_.value()];
                        } else if (temp.posIndex2_ && temp.negIndex2_) {
                            temp.pn1_ = x_[temp.posIndex2_.value()] - x_[temp.negIndex2_.value()];
                        } else {
                            temp.pn1_ = 0.0;
                        }
                        b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
                        temp.pn4_                                = temp.pn3_;
                        temp.pn3_                                = temp.pn2_;
                        temp.pn2_                                = temp.pn1_;
                    }
                }

                // Transmission Line Port Halves (RHS setup)
                for (int64_t tIdx : part.txPort1Indices) {
                    auto&        tx  = txlines_[tIdx];
                    const double rem = txRem1[tIdx * w + s];
                    if (atyp_ == AnalysisType::Voltage) {
                        b_[tx.indexInfo.currentIndex_.value()] = rem;
                    } else {
                        b_[tx.indexInfo.currentIndex_.value()]
                                = (4.0 / 3.0) * tx.n1_1_ - (1.0 / 3.0) * tx.n2_1_ + rem;
                    }
                }
                for (int64_t tIdx : part.txPort2Indices) {
                    auto&        tx  = txlines_[tIdx];
                    const double rem = txRem2[tIdx * w + s];
                    if (atyp_ == AnalysisType::Voltage) {
                        b_[tx.currentIndex2_] = rem;
                    } else {
                        b_[tx.currentIndex2_] = (4.0 / 3.0) * tx.n1_2_ - (1.0 / 3.0) * tx.n2_2_ + rem;
                    }
                }

                // Partition KLU Solve
                for (int64_t r = 0; r < part.n; ++r) { part.xLocal[r] = b_[part.globalRows[r]]; }
                int64_t ok = klu_l_tsolve(part.symbolic, part.numeric, part.n, 1, part.xLocal.data(), &part.common);
                if (!ok) {
                    windowSingErr = true;
                    break;
                }
                for (int64_t r = 0; r < part.n; ++r) { x_[part.globalRows[r]] = part.xLocal[r]; }

                // Record completed step i into transmission line circular history and shift local port state
                for (int64_t tIdx : part.txPort1Indices) {
                    auto&             tx   = txlines_[tIdx];
                    JoSIM::NodeConfig nc   = tx.indexInfo.nodeConfig_;
                    double            v1   = 0.0;
                    if (nc == NodeConfig::POSGND) {
                        v1 = x_[tx.indexInfo.posIndex_.value()];
                    } else if (nc == NodeConfig::GNDNEG) {
                        v1 = -x_[tx.indexInfo.negIndex_.value()];
                    } else if (nc == NodeConfig::POSNEG) {
                        v1 = x_[tx.indexInfo.posIndex_.value()] - x_[tx.indexInfo.negIndex_.value()];
                    }
                    const int64_t slot    = (tx.loadedSteps_ + i) % tx.histSize_;
                    tx.hist_[slot].v1     = v1;
                    tx.hist_[slot].i1     = x_[tx.indexInfo.currentIndex_.value()];
                    if (atyp_ == AnalysisType::Phase) {
                        tx.n2_1_ = tx.n1_1_;
                        tx.n1_1_ = v1;
                    }
                }
                for (int64_t tIdx : part.txPort2Indices) {
                    auto&             tx   = txlines_[tIdx];
                    JoSIM::NodeConfig nc2  = tx.nodeConfig2_;
                    double            v2   = 0.0;
                    if (nc2 == NodeConfig::POSGND) {
                        v2 = x_[tx.posIndex2_.value()];
                    } else if (nc2 == NodeConfig::GNDNEG) {
                        v2 = -x_[tx.negIndex2_.value()];
                    } else if (nc2 == NodeConfig::POSNEG) {
                        v2 = x_[tx.posIndex2_.value()] - x_[tx.negIndex2_.value()];
                    }
                    const int64_t slot    = (tx.loadedSteps_ + i) % tx.histSize_;
                    tx.hist_[slot].v2     = v2;
                    tx.hist_[slot].i2     = x_[tx.currentIndex2_];
                    if (atyp_ == AnalysisType::Phase) {
                        tx.n2_2_ = tx.n1_2_;
                        tx.n1_2_ = v2;
                    }
                }

                // Store kept trace outputs for step i
                const int32_t kSlot = keptSlot_[i];
                if (kSlot >= 0) {
                    for (int64_t globR : part.traceGlobalRows) {
                        (*results.xVector[globR])[kSlot] = x_[globR];
                    }
                }
            }
        }
        if (windowSingErr) { Errors::simulation_errors(SimulationErrors::MATRIX_SINGULAR); }
        if (windowNeedsTR) {
            needsTR_ = true;
            if (!minOut_) {
                bar.complete();
                std::cout << "\n";
            }
            return;
        }
    }

    // Sync device states back to mObj.components.devices for callers that reuse mObj (e.g. IV)
    for (size_t idx = 0; idx < mObj.components.junctionIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.junctionIndices[idx]] = jjs_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.resistorIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.resistorIndices[idx]] = resistors_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.inductorIndices.size(); ++idx) {
        auto& dst = std::get<Inductor>(mObj.components.devices[mObj.components.inductorIndices[idx]]);
        dst.In2_  = inductors_[idx].In2_;
        dst.In3_  = inductors_[idx].In3_;
        dst.In4_  = inductors_[idx].In4_;
    }
    for (size_t idx = 0; idx < mObj.components.capacitorIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.capacitorIndices[idx]] = capacitors_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.txIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.txIndices[idx]] = txlines_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.vsIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.vsIndices[idx]] = vsources_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.psIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.psIndices[idx]] = psources_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.ccvsIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.ccvsIndices[idx]] = ccvss_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.vccsIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.vccsIndices[idx]] = vccss_[idx];
    }

    if (!minOut_) {
        bar.complete();
        std::cout << "\n";
    }
#else
    (void) mObj;
#endif
}

void Simulation::trans_sim(Matrix& mObj) {
    // Ensure the b vector is sized once
    b_.assign(mObj.rp.size(), 0.0);
    if (loadStateFile_.has_value()) {
        load_state(loadStateFile_.value(), mObj);
    } else if (startup_) {
        bool nonZeroInit = hasJJThermalNoise_ || hasResistorThermalNoise_;
        if (!nonZeroInit) {
            for (const auto& jj : jjs_) {
                if (jj.model_.phiOff() != 0.0) {
                    nonZeroInit = true;
                    break;
                }
            }
        }
        if (!nonZeroInit) {
            for (auto& src : mObj.sourcegen) {
                if (src.value(-stepSize_) != 0.0 || src.value(0.0) != 0.0) {
                    nonZeroInit = true;
                    break;
                }
            }
        }
        if (nonZeroInit) {
            // Stabilize the simulation before starting at t=0
            int64_t startup = static_cast<int64_t>(2 * pow(10, (abs(log10(stepSize_)) - 12) * 2 + 1));
            if (startup > 1000) { startup = 1000; }
            for (int64_t i = -startup; i < 0; ++i) {
                // Setup the b matrix
                setup_b(mObj, i, i * stepSize_);
                if (needsTR_) { return; }
                // Solve Ax=b, storing the results in x_
                solve_lu(mObj);
                if (correctorIter_ > 0 && !jjs_.empty()) { corrector_jj(mObj); }
            }
        }
    }

#ifndef SLU
    if (mcRuns_ > 1 && usePartitions_ && windowSize_ >= 1 && !hasJJThermalNoise_ && !hasResistorThermalNoise_
        && correctorIter_ == 0 && !saveStateFile_.has_value()) {
        trans_sim_mc(mObj);
        return;
    }
#ifdef JOSIM_CUDA
    if (gpu_) {
        if (gpuMode_ == 3) {
            if (setup_gpu_monolithic_full(mObj) && cudaSolver_) {
                trans_sim_gpu_monolithic_full(mObj);
                return;
            }
        } else if (gpuMode_ == 2) {
            if (setup_gpu_monolithic(mObj) && cudaSolver_) {
                trans_sim_gpu_monolithic(mObj);
                return;
            }
        } else {
            if (setup_gpu_solver(mObj) && cudaSolver_) {
                trans_sim_gpu_windowed(mObj);
                return;
            }
            const int64_t N      = static_cast<int64_t>(mObj.rp.size()) - 1;
            const int64_t luNnz  = (Numeric_ != nullptr) ? (Numeric_->lnz + Numeric_->unz) : 0;
            const bool    mesh2D = (N >= 10000 && luNnz >= 10 * N && luNnz >= 200000);
            if (mesh2D && setup_gpu_monolithic_full(mObj) && cudaSolver_) {
                trans_sim_gpu_monolithic_full(mObj);
                return;
            }
            if (setup_gpu_monolithic(mObj) && cudaSolver_) {
                trans_sim_gpu_monolithic(mObj);
                return;
            }
        }
    }
#endif
    if (parallel_ && usePartitions_ && windowSize_ >= 2 && partitions_.size() >= 2 && !hasJJThermalNoise_
        && !hasResistorThermalNoise_ && correctorIter_ == 0 && !saveStateFile_.has_value()) {
        trans_sim_omp_windowed(mObj);
        return;
    }
#endif
    // Ensure time axis is cleared and pre-allocated
    results.timeAxis.clear();
    results.timeAxis.reserve(useDecimation_ ? results.printIndices.size() * 3 : simSize_);
    ProgressBar bar;
    if (!minOut_) {
        bar.create_thread();
        bar.set_bar_width(30);
        bar.fill_bar_progress_with("O");
        bar.fill_bar_remainder_with(" ");
        bar.set_status_text("Simulating");
        bar.set_total((float) simSize_);
    }
    const bool hasRelevantTraces = !mObj.relevantTraces.empty();
    bool       stateSaved        = false;
    // Start the simulation loop
    for (int64_t i = 0; i < simSize_; ++i) {
        double step = i * stepSize_;
        // If not minimal printing report progress
        if (!minOut_) { bar.update(static_cast<float>(i)); }
        // Setup the b matrix
        setup_b(mObj, i, step);
        if (needsTR_) { return; }
        // Solve Ax=b, storing the results in x_
        solve_lu(mObj);
        if (correctorIter_ > 0 && !jjs_.empty()) { corrector_jj(mObj); }
        // Store results (only requested, and only kept steps when decimation is active)
        if (!useDecimation_ || keepStep_[i] != 0) {
            if (hasRelevantTraces) {
                for (const auto& j : mObj.relevantIndices) { results.xVector[j]->emplace_back(x_[j]); }
            } else {
                for (size_t j = 0; j < results.xVector.size(); ++j) {
                    if (results.xVector[j]) { results.xVector[j]->emplace_back(x_[j]); }
                }
            }
            // Store the time step
            results.timeAxis.emplace_back(step);
        }
        if (!stateSaved && saveStateFile_.has_value() && saveStateFile_->second.has_value()
            && step >= saveStateFile_->second.value()) {
            save_state(saveStateFile_->first, mObj, i);
            stateSaved = true;
        }
    }
    if (!stateSaved && saveStateFile_.has_value()) {
        save_state(saveStateFile_->first, mObj, simSize_ - 1);
        stateSaved = true;
    }

    // Sync device states back to mObj.components.devices for callers that reuse mObj (e.g. IV)
    for (size_t idx = 0; idx < mObj.components.junctionIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.junctionIndices[idx]] = jjs_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.resistorIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.resistorIndices[idx]] = resistors_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.inductorIndices.size(); ++idx) {
        auto& dst = std::get<Inductor>(mObj.components.devices[mObj.components.inductorIndices[idx]]);
        dst.In2_  = inductors_[idx].In2_;
        dst.In3_  = inductors_[idx].In3_;
        dst.In4_  = inductors_[idx].In4_;
    }
    for (size_t idx = 0; idx < mObj.components.capacitorIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.capacitorIndices[idx]] = capacitors_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.txIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.txIndices[idx]] = txlines_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.vsIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.vsIndices[idx]] = vsources_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.psIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.psIndices[idx]] = psources_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.ccvsIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.ccvsIndices[idx]] = ccvss_[idx];
    }
    for (size_t idx = 0; idx < mObj.components.vccsIndices.size(); ++idx) {
        mObj.components.devices[mObj.components.vccsIndices[idx]] = vccss_[idx];
    }

    if (!minOut_) {
        bar.complete();
        std::cout << "\n";
    }
}

void Simulation::reduce_step(Input& iObj, Matrix& mObj) {
    if (fixedStep_) {
        Errors::simulation_errors(SimulationErrors::PHASEGUESS_TOO_LARGE,
                                  offendingJJ_.empty() ? "unknown" : offendingJJ_);
    }
    const double oldStep = iObj.transSim.tstep();
    const double newStep = oldStep / 2.0;
    ++stepHalvings_;
    if (verb_ >= 1 || warnStep_) {
        std::cout << "JoSIM: Step halving #" << stepHalvings_ << " triggered by "
                  << (offendingJJ_.empty() ? "JJ" : offendingJJ_) << " (tstep " << oldStep << " -> " << newStep
                  << ")\n";
    }
    iObj.transSim.tstep(newStep);
    Rng::rewind();
    bool tempMinOut = iObj.argMin;
    if (!iObj.argMin) { iObj.argMin = true; }
    Matrix newmObj;
    mObj = newmObj;
    // Create the matrix in csr format
    for (auto& i : mObj.sourcegen) { i.clearMisc(); }
    mObj.create_matrix(iObj);
    find_relevant_traces(iObj, mObj);
    //// Dump expanded Netlist since it is no longer needed
    // iObj.netlist.expNetlist.clear();
    // iObj.netlist.expNetlist.shrink_to_fit();
    if (!tempMinOut) { iObj.argMin = tempMinOut; }
    results.xVector.clear();
    results.timeAxis.clear();
}

void Simulation::setup_b(Matrix& mObj, int64_t i, double step, double factor) {
#ifdef _OPENMP
    const int64_t nJJ  = static_cast<int64_t>(jjs_.size());
    const int64_t nRes = (atyp_ == AnalysisType::Phase) ? static_cast<int64_t>(resistors_.size()) : 0;
    const int64_t nInd = (atyp_ == AnalysisType::Voltage) ? static_cast<int64_t>(inductors_.size()) : 0;
    const int64_t nCap = static_cast<int64_t>(capacitors_.size());
    if (parallel_ && !hasJJThermalNoise_ && !hasResistorThermalNoise_ && factor == 1.0
        && (nJJ + nRes + nInd * 4 + nCap >= 4096)) {
        const int64_t bSize        = static_cast<int64_t>(b_.size());
        const bool    checkPhase   = (static_cast<double>(i) / static_cast<double>(simSize_) > 0.01);
        bool          localNeedsTR = false;
        bool          localNeedsLU = false;
        const int     nThreads     = std::min<int>(16, omp_get_max_threads());

#pragma omp parallel num_threads(nThreads) reduction(|| : localNeedsTR, localNeedsLU)
        {
#pragma omp for schedule(static)
            for (int64_t r = 0; r < bSize; ++r) { b_[r] = 0.0; }

#pragma omp for schedule(static) nowait
            for (int64_t idx = 0; idx < nJJ; ++idx) {
                if (localNeedsTR) { continue; }
                auto&       temp  = jjs_[idx];
                const auto& model = temp.model_;
                if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
                    temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
                } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                    temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
                } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                    temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
                } else {
                    temp.pn1_ = 0.0;
                }
                if (i > 0) {
                    if (atyp_ == AnalysisType::Voltage) {
                        temp.vn1_ = temp.pn1_;
                        temp.pn1_ = x_[temp.variableIndex_];
                    } else if (atyp_ == AnalysisType::Phase) {
                        temp.vn1_ = x_[temp.variableIndex_];
                    }
                }
                double v0  = (5.0 / 2.0) * temp.vn1_ - 2.0 * temp.vn2_ + (1.0 / 2.0) * temp.vn3_;
                temp.phi0_ = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_
                             + ((1.0 / Constants::SIGMA) * ((2.0 * stepSize_) / 3.0)) * v0;
                if (checkPhase && std::abs(temp.phi0_ - temp.pn1_) > (0.20 * 2 * Constants::PI)) {
                    localNeedsTR = true;
                    continue;
                }
                if (atyp_ == AnalysisType::Voltage) {
                    b_[temp.variableIndex_] = Constants::SIGMA
                                              * (-(2.0 / stepSize_) * temp.pn1_ + (1.0 / (2.0 * stepSize_)) * temp.pn2_);
                } else if (atyp_ == AnalysisType::Phase) {
                    b_[temp.variableIndex_] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
                }
                temp.pn4_ = temp.pn3_;
                temp.pn3_ = temp.pn2_;
                temp.pn2_ = temp.pn1_;
                temp.vn6_ = temp.vn5_;
                temp.vn5_ = temp.vn4_;
                temp.vn4_ = temp.vn3_;
                temp.vn3_ = temp.vn2_;
                if (model.rtype() == 1) {
                    if (temp.update_value(v0)) {
                        const double newNz         = temp.matrixInfo.nonZeros_.back();
                        mObj.nz[temp.nzBackIndex_] = newNz;
#ifndef SLU
                        if (usePartitions_) {
                            const int64_t p                      = jjPartIdx_[idx];
                            partitions_[p].nz[jjPartNzIdx_[idx]] = newNz;
                            partitions_[p].needsLU               = true;
                        }
#endif
                        localNeedsLU = true;
                    }
                }
                const auto&  cpr        = model.cpr();
                const double phiOff     = model.phiOff();
                const double ic         = model.ic();
                double       ic_sin_phi = 0.0;
                if (cpr.size() == 1) {
                    ic_sin_phi += ic * (cpr[0] * sin(1 * (temp.phi0_ - phiOff)));
                } else {
                    for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                        ic_sin_phi += ic * (cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff)));
                    }
                }
                temp.cTerm_ = -(((2 * model.c()) / stepSize_) * temp.vn1_) + ((model.c() / (2.0 * stepSize_)) * temp.vn2_);
                if (!model.tDep()) {
                    b_[temp.indexInfo.currentIndex_.value()]
                            = temp.matrixInfo.nonZeros_.back()
                              * (ic_sin_phi - (((2 * model.c()) / stepSize_) * temp.vn1_)
                                 + ((model.c() / (2.0 * stepSize_)) * temp.vn2_) + temp.it_);
                } else {
                    double sin2_half_phi = 0.0;
                    for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                        sin2_half_phi += cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff) / 2);
                    }
                    sin2_half_phi  = sin2_half_phi * sin2_half_phi;
                    double sin_phi = 0.0;
                    for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                        sin_phi += cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff));
                    }
                    double sqrt_part                         = sqrt(1 - model.d() * sin2_half_phi);
                    b_[temp.indexInfo.currentIndex_.value()] = temp.matrixInfo.nonZeros_.back()
                                                               * ((((Constants::PI * temp.del_) / (2 * Constants::EV * model.rn()))
                                                                   * (sin_phi / sqrt_part)
                                                                   * tanh(temp.del_ / (2 * Constants::BOLTZMANN * model.t()) * sqrt_part))
                                                                  - (((2 * model.c()) / stepSize_) * temp.vn1_)
                                                                  + ((model.c() / (2.0 * stepSize_)) * temp.vn2_) + temp.it_);
                }
                temp.vn2_ = temp.vn1_;
            }

            if (nRes > 0) {
#pragma omp for schedule(static) nowait
                for (int64_t idx = 0; idx < nRes; ++idx) {
                    auto&       temp = resistors_[idx];
                    NodeConfig& nc   = temp.indexInfo.nodeConfig_;
                    if (nc == NodeConfig::POSGND) {
                        temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
                    } else if (nc == NodeConfig::GNDNEG) {
                        temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
                    } else if (nc == NodeConfig::POSNEG) {
                        temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
                    } else {
                        temp.pn1_ = 0.0;
                    }
                    b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
                    temp.pn4_                                = temp.pn3_;
                    temp.pn3_                                = temp.pn2_;
                    temp.pn2_                                = temp.pn1_;
                }
            }

            if (nCap > 0) {
#pragma omp for schedule(static) nowait
                for (int64_t idx = 0; idx < nCap; ++idx) {
                    auto& temp = capacitors_[idx];
                    if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
                        temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
                    } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                        temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
                    } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                        temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
                    } else {
                        temp.pn1_ = 0.0;
                    }
                    if (atyp_ == AnalysisType::Voltage) {
                        b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
                    } else if (atyp_ == AnalysisType::Phase) {
                        b_[temp.indexInfo.currentIndex_.value()] = (8.0 / 3.0) * temp.pn1_ - (22.0 / 9.0) * temp.pn2_
                                                                   + (8.0 / 9.0) * temp.pn3_ - (1.0 / 9.0) * temp.pn4_;
                        temp.pn7_ = temp.pn6_;
                        temp.pn6_ = temp.pn5_;
                        temp.pn5_ = temp.pn4_;
                        temp.pn4_ = temp.pn3_;
                        temp.pn3_ = temp.pn2_;
                    }
                    temp.pn2_ = temp.pn1_;
                }
            }

            if (nInd > 0) {
                const double h = stepSize_;
#pragma omp for schedule(static)
                for (int64_t idx = 0; idx < nInd; ++idx) {
                    auto&         temp   = inductors_[idx];
                    const int64_t curIdx = temp.indexInfo.currentIndex_.value();
                    double        bVal   = -(2.0 * temp.netlistInfo.value_ / h) * x_[curIdx]
                                  + ((temp.netlistInfo.value_ / (2.0 * h)) * temp.In2_);
                    for (const auto& m : temp.get_mutualInductance()) {
                        const Inductor& mi   = inductors_[m.first];
                        const int64_t   mCur = mi.indexInfo.currentIndex_.value();
                        const double    mIn2 = (static_cast<int64_t>(m.first) < idx) ? x_[mCur] : mi.In2_;
                        bVal                += (-((2.0 * m.second) / h) * x_[mCur] + (m.second / (2.0 * h)) * mIn2);
                    }
                    b_[curIdx] = bVal;
                }
#pragma omp for schedule(static) nowait
                for (int64_t idx = 0; idx < nInd; ++idx) {
                    auto& temp = inductors_[idx];
                    temp.In4_  = temp.In3_;
                    temp.In3_  = temp.In2_;
                    temp.In2_  = x_[temp.indexInfo.currentIndex_.value()];
                }
            }
        }

        if (localNeedsTR) {
            needsTR_ = true;
            return;
        }
        if (localNeedsLU) {
            refactor_lu(mObj);
            needsLU_ = false;
        }
        handle_cs(mObj, step, i);
        handle_vs(mObj, i, step, factor);
        handle_ps(mObj, i, step, factor);
        handle_ccvs(mObj);
        handle_vccs(mObj);
        handle_tx(mObj, i, step);
        return;
    }
#endif
    // Reset b vector in place without deallocating capacity
    std::fill(b_.begin(), b_.end(), 0.0);
    // Handle jj
    handle_jj(mObj, i, step, factor);
    if (needsTR_) { return; }
    // Re-factorize the LU if any jj transitions (mObj.nz is updated in-place in handle_jj)
    if (needsLU_) {
        refactor_lu(mObj);
        needsLU_ = false;
    }
    // Handle current sources
    handle_cs(mObj, step, i);
    // Handle resistors
    handle_resistors(mObj, step);
    // Handle inductors
    handle_inductors(mObj, factor);
    // Handle capacitors
    handle_capacitors(mObj);
    // Handle voltage sources
    handle_vs(mObj, i, step, factor);
    // Handle phase sources
    handle_ps(mObj, i, step, factor);
    // Handle ccvs
    handle_ccvs(mObj);
    // Handle vccs
    handle_vccs(mObj);
    // Handle transmission lines
    handle_tx(mObj, i, step);
}

void Simulation::handle_cs(Matrix& mObj, double& step, const int64_t& i) {
    for (const auto& j : mObj.components.currentsources) {
        double srcVal = mObj.sourcegen[j.sourceIndex_].value(step);
        if (j.indexInfo.nodeConfig_ == NodeConfig::POSGND) {
            b_[j.indexInfo.posIndex_.value()] -= srcVal;
        } else if (j.indexInfo.nodeConfig_ == NodeConfig::GNDNEG) {
            b_[j.indexInfo.negIndex_.value()] += srcVal;
        } else if (j.indexInfo.nodeConfig_ == NodeConfig::POSNEG) {
            b_[j.indexInfo.posIndex_.value()] -= srcVal;
            b_[j.indexInfo.negIndex_.value()] += srcVal;
        }
    }
}

void Simulation::handle_resistors(Matrix& mObj, double& step) {
    if (resistors_.empty()) { return; }
    if (!hasResistorThermalNoise_) {
        if (atyp_ != AnalysisType::Phase) { return; }
        const int64_t nRes = static_cast<int64_t>(resistors_.size());
#ifdef _OPENMP
        if (parallel_ && nRes >= 8192) {
            const int nThreads = std::min<int>(8, std::min<int>(omp_get_max_threads(), static_cast<int>(nRes / 4096)));
#pragma omp parallel for schedule(static) num_threads(nThreads)
            for (int64_t idx = 0; idx < nRes; ++idx) {
                auto&       temp = resistors_[idx];
                NodeConfig& nc   = temp.indexInfo.nodeConfig_;
                if (nc == NodeConfig::POSGND) {
                    temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
                } else if (nc == NodeConfig::GNDNEG) {
                    temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
                } else if (nc == NodeConfig::POSNEG) {
                    temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
                } else {
                    temp.pn1_ = 0.0;
                }
                b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
                temp.pn4_                                = temp.pn3_;
                temp.pn3_                                = temp.pn2_;
                temp.pn2_                                = temp.pn1_;
            }
            return;
        }
#endif
        for (int64_t idx = 0; idx < nRes; ++idx) {
            auto&       temp = resistors_[idx];
            NodeConfig& nc   = temp.indexInfo.nodeConfig_;
            if (nc == NodeConfig::POSGND) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
            } else if (nc == NodeConfig::GNDNEG) {
                temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
            } else if (nc == NodeConfig::POSNEG) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
            } else {
                temp.pn1_ = 0.0;
            }
            // 4/3 φp1 - 1/3 φp2
            b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            temp.pn4_                                = temp.pn3_;
            temp.pn3_                                = temp.pn2_;
            temp.pn2_                                = temp.pn1_;
        }
        return;
    }

    for (auto& temp : resistors_) {
        NodeConfig& nc = temp.indexInfo.nodeConfig_;
        if (nc == NodeConfig::POSGND) {
            if (temp.thermalNoise) { b_[temp.indexInfo.posIndex_.value()] -= temp.thermalNoise.value().value(step); }
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
        } else if (nc == NodeConfig::GNDNEG) {
            if (temp.thermalNoise) { b_[temp.indexInfo.negIndex_.value()] += temp.thermalNoise.value().value(step); }
            temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
        } else if (nc == NodeConfig::POSNEG) {
            if (temp.thermalNoise) {
                b_[temp.indexInfo.posIndex_.value()] -= temp.thermalNoise.value().value(step);
                b_[temp.indexInfo.negIndex_.value()] += temp.thermalNoise.value().value(step);
            }
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
        } else {
            temp.pn1_ = 0.0;
        }
        if (atyp_ == AnalysisType::Phase) {
            // 4/3 φp1 - 1/3 φp2
            b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            temp.pn4_                                = temp.pn3_;
            temp.pn3_                                = temp.pn2_;
            temp.pn2_                                = temp.pn1_;
        }
    }
}

void Simulation::handle_inductors(Matrix& mObj, double factor) {
    if (atyp_ != AnalysisType::Voltage) { return; }
    const double h = stepSize_ * factor;
    for (auto& temp : inductors_) {
        const int64_t curIdx = temp.indexInfo.currentIndex_.value();
        // -2L/h Ip + L/2h Ip2
        b_[curIdx]           = -(2.0 * temp.netlistInfo.value_ / h) * x_[curIdx]
                     + ((temp.netlistInfo.value_ / (2.0 * h)) * temp.In2_);
        // -2M/h Im + M/2h Im2
        for (const auto& m : temp.get_mutualInductance()) {
            Inductor& mi  = inductors_[m.first];
            b_[curIdx]   += (-((2 * m.second) / h) * x_[mi.indexInfo.currentIndex_.value()]
                           + (m.second / (2.0 * h)) * mi.In2_);
        }
        temp.In4_ = temp.In3_;
        temp.In3_ = temp.In2_;
        temp.In2_ = x_[curIdx];
    }
}

void Simulation::handle_capacitors(Matrix& mObj) {
    if (capacitors_.empty()) { return; }
    const int64_t nCap = static_cast<int64_t>(capacitors_.size());
#ifdef _OPENMP
    if (parallel_ && nCap >= 8192) {
        const int nThreads = std::min<int>(8, std::min<int>(omp_get_max_threads(), static_cast<int>(nCap / 4096)));
#pragma omp parallel for schedule(static) num_threads(nThreads)
        for (int64_t idx = 0; idx < nCap; ++idx) {
            auto& temp = capacitors_[idx];
            if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
            } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
            } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
            } else {
                temp.pn1_ = 0.0;
            }
            if (atyp_ == AnalysisType::Voltage) {
                b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            } else if (atyp_ == AnalysisType::Phase) {
                b_[temp.indexInfo.currentIndex_.value()] = (8.0 / 3.0) * temp.pn1_ - (22.0 / 9.0) * temp.pn2_
                                                           + (8.0 / 9.0) * temp.pn3_ - (1.0 / 9.0) * temp.pn4_;
                temp.pn7_ = temp.pn6_;
                temp.pn6_ = temp.pn5_;
                temp.pn5_ = temp.pn4_;
                temp.pn4_ = temp.pn3_;
                temp.pn3_ = temp.pn2_;
            }
            temp.pn2_ = temp.pn1_;
        }
        return;
    }
#endif
    for (int64_t idx = 0; idx < nCap; ++idx) {
        auto& temp = capacitors_[idx];
        if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
        } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
        } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
        } else {
            temp.pn1_ = 0.0;
        }
        if (atyp_ == AnalysisType::Voltage) {
            // 4/3 Vp1 - 1/3 Vp2
            b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
        } else if (atyp_ == AnalysisType::Phase) {
            // (8/3)φn-1 - (22/9)φn-2 + (8/9)φn-3 - (1/9)φn-4
            b_[temp.indexInfo.currentIndex_.value()] = (8.0 / 3.0) * temp.pn1_ - (22.0 / 9.0) * temp.pn2_
                                                       + (8.0 / 9.0) * temp.pn3_ - (1.0 / 9.0) * temp.pn4_;
            temp.pn7_ = temp.pn6_;
            temp.pn6_ = temp.pn5_;
            temp.pn5_ = temp.pn4_;
            temp.pn4_ = temp.pn3_;
            temp.pn3_ = temp.pn2_;
        }
        temp.pn2_ = temp.pn1_;
    }
}

void Simulation::handle_jj(Matrix& mObj, int64_t& i, double& step, double factor) {
    if (jjs_.empty()) { return; }
    const int64_t nJJ        = static_cast<int64_t>(jjs_.size());
    const bool    checkPhase = ((double) i / (double) simSize_ > 0.01);

#ifdef _OPENMP
    if (parallel_ && !hasJJThermalNoise_ && nJJ >= 8192) {
        const int nThreads     = std::min<int>(8, std::min<int>(omp_get_max_threads(), static_cast<int>(nJJ / 4096)));
        bool      localNeedsTR = false;
        bool      localNeedsLU = false;
#pragma omp parallel for schedule(static) num_threads(nThreads) reduction(|| : localNeedsTR, localNeedsLU)
        for (int64_t idx = 0; idx < nJJ; ++idx) {
            if (localNeedsTR) { continue; }
            auto&       temp  = jjs_[idx];
            const auto& model = temp.model_;
            if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
            } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
            } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
            } else {
                temp.pn1_ = 0.0;
            }
            if (i > 0) {
                if (atyp_ == AnalysisType::Voltage) {
                    temp.vn1_ = temp.pn1_;
                    temp.pn1_ = x_[temp.variableIndex_];
                } else if (atyp_ == AnalysisType::Phase) {
                    temp.vn1_ = x_[temp.variableIndex_];
                }
            }
            double v0  = (5.0 / 2.0) * temp.vn1_ - 2.0 * temp.vn2_ + (1.0 / 2.0) * temp.vn3_;
            temp.phi0_ = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_
                         + ((1.0 / Constants::SIGMA) * ((2.0 * (stepSize_)) / 3.0)) * v0;
            if (checkPhase) {
                if (abs(temp.phi0_ - temp.pn1_) > (0.20 * 2 * Constants::PI)) {
                    localNeedsTR = true;
                    continue;
                }
            }
            if (atyp_ == AnalysisType::Voltage) {
                b_[temp.variableIndex_] = (Constants::SIGMA)
                                          * (-(2.0 / (stepSize_)) * temp.pn1_ + (1.0 / (2.0 * (stepSize_))) * temp.pn2_);
            } else if (atyp_ == AnalysisType::Phase) {
                b_[temp.variableIndex_] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            }
            temp.pn4_ = temp.pn3_;
            temp.pn3_ = temp.pn2_;
            temp.pn2_ = temp.pn1_;
            temp.vn6_ = temp.vn5_;
            temp.vn5_ = temp.vn4_;
            temp.vn4_ = temp.vn3_;
            temp.vn3_ = temp.vn2_;
            if (model.rtype() == 1) {
                auto testLU = temp.update_value(v0);
                if (testLU) {
                    const double newNz         = temp.matrixInfo.nonZeros_.back();
                    mObj.nz[temp.nzBackIndex_] = newNz;
#ifndef SLU
                    if (usePartitions_) {
                        const int64_t p                     = jjPartIdx_[idx];
                        partitions_[p].nz[jjPartNzIdx_[idx]] = newNz;
                        partitions_[p].needsLU              = true;
                    }
#endif
                    localNeedsLU = true;
                }
            }
            const auto&  cpr        = model.cpr();
            const double phiOff     = model.phiOff();
            const double ic         = model.ic();
            double       ic_sin_phi = 0.0;
            if (cpr.size() == 1) {
                ic_sin_phi += ic * (cpr[0] * sin(1 * (temp.phi0_ - phiOff)));
            } else {
                for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                    ic_sin_phi += ic * (cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff)));
                }
            }
            temp.cTerm_ = -(((2 * model.c()) / (stepSize_)) * temp.vn1_)
                          + ((model.c() / (2.0 * (stepSize_))) * temp.vn2_);
            if (!model.tDep()) {
                b_[temp.indexInfo.currentIndex_.value()]
                        = (temp.matrixInfo.nonZeros_.back())
                          * (ic_sin_phi - (((2 * model.c()) / (stepSize_)) * temp.vn1_)
                             + ((model.c() / (2.0 * (stepSize_))) * temp.vn2_) + temp.it_);
            } else {
                double sin2_half_phi = 0.0;
                for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                    sin2_half_phi += cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff) / 2);
                }
                sin2_half_phi  = sin2_half_phi * sin2_half_phi;
                double sin_phi = 0.0;
                for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                    sin_phi += cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff));
                }
                double sqrt_part                         = sqrt(1 - model.d() * sin2_half_phi);
                b_[temp.indexInfo.currentIndex_.value()] = (temp.matrixInfo.nonZeros_.back())
                                                           * (((((Constants::PI * temp.del_) / (2 * Constants::EV * model.rn()))
                                                                * (sin_phi / sqrt_part)
                                                                * tanh(temp.del_ / (2 * Constants::BOLTZMANN * model.t())
                                                                       * sqrt_part))
                                                               - (((2 * model.c()) / stepSize_) * temp.vn1_)
                                                               + ((model.c() / (2.0 * stepSize_)) * temp.vn2_)
                                                               + temp.it_));
            }
            temp.vn2_ = temp.vn1_;
        }
        if (localNeedsTR) {
            needsTR_ = true;
            return;
        }
        if (localNeedsLU) { needsLU_ = true; }
        return;
    }
#endif

    for (int64_t idx = 0; idx < nJJ; ++idx) {
        auto&       temp  = jjs_[idx];
        const auto& model = temp.model_;
        if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
            if (temp.thermalNoise) { b_[temp.indexInfo.posIndex_.value()] -= temp.thermalNoise.value().value(step); }
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
        } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            if (temp.thermalNoise) { b_[temp.indexInfo.negIndex_.value()] += temp.thermalNoise.value().value(step); }
            temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
        } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            if (temp.thermalNoise) {
                b_[temp.indexInfo.posIndex_.value()] -= temp.thermalNoise.value().value(step);
                b_[temp.indexInfo.negIndex_.value()] += temp.thermalNoise.value().value(step);
            }
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
        } else {
            temp.pn1_ = 0.0;
        }
        if (i > 0) {
            if (atyp_ == AnalysisType::Voltage) {
                temp.vn1_ = temp.pn1_;
                temp.pn1_ = x_[temp.variableIndex_];
            } else if (atyp_ == AnalysisType::Phase) {
                temp.vn1_ = x_[temp.variableIndex_];
                temp.pn1_ = temp.pn1_;
            }
        }
        // Guess voltage (V0)
        double v0  = (5.0 / 2.0) * temp.vn1_ - 2.0 * temp.vn2_ + (1.0 / 2.0) * temp.vn3_;
        // Phase guess (P0)
        temp.phi0_ = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_
                     + ((1.0 / Constants::SIGMA) * ((2.0 * (stepSize_)) / 3.0)) * v0;
        // Ensure timestep is not too large
        if (checkPhase) {
            if (abs(temp.phi0_ - temp.pn1_) > (0.20 * 2 * Constants::PI)) {
                offendingJJ_ = temp.netlistInfo.label_;
                needsTR_     = true;
                return;
                /*Errors::simulation_errors(
                  SimulationErrors::PHASEGUESS_TOO_LARGE, temp.netlistInfo.label_);*/
            }
        }
        // (hbar / 2 * e) ( -(2 / h) φp1 + (1 / 2h) φp2 )
        if (atyp_ == AnalysisType::Voltage) {
            b_[temp.variableIndex_]
                    = (Constants::SIGMA) * (-(2.0 / (stepSize_)) * temp.pn1_ + (1.0 / (2.0 * (stepSize_))) * temp.pn2_);
            // (4 / 3) φp1 - (1/3) φp2
        } else if (atyp_ == AnalysisType::Phase) {
            b_[temp.variableIndex_] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
        }
        temp.pn4_ = temp.pn3_;
        temp.pn3_ = temp.pn2_;
        temp.pn2_ = temp.pn1_;
        temp.vn6_ = temp.vn5_;
        temp.vn5_ = temp.vn4_;
        temp.vn4_ = temp.vn3_;
        temp.vn3_ = temp.vn2_;
        // Update junction transition
        if (model.rtype() == 1) {
            auto testLU = temp.update_value(v0);
            if (testLU) {
                const double newNz         = temp.matrixInfo.nonZeros_.back();
                mObj.nz[temp.nzBackIndex_] = newNz;
#ifndef SLU
                if (usePartitions_) {
                    const int64_t p                      = jjPartIdx_[idx];
                    partitions_[p].nz[jjPartNzIdx_[idx]] = newNz;
                    partitions_[p].needsLU               = true;
                }
#endif
                if (!needsLU_) { needsLU_ = true; }
            }
        }
        // Ic * sin (phi * (φ0 - φ))
        const auto&  cpr        = model.cpr();
        const double phiOff     = model.phiOff();
        const double ic         = model.ic();
        double       ic_sin_phi = 0.0;
        if (cpr.size() == 1) {
            ic_sin_phi += ic * (cpr[0] * sin(1 * (temp.phi0_ - phiOff)));
        } else {
            for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                ic_sin_phi += ic * (cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff)));
            }
        }
        temp.cTerm_ = -(((2 * model.c()) / (stepSize_)) * temp.vn1_)
                      + ((model.c() / (2.0 * (stepSize_))) * temp.vn2_);
        if (!model.tDep()) {
            // -(hR / h + 2RC) * (Ic sin (φ0) - 2C / h Vp1 + C/2h Vp2 + It)
            b_[temp.indexInfo.currentIndex_.value()]
                    = (temp.matrixInfo.nonZeros_.back())
                      * (ic_sin_phi - (((2 * model.c()) / (stepSize_)) * temp.vn1_)
                         + ((model.c() / (2.0 * (stepSize_))) * temp.vn2_) + temp.it_);
        } else {
            double sin2_half_phi = 0.0;
            for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                sin2_half_phi += cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff) / 2);
            }
            sin2_half_phi  = sin2_half_phi * sin2_half_phi;
            double sin_phi = 0.0;
            for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                sin_phi += cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff));
            }
            double sqrt_part                            = sqrt(1 - model.d() * sin2_half_phi);
            b_[temp.indexInfo.currentIndex_.value()] =
                    // -(hR / h + 2RC) *(
                    (temp.matrixInfo.nonZeros_.back())
                    * ((
                               // (π * Δ / 2 * e * Rn)
                               ((Constants::PI * temp.del_) / (2 * Constants::EV * model.rn()))
                               // * (sin(φ0 - φ) / √(1 - D * sin²((φ0 - φ) / 2))
                               * (sin_phi / sqrt_part)
                               // * tanh(Δ / (2 * kB * T) * √(1 - D * sin²((φ0 - φ) / 2)))
                               * tanh(temp.del_ / (2 * Constants::BOLTZMANN * model.t()) * sqrt_part))
                       // - 2C / h Vp1
                       - (((2 * model.c()) / stepSize_) * temp.vn1_)
                       // + C/2h Vp2
                       + ((model.c() / (2.0 * stepSize_)) * temp.vn2_)
                       // + It)
                       + temp.it_);
        }
        temp.vn2_ = temp.vn1_;
    }
}

void Simulation::corrector_jj(Matrix& mObj) {
    const int64_t nJJ = static_cast<int64_t>(jjs_.size());
    for (int64_t iter = 0; iter < correctorIter_; ++iter) {
        double maxDiff     = 0.0;
        bool   corrNeedsLU = false;
        for (int64_t idx = 0; idx < nJJ; ++idx) {
            auto&  temp    = jjs_[idx];
            double nodeVal = 0.0;
            if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
                nodeVal = x_[temp.indexInfo.posIndex_.value()];
            } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                nodeVal = -x_[temp.indexInfo.negIndex_.value()];
            } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
                nodeVal = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
            }
            const double phiNew = (atyp_ == AnalysisType::Phase) ? nodeVal : x_[temp.variableIndex_];
            const double vNew   = (atyp_ == AnalysisType::Phase) ? x_[temp.variableIndex_] : nodeVal;
            const double diff   = std::abs(phiNew - temp.phi0_);
            if (diff > maxDiff) { maxDiff = diff; }
            temp.phi0_        = 0.5 * (temp.phi0_ + phiNew);
            const auto& model = temp.model_;
            if (model.rtype() == 1) {
                if (temp.update_value(vNew)) {
                    const double newNz         = temp.matrixInfo.nonZeros_.back();
                    mObj.nz[temp.nzBackIndex_] = newNz;
#ifndef SLU
                    if (usePartitions_) {
                        const int64_t p                      = jjPartIdx_[idx];
                        partitions_[p].nz[jjPartNzIdx_[idx]] = newNz;
                        partitions_[p].needsLU               = true;
                    }
#endif
                    corrNeedsLU = true;
                }
            }
            const auto&  cpr        = model.cpr();
            const double phiOff     = model.phiOff();
            const double ic         = model.ic();
            double       ic_sin_phi = 0.0;
            if (cpr.size() == 1) {
                ic_sin_phi += ic * (cpr[0] * sin(1 * (temp.phi0_ - phiOff)));
            } else {
                for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                    ic_sin_phi += ic * (cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff)));
                }
            }
            if (!model.tDep()) {
                b_[temp.indexInfo.currentIndex_.value()]
                        = (temp.matrixInfo.nonZeros_.back()) * (ic_sin_phi + temp.cTerm_ + temp.it_);
            } else {
                double sin2_half_phi = 0.0;
                for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                    sin2_half_phi += cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff) / 2);
                }
                sin2_half_phi  = sin2_half_phi * sin2_half_phi;
                double sin_phi = 0.0;
                for (int harm = 0; harm < static_cast<int>(cpr.size()); ++harm) {
                    sin_phi += cpr[harm] * sin((harm + 1) * (temp.phi0_ - phiOff));
                }
                double sqrt_part = sqrt(1 - model.d() * sin2_half_phi);
                b_[temp.indexInfo.currentIndex_.value()]
                        = (temp.matrixInfo.nonZeros_.back())
                          * ((((Constants::PI * temp.del_) / (2 * Constants::EV * model.rn())) * (sin_phi / sqrt_part)
                              * tanh(temp.del_ / (2 * Constants::BOLTZMANN * model.t()) * sqrt_part))
                             + temp.cTerm_ + temp.it_);
            }
        }
        if (maxDiff <= correctorTol_ && !corrNeedsLU) { break; }
        if (corrNeedsLU) { refactor_lu(mObj); }
        solve_lu(mObj);
    }
}

void Simulation::handle_vs(Matrix& mObj, const int64_t& i, double& step, double factor) {
    for (auto& temp : vsources_) {
        JoSIM::NodeConfig& nc = temp.indexInfo.nodeConfig_;
        if (atyp_ == AnalysisType::Voltage) {
            // Vn
            b_[temp.indexInfo.currentIndex_.value()] = mObj.sourcegen[temp.sourceIndex_].value(step);
        } else if (atyp_ == AnalysisType::Phase) {
            if (nc == NodeConfig::POSGND) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
            } else if (nc == NodeConfig::GNDNEG) {
                temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
            } else if (nc == NodeConfig::POSNEG) {
                temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
            } else {
                // This branch is only taken as a result of a GND-GND short-circuited voltage source.
                // In this case, the matrix is singular and the program will throw an error.
                // Without this branch, the program could crash due to std::bad_optional_access.
                temp.pn1_ = 0.0;
            }
            // (2e/hbar)(2h/3)Vn + (4/3)φn-1 - (1/3)φn-2
            b_[temp.indexInfo.currentIndex_.value()] = ((2 * stepSize_ * factor) / (3 * Constants::SIGMA))
                                                               * (mObj.sourcegen[temp.sourceIndex_].value(step))
                                                       + (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
            temp.pn4_ = temp.pn3_;
            temp.pn3_ = temp.pn2_;
            temp.pn2_ = temp.pn1_;
        }
    }
}

void Simulation::handle_ps(Matrix& mObj, const int64_t& i, double& step, double factor) {
    for (auto& temp : psources_) {
        if (atyp_ == AnalysisType::Phase) {
            // φn
            b_[temp.indexInfo.currentIndex_.value()] = mObj.sourcegen[temp.sourceIndex_].value(step);
        } else if (atyp_ == AnalysisType::Voltage) {
            if (i == 0) {
                b_[temp.indexInfo.currentIndex_.value()]
                        = (Constants::SIGMA / (stepSize_ * factor))
                          * ((3.0 / 2.0) * (mObj.sourcegen[temp.sourceIndex_].value(step)));
            } else if (i == 1) {
                b_[temp.indexInfo.currentIndex_.value()]
                        = (Constants::SIGMA / (stepSize_ * factor))
                          * ((3.0 / 2.0) * (mObj.sourcegen[temp.sourceIndex_].value(step))
                             - 2.0 * (mObj.sourcegen[temp.sourceIndex_].value(step - (stepSize_ * factor))));
            } else {
                b_[temp.indexInfo.currentIndex_.value()]
                        = (Constants::SIGMA / (stepSize_ * factor))
                          * ((3.0 / 2.0) * (mObj.sourcegen[temp.sourceIndex_].value(step))
                             - 2.0 * (mObj.sourcegen[temp.sourceIndex_].value(step - (stepSize_ * factor)))
                             + 0.5 * (mObj.sourcegen[temp.sourceIndex_].value(step - (2 * stepSize_ * factor))));
            }
        }
    }
}

void Simulation::handle_ccvs(Matrix& mObj) {
    if (atyp_ != AnalysisType::Phase) { return; }
    for (auto& temp : ccvss_) {
        if (temp.indexInfo.posIndex_ && !temp.indexInfo.negIndex_) {
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()];
        } else if (!temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            temp.pn1_ = -x_[temp.indexInfo.negIndex_.value()];
        } else if (temp.indexInfo.posIndex_ && temp.indexInfo.negIndex_) {
            temp.pn1_ = x_[temp.indexInfo.posIndex_.value()] - x_[temp.indexInfo.negIndex_.value()];
        } else {
            temp.pn1_ = 0.0;
        }
        b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
        temp.pn4_                                = temp.pn3_;
        temp.pn3_                                = temp.pn2_;
        temp.pn2_                                = temp.pn1_;
    }
}

void Simulation::handle_vccs(Matrix& mObj) {
    if (atyp_ != AnalysisType::Phase) { return; }
    for (auto& temp : vccss_) {
        if (temp.posIndex2_ && !temp.negIndex2_) {
            temp.pn1_ = x_[temp.posIndex2_.value()];
        } else if (!temp.posIndex2_ && temp.negIndex2_) {
            temp.pn1_ = -x_[temp.negIndex2_.value()];
        } else if (temp.posIndex2_ && temp.negIndex2_) {
            temp.pn1_ = x_[temp.posIndex2_.value()] - x_[temp.negIndex2_.value()];
        } else {
            temp.pn1_ = 0.0;
        }
        b_[temp.indexInfo.currentIndex_.value()] = (4.0 / 3.0) * temp.pn1_ - (1.0 / 3.0) * temp.pn2_;
        temp.pn4_                                = temp.pn3_;
        temp.pn3_                                = temp.pn2_;
        temp.pn2_                                = temp.pn1_;
    }
}

void Simulation::handle_tx(Matrix& mObj, const int64_t& i, double& step, double factor) {
    (void) mObj;
    (void) step;
    for (auto& temp : txlines_) {
        // Z0
        double&            Z       = temp.netlistInfo.value_;
        // Td == k
        int64_t&           k       = temp.timestepDelay_;
        // Shorthands
        JoSIM::NodeConfig &nc      = temp.indexInfo.nodeConfig_, &nc2 = temp.nodeConfig2_;
        int_o &            posInd  = temp.indexInfo.posIndex_, &negInd = temp.indexInfo.negIndex_, &posInd2 = temp.posIndex2_,
              &negInd2             = temp.negIndex2_;
        int64_t &          curInd  = temp.indexInfo.currentIndex_.value(), &curInd2 = temp.currentIndex2_;
        const int64_t      effStep = temp.loadedSteps_ + i;

        // Record the completed previous step (i - 1) into the transmission line's circular ring buffer
        if (i > 0) {
            double v1_prev = 0.0;
            if (nc == NodeConfig::POSGND) {
                v1_prev = x_[posInd.value()];
            } else if (nc == NodeConfig::GNDNEG) {
                v1_prev = -x_[negInd.value()];
            } else if (nc == NodeConfig::POSNEG) {
                v1_prev = x_[posInd.value()] - x_[negInd.value()];
            }
            double v2_prev = 0.0;
            if (nc2 == NodeConfig::POSGND) {
                v2_prev = x_[posInd2.value()];
            } else if (nc2 == NodeConfig::GNDNEG) {
                v2_prev = -x_[negInd2.value()];
            } else if (nc2 == NodeConfig::POSNEG) {
                v2_prev = x_[posInd2.value()] - x_[negInd2.value()];
            }
            temp.hist_[(effStep - 1) % temp.histSize_] = {v1_prev, v2_prev, x_[curInd], x_[curInd2]};
            if (atyp_ == AnalysisType::Phase) {
                temp.n2_1_ = temp.n1_1_;
                temp.n2_2_ = temp.n1_2_;
                temp.n1_1_ = v1_prev;
                temp.n1_2_ = v2_prev;
            }
        }

        if (temp.fracDelay_ > 0.0) {
            const double alpha     = temp.fracDelay_;
            const double oma       = 1.0 - alpha;
            auto         interp_v1 = [&](int64_t m) -> double {
                if (m < 0) { return 0.0; }
                const double s0 = temp.hist_[m % temp.histSize_].v1;
                const double s1 = (m >= 1) ? temp.hist_[(m - 1) % temp.histSize_].v1 : 0.0;
                return oma * s0 + alpha * s1;
            };
            auto interp_v2 = [&](int64_t m) -> double {
                if (m < 0) { return 0.0; }
                const double s0 = temp.hist_[m % temp.histSize_].v2;
                const double s1 = (m >= 1) ? temp.hist_[(m - 1) % temp.histSize_].v2 : 0.0;
                return oma * s0 + alpha * s1;
            };
            auto interp_i1 = [&](int64_t m) -> double {
                if (m < 0) { return 0.0; }
                const double s0 = temp.hist_[m % temp.histSize_].i1;
                const double s1 = (m >= 1) ? temp.hist_[(m - 1) % temp.histSize_].i1 : 0.0;
                return oma * s0 + alpha * s1;
            };
            auto interp_i2 = [&](int64_t m) -> double {
                if (m < 0) { return 0.0; }
                const double s0 = temp.hist_[m % temp.histSize_].i2;
                const double s1 = (m >= 1) ? temp.hist_[(m - 1) % temp.histSize_].i2 : 0.0;
                return oma * s0 + alpha * s1;
            };
            if (atyp_ == AnalysisType::Voltage) {
                if (effStep >= k) {
                    const int64_t m    = effStep - k;
                    temp.nk_1_         = interp_v1(m);
                    temp.nk_2_         = interp_v2(m);
                    const double  I1nk = interp_i1(m);
                    const double  I2nk = interp_i2(m);
                    b_[curInd]         = Z * I2nk + temp.nk_2_;
                    b_[curInd2]        = Z * I1nk + temp.nk_1_;
                }
            } else if (atyp_ == AnalysisType::Phase) {
                if (effStep >= k) {
                    const int64_t m    = effStep - k;
                    temp.nk_1_         = interp_v1(m);
                    temp.nk_2_         = interp_v2(m);
                    temp.nk1_1_        = interp_v1(m - 1);
                    temp.nk1_2_        = interp_v2(m - 1);
                    temp.nk2_1_        = interp_v1(m - 2);
                    temp.nk2_2_        = interp_v2(m - 2);
                    const double  I1nk = interp_i1(m);
                    const double  I2nk = interp_i2(m);
                    b_[curInd]         = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I2nk
                                 + (4.0 / 3.0) * temp.n1_1_ - (1.0 / 3.0) * temp.n2_1_ + temp.nk_2_
                                 - (4.0 / 3.0) * temp.nk1_2_ + (1.0 / 3.0) * temp.nk2_2_;
                    b_[curInd2] = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I1nk
                                  + (4.0 / 3.0) * temp.n1_2_ - (1.0 / 3.0) * temp.n2_2_ + temp.nk_1_
                                  - (4.0 / 3.0) * temp.nk1_1_ + (1.0 / 3.0) * temp.nk2_1_;
                } else {
                    b_[curInd]  = (4.0 / 3.0) * temp.n1_1_ - (1.0 / 3.0) * temp.n2_1_;
                    b_[curInd2] = (4.0 / 3.0) * temp.n1_2_ - (1.0 / 3.0) * temp.n2_2_;
                }
            }
            continue;
        }
        if (atyp_ == AnalysisType::Voltage) {
            if (effStep >= k) {
                const auto& h_k = temp.hist_[(effStep - k) % temp.histSize_];
                temp.nk_1_      = h_k.v1;
                temp.nk_2_      = h_k.v2;
                // I1 = ZI2n-k + V2n-k
                b_[curInd]      = Z * h_k.i2 + temp.nk_2_;
                // I2 = ZI1n-k + V1n-k
                b_[curInd2]     = Z * h_k.i1 + temp.nk_1_;
            }
        } else if (atyp_ == AnalysisType::Phase) {
            if (effStep >= k) {
                const int64_t m    = effStep - k;
                const auto&   h_k  = temp.hist_[m % temp.histSize_];
                temp.nk_1_         = h_k.v1;
                temp.nk_2_         = h_k.v2;
                const double  I1nk = h_k.i1;
                const double  I2nk = h_k.i2;
                if (effStep == k) {
                    // I1 = Z(2e/hbar)(2h/3)I2n-k + (4/3)φ1n-1 - (1/3)φ1n-2 + φ2n-k
                    b_[curInd]  = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I2nk
                                 + (4.0 / 3.0) * temp.n1_1_ - (1.0 / 3.0) * temp.n2_1_ + temp.nk_2_;
                    // I2 = Z(2e/hbar)(2h/3)I1n-k + (4/3)φ2n-1 - (1/3)φ2n-2 + φ1n-k
                    b_[curInd2] = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I1nk
                                  + (4.0 / 3.0) * temp.n1_2_ - (1.0 / 3.0) * temp.n2_2_ + temp.nk_1_;
                } else if (effStep == k + 1) {
                    const auto& h_k1 = temp.hist_[(m - 1) % temp.histSize_];
                    temp.nk1_1_      = h_k1.v1;
                    temp.nk1_2_      = h_k1.v2;
                    // I1 = Z(2e/hbar)(2h/3)I2n-k + (4/3)φ1n-1 - (1/3)φ1n-2 +
                    //      φ2n-k - (4/3)φ2n-k-1
                    b_[curInd]       = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I2nk
                                 + (4.0 / 3.0) * temp.n1_1_ - (1.0 / 3.0) * temp.n2_1_ + temp.nk_2_
                                 - (4.0 / 3.0) * temp.nk1_2_;
                    // I2 = Z(2e/hbar)(2h/3)I1n-k + (4/3)φ2n-1 - (1/3)φ2n-2 +
                    //      φ1n-k - (4/3)φ1n-k-1
                    b_[curInd2] = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I1nk
                                  + (4.0 / 3.0) * temp.n1_2_ - (1.0 / 3.0) * temp.n2_2_ + temp.nk_1_
                                  - (4.0 / 3.0) * temp.nk1_1_;
                } else if (effStep > k + 1) {
                    const auto& h_k1 = temp.hist_[(m - 1) % temp.histSize_];
                    const auto& h_k2 = temp.hist_[(m - 2) % temp.histSize_];
                    temp.nk1_1_      = h_k1.v1;
                    temp.nk1_2_      = h_k1.v2;
                    temp.nk2_1_      = h_k2.v1;
                    temp.nk2_2_      = h_k2.v2;
                    // I1 = Z(2e/hbar)(2h/3)I2n-k + (4/3)φ1n-1 - (1/3)φ1n-2 +
                    //      φ2n-k - (4/3)φ2n-k-1 + (1/3)φ2n-k-2
                    b_[curInd]       = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I2nk
                                 + (4.0 / 3.0) * temp.n1_1_ - (1.0 / 3.0) * temp.n2_1_ + temp.nk_2_
                                 - (4.0 / 3.0) * temp.nk1_2_ + (1.0 / 3.0) * temp.nk2_2_;
                    // I2 = Z(2e/hbar)(2h/3)I1n-k + (4/3)φ2n-1 - (1/3)φ2n-2 +
                    //      φ1n-k - (4/3)φ1n-k-1 + (1/3)φ1n-k-2
                    b_[curInd2] = (Z / Constants::SIGMA) * ((2.0 * stepSize_ * factor) / 3.0) * I1nk
                                  + (4.0 / 3.0) * temp.n1_2_ - (1.0 / 3.0) * temp.n2_2_ + temp.nk_1_
                                  - (4.0 / 3.0) * temp.nk1_1_ + (1.0 / 3.0) * temp.nk2_1_;
                }
            } else {
                // I1 = (4/3)φ1n-1 - (1/3)φ1n-2
                b_[curInd]  = (4.0 / 3.0) * temp.n1_1_ - (1.0 / 3.0) * temp.n2_1_;
                // I2 = (4/3)φ2n-1 - (1/3)φ2n-2
                b_[curInd2] = (4.0 / 3.0) * temp.n1_2_ - (1.0 / 3.0) * temp.n2_2_;
            }
        }
    }
}

void Simulation::save_state(const std::string& filename, const Matrix& mObj, int64_t currentStep) const {
    std::ofstream out(filename, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
        Errors::output_errors(OutputErrors::CANNOT_OPEN_FILE, filename);
        return;
    }
    const char     magic[8] = {'J', 'O', 'S', 'I', 'M', 'S', 'T', 'A'};
    const uint32_t version  = 2;
    out.write(magic, 8);
    out.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    auto write_u64 = [&](uint64_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(uint64_t)); };
    auto write_f64 = [&](double v) { out.write(reinterpret_cast<const char*>(&v), sizeof(double)); };

    write_u64(static_cast<uint64_t>(x_.size()));
    out.write(reinterpret_cast<const char*>(x_.data()), static_cast<std::streamsize>(x_.size() * sizeof(double)));

    write_u64(static_cast<uint64_t>(jjs_.size()));
    for (const auto& jj : jjs_) {
        const double vals[13] = {jj.pn1_,
                                 jj.pn2_,
                                 jj.pn3_,
                                 jj.pn4_,
                                 jj.vn1_,
                                 jj.vn2_,
                                 jj.vn3_,
                                 jj.vn4_,
                                 jj.vn5_,
                                 jj.vn6_,
                                 jj.phi0_,
                                 jj.it_,
                                 jj.matrixInfo.nonZeros_.back()};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(inductors_.size()));
    for (const auto& ind : inductors_) {
        const double vals[3] = {ind.In2_, ind.In3_, ind.In4_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(resistors_.size()));
    for (const auto& res : resistors_) {
        const double vals[4] = {res.pn1_, res.pn2_, res.pn3_, res.pn4_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(capacitors_.size()));
    for (const auto& cap : capacitors_) {
        const double vals[7] = {cap.pn1_, cap.pn2_, cap.pn3_, cap.pn4_, cap.pn5_, cap.pn6_, cap.pn7_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(vsources_.size()));
    for (const auto& vs : vsources_) {
        const double vals[4] = {vs.pn1_, vs.pn2_, vs.pn3_, vs.pn4_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(ccvss_.size()));
    for (const auto& ccvs : ccvss_) {
        const double vals[4] = {ccvs.pn1_, ccvs.pn2_, ccvs.pn3_, ccvs.pn4_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(vccss_.size()));
    for (const auto& vccs : vccss_) {
        const double vals[4] = {vccs.pn1_, vccs.pn2_, vccs.pn3_, vccs.pn4_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    write_u64(static_cast<uint64_t>(txlines_.size()));
    for (const auto& tx : txlines_) {
        std::vector<TransmissionLine::TxHistEntry> histCopy = tx.hist_;
        double                                     n1_1     = tx.n1_1_;
        double                                     n1_2     = tx.n1_2_;
        double                                     n2_1     = tx.n2_1_;
        double                                     n2_2     = tx.n2_2_;
        int64_t                                    effSteps = tx.loadedSteps_;
        if (currentStep >= 0 && tx.histSize_ > 0) {
            const JoSIM::NodeConfig nc     = tx.indexInfo.nodeConfig_;
            const JoSIM::NodeConfig nc2    = tx.nodeConfig2_;
            double                  v1Curr = 0.0;
            if (nc == NodeConfig::POSGND) {
                v1Curr = x_[tx.indexInfo.posIndex_.value()];
            } else if (nc == NodeConfig::GNDNEG) {
                v1Curr = -x_[tx.indexInfo.negIndex_.value()];
            } else if (nc == NodeConfig::POSNEG) {
                v1Curr = x_[tx.indexInfo.posIndex_.value()] - x_[tx.indexInfo.negIndex_.value()];
            }
            double v2Curr = 0.0;
            if (nc2 == NodeConfig::POSGND) {
                v2Curr = x_[tx.posIndex2_.value()];
            } else if (nc2 == NodeConfig::GNDNEG) {
                v2Curr = -x_[tx.negIndex2_.value()];
            } else if (nc2 == NodeConfig::POSNEG) {
                v2Curr = x_[tx.posIndex2_.value()] - x_[tx.negIndex2_.value()];
            }
            const int64_t slot = (tx.loadedSteps_ + currentStep) % tx.histSize_;
            histCopy[slot] = {v1Curr, v2Curr, x_[tx.indexInfo.currentIndex_.value()], x_[tx.currentIndex2_]};
            effSteps       = tx.loadedSteps_ + currentStep + 1;
            n2_1           = tx.n1_1_;
            n2_2           = tx.n1_2_;
            n1_1           = v1Curr;
            n1_2           = v2Curr;
        }
        const double vals[10] = {n1_1,
                                 n1_2,
                                 n2_1,
                                 n2_2,
                                 tx.nk_1_,
                                 tx.nk_2_,
                                 tx.nk1_1_,
                                 tx.nk1_2_,
                                 tx.nk2_1_,
                                 tx.nk2_2_};
        out.write(reinterpret_cast<const char*>(vals), sizeof(vals));
        write_u64(static_cast<uint64_t>(effSteps));
        write_u64(static_cast<uint64_t>(histCopy.size()));
        if (!histCopy.empty()) {
            out.write(reinterpret_cast<const char*>(histCopy.data()),
                      static_cast<std::streamsize>(histCopy.size() * sizeof(TransmissionLine::TxHistEntry)));
        }
    }
    (void) write_f64;
    (void) mObj;
}

void Simulation::load_state(const std::string& filename, Matrix& mObj) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        Errors::input_errors(InputErrors::CANNOT_OPEN_FILE, filename);
        return;
    }
    char     magic[8] = {};
    uint32_t version  = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (std::string(magic, 8) != "JOSIMSTA" || (version != 1 && version != 2)) {
        Errors::input_errors(InputErrors::CANNOT_OPEN_FILE, filename);
        return;
    }
    auto read_u64 = [&]() -> uint64_t {
        uint64_t v = 0;
        in.read(reinterpret_cast<char*>(&v), sizeof(uint64_t));
        return v;
    };

    const uint64_t xLen = read_u64();
    if (xLen == x_.size()) {
        in.read(reinterpret_cast<char*>(x_.data()), static_cast<std::streamsize>(xLen * sizeof(double)));
    } else {
        std::vector<double> tmp(xLen);
        in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(xLen * sizeof(double)));
        const size_t copyLen = std::min<size_t>(x_.size(), tmp.size());
        std::copy(tmp.begin(), tmp.begin() + copyLen, x_.begin());
    }

    const uint64_t nJJ      = read_u64();
    bool           nzChange = false;
    for (uint64_t i = 0; i < nJJ; ++i) {
        double vals[13] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < jjs_.size()) {
            auto& jj = jjs_[i];
            jj.pn1_  = vals[0];
            jj.pn2_  = vals[1];
            jj.pn3_  = vals[2];
            jj.pn4_  = vals[3];
            jj.vn1_  = vals[4];
            jj.vn2_  = vals[5];
            jj.vn3_  = vals[6];
            jj.vn4_  = vals[7];
            jj.vn5_  = vals[8];
            jj.vn6_  = vals[9];
            jj.phi0_ = vals[10];
            jj.it_   = vals[11];
            if (jj.matrixInfo.nonZeros_.back() != vals[12]) {
                jj.matrixInfo.nonZeros_.back() = vals[12];
                mObj.nz[jj.nzBackIndex_]       = vals[12];
#ifndef SLU
                if (usePartitions_) {
                    const int64_t p               = jjPartIdx_[i];
                    partitions_[p].nz[jjPartNzIdx_[i]] = vals[12];
                    partitions_[p].needsLU        = true;
                }
#endif
                nzChange = true;
            }
        }
    }

    const uint64_t nInd = read_u64();
    for (uint64_t i = 0; i < nInd; ++i) {
        double vals[3] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < inductors_.size()) {
            inductors_[i].In2_ = vals[0];
            inductors_[i].In3_ = vals[1];
            inductors_[i].In4_ = vals[2];
        }
    }

    const uint64_t nRes = read_u64();
    for (uint64_t i = 0; i < nRes; ++i) {
        double vals[4] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < resistors_.size()) {
            resistors_[i].pn1_ = vals[0];
            resistors_[i].pn2_ = vals[1];
            resistors_[i].pn3_ = vals[2];
            resistors_[i].pn4_ = vals[3];
        }
    }

    const uint64_t nCap = read_u64();
    for (uint64_t i = 0; i < nCap; ++i) {
        double vals[7] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < capacitors_.size()) {
            capacitors_[i].pn1_ = vals[0];
            capacitors_[i].pn2_ = vals[1];
            capacitors_[i].pn3_ = vals[2];
            capacitors_[i].pn4_ = vals[3];
            capacitors_[i].pn5_ = vals[4];
            capacitors_[i].pn6_ = vals[5];
            capacitors_[i].pn7_ = vals[6];
        }
    }

    const uint64_t nVS = read_u64();
    for (uint64_t i = 0; i < nVS; ++i) {
        double vals[4] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < vsources_.size()) {
            vsources_[i].pn1_ = vals[0];
            vsources_[i].pn2_ = vals[1];
            vsources_[i].pn3_ = vals[2];
            vsources_[i].pn4_ = vals[3];
        }
    }

    const uint64_t nCCVS = read_u64();
    for (uint64_t i = 0; i < nCCVS; ++i) {
        double vals[4] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < ccvss_.size()) {
            ccvss_[i].pn1_ = vals[0];
            ccvss_[i].pn2_ = vals[1];
            ccvss_[i].pn3_ = vals[2];
            ccvss_[i].pn4_ = vals[3];
        }
    }

    const uint64_t nVCCS = read_u64();
    for (uint64_t i = 0; i < nVCCS; ++i) {
        double vals[4] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        if (i < vccss_.size()) {
            vccss_[i].pn1_ = vals[0];
            vccss_[i].pn2_ = vals[1];
            vccss_[i].pn3_ = vals[2];
            vccss_[i].pn4_ = vals[3];
        }
    }

    const uint64_t nTX = read_u64();
    for (uint64_t i = 0; i < nTX; ++i) {
        double vals[10] = {};
        in.read(reinterpret_cast<char*>(vals), sizeof(vals));
        uint64_t                                   effSteps = 0;
        std::vector<TransmissionLine::TxHistEntry> histRead;
        if (version >= 2) {
            effSteps                = read_u64();
            const uint64_t histSize = read_u64();
            histRead.resize(histSize);
            if (histSize > 0) {
                in.read(reinterpret_cast<char*>(histRead.data()),
                        static_cast<std::streamsize>(histSize * sizeof(TransmissionLine::TxHistEntry)));
            }
        }
        if (i < txlines_.size()) {
            txlines_[i].n1_1_  = vals[0];
            txlines_[i].n1_2_  = vals[1];
            txlines_[i].n2_1_  = vals[2];
            txlines_[i].n2_2_  = vals[3];
            txlines_[i].nk_1_  = vals[4];
            txlines_[i].nk_2_  = vals[5];
            txlines_[i].nk1_1_ = vals[6];
            txlines_[i].nk1_2_ = vals[7];
            txlines_[i].nk2_1_ = vals[8];
            txlines_[i].nk2_2_ = vals[9];
            if (version >= 2 && histRead.size() == static_cast<size_t>(txlines_[i].histSize_)) {
                txlines_[i].loadedSteps_ = static_cast<int64_t>(effSteps);
                txlines_[i].hist_        = std::move(histRead);
            }
        }
    }

    if (nzChange) { refactor_lu(mObj); }
}
